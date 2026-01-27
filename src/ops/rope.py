import torch
import triton
import triton.language as tl

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def rope_ref(q, k, freqs):
    # freqs shape: (S, D//2) -> broadcast to (1, 1, S, D)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


@triton.jit
def _rope_forward_kernel(
    q_ptr, k_ptr, freqs_ptr,
    q_out_ptr, k_out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_fs, stride_fd,
    S, H, D,
    D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % S
    h = (pid // S) % H
    b = pid // (S * H)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D_HALF

    # Load cos/sin freqs
    freqs = tl.load(freqs_ptr + s * stride_fs + offs * stride_fd, mask=mask)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    # Offsets
    q_low_ptr = q_ptr + b * stride_qb + h * stride_qh + s * stride_qs + offs * stride_qd
    q_high_ptr = q_low_ptr + D_HALF * stride_qd
    k_low_ptr = k_ptr + b * stride_kb + h * stride_kh + s * stride_ks + offs * stride_kd
    k_high_ptr = k_low_ptr + D_HALF * stride_kd

    q1 = tl.load(q_low_ptr, mask=mask)
    q2 = tl.load(q_high_ptr, mask=mask)
    k1 = tl.load(k_low_ptr, mask=mask)
    k2 = tl.load(k_high_ptr, mask=mask)

    # Standard RoPE: [q1*cos - q2*sin, q2*cos + q1*sin]
    qo1 = q1 * cos - q2 * sin
    qo2 = q2 * cos + q1 * sin
    ko1 = k1 * cos - k2 * sin
    ko2 = k2 * cos + k1 * sin

    # Store
    out_q_low_ptr = q_out_ptr + b * stride_qb + h * stride_qh + s * stride_qs + offs * stride_qd
    out_q_high_ptr = out_q_low_ptr + D_HALF * stride_qd
    out_k_low_ptr = k_out_ptr + b * stride_kb + h * stride_kh + s * stride_ks + offs * stride_kd
    out_k_high_ptr = out_k_low_ptr + D_HALF * stride_kd

    tl.store(out_q_low_ptr, qo1, mask=mask)
    tl.store(out_q_high_ptr, qo2, mask=mask)
    tl.store(out_k_low_ptr, ko1, mask=mask)
    tl.store(out_k_high_ptr, ko2, mask=mask)

@triton.jit
def _rope_backward_kernel(
    gq_ptr, gk_ptr, freqs_ptr,
    dq_ptr, dk_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_fs, stride_fd,
    S, H, D,
    D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % S
    h = (pid // S) % H
    b = pid // (S * H)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D_HALF

    # Load freqs
    freqs = tl.load(freqs_ptr + s * stride_fs + offs * stride_fd, mask=mask)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    # Load gradient of outputs
    gq_low_ptr = gq_ptr + b * stride_qb + h * stride_qh + s * stride_qs + offs * stride_qd
    gq_high_ptr = gq_low_ptr + D_HALF * stride_qd
    gk_low_ptr = gk_ptr + b * stride_kb + h * stride_kh + s * stride_ks + offs * stride_kd
    gk_high_ptr = gk_low_ptr + D_HALF * stride_kd

    gq1 = tl.load(gq_low_ptr, mask=mask)
    gq2 = tl.load(gq_high_ptr, mask=mask)
    gk1 = tl.load(gk_low_ptr, mask=mask)
    gk2 = tl.load(gk_high_ptr, mask=mask)

    # dq1 = gq1 * cos + gq2 * sin
    # dq2 = gq2 * cos - gq1 * sin
    dq1 = gq1 * cos + gq2 * sin
    dq2 = gq2 * cos - gq1 * sin
    dk1 = gk1 * cos + gk2 * sin
    dk2 = gk2 * cos - gk1 * sin

    # Store gradients of inputs
    dq_low_ptr = dq_ptr + b * stride_qb + h * stride_qh + s * stride_qs + offs * stride_qd
    dq_high_ptr = dq_low_ptr + D_HALF * stride_qd
    dk_low_ptr = dk_ptr + b * stride_kb + h * stride_kh + s * stride_ks + offs * stride_kd
    dk_high_ptr = dk_low_ptr + D_HALF * stride_kd

    tl.store(dq_low_ptr, dq1, mask=mask)
    tl.store(dq_high_ptr, dq2, mask=mask)
    tl.store(dk_low_ptr, dk1, mask=mask)
    tl.store(dk_high_ptr, dk2, mask=mask)

class TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs):
        # q, k: (B, H, S, D)
        B, H, S, D = q.shape
        D_HALF = D // 2
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        BLOCK_SIZE = triton.next_power_of_2(D_HALF)

        grid = (B * H * S,)
        _rope_forward_kernel[grid](
            q, k, freqs, q_out, k_out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            freqs.stride(0), freqs.stride(1),
            S, H, D, D_HALF=D_HALF, BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(freqs)
        ctx.params = (B, H, S, D, D_HALF, BLOCK_SIZE)
        return q_out, k_out

    @staticmethod
    def backward(ctx, gq, gk):
        freqs, = ctx.saved_tensors
        B, H, S, D, D_HALF, BLOCK_SIZE = ctx.params
        
        # Ensure gradients are contiguous or have expected layout
        dq = torch.empty_like(gq)
        dk = torch.empty_like(gk)

        grid = (B * H * S,)
        _rope_backward_kernel[grid](
            gq, gk, freqs, dq, dk,
            gq.stride(0), gq.stride(1), gq.stride(2), gq.stride(3),
            gk.stride(0), gk.stride(1), gk.stride(2), gk.stride(3),
            freqs.stride(0), freqs.stride(1),
            S, H, D, D_HALF=D_HALF, BLOCK_SIZE=BLOCK_SIZE,
        )
        return dq, dk, None

def test():
    torch.manual_seed(42)
    device = "cuda"
    B, H, S, D = 2, 4, 16, 64
    dtype = torch.float32 # Use float64 for near-zero error

    q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    freqs = torch.randn(S, D // 2, device=device, dtype=dtype)

    # Reference
    q_ref, k_ref = rope_ref(q, k, freqs)
    grad_output = torch.randn_like(q_ref) # Use random grad instead of .sum() for better coverage
    torch.autograd.backward([q_ref, k_ref], [grad_output, grad_output])
    
    dq_ref = q.grad.clone()
    dk_ref = k.grad.clone()
    q.grad, k.grad = None, None

    # Triton
    q_tri, k_tri = TritonRoPE.apply(q, k, freqs)
    torch.autograd.backward([q_tri, k_tri], [grad_output, grad_output])

    print(f"Forward Max Error (Q): {torch.max(torch.abs(q_ref - q_tri)):.2e}")
    print(f"k fwd error: {torch.max(torch.abs(k_ref - k_tri)):.2e}")
    print(f"Backward Max Error (Q): {torch.max(torch.abs(dq_ref - q.grad)):.2e}")
    print(f"Backward Max Error (K): {torch.max(torch.abs(dk_ref - k.grad)):.2e}")

if __name__ == "__main__":
    test()