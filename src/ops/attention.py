import torch
import triton
import triton.language as tl

# --- 1. KERNELS ---

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, Mask, sm_scale, L, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_ob, stride_oh, stride_om, stride_ok,
    BATCH, HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :], mask=rm[:, None] < SEQ_LEN, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    mask_ptr = Mask + (pid_hz // HEADS) * stride_mb

    for start_n in range(0, SEQ_LEN, BLOCK_N):
        cols = start_n + rn
        k = tl.load(K + pid_hz * stride_kh + cols[None, :] * stride_kn + rk[:, None] * stride_kk, mask=cols[None, :] < SEQ_LEN, other=0.0).to(tl.float32)
        v = tl.load(V + pid_hz * stride_vh + cols[:, None] * stride_vn + rk[None, :], mask=cols[:, None] < SEQ_LEN, other=0.0).to(tl.float32)

        qk = tl.dot(q, k) * sm_scale
        m_tile = tl.load(mask_ptr + rm[:, None] * stride_mm + cols[None, :] * stride_mn, mask=(rm[:, None] < SEQ_LEN) & (cols[None, :] < SEQ_LEN), other=-float("inf"))
        qk += m_tile

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = acc / l_i[:, None]
    tl.store(Out + pid_hz * stride_oh + rm[:, None] * stride_om + rk[None, :], out.to(tl.float16), mask=rm[:, None] < SEQ_LEN)
    tl.store(L + pid_hz * SEQ_LEN + rm, m_i + tl.log(l_i), mask=rm < SEQ_LEN)

@triton.jit
def _bwd_preprocess_kernel(Out, dOut, D, stride_ob, stride_oh, stride_om, stride_ok, BATCH, HEADS, SEQ_LEN, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m, pid_hz = tl.program_id(0), tl.program_id(1)
    rm, rk = pid_m * BLOCK_M + tl.arange(0, BLOCK_M), tl.arange(0, HEAD_DIM)
    off_o = pid_hz * stride_oh + rm[:, None] * stride_om + rk[None, :]
    o = tl.load(Out + off_o, mask=rm[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
    do = tl.load(dOut + off_o, mask=rm[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
    tl.store(D + pid_hz * SEQ_LEN + rm, tl.sum(o * do, axis=1), mask=rm < SEQ_LEN)

@triton.jit
def _bwd_kernel_dq(Q, K, V, Mask, sm_scale, dO, dQ, L, D, stride_qb, stride_qh, stride_qm, stride_qk, stride_kb, stride_kh, stride_kn, stride_kk, stride_mb, stride_mh, stride_mm, stride_mn, BATCH, HEADS, SEQ_LEN, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m, pid_hz = tl.program_id(0), tl.program_id(1)
    rm, rk = pid_m * BLOCK_M + tl.arange(0, BLOCK_M), tl.arange(0, HEAD_DIM)
    q = tl.load(Q + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :], mask=rm[:, None] < SEQ_LEN, other=0.0)
    do = tl.load(dO + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :], mask=rm[:, None] < SEQ_LEN, other=0.0)
    lse = tl.load(L + pid_hz * SEQ_LEN + rm, mask=rm < SEQ_LEN)
    di = tl.load(D + pid_hz * SEQ_LEN + rm, mask=rm < SEQ_LEN)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    mask_ptr = Mask + (pid_hz // HEADS) * stride_mb
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + pid_hz * stride_kh + rn[None, :] * stride_kn + rk[:, None] * stride_kk, mask=rn[None, :] < SEQ_LEN, other=0.0)
        v = tl.load(V + pid_hz * stride_kh + rn[:, None] * stride_kn + rk[None, :], mask=rn[:, None] < SEQ_LEN, other=0.0)
        qk = tl.dot(q, k) * sm_scale
        m_tile = tl.load(mask_ptr + rm[:, None] * stride_mm + rn[None, :] * stride_mn, mask=(rm[:, None] < SEQ_LEN) & (rn[None, :] < SEQ_LEN), other=-float('inf'))
        p = tl.exp(qk + m_tile - lse[:, None])
        dp = (tl.dot(do, tl.trans(v)) - di[:, None]) * p
        dq += tl.dot(dp.to(tl.float16), tl.trans(k))
    tl.store(dQ + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :], (dq * sm_scale).to(tl.float16), mask=rm[:, None] < SEQ_LEN)

@triton.jit
def _bwd_kernel_dkdv(Q, K, V, Mask, sm_scale, dO, dK, dV, L, D, stride_qb, stride_qh, stride_qm, stride_qk, stride_kb, stride_kh, stride_kn, stride_kk, stride_mb, stride_mh, stride_mm, stride_mn, BATCH, HEADS, SEQ_LEN, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_n, pid_hz = tl.program_id(0), tl.program_id(1)
    rn, rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N), tl.arange(0, HEAD_DIM)
    k = tl.load(K + pid_hz * stride_kh + rn[:, None] * stride_kn + rk[None, :], mask=rn[:, None] < SEQ_LEN, other=0.0)
    v = tl.load(V + pid_hz * stride_kh + rn[:, None] * stride_kn + rk[None, :], mask=rn[:, None] < SEQ_LEN, other=0.0)
    dk, dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32), tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    mask_ptr = Mask + (pid_hz // HEADS) * stride_mb
    for start_m in range(0, SEQ_LEN, BLOCK_M):
        rm = start_m + tl.arange(0, BLOCK_M)
        q = tl.load(Q + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :], mask=rm[:, None] < SEQ_LEN, other=0.0)
        do = tl.load(dO + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :], mask=rm[:, None] < SEQ_LEN, other=0.0)
        lse, di = tl.load(L + pid_hz * SEQ_LEN + rm, mask=rm < SEQ_LEN), tl.load(D + pid_hz * SEQ_LEN + rm, mask=rm < SEQ_LEN)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        m_tile = tl.load(mask_ptr + rm[:, None] * stride_mm + rn[None, :] * stride_mn, mask=(rm[:, None] < SEQ_LEN) & (rn[None, :] < SEQ_LEN), other=-float('inf'))
        p = tl.exp(qk + m_tile - lse[:, None])
        dv += tl.dot(tl.trans(p.to(tl.float16)), do)
        dp = (tl.dot(do, tl.trans(v)) - di[:, None]) * p
        dk += tl.dot(tl.trans(dp.to(tl.float16)), q)
    tl.store(dK + pid_hz * stride_kh + rn[:, None] * stride_kn + rk[None, :], (dk * sm_scale).to(tl.float16), mask=rn[:, None] < SEQ_LEN)
    tl.store(dV + pid_hz * stride_kh + rn[:, None] * stride_kn + rk[None, :], dv.to(tl.float16), mask=rn[:, None] < SEQ_LEN)

# --- 2. WRAPPER ---

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, sm_scale):
        BLOCK_M, BLOCK_N = 64, 64
        B, H, S, D = q.shape
        out = torch.empty_like(q)
        L = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(S, BLOCK_M), B * H)
        _attn_fwd_kernel[grid](
            q, k, v, mask, sm_scale, L, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, S, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(q, k, v, mask, L, out)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, mask, L, out = ctx.saved_tensors
        B, H, S, D = q.shape
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N = 64, 64
        grid_prep = (triton.cdiv(S, BLOCK_M), B * H)
        _bwd_preprocess_kernel[grid_prep](out, do, delta, out.stride(0), out.stride(1), out.stride(2), out.stride(3), B, H, S, BLOCK_M=BLOCK_M, HEAD_DIM=D)
        _bwd_kernel_dq[(triton.cdiv(S, BLOCK_M), B * H)](q, k, v, mask, ctx.sm_scale, do, dq, L, delta, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3), B, H, S, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D)
        _bwd_kernel_dkdv[(triton.cdiv(S, BLOCK_N), B * H)](q, k, v, mask, ctx.sm_scale, do, dk, dv, L, delta, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3), B, H, S, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D)
        return dq, dk, dv, None, None




def test_flash_attn_full():
    B, H, S, D = 2, 4, 128, 64
    dtype = torch.float16
    device = "cuda"
    
    # Init inputs
    q = torch.randn((B, H, S, D), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((B, H, S, D), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((B, H, S, D), device=device, dtype=dtype, requires_grad=True)
    
    # 1. Mask for Triton and Manual Ref (Explicit -inf mask)
    mask = torch.tril(torch.ones((B, 1, S, S), device=device))
    mask = torch.where(mask > 0, 0.0, float("-inf")).to(dtype)
    
    sm_scale = D**-0.5
    do = torch.randn_like(q)

    # --- A. TRITON ---
    out_tri = FlashAttention.apply(q, k, v, mask, sm_scale)
    out_tri.backward(do, retain_graph=True)
    grads_tri = [q.grad.clone(), k.grad.clone(), v.grad.clone()]
    q.grad, k.grad, v.grad = None, None, None

    # --- B. PYTORCH MANUAL REF ---
    # Apply mask manually before softmax
    p = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    p += mask # Broadcasting add
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    out_ref = torch.matmul(p, v)
    out_ref.backward(do, retain_graph=True)
    grads_ref = [q.grad.clone(), k.grad.clone(), v.grad.clone()]
    q.grad, k.grad, v.grad = None, None, None

    # --- C. PYTORCH SDPA ---
    # We use is_causal=True which is mathematically equivalent to the lower-triangular mask
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=sm_scale)
    out_sdpa.backward(do, retain_graph=True)
    grads_sdpa = [q.grad.clone(), k.grad.clone(), v.grad.clone()]

    # --- REPORT ---
    print(f"{'Variable':<10} | {'Triton vs Manual':<20} | {'Triton vs SDPA':<20}")
    print("-" * 60)
    
    def check_diff(a, b):
        # Allow slightly higher tolerance for backward pass due to atomic adds in GPU
        return torch.allclose(a, b, atol=1e-2, rtol=1e-2)

    print(f"{'Output':<10} | {str(check_diff(out_tri, out_ref)):<20} | {str(check_diff(out_tri, out_sdpa)):<20}")
    
    for name, g_t, g_r, g_s in zip(['dQ', 'dK', 'dV'], grads_tri, grads_ref, grads_sdpa):
        match_ref = check_diff(g_t, g_r)
        match_sdpa = check_diff(g_t, g_s)
        
        # Calculate max absolute error for visibility
        err_ref = (g_t - g_r).abs().max().item()
        err_sdpa = (g_t - g_s).abs().max().item()
        
        print(f"{name:<10} | {str(match_ref):<5} (Err: {err_ref:.4f})   | {str(match_sdpa):<5} (Err: {err_sdpa:.4f})")

if __name__ == "__main__":
    test_flash_attn_full()