import torch
import triton
import triton.language as tl

@triton.jit
def _rope_forward_kernel(
    q_ptr, k_ptr, freqs_ptr,
    q_out_ptr, k_out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_fs, stride_fd,
    S, H, D,
    D_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_D: tl.constexpr, 
):
    # Grid
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_s = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    mask_s = offs_s < S
    mask_d = offs_d < D_HALF
    # 2D mask
    mask2d = mask_s[:, None] & mask_d[None, :]

    # Load cos/sin freqs shape (BLOCK_M, BLOCK_D)
    f_p = freqs_ptr + (offs_s[:, None] * stride_fs + offs_d[None, :] * stride_fd)
    freqs = tl.load(f_p, mask=mask2d)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    
   
   
    offs_q = offs_s[:, None] * stride_qs + offs_d[None, :] * stride_qd
    offs_k = offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
    
    # Load Q and K q1,q2 
    q1 = tl.load(q_base + offs_q, mask=mask2d)
    q2 = tl.load(q_base + offs_q + D_HALF * stride_qd, mask=mask2d)
    k1 = tl.load(k_base + offs_k, mask=mask2d)
    k2 = tl.load(k_base + offs_k + D_HALF * stride_qd, mask=mask2d)

    # RoPE 
    qo1 = q1 * cos - q2 * sin
    qo2 = q2 * cos + q1 * sin
    ko1 = k1 * cos - k2 * sin
    ko2 = k2 * cos + k1 * sin

   
    qo_base = q_out_ptr + b * stride_qb + h * stride_qh
    ko_base = k_out_ptr + b * stride_kb + h * stride_kh
    
    tl.store(qo_base + offs_q, qo1, mask=mask2d)
    tl.store(qo_base + offs_q + D_HALF * stride_qd, qo2, mask=mask2d)
    tl.store(ko_base + offs_k, ko1, mask=mask2d)
    tl.store(ko_base + offs_k + D_HALF * stride_qd, ko2, mask=mask2d)

@triton.jit
def _rope_backward_kernel(
    gq_ptr, gk_ptr, freqs_ptr,
    dq_ptr, dk_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_fs, stride_fd,
    S, H, D,
    D_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_s = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask2d = (offs_s[:, None] < S) & (offs_d[None, :] < D_HALF)

    # freqs
    f_p = freqs_ptr + (offs_s[:, None] * stride_fs + offs_d[None, :] * stride_fd)
    freqs = tl.load(f_p, mask=mask2d)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    
    gq_base = gq_ptr + b * stride_qb + h * stride_qh
    gk_base = gk_ptr + b * stride_kb + h * stride_kh

    offs_q = offs_s[:, None] * stride_qs + offs_d[None, :] * stride_qd
    offs_k = offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd

    gq1 = tl.load(gq_base + offs_q, mask=mask2d)
    gq2 = tl.load(gq_base + offs_q + D_HALF * stride_qd, mask=mask2d)
    gk1 = tl.load(gk_base + offs_k, mask=mask2d)
    gk2 = tl.load(gk_base + offs_k + D_HALF * stride_qd, mask=mask2d)

    # Backward RoPE 
    dq1 = gq1 * cos + gq2 * sin
    dq2 = gq2 * cos - gq1 * sin
    dk1 = gk1 * cos + gk2 * sin
    dk2 = gk2 * cos - gk1 * sin

    dq_base = dq_ptr + b * stride_qb + h * stride_qh
    dk_base = dk_ptr + b * stride_kb + h * stride_kh
    tl.store(dq_base + offs_q, dq1, mask=mask2d)
    tl.store(dq_base + offs_q + D_HALF * stride_qd, dq2, mask=mask2d)
    tl.store(dk_base + offs_k, dk1, mask=mask2d)
    tl.store(dk_base + offs_k + D_HALF * stride_qd, dk2, mask=mask2d)

class TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs):
        B, H, S, D = q.shape
        D_HALF = D // 2
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        
        BLOCK_M = 32 
        BLOCK_D = triton.next_power_of_2(D_HALF) 

        grid = (triton.cdiv(S, BLOCK_M), B * H)
        
        _rope_forward_kernel[grid](
            q, k, freqs, q_out, k_out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            freqs.stride(0), freqs.stride(1),
            S, H, D, D_HALF=D_HALF, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(freqs)
        ctx.params = (B, H, S, D, D_HALF, BLOCK_M, BLOCK_D)
        return q_out, k_out

    @staticmethod
    def backward(ctx, gq, gk):
        freqs, = ctx.saved_tensors
        B, H, S, D, D_HALF, BLOCK_M, BLOCK_D = ctx.params
        
        dq = torch.empty_like(gq)
        dk = torch.empty_like(gk)

        grid = (triton.cdiv(S, BLOCK_M), B * H)
        _rope_backward_kernel[grid](
            gq, gk, freqs, dq, dk,
            gq.stride(0), gq.stride(1), gq.stride(2), gq.stride(3),
            gk.stride(0), gk.stride(1), gk.stride(2), gk.stride(3),
            freqs.stride(0), freqs.stride(1),
            S, H, D, D_HALF=D_HALF, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        )
        return dq, dk, None
    
class TritonRopeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, freqs):
        """
        Args:
            q, k, v: Tensors of shape (Batch, Heads, Seq_Len, Head_Dim)
            freqs: Tensor of shape (Seq_Len, Head_Dim//2)
        """
        return TritonRoPE.apply(q, k, freqs)