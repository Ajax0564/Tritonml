import torch
import triton
import triton.language as tl
import math

@triton.jit
def _attn_fwd_varlen_kernel(
    Q, K, V, cu_seqlens, sm_scale, Out,
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vk,
    stride_ot, stride_oh, stride_ok,
    HEADS, 
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // HEADS
    head_idx = pid_bh % HEADS

    # bounds
    start_idx = tl.load(cu_seqlens + batch_idx)
    end_idx = tl.load(cu_seqlens + batch_idx + 1)
    seq_len = end_idx - start_idx

    if pid_m * BLOCK_M >= seq_len:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_DMODEL)
    rn_ =  tl.arange(0, BLOCK_N)
    
    # Q Pointer ->  (Base + TokenOffset + HeadOffset + DimOffset) same pattern for K, V
    q_ptr = Q + (start_idx + rm[:, None]) * stride_qt + head_idx * stride_qh + rk[None, :] * stride_qk
    q_mask = (rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
    q = tl.load(q_ptr, mask=q_mask, other=0.0) # load [N,D]

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)
    

    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + rn_
        
        # K load as [D, N])
        k_ptr = K + (start_idx + rn[None, :]) * stride_kt + head_idx * stride_kh + rk[:, None] * stride_kk
        k_mask = (rn[None, :] < seq_len) & (rk[:, None] < HEAD_DIM)
        k = tl.load(k_ptr, mask=k_mask, other=0.0)

        # V Load ([N, D])
        v_ptr = V + (start_idx + rn[:, None]) * stride_vt + head_idx * stride_vh + rk[None, :] * stride_vk
        v_mask = (rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
        v = tl.load(v_ptr, mask=v_mask, other=0.0)

        # Attention 
        qk = tl.dot(q, k,out_dtype=tl.float32) * sm_scale #[N,N]
        qk = tl.where(rn[None, :] < seq_len, qk, -float("inf"))
        qk = tl.where((rm[:, None]) >= rn[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v,out_dtype=tl.float32) #[N,N]dot[N,D]> [N,D]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Store Output
    off_o = (start_idx + rm[:, None]) * stride_ot + head_idx * stride_oh + rk[None, :] * stride_ok
    tl.store(Out + off_o, (acc / l_i[:, None]).to(Out.dtype.element_ty), mask=q_mask)
    
class FlashVarLenCausal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, sm_scale):
        T, H, D = q.shape
        num_seqs = len(cu_seqlens) - 1
        # max_s 
        max_s = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        
        BLOCK_M, BLOCK_N = 32, 32
        out = torch.empty_like(q)
        
        grid = (triton.cdiv(max_s, BLOCK_M), num_seqs * H)
        _attn_fwd_varlen_kernel[grid](
            q, k, v, cu_seqlens, sm_scale, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            H, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
            BLOCK_DMODEL=triton.next_power_of_2(D)
        )
        return out