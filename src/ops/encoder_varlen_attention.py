import torch
import triton
import triton.language as tl
import math

@triton.jit
def _attn_fwd_varlen_kernel(
    Q, K, V, cu_seqlens, sm_scale, L, Out,
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

    # Sequence bounds
    start_idx = tl.load(cu_seqlens + batch_idx)
    end_idx = tl.load(cu_seqlens + batch_idx + 1)
    seq_len = end_idx - start_idx

    if pid_m * BLOCK_M >= seq_len:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_DMODEL)
    
    # Q Pointer: (Base + TokenOffset + HeadOffset + DimOffset)
    q_ptr = Q + (start_idx + rm[:, None]) * stride_qt + head_idx * stride_qh + rk[None, :] * stride_qk
    q_mask = (rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
    q = tl.load(q_ptr, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)
        
        # K transpose load [D, N])
        k_ptr = K + (start_idx + rn[None, :]) * stride_kt + head_idx * stride_kh + rk[:, None] * stride_kk
        k_mask = (rn[None, :] < seq_len) & (rk[:, None] < HEAD_DIM)
        k = tl.load(k_ptr, mask=k_mask, other=0.0)

        # V Load ([N, D])
        v_ptr = V + (start_idx + rn[:, None]) * stride_vt + head_idx * stride_vh + rk[None, :] * stride_vk
        v_mask = (rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
        v = tl.load(v_ptr, mask=v_mask, other=0.0)

        # Attention 
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(rn[None, :] < seq_len, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Store Output
    off_o = (start_idx + rm[:, None]) * stride_ot + head_idx * stride_oh + rk[None, :] * stride_ok
    tl.store(Out + off_o, acc / l_i[:, None], mask=q_mask)
    
    # Store LSE (LogSumExp) for backward
    off_l = (start_idx + rm) * HEADS + head_idx
    tl.store(L + off_l, m_i + tl.log(l_i), mask=rm < seq_len)

@triton.jit
def _bwd_preprocess_varlen(
    Out, dOut, D_out, T_size, HEADS, stride_ot, stride_oh, stride_ok, 
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    head_idx = pid_bh % HEADS
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_DMODEL)
    
    mask_m = rm < T_size
    off = rm[:, None] * stride_ot + head_idx * stride_oh + rk[None, :] * stride_ok
    mask_md = mask_m[:, None] & (rk[None, :] < HEAD_DIM)
    
    o = tl.load(Out + off, mask=mask_md, other=0.0)
    do = tl.load(dOut + off, mask=mask_md, other=0.0)
    
    delta = tl.sum(o * do, axis=1)
    # Store delta as (T, H)
    tl.store(D_out + rm * HEADS + head_idx, delta, mask=mask_m)


@triton.jit
def _attn_bwd_dq_varlen_kernel(
    Q, K, V, cu_seqlens, sm_scale, dO, dQ, L, D,
    stride_qt, stride_qh, stride_qk, stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vk, HEADS, 
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // HEADS
    head_idx = pid_bh % HEADS
    start_idx = tl.load(cu_seqlens + batch_idx)
    end_idx = tl.load(cu_seqlens + batch_idx + 1)
    seq_len = end_idx - start_idx

    if pid_m * BLOCK_M >= seq_len: return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_DMODEL)
    mask_md = (rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
    
    q = tl.load(Q + (start_idx + rm[:, None]) * stride_qt + head_idx * stride_qh + rk[None, :] * stride_qk, mask=mask_md, other=0.0)
    do = tl.load(dO + (start_idx + rm[:, None]) * stride_qt + head_idx * stride_qh + rk[None, :] * stride_qk, mask=mask_md, other=0.0)
    lse = tl.load(L + (start_idx + rm) * HEADS + head_idx, mask=rm < seq_len)
    di = tl.load(D + (start_idx + rm) * HEADS + head_idx, mask=rm < seq_len)
    
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + (start_idx + rn[None, :]) * stride_kt + head_idx * stride_kh + rk[:, None] * stride_kk, 
                    mask=(rn[None, :] < seq_len) & (rk[:, None] < HEAD_DIM), other=0.0)
        v = tl.load(V + (start_idx + rn[:, None]) * stride_vt + head_idx * stride_vh + rk[None, :] * stride_vk, 
                    mask=(rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM), other=0.0)
        
        qk = tl.dot(q, k) * sm_scale
        p = tl.exp(qk - lse[:, None])
        p = tl.where(rn[None, :] < seq_len, p, 0.0)
        
        dp = (tl.dot(do, tl.trans(v)) - di[:, None]) * p
        dq += tl.dot(dp, tl.trans(k))
        
    tl.store(dQ + (start_idx + rm[:, None]) * stride_qt + head_idx * stride_qh + rk[None, :] * stride_qk, dq * sm_scale, mask=mask_md)


@triton.jit
def _attn_bwd_dkv_varlen_kernel(
    Q, K, V, cu_seqlens, sm_scale, dO, dK, dV, L, D,
    stride_qt, stride_qh, stride_qk, stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vk, HEADS,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // HEADS
    head_idx = pid_bh % HEADS
    start_idx = tl.load(cu_seqlens + batch_idx)
    end_idx = tl.load(cu_seqlens + batch_idx + 1)
    seq_len = end_idx - start_idx

    if pid_n * BLOCK_N >= seq_len: return
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_DMODEL)
    mask_nd = (rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
    
    k = tl.load(K + (start_idx + rn[:, None]) * stride_kt + head_idx * stride_kh + rk[None, :] * stride_kk, mask=mask_nd, other=0.0)
    v = tl.load(V + (start_idx + rn[:, None]) * stride_vt + head_idx * stride_vh + rk[None, :] * stride_vk, mask=mask_nd, other=0.0)
    
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    
    for start_m in range(0, seq_len, BLOCK_M):
        rm = start_m + tl.arange(0, BLOCK_M)
        q = tl.load(Q + (start_idx + rm[:, None]) * stride_qt + head_idx * stride_qh + rk[None, :] * stride_qk, 
                    mask=(rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM), other=0.0)
        do = tl.load(dO + (start_idx + rm[:, None]) * stride_qt + head_idx * stride_qh + rk[None, :] * stride_qk, 
                     mask=(rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM), other=0.0)
        lse = tl.load(L + (start_idx + rm) * HEADS + head_idx, mask=rm < seq_len)
        di = tl.load(D + (start_idx + rm) * HEADS + head_idx, mask=rm < seq_len)
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        p = tl.where(rm[:, None] < seq_len, p, 0.0)
        
        dv += tl.dot(tl.trans(p), do)
        dp = (tl.dot(do, tl.trans(v)) - di[:, None]) * p
        dk += tl.dot(tl.trans(dp), q)
        
    tl.store(dK + (start_idx + rn[:, None]) * stride_kt + head_idx * stride_kh + rk[None, :] * stride_kk, dk * sm_scale, mask=mask_nd)
    tl.store(dV + (start_idx + rn[:, None]) * stride_vt + head_idx * stride_vh + rk[None, :] * stride_vk, dv, mask=mask_nd)

class FlashVarLen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, sm_scale):
        T, H, D = q.shape
        num_seqs = len(cu_seqlens) - 1
        # max_s safely
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
        max_s = max(seqlens)
        
        BLOCK_M, BLOCK_N = 32, 32
        out = torch.empty_like(q)
        L = torch.empty((T, H), device=q.device, dtype=torch.float32)
        
        grid = (triton.cdiv(max_s, BLOCK_M), num_seqs * H)
        _attn_fwd_varlen_kernel[grid](
            q, k, v, cu_seqlens, sm_scale, L, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            H, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
            BLOCK_DMODEL=triton.next_power_of_2(D)
        )
        ctx.save_for_backward(q, k, v, cu_seqlens, L, out)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, cu_seqlens, L, out = ctx.saved_tensors
        T, H, D = q.shape
        num_seqs = len(cu_seqlens) - 1
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
        max_s = max(seqlens)
        
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        delta = torch.empty((T, H), device=q.device, dtype=torch.float32)
        
        BM, BN = 32, 32
        BLOCK_DMODEL = triton.next_power_of_2(D)
        
        # Delta
        _bwd_preprocess_varlen[(triton.cdiv(T, BM), H)](
            out, do, delta, T, H, 
            out.stride(0), out.stride(1), out.stride(2), 
            BLOCK_M=BM, HEAD_DIM=D, BLOCK_DMODEL=BLOCK_DMODEL
        )
        
        # dQ
        grid_dq = (triton.cdiv(max_s, BM), num_seqs * H)
        _attn_bwd_dq_varlen_kernel[grid_dq](
            q, k, v, cu_seqlens, ctx.sm_scale, do, dq, L, delta,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            H, HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, BLOCK_DMODEL=BLOCK_DMODEL
        )
        
        # dK, dV
        grid_dkv = (triton.cdiv(max_s, BN), num_seqs * H)
        _attn_bwd_dkv_varlen_kernel[grid_dkv](
            q, k, v, cu_seqlens, ctx.sm_scale, do, dk, dv, L, delta,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            H, HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, BLOCK_DMODEL=BLOCK_DMODEL
        )
        return dq, dk, dv, None, None

class TritonVerlenAttention(torch.nn.Module):
    def __init__(self, sm_scale=None):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, q, k, v,cu_seqlens):
        """
        Args:
            q, k, v: Tensors of shape (Batch, Heads, Seq_Len, Head_Dim)
            cu_seqlens: Tensor of shape (no of seqs+1)
        """
        scale = self.sm_scale if self.sm_scale is not None else 1.0 / torch.sqrt(q.size(-1))
        
        return FlashVarLen.apply(q, k, v, cu_seqlens, scale)