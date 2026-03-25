import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_varlen_kernel(
    Q, K, V, cu_seqlens, sm_scale, L, Out,
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vk,
    stride_ot, stride_oh, stride_ok,
    QHEADS, KV_GROUPS,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // QHEADS
    q_head_idx = pid_bh % QHEADS

    kv_head_idx = q_head_idx // KV_GROUPS
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
    q_ptr = Q + (start_idx + rm[:, None]) * stride_qt + q_head_idx * stride_qh + rk[None, :] * stride_qk
    q_mask = (rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
    q = tl.load(q_ptr, mask=q_mask, other=0.0) # load [N,D]

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + rn_
        
        # K load as [D, N])
        k_ptr = K + (start_idx + rn[None, :]) * stride_kt + kv_head_idx * stride_kh + rk[:, None] * stride_kk
        k_mask = (rn[None, :] < seq_len) & (rk[:, None] < HEAD_DIM)
        k = tl.load(k_ptr, mask=k_mask, other=0.0)

        # V Load ([N, D])
        v_ptr = V + (start_idx + rn[:, None]) * stride_vt + kv_head_idx * stride_vh + rk[None, :] * stride_vk
        v_mask = (rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
        v = tl.load(v_ptr, mask=v_mask, other=0.0)

        # Attention 
        qk = tl.dot(q, k,out_dtype=tl.float32) * sm_scale #[N,N]
        qk = tl.where((rn[None, :] < seq_len), qk, -float("inf"))
    
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v,out_dtype=tl.float32) #[N,N]dot[N,D]> [N,D]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Store Output
    off_o = (start_idx + rm[:, None]) * stride_ot + q_head_idx * stride_oh + rk[None, :] * stride_ok
    tl.store(Out + off_o, (acc / l_i[:, None]).to(Out.dtype.element_ty), mask=q_mask)
    
    # Store LSE (LogSumExp) for backward
    off_l = (start_idx + rm) * QHEADS + q_head_idx
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
    stride_vt, stride_vh, stride_vk, QHEADS, KV_GROUPS,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // QHEADS
    q_head_idx = pid_bh % QHEADS

    kv_head_idx = q_head_idx // KV_GROUPS

    start_idx = tl.load(cu_seqlens + batch_idx)
    end_idx = tl.load(cu_seqlens + batch_idx + 1)
    seq_len = end_idx - start_idx

    if pid_m * BLOCK_M >= seq_len: return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_DMODEL)
    mask_md = (rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM)
    rn_ = tl.arange(0, BLOCK_N)
    
    q = tl.load(Q + (start_idx + rm[:, None]) * stride_qt + q_head_idx * stride_qh + rk[None, :] * stride_qk, mask=mask_md, other=0.0)
    do = tl.load(dO + (start_idx + rm[:, None]) * stride_qt + q_head_idx * stride_qh + rk[None, :] * stride_qk, mask=mask_md, other=0.0)
    lse = tl.load(L + (start_idx + rm) * QHEADS + q_head_idx, mask=rm < seq_len)
    di = tl.load(D + (start_idx + rm) * QHEADS + q_head_idx, mask=rm < seq_len)
    
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)
   
    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + rn_
        k = tl.load(K + (start_idx + rn[None, :]) * stride_kt + kv_head_idx * stride_kh + rk[:, None] * stride_kk, 
                    mask=(rn[None, :] < seq_len) & (rk[:, None] < HEAD_DIM), other=0.0)
        v = tl.load(V + (start_idx + rn[:, None]) * stride_vt + kv_head_idx * stride_vh + rk[None, :] * stride_vk, 
                    mask=(rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM), other=0.0)
        
        qk = tl.dot(q, k,out_dtype=tl.float32) * sm_scale
        p = tl.exp(qk - lse[:, None])
        p = tl.where(rn[None, :] < seq_len, p, 0.0)
        
        dp = (tl.dot(do, tl.trans(v),out_dtype=tl.float32) - di[:, None]) * p
        dq += tl.dot(dp.to(k.dtype), tl.trans(k),out_dtype=tl.float32)
        
    tl.store(dQ + (start_idx + rm[:, None]) * stride_qt + q_head_idx * stride_qh + rk[None, :] * stride_qk, (dq * sm_scale).to(dQ.dtype.element_ty), mask=mask_md)

@triton.jit
def _attn_bwd_dkv_varlen_kernel(
    Q, K, V, cu_seqlens, sm_scale, dO, dK, dV, L, D,
    stride_qt, stride_qh, stride_qk, stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vk, QHEADS, KV_GROUPS,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_kvh = tl.program_id(1)
    
    KV_HEADS = QHEADS // KV_GROUPS
    batch_idx = pid_kvh // KV_HEADS
    kv_head_idx = pid_kvh % KV_HEADS

    start_idx = tl.load(cu_seqlens + batch_idx)
    end_idx = tl.load(cu_seqlens + batch_idx + 1)
    seq_len = end_idx - start_idx

    if pid_n * BLOCK_N >= seq_len: return
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_DMODEL)
    
    k = tl.load(K + (start_idx + rn[:, None]) * stride_kt + kv_head_idx * stride_kh + rk[None, :] * stride_kk, mask=(rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM), other=0.0)
    v = tl.load(V + (start_idx + rn[:, None]) * stride_vt + kv_head_idx * stride_vh + rk[None, :] * stride_vk, mask=(rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM), other=0.0)
    
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    
    # Internal loop over Q-heads to avoid atomic adds
    for g in range(KV_GROUPS):
        q_head_idx = kv_head_idx * KV_GROUPS + g
        
        for start_m in range(0, seq_len, BLOCK_M):
            rm = start_m + tl.arange(0, BLOCK_M)
            q = tl.load(Q + (start_idx + rm[:, None]) * stride_qt + q_head_idx * stride_qh + rk[None, :] * stride_qk, mask=(rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM), other=0.0)
            do = tl.load(dO + (start_idx + rm[:, None]) * stride_qt + q_head_idx * stride_qh + rk[None, :] * stride_qk, mask=(rm[:, None] < seq_len) & (rk[None, :] < HEAD_DIM), other=0.0)
            lse = tl.load(L + (start_idx + rm) * QHEADS + q_head_idx, mask=rm < seq_len)
            di = tl.load(D + (start_idx + rm) * QHEADS + q_head_idx, mask=rm < seq_len)
            
            qk = tl.dot(q, tl.trans(k),out_dtype=tl.float32) * sm_scale
            p = tl.exp(qk.to(tl.float32) - lse[:, None])
            p = tl.where(rm[:, None] < seq_len, p, 0.0)
            
            dv += tl.dot(tl.trans(p.to(do.dtype)), do,out_dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v),out_dtype=tl.float32)
            ds = p * (dp - di[:, None]) * sm_scale
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q,out_dtype=tl.float32)
        
    tl.store(dK + (start_idx + rn[:, None]) * stride_kt + kv_head_idx * stride_kh + rk[None, :] * stride_kk, dk.to(dK.dtype.element_ty), mask=(rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM))
    tl.store(dV + (start_idx + rn[:, None]) * stride_vt + kv_head_idx * stride_vh + rk[None, :] * stride_vk, dv.to(dV.dtype.element_ty), mask=(rn[:, None] < seq_len) & (rk[None, :] < HEAD_DIM))

class FlashVarLenGqa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, sm_scale,groups,kv_heads):
        T, H, D = q.shape
        num_seqs = len(cu_seqlens) - 1
        # max_s 
        max_s = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        
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
            H,groups, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
            BLOCK_DMODEL=triton.next_power_of_2(D)
        )
        ctx.save_for_backward(q, k, v, cu_seqlens, L, out)
        ctx.sm_scale = sm_scale
        ctx.groups = groups
        ctx.kv_heads = kv_heads
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, cu_seqlens, L, out = ctx.saved_tensors
        T, H, D = q.shape
        num_seqs = len(cu_seqlens) - 1
         # max_s 
        max_s = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        
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
            H,ctx.groups, HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, BLOCK_DMODEL=BLOCK_DMODEL
        )
        
        # dK, dV
        grid_dkv = (triton.cdiv(max_s, BN), num_seqs * ctx.kv_heads)
        _attn_bwd_dkv_varlen_kernel[grid_dkv](
            q, k, v, cu_seqlens, ctx.sm_scale, do, dk, dv, L, delta,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            H, ctx.groups, 
            HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, BLOCK_DMODEL=BLOCK_DMODEL
        )
        return dq, dk, dv, None, None,None,None

class TritonVarlenAttentionGqa(torch.nn.Module):
    def __init__(self,q_heads, kv_heads, sm_scale=None):
        super().__init__()
        self.q_heads, self.kv_heads = q_heads, kv_heads
        self.groups = q_heads // kv_heads
        self.sm_scale = sm_scale

    def forward(self, q, k, v,cu_seqlens):
        """
        Args:
            q, k, v: Tensors of shape (Batch, Heads, Seq_Len, Head_Dim)
            cu_seqlens: Tensor of shape (no of seqs+1)
        """
        scale = self.sm_scale if self.sm_scale is not None else q.size(-1)**-0.5
        
        return FlashVarLenGqa.apply(q, k, v, cu_seqlens, scale,self.groups,self.kv_heads)