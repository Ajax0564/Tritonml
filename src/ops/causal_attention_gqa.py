import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, LSE, Out,
    MASK,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    stride_mb, stride_ms,
    B,QHEADS, KV_GROUPS, S_Q, S_K,
    OFFSET,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    pid_m, pid_hz = tl.program_id(0), tl.program_id(1)
    batch_idx = pid_hz // QHEADS
    q_head_idx = pid_hz % QHEADS
    kv_head_idx = q_head_idx // KV_GROUPS
    KV_HEADS = QHEADS // KV_GROUPS
    pid_kvh = batch_idx * KV_HEADS + kv_head_idx
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_ = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_DMODEL)
    rk_mask = rk < HEAD_DIM

    q_ptr = Q + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :] * stride_qk
    mask_ptr = MASK + batch_idx * stride_mb
    
    q = tl.load(q_ptr, mask=(rm[:, None] < S_Q) & (rk_mask[None, :]), other=0.0)
    
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    for start_n in range(0, S_K, BLOCK_N):
        cols = start_n + rn_
        
        k_ptr = K + pid_kvh * stride_kh + cols[None, :] * stride_kn + rk[:, None] * stride_kk
        k = tl.load(k_ptr, mask=(cols[None, :] < S_K) & (rk_mask[:, None]), other=0.0)
        
        qk = tl.dot(q, k,out_dtype=tl.float32) * sm_scale
        m_tile = tl.load(mask_ptr + cols[None, :] * stride_ms, mask=cols[None, :] < S_K, other=-float("inf"))
        qk += m_tile 
        
        if IS_CAUSAL:
            qk = tl.where((OFFSET + rm[:, None]) >= cols[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        
        v_ptr = V + pid_kvh * stride_vh + cols[:, None] * stride_vn + rk[None, :] * stride_vk
        v = tl.load(v_ptr, mask=(cols[:, None] < S_K) & (rk_mask[None, :]), other=0.0)
        
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v,out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    out_ptr = Out + pid_hz * stride_oh + rm[:, None] * stride_om + rk[None, :] * stride_ok
    tl.store(out_ptr, (acc / l_i[:, None]).to(Out.dtype.element_ty), mask=(rm[:, None] < S_Q) & (rk_mask[None, :]))
    tl.store(LSE + pid_hz * S_Q + rm, m_i + tl.log(l_i), mask=rm < S_Q)

@triton.jit
def _bwd_preprocess_kernel(
    Out, dOut, D_vec, 
    stride_oh, stride_om, stride_ok, 
    stride_doh, stride_dom, stride_dok,
    S, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m, pid_hz = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_DMODEL)
    rk_mask = rk < HEAD_DIM
    
    o = tl.load(Out + pid_hz * stride_oh + rm[:, None] * stride_om + rk[None, :]*stride_ok, mask=(rm[:, None] < S) & (rk_mask[None, :]), other=0.0)
    do = tl.load(dOut + pid_hz * stride_doh + rm[:, None] * stride_dom + rk[None, :]*stride_dok, mask=(rm[:, None] < S) & (rk_mask[None, :]), other=0.0)
    
    tl.store(D_vec + pid_hz * S + rm, tl.sum(o * do, axis=1), mask=rm < S)

@triton.jit
def _bwd_kernel_dq(
    Q, K, V, sm_scale, dO, dQ, LSE, D_vec, MASK,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_doh, stride_dom, stride_dok,
    stride_mb, stride_ms,
    B,QHEADS, KV_GROUPS,S_Q, S_K,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m, pid_hz = tl.program_id(0), tl.program_id(1)
    batch_idx = pid_hz // QHEADS
    q_head_idx = pid_hz % QHEADS
    kv_head_idx = q_head_idx // KV_GROUPS
    KV_HEADS = QHEADS // KV_GROUPS
    pid_kvh = batch_idx * KV_HEADS + kv_head_idx
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_DMODEL)
    rk_mask = rk < HEAD_DIM
    rn_ = tl.arange(0, BLOCK_N)
    
    q = tl.load(Q + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :]*stride_qk, mask=(rm[:, None] < S_Q) & (rk_mask[None, :]), other=0.0)
    do = tl.load(dO + pid_hz * stride_doh + rm[:, None] * stride_dom + rk[None, :]*stride_dok, mask=(rm[:, None] < S_Q) & (rk_mask[None, :]), other=0.0)
    lse = tl.load(LSE + pid_hz * S_Q + rm, mask=rm < S_Q)
    di = tl.load(D_vec + pid_hz * S_Q + rm, mask=rm < S_Q)
    mask_ptr = MASK + batch_idx * stride_mb

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)
    for start_n in range(0, S_K, BLOCK_N):
        rn = start_n + rn_
        
        k = tl.load(K + pid_kvh * stride_kh + rn[None, :] * stride_kn + rk[:, None] * stride_kk, mask=(rn[None, :] < S_K) & (rk_mask[:, None]), other=0.0)
        v = tl.load(V + pid_kvh * stride_vh + rn[:, None] * stride_vn + rk[None, :] * stride_vk, mask=(rn[:, None] < S_K) & (rk_mask[None, :]), other=0.0)
        
        qk = tl.dot(q, k,out_dtype=tl.float32) * sm_scale
        m_tile = tl.load(mask_ptr + rn[None, :] * stride_ms, mask=rn[None, :] < S_K, other=-float('inf'))
        
        qk = tl.where(rm[:, None] >= rn[None, :], qk, -float('inf'))
        p = tl.exp(qk+m_tile - lse[:, None])
        
        dp = (tl.dot(do, tl.trans(v),out_dtype=tl.float32) - di[:, None]) * p
        dq += tl.dot(dp.to(k.dtype), tl.trans(k),out_dtype=tl.float32)
    
    tl.store(dQ + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :], (dq * sm_scale).to(dQ.dtype.element_ty), mask=(rm[:, None] < S_K) & (rk_mask[None, :]))

@triton.jit
def _bwd_kernel_dkdv(
    Q, K, V, sm_scale, dO, dK, dV, LSE, D_vec, MASK,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_doh, stride_dom, stride_dok,
    stride_mb, stride_ms,
    B,QHEADS, KV_GROUPS, S_Q, S_K,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_n, pid_bk_vh = tl.program_id(0), tl.program_id(1)
    # Now pid_bk_vh represents (batch_idx * KV_HEADS + kv_head_idx)
    KV_HEADS = QHEADS // KV_GROUPS
    batch_idx = pid_bk_vh // KV_HEADS
    kv_head_idx = pid_bk_vh % KV_HEADS
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_DMODEL)
    rk_mask = rk < HEAD_DIM
    rm_ = tl.arange(0, BLOCK_M)
   
    mask_ptr = MASK + batch_idx * stride_mb
    m_tile = tl.load(mask_ptr + rn[None, :] * stride_ms, mask=rn[None, :] < S_K, other=-float('inf'))
    
    k = tl.load(K + pid_bk_vh * stride_kh + rn[:, None] * stride_kn + rk[None, :] * stride_kk, mask=(rn[:, None] < S_K) & (rk_mask[None, :]), other=0.0)
    v = tl.load(V + pid_bk_vh * stride_vh + rn[:, None] * stride_vn + rk[None, :] * stride_vk, mask=(rn[:, None] < S_K) & (rk_mask[None, :]), other=0.0)
    
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    
    for g in range(KV_GROUPS):
        q_head_idx = kv_head_idx * KV_GROUPS + g
        cur_pid_hz = batch_idx * QHEADS + q_head_idx
        for start_m in range(0, S_Q, BLOCK_M):
            rm = start_m + rm_
            q = tl.load(Q + cur_pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :] * stride_qk, mask=(rm[:, None] < S_Q) & (rk_mask[None, :]), other=0.0)
            do = tl.load(dO + cur_pid_hz * stride_doh + rm[:, None] * stride_dom + rk[None, :] * stride_dok, mask=(rm[:, None] < S_Q) & (rk_mask[None, :]), other=0.0)
            lse = tl.load(LSE + cur_pid_hz * S_Q + rm, mask=rm < S_Q)
            di = tl.load(D_vec + cur_pid_hz * S_Q + rm, mask=rm < S_Q)

            qk = tl.dot(q, tl.trans(k),out_dtype=tl.float32) * sm_scale
        
            qk = tl.where(rm[:, None] >= rn[None, :], qk, -float('inf'))
            p = tl.exp(qk+m_tile - lse[:, None])

            dv += tl.dot(tl.trans(p.to(do.dtype)), do,out_dtype=tl.float32)
            dp = (tl.dot(do, tl.trans(v),out_dtype=tl.float32) - di[:, None]) * p
            dk += tl.dot(tl.trans(dp.to(q.dtype)), q,out_dtype=tl.float32)

    tl.store(dK + pid_bk_vh * stride_kh + rn[:, None] * stride_kn + rk[None, :] * stride_kk, (dk * sm_scale).to(dK.dtype.element_ty), mask=(rn[:, None] < S_K) & (rk_mask[None, :]))
    tl.store(dV + pid_bk_vh * stride_vh + rn[:, None] * stride_vn + rk[None, :] * stride_vk,dv.to(dV.dtype.element_ty), mask=(rn[:, None] < S_K) & (rk_mask[None, :]))

class FlashAttentionCausalGqa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, q_heads, kv_heads, offset, sm_scale, causal):
        
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        mask = mask.contiguous()
        
        B, H, S_Q, D = q.shape
        S_K = k.shape[2]
        kv_groups = q_heads // kv_heads
        BLOCK_DMODEL = triton.next_power_of_2(D)
        
        out = torch.empty_like(q)
        lse = torch.empty((B, H, S_Q), device=q.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N = 32, 32

        _attn_fwd_kernel[(triton.cdiv(S_Q, BLOCK_M), B * H)](
            q, k, v, sm_scale, lse, out, mask,
            q.stride(1), q.stride(2), q.stride(3),
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            out.stride(1), out.stride(2), out.stride(3),
            mask.stride(0), mask.stride(1),
            B, H, kv_groups, S_Q, S_K, int(offset),
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
            BLOCK_DMODEL=BLOCK_DMODEL, IS_CAUSAL=causal
        )
        
        ctx.save_for_backward(q, k, v, lse, out, mask)
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = BLOCK_DMODEL
        ctx.kv_heads = kv_heads 
        ctx.kv_groups = kv_groups
        ctx.offset = offset
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, lse, out, mask = ctx.saved_tensors
        do = do.contiguous()
        B, H, S_Q, D = q.shape
        S_K = k.shape[2]
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty((B, H, S_Q), device=q.device, dtype=torch.float32)
        
        BLOCK_M, BLOCK_N = 32, 32
        grid = (triton.cdiv(S_Q, BLOCK_M), B * H)

        _bwd_preprocess_kernel[grid](
            out, do, delta, 
            out.stride(1), out.stride(2), out.stride(3),
            do.stride(1), do.stride(2), do.stride(3),
            S_Q, BLOCK_M=BLOCK_M, HEAD_DIM=D, BLOCK_DMODEL=ctx.BLOCK_DMODEL
        )

        _bwd_kernel_dq[grid](
            q, k, v, ctx.sm_scale, do, dq, lse, delta, mask,
            q.stride(1), q.stride(2), q.stride(3),
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            do.stride(1), do.stride(2), do.stride(3),
            mask.stride(0), mask.stride(1),
            B, H, ctx.kv_groups,S_Q, S_K, 
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=ctx.BLOCK_DMODEL
        )

        _bwd_kernel_dkdv[(triton.cdiv(S_K, BLOCK_N), B * ctx.kv_heads)](
            q, k, v, ctx.sm_scale, do, dk, dv, lse, delta, mask,
            q.stride(1), q.stride(2), q.stride(3),
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            do.stride(1), do.stride(2), do.stride(3),
            mask.stride(0), mask.stride(1),
            B, H, ctx.kv_groups,S_Q, S_K, 
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=ctx.BLOCK_DMODEL
        )
        
        return dq, dk, dv, None, None, None, None, None, None

class TritonCausalAttentionGqa(torch.nn.Module):
    def __init__(self, q_heads, kv_heads, sm_scale=None, causal=True):
        super().__init__()
        self.sm_scale = sm_scale
        self.causal = causal
        self.q_heads, self.kv_heads = q_heads, kv_heads

    def forward(self, q, k, v, mask, start_pos=0):
        scale = self.sm_scale if self.sm_scale is not None else q.size(-1)**-0.5

        return FlashAttentionCausalGqa.apply(
            q, k, v, mask, 
            self.q_heads, self.kv_heads, 
            start_pos, scale, self.causal
        )