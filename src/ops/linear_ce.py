import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F

@triton.jit
def fused_linear_ce_fwd_kernel(
    x_ptr, w_ptr, b_ptr, targets_ptr, loss_ptr, lse_ptr,
    M, K, N,
    stride_xm, stride_xk, stride_wn, stride_wk,
    ignore_index, HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    row_idx = tl.program_id(0)
    target = tl.load(targets_ptr + row_idx)
    if target == ignore_index:
        tl.store(loss_ptr + row_idx, 0.0)
        tl.store(lse_ptr + row_idx, 0.0)
        return

    m_row = -float('inf')
    d_row = 0.0
    target_logit = 0.0

    for n_st in range(0, N, BLOCK_N):
        cols = n_st + tl.arange(0, BLOCK_N)
        mask_n = cols < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_st in range(0, K, BLOCK_K):
            offs_k = k_st + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            xv = tl.load(x_ptr + row_idx * stride_xm + offs_k * stride_xk, mask=mask_k, other=0.0)
            wv = tl.load(w_ptr + cols[:, None] * stride_wn + offs_k[None, :] * stride_wk, 
                         mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc += tl.sum(xv[None, :] * wv, axis=1)

        if HAS_BIAS:
            acc += tl.load(b_ptr + cols, mask=mask_n, other=0.0)

        m_tile = tl.max(tl.where(mask_n, acc, -float('inf')), axis=0)
        m_new = tl.maximum(m_row, m_tile)
        d_row = d_row * tl.exp(m_row - m_new) + tl.sum(tl.where(mask_n, tl.exp(acc - m_new), 0.0), axis=0)
        m_row = m_new

        if (target >= n_st) & (target < n_st + BLOCK_N):
            target_logit = tl.sum(tl.where(cols == target, acc, 0.0))

    lse = tl.log(d_row) + m_row
    tl.store(lse_ptr + row_idx, lse)
    tl.store(loss_ptr + row_idx, lse - target_logit)


@triton.jit
def fused_linear_ce_bwd_dx_kernel(
    x_ptr, w_ptr, b_ptr, targets_ptr, lse_ptr, grad_x_ptr,
    M, K, N, scale,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_gxm, stride_gxk,
    ignore_index, HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    row_idx = tl.program_id(0)
    target = tl.load(targets_ptr + row_idx)
    if target == ignore_index:
        for k_st in range(0, K, BLOCK_K):
            offs_k = k_st + tl.arange(0, BLOCK_K)
            tl.store(grad_x_ptr + row_idx * stride_gxm + offs_k * stride_gxk, 0.0, mask=offs_k < K)
        return
    
    lse = tl.load(lse_ptr + row_idx)

    for k_tile_st in range(0, K, BLOCK_K):
        offs_k_tile = k_tile_st + tl.arange(0, BLOCK_K)
        mask_k_tile = offs_k_tile < K
        grad_x_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        for n_st in range(0, N, BLOCK_N):
            cols = n_st + tl.arange(0, BLOCK_N)
            mask_n = cols < N
            logits = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k_inner in range(0, K, BLOCK_K):
                offs_ki = k_inner + tl.arange(0, BLOCK_K)
                mask_ki = offs_ki < K
                xv = tl.load(x_ptr + row_idx * stride_xm + offs_ki * stride_xk, mask=mask_ki, other=0.0)
                wv = tl.load(w_ptr + cols[:, None] * stride_wn + offs_ki[None, :] * stride_wk, 
                             mask=mask_n[:, None] & mask_ki[None, :], other=0.0)
                logits += tl.sum(xv[None, :] * wv, axis=1)
            
            if HAS_BIAS: logits += tl.load(b_ptr + cols, mask=mask_n, other=0.0)
            probs = tl.exp(logits - lse)
            grad_logit = tl.where(cols == target, probs - 1.0, probs) * scale
            grad_logit = tl.where(mask_n, grad_logit, 0.0) 
            w_chunk = tl.load(w_ptr + cols[:, None] * stride_wn + offs_k_tile[None, :] * stride_wk, 
                              mask=mask_n[:, None] & mask_k_tile[None, :], other=0.0)
            grad_x_acc += tl.sum(grad_logit[:, None] * w_chunk, axis=0)
            
        tl.store(grad_x_ptr + row_idx * stride_gxm + offs_k_tile * stride_gxk, grad_x_acc, mask=mask_k_tile)

# GRAD W & BIAS
@triton.jit
def fused_linear_ce_bwd_dw_kernel(
    x_ptr, w_ptr, b_ptr, targets_ptr, lse_ptr, dw_ptr, db_ptr,
    M, K, N, scale,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_dwn, stride_dwk,
    ignore_index, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_k = offs_k < K

    dw_acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_st in range(0, M, BLOCK_M):
        offs_m = m_st + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        
        # Load targets and LSE for this batch block
        targets = tl.load(targets_ptr + offs_m, mask=mask_m, other=ignore_index)
        lse = tl.load(lse_ptr + offs_m, mask=mask_m, other=0.0)
        
        # Load X block
        xv = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, 
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # We need logits for the specific N-block we are calculating dW for
        logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_inner in range(0, K, BLOCK_K):
            offs_ki = k_inner + tl.arange(0, BLOCK_K)
            mask_ki = offs_ki < K
            xv_logit = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_ki[None, :] * stride_xk, 
                               mask=mask_m[:, None] & mask_ki[None, :], other=0.0)
            wv_logit = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_ki[None, :] * stride_wk, 
                               mask=mask_n[:, None] & mask_ki[None, :], other=0.0)
            logits += tl.dot(xv_logit, tl.trans(wv_logit))

        if HAS_BIAS:
            bv = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
            logits += bv[None, :]

        #  (Prob - Delta)
        probs = tl.exp(logits - lse[:, None])
        #  where targets[m] == offs_n[n]
        targets_mask = (targets[:, None] == offs_n[None, :])
        grad_logit = tl.where(targets_mask, probs - 1.0, probs) * scale
        # Handle ignore_index and padding
        grad_logit = tl.where(mask_m[:, None] & mask_n[None, :] & (targets[:, None] != ignore_index), grad_logit, 0.0)

        # dW: (Prob-Delta)^T @ X
        dw_acc += tl.dot(tl.trans(grad_logit), xv)
        if HAS_BIAS:
            db_acc += tl.sum(grad_logit, axis=0)

    tl.store(dw_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk, dw_acc, mask=mask_n[:, None] & mask_k[None, :])
    if HAS_BIAS and pid_k == 0:
        tl.store(db_ptr + offs_n, db_acc, mask=mask_n)


class FusedLinearCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, targets, ignore_index=-100):
        ctx.orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        M, K = x_flat.shape
        N = weight.shape[0]
        loss_vec = torch.empty(M, device=x.device, dtype=torch.float32)
        lse = torch.empty(M, device=x.device, dtype=torch.float32)
        fused_linear_ce_fwd_kernel[(M,)](
            x_flat, weight, bias, targets.view(-1), loss_vec, lse,
            M, K, N, x_flat.stride(0), x_flat.stride(1), weight.stride(0), weight.stride(1),
            ignore_index, bias is not None, BLOCK_N=64, BLOCK_K=32
        )
        num_valid = (targets.view(-1) != ignore_index).sum().clamp(min=1)
        ctx.save_for_backward(x_flat, weight, bias, targets.view(-1), lse)
        ctx.ignore_index, ctx.num_valid = ignore_index, num_valid
        return loss_vec.sum() / num_valid

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, targets, lse = ctx.saved_tensors
        M, K = x.shape
        N = weight.shape[0]
        scale = (grad_output / ctx.num_valid).item()

        grad_x = torch.empty_like(x)
        fused_linear_ce_bwd_dx_kernel[(M,)](
            x, weight, bias, targets, lse, grad_x,
            M, K, N, scale,
            x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            ctx.ignore_index, bias is not None, BLOCK_N=64, BLOCK_K=32
        )

        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty((N,), device=x.device) if bias is not None else None
        
        # Grid covers the Weight Matrix (N x K)
        grid_dw = (triton.cdiv(N, 64), triton.cdiv(K, 32))
        fused_linear_ce_bwd_dw_kernel[grid_dw](
            x, weight, bias, targets, lse, grad_weight, grad_bias,
            M, K, N, scale,
            x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
            ctx.ignore_index, bias is not None,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )
        return grad_x.view(ctx.orig_shape), grad_weight, grad_bias, None, None

def test():
    torch.manual_seed(42)
    B, T, C, V = 2, 4, 16, 32000 # Small V for precision check, set to 32000 for speed test
    device = "cuda"
    x = torch.randn(B, T, C, device=device, requires_grad=True)
    targets = torch.randint(0, V, (B, T), device=device)
    targets[0, 1] = -100
    weight = torch.randn(V, C, device=device, requires_grad=True)
    bias = torch.randn(V, device=device, requires_grad=True)

    ref_x, ref_w, ref_b = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_(), bias.detach().clone().requires_grad_()
    ref_loss = F.cross_entropy(F.linear(ref_x, ref_w, ref_b).view(-1, V), targets.view(-1), ignore_index=-100)
    
    tri_loss = FusedLinearCEFunction.apply(x, weight, bias, targets, -100)
    
    print(f"Loss Diff: {torch.abs(ref_loss - tri_loss).item():.6e}")
    ref_loss.backward(); tri_loss.backward()
    
    dx_diff = torch.abs(ref_x.grad - x.grad).max().item()
    dw_diff = torch.abs(ref_w.grad - weight.grad).max().item()
    db_diff = torch.abs(ref_b.grad - bias.grad).max().item() if bias is not None else 0
    
    print(f"dX Diff: {dx_diff:.6e} | dW Diff: {dw_diff:.6e} | dB Diff: {db_diff:.6e}")
    assert dx_diff < 1e-4 and dw_diff < 1e-4
    print(" SUCCESS: Fully fused forward and backward passed!")

if __name__ == "__main__":
    test()