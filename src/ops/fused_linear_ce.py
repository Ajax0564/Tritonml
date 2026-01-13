import torch
import triton
import triton.language as tl

@triton.jit
def linear_softmax_ce_fwd(
    X, W, B, T,
    LOSS,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_b,
    stride_t,
    stride_l,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    pid = tl.program_id(0) # Row index (M)
    
    # Load target
    target = tl.load(T + pid * stride_t)
    if target == IGNORE_INDEX:
        tl.store(LOSS + pid * stride_l, 0.0)
        return

    # Load input row X[pid, :]
    offs_k = tl.arange(0, K)
    x = tl.load(X + pid * stride_xm + offs_k * stride_xk)

    # -----------------------------------------------------------
    # Pass 1: Find Global Max Logit for numerical stability
    # -----------------------------------------------------------
    max_logit = -float("inf")
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        
        # W is (K, N), so we load (K, BLOCK_N)
        w = tl.load(W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn, mask=mask[None, :], other=0.0)
        logits = tl.sum(x[:, None] * w, axis=0)
        if B is not None:
            logits += tl.load(B + offs_n * stride_b, mask=mask, other=0.0)
        
        max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

    # -----------------------------------------------------------
    # Pass 2: Compute Denominator (Sum of Exps) and Target Logit
    # -----------------------------------------------------------
    sum_exp = 0.0
    target_logit = 0.0

    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask = offs_n < N

        w = tl.load(W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn, mask=mask[None, :], other=0.0)
        logits = tl.sum(x[:, None] * w, axis=0)
        if B is not None:
            logits += tl.load(B + offs_n * stride_b, mask=mask, other=0.0)

        # Extract target logit before subtracting max
        is_target = offs_n == target
        # Sum only the logit where mask is true and it is the target
        target_logit += tl.sum(tl.where(is_target, logits, 0.0), axis=0)

        # Log-Sum-Exp part
        sum_exp += tl.sum(tl.exp(logits - max_logit), axis=0)

    # Loss = log(sum(exp(x - max))) + max - x_target
    loss = tl.log(sum_exp) + max_logit - target_logit
    tl.store(LOSS + pid * stride_l, loss)


def test_linear_softmax_ce(M=128, K=512, N=1024, block_n=128):
    torch.manual_seed(42)
    
    # Inputs
    X = torch.randn((M, K), device='cuda', dtype=torch.float32)
    W = torch.randn((K, N), device='cuda', dtype=torch.float32)
    B = torch.randn((N,), device='cuda', dtype=torch.float32)
    T = torch.randint(0, N, (M,), device='cuda', dtype=torch.int64)
    
    # Triton Output
    loss_triton = torch.zeros(M, device='cuda', dtype=torch.float32)
    
    grid = (M,)
    linear_softmax_ce_fwd[grid](
        X, W, B, T,
        loss_triton,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        B.stride(0),
        T.stride(0),
        loss_triton.stride(0),
        K=K, N=N, BLOCK_N=block_n, IGNORE_INDEX=-100
    )

    # PyTorch Reference
    # Reference does: (X @ W + B) then CrossEntropy
    logits_ref = torch.matmul(X, W) + B
    loss_ref = torch.nn.functional.cross_entropy(logits_ref, T, reduction='none')

    # Comparison
    print(f"Max Difference: {torch.max(torch.abs(loss_triton - loss_ref)).item()}")
    torch.testing.assert_close(loss_triton, loss_ref, atol=1e-4, rtol=1e-4)
    print("✅ Triton kernel matches PyTorch reference!")

@triton.jit
def linear_softmax_ce_bwd(
    X, W, B, T,
    DX, DW, DB,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_b,
    stride_t,
    stride_dxm, stride_dxk,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    pid = tl.program_id(0) # Row index
    
    target = tl.load(T + pid * stride_t)
    if target == IGNORE_INDEX:
        return

    offs_k = tl.arange(0, K)
    x = tl.load(X + pid * stride_xm + offs_k * stride_xk)

    # --- Recompute LogSumExp (Pass 1 & 2 logic from FWD) ---
    # In a real system, you'd pass the LSE from FWD to save time.
    max_logit = -float("inf")
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        w = tl.load(W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn, mask=mask[None, :], other=0.0)
        logits = tl.sum(x[:, None] * w, axis=0)
        if B is not None:
            logits += tl.load(B + offs_n * stride_b, mask=mask, other=0.0)
        max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

    sum_exp = 0.0
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        w = tl.load(W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn, mask=mask[None, :], other=0.0)
        logits = tl.sum(x[:, None] * w, axis=0)
        if B is not None:
            logits += tl.load(B + offs_n * stride_b, mask=mask, other=0.0)
        sum_exp += tl.sum(tl.exp(logits - max_logit), axis=0)

    # --- Pass 3: Compute Gradients ---
    lse = tl.log(sum_exp) + max_logit
    dx_accumulator = tl.zeros([K], dtype=tl.float32)

    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask = offs_n < N

        w_ptr = W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptr, mask=mask[None, :], other=0.0)
        
        logits = tl.sum(x[:, None] * w, axis=0)
        if B is not None:
            logits += tl.load(B + offs_n * stride_b, mask=mask, other=0.0)

        # Gradient of Logits: p - y
        probs = tl.exp(logits - lse)
        is_target = offs_n == target
        grad_logits = probs - tl.where(is_target, 1.0, 0.0)
        grad_logits = tl.where(mask, grad_logits, 0.0)

        # Update DX: grad_logits @ W.T
        dx_accumulator += tl.sum(grad_logits[None, :] * w, axis=1)

        # Update DW: X.T @ grad_logits (using atomics)
        dw_ptr = DW + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        tl.atomic_add(dw_ptr, x[:, None] * grad_logits[None, :], mask=mask[None, :])

        # Update DB: sum(grad_logits) (using atomics)
        if B is not None:
            tl.atomic_add(DB + offs_n * stride_b, grad_logits, mask=mask)

    # Store DX row
    tl.store(DX + pid * stride_dxm + offs_k * stride_dxk, dx_accumulator)

import torch

def test_backward(M=64, K=256, N=512):
    # Setup
    X = torch.randn((M, K), device='cuda', requires_grad=True)
    W = torch.randn((K, N), device='cuda', requires_grad=True)
    B = torch.randn((N,), device='cuda', requires_grad=True)
    T = torch.randint(0, N, (M,), device='cuda')

    # --- PyTorch Reference ---
    logits = X @ W + B
    loss = torch.nn.functional.cross_entropy(logits, T)
    loss.backward()
    
    ref_dx, ref_dw, ref_db = X.grad.clone(), W.grad.clone(), B.grad.clone()
    X.grad, W.grad, B.grad = None, None, None

    # --- Triton Implementation ---
    DX = torch.zeros_like(X)
    DW = torch.zeros_like(W)
    DB = torch.zeros_like(B)
    
    # We use M (rows) as the grid
    grid = (M,)
    linear_softmax_ce_bwd[grid](
        X, W, B, T,
        DX, DW, DB,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        B.stride(0), T.stride(0),
        DX.stride(0), DX.stride(1),
        K=K, N=N, BLOCK_N=128, IGNORE_INDEX=-100
    )
    
    # Normalize by batch size since PyTorch CE defaults to 'mean' reduction
    DX /= M
    DW /= M
    DB /= M

    # Compare
    print(f"DX match: {torch.allclose(DX, ref_dx, atol=1e-5)}")
    print(f"DW match: {torch.allclose(DW, ref_dw, atol=1e-5)}")
    print(f"DB match: {torch.allclose(DB, ref_db, atol=1e-5)}")
