import torch
import triton
import triton.language as tl
import triton.testing
import gc

@triton.jit
def rms_norm_forward_3d_kernel(
    input_ptr, output_ptr, weight_ptr, rstd_ptr,
    stride_xb, stride_xm, stride_xn,
    stride_rb, stride_rm,
    M, N, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (num_blocks_m, batch)
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M

    # Batch pointers
    batch_input_ptr = input_ptr + pid_b * stride_xb
    batch_output_ptr = output_ptr + pid_b * stride_xb
    batch_rstd_ptr = rstd_ptr + pid_b * stride_rb

    # 1. RMS Reduction
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        c = n + cols
        mask = row_mask[:, None] & (c[None, :] < N)
        x = tl.load(batch_input_ptr + rows[:, None] * stride_xm + c[None, :], mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=1)

    var = acc / N
    rstd = tl.rsqrt(var + eps)
    tl.store(batch_rstd_ptr + rows, rstd, mask=row_mask)

    # 2. Normalize + Scale
    for n in range(0, N, BLOCK_N):
        c = n + cols
        mask = row_mask[:, None] & (c[None, :] < N)
        x = tl.load(batch_input_ptr + rows[:, None] * stride_xm + c[None, :], mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + c, mask=c < N).to(tl.float32)
        
        y = x * rstd[:, None] * w[None, :]
        tl.store(batch_output_ptr + rows[:, None] * stride_xm + c[None, :], y, mask=mask)

@triton.jit
def rms_norm_backward_3d_kernel(
    dY_ptr, X_ptr, W_ptr, RSTD_ptr, dX_ptr, dW_ptr,
    stride_xb, stride_xm, stride_xn,
    stride_rb, stride_rm,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M

    batch_x_ptr = X_ptr + pid_b * stride_xb
    batch_dy_ptr = dY_ptr + pid_b * stride_xb
    batch_dx_ptr = dX_ptr + pid_b * stride_xb
    batch_rstd_ptr = RSTD_ptr + pid_b * stride_rb

    rstd = tl.load(batch_rstd_ptr + rows, mask=row_mask, other=0.0)

    # Pass 1: Row-wise dot
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        c = n + cols
        mask = row_mask[:, None] & (c[None, :] < N)
        x = tl.load(batch_x_ptr + rows[:, None] * stride_xm + c[None, :], mask=mask, other=0.0).to(tl.float32)
        do = tl.load(batch_dy_ptr + rows[:, None] * stride_xm + c[None, :], mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + c, mask=c < N).to(tl.float32)
        
        x_hat = x * rstd[:, None]
        row_sum += tl.sum(do * w * x_hat, axis=1)

    row_sum = row_sum / N

    # Pass 2: dX + dW
    for n in range(0, N, BLOCK_N):
        c = n + cols
        mask = row_mask[:, None] & (c[None, :] < N)
        x = tl.load(batch_x_ptr + rows[:, None] * stride_xm + c[None, :], mask=mask, other=0.0).to(tl.float32)
        do = tl.load(batch_dy_ptr + rows[:, None] * stride_xm + c[None, :], mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + c, mask=c < N).to(tl.float32)
        
        x_hat = x * rstd[:, None]
        dx = rstd[:, None] * (do * w - x_hat * row_sum[:, None])
        tl.store(batch_dx_ptr + rows[:, None] * stride_xm + c[None, :], dx, mask=mask)

        dw_local = tl.sum(do * x_hat, axis=0)
        tl.atomic_add(dW_ptr + c, dw_local, mask=c < N)

class TritonRMSNorm3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        B, M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty((B, M), device=x.device, dtype=torch.float32)
        
        BLOCK_M, BLOCK_N = 16, 1024
        grid = (triton.cdiv(M, BLOCK_M), B)
        
        rms_norm_forward_3d_kernel[grid](
            x, y, weight, rstd,
            x.stride(0), x.stride(1), x.stride(2),
            rstd.stride(0), rstd.stride(1),
            M, N, eps,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        B, M, N = x.shape
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        
        BLOCK_M, BLOCK_N = 16, 1024
        grid = (triton.cdiv(M, BLOCK_M), B)
        
        rms_norm_backward_3d_kernel[grid](
            dy, x, weight, rstd, dx, dw,
            x.stride(0), x.stride(1), x.stride(2),
            rstd.stride(0), rstd.stride(1),
            M, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        return dx, dw.to(weight.dtype), None


def test_correctness():
    B, M, N = 4, 64, 2048
    dtype = torch.float32  # Use float32 for strict correctness checks
    eps = 1e-6

    # 1. Setup Input
    x = torch.randn((B, M, N), device='cuda', dtype=dtype, requires_grad=True)
    w = torch.randn(N, device='cuda', dtype=dtype, requires_grad=True)
    dy = torch.randn((B, M, N), device='cuda', dtype=dtype)

    # 2. Reference (PyTorch)
    # We use a pure torch implementation for the ground truth
    def torch_rmsnorm(x, w, eps):
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * w
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * rms * w

    y_ref = torch_rmsnorm(x, w, eps)
    y_ref.backward(dy)
    
    dx_ref = x.grad.clone()
    dw_ref = w.grad.clone()
    
    # 3. Triton
    x.grad.zero_()
    w.grad.zero_()
    
    y_tri = TritonRMSNorm3D.apply(x, w, eps)
    y_tri.backward(dy)
    
    dx_tri = x.grad.clone()
    dw_tri = w.grad.clone()

    # 4. Verification
    # Check Forward
    fwd_max_diff = (y_tri - y_ref).abs().max().item()
    fwd_close = torch.allclose(y_tri, y_ref, atol=1e-5)
    
    # Check Backward dX
    dx_max_diff = (dx_tri - dx_ref).abs().max().item()
    dx_close = torch.allclose(dx_tri, dx_ref, atol=1e-5)
    
    # Check Backward dW
    dw_max_diff = (dw_tri - dw_ref).abs().max().item()
    dw_close = torch.allclose(dw_tri, dw_ref, atol=1e-5)

    print("--- Correctness Report ---")
    print(f"Forward: {'PASS' if fwd_close else 'FAIL'} (Max Diff: {fwd_max_diff:.2e})")
    print(f"dX:      {'PASS' if dx_close else 'FAIL'} (Max Diff: {dx_max_diff:.2e})")
    print(f"dW:      {'PASS' if dw_close else 'FAIL'} (Max Diff: {dw_max_diff:.2e})")

if __name__ == "__main__":
    test_correctness()