import torch
import triton
import triton.language as tl
import triton.testing
import gc

@triton.jit
def rms_norm_forward_kernel(
    input_ptr, output_ptr, weight_ptr, rstd_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_rm,
    M, N, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M

    # calculate (x*x)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, N, BLOCK_N):
        c = k+ tl.arange(0, BLOCK_N)
        mask = (rows[:, None]<M) & (c[None, :] < N)
        x = tl.load(input_ptr + rows[:, None] * stride_xm + c[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=1)

    var = acc / N
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + rows * stride_rm, rstd, mask=row_mask)

    # Normalize + Scale
    for k in range(0, N, BLOCK_N):
        c =  k  + tl.arange(0, BLOCK_N)
        mask = (rows[:, None]<M) & (c[None, :] < N)
        x = tl.load(input_ptr + rows[:, None] * stride_xm + c[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + c, mask=c < N).to(tl.float32)
        
        y = x * rstd[:, None] * w[None, :]
        tl.store(output_ptr + rows[:, None] * stride_ym + c[None, :] * stride_yn, y, mask=mask)

@triton.jit
def rms_norm_backward_dx_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
    stride_dym, stride_dyn,
    stride_xm, stride_xn,
    stride_dxm, stride_dxn,
    stride_rm,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
   
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M

    rstd = tl.load(rstd_ptr + rows * stride_rm, mask=row_mask, other=0.0)

    # Row-wise dot product: sum(dY * W * X * rstd)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0,N, BLOCK_N):
        c =  k + tl.arange(0, BLOCK_N)
        mask = (rows[:, None]<M) & (c[None, :] < N)
        
        dy = tl.load(dy_ptr + rows[:, None] * stride_dym + c[None, :] * stride_dyn, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + rows[:, None] * stride_xm + c[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c, mask=c < N).to(tl.float32)
        
        row_sum += tl.sum(dy * w * x, axis=1)

    row_sum = row_sum * (rstd * rstd * rstd / N)

    # dX
    for k in range(0,N, BLOCK_N):
        c =  k + tl.arange(0, BLOCK_N)
        mask =(rows[:, None]<M) & (c[None, :] < N)
        
        dy = tl.load(dy_ptr + rows[:, None] * stride_dym + c[None, :] * stride_dyn, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + rows[:, None] * stride_xm + c[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c, mask=c < N).to(tl.float32)

        dx = dy * w * rstd[:, None] - x * row_sum[:, None]
        tl.store(dx_ptr + rows[:, None] * stride_dxm + c[None, :] * stride_dxn, dx, mask=mask)

@triton.jit
def rms_norm_backward_dw_kernel(
    dy_ptr, x_ptr, rstd_ptr, dw_ptr,
    stride_dym, stride_dyn,
    stride_xm, stride_xn,
    stride_rm,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m in range(0, M, BLOCK_M):
        rows = m + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]

        dy = tl.load(dy_ptr + rows[:, None] * stride_dym + cols[None, :] * stride_dyn, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows * stride_rm, mask=row_mask, other=0.0)

        acc += tl.sum(dy * (x * rstd[:, None]), axis=0)

    tl.store(dw_ptr + cols, acc, mask=col_mask)

class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty((M), device=x.device, dtype=torch.float32)
        
        BLOCK_M, BLOCK_N = 16, 64
        grid = (triton.cdiv(M, BLOCK_M),)
        
        rms_norm_forward_kernel[grid](
            x, y, weight, rstd,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            rstd.stride(0),
            M, N, eps,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        M, N = x.shape
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        
        BLOCK_M, BLOCK_N = 16, 64
        
        # Kernel 1: dX
        grid_dx = (triton.cdiv(M, BLOCK_M),)
        rms_norm_backward_dx_kernel[grid_dx](
            dy, x, weight, rstd, dx,
            dy.stride(0), dy.stride(1),
            x.stride(0), x.stride(1),
            dx.stride(0), dx.stride(1),
            rstd.stride(0),
            M, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

        # Kernel 2: dW
        grid_dw = (triton.cdiv(N, BLOCK_N),)
        rms_norm_backward_dw_kernel[grid_dw](
            dy, x, rstd, dw,
            dy.stride(0), dy.stride(1),
            x.stride(0), x.stride(1),
            rstd.stride(0),
            M, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        
        return dx, dw.to(weight.dtype), None



def test_correctness():
    M, N = 512, 2048
    dtype = torch.float32  # Use float32 for strict correctness checks
    eps = 1e-6

    # 1. Setup Input
    x = torch.randn((M, N), device='cuda', dtype=dtype, requires_grad=True)
    w = torch.randn(N, device='cuda', dtype=dtype, requires_grad=True)
    dy = torch.randn((M, N), device='cuda', dtype=dtype)

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
    
    y_tri = TritonRMSNorm.apply(x, w, eps)
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