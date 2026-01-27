import torch
import triton
import triton.language as tl

@triton.jit
def linear_kernel_fwd(
    x_ptr, w_ptr, y_ptr, b_ptr,
    B, M, K, N,
    stride_xb, stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptr += pid_b * stride_xb
    y_ptr += pid_b * stride_yb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        w = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )

        acc += tl.dot(x, w)

    if ADD_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def linear_dx_kernel(
    dy_ptr, w_ptr, dx_ptr,
    B, M, K, N,
    stride_dyb, stride_dym, stride_dyn,
    stride_wn, stride_wk,
    stride_dxb, stride_dxm, stride_dxk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    dy_ptr += pid_b * stride_dyb
    dx_ptr += pid_b * stride_dxb

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)

        dy = tl.load(
            dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )

        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc += tl.dot(dy, w)

    tl.store(
        dx_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk,
        acc.to(dx_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )

@triton.jit
def linear_dw_kernel(
    dy_ptr, x_ptr, dw_ptr,
    B, M, K, N,
    stride_dyb, stride_dym, stride_dyn,
    stride_xb, stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    dy_ptr += pid_b * stride_dyb
    x_ptr += pid_b * stride_xb

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m in range(0, tl.cdiv(M, BLOCK_M)):
        offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)

        dy = tl.load(
            dy_ptr + offs_m[None, :] * stride_dym + offs_n[:, None] * stride_dyn,
            mask=(offs_n[:, None] < N) & (offs_m[None, :] < M),
            other=0.0,
        )

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc += tl.dot(dy, x)

    tl.atomic_add(
        dw_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
        acc.to(dw_ptr.dtype.element_ty),
        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
    )

class TritonLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b=None):
        B, M, K = x.shape
        N, _ = w.shape

        y = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64), B)

        linear_kernel_fwd[grid](
            x, w, y, b,
            B, M, K, N,
            *x.stride(),
            w.stride(1), w.stride(0),
            *y.stride(),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            ADD_BIAS=b is not None,
        )

        ctx.save_for_backward(x, w, b)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b = ctx.saved_tensors
        B, M, K = x.shape
        N, _ = w.shape

        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)

        grid_dx = (triton.cdiv(M, 64), triton.cdiv(K, 64), B)
        linear_dx_kernel[grid_dx](
            dy, w, dx,
            B, M, K, N,
            *dy.stride(),
            *w.stride(),
            *dx.stride(),
            BLOCK_M=64, BLOCK_K=64, BLOCK_N=32,
        )

        grid_dw = (triton.cdiv(N, 64), triton.cdiv(K, 64), B)
        linear_dw_kernel[grid_dw](
            dy, x, dw,
            B, M, K, N,
            *dy.stride(),
            *x.stride(),
            *w.stride(),
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=64,
        )

        db = dy.sum(dim=(0, 1)) if b is not None else None
        return dx, dw, db

def test():
    B, M, K, N = 8, 128, 256, 512
    device = "cuda"
    dtype=torch.float32
    x = torch.randn(B, M, K, device=device,dtype=dtype, requires_grad=True)
    w = torch.randn(N, K, device=device,dtype=dtype, requires_grad=True)
    b = torch.randn(N, device=device, dtype=dtype,requires_grad=True)

    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()

    y = TritonLinear.apply(x, w, b)
    y_ref = x_ref @ w_ref.t() + b_ref

    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
    print("Forward OK")

    y.sum().backward()
    y_ref.sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w.grad, w_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(b.grad, b_ref.grad, atol=2e-2, rtol=2e-2)

    print("Backward OK")


if __name__ == "__main__":
    test()
