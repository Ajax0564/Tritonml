import triton
import triton.language as tl
import torch
from torch import nn
import torch.nn.functional as F

@triton.jit
def matmul_kernel_fwd(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ADD_BIAS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
   

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr+offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=(offs_k[None, :] < K) & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptr+offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)


    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
       
        a = tl.load(a_ptr+offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=(offs_k[None, :]<K) & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptr+offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=(offs_k[:, None]<K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

# Bias Gradient Kernel (Reduction)
@triton.jit
def linear_bw_db_kernel(
    dy_ptr, db_ptr, 
    M, N, 
    stride_dym, stride_dyn, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m in range(0, tl.cdiv(M, BLOCK_M)):
        offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        dy = tl.load(dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn, mask=mask, other=0.0)
        acc += tl.sum(dy, axis=0)

    tl.store(db_ptr + offs_n, acc, mask=offs_n < N)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: (M, K), weight: (N, K) -> y: (M, N)
        M, K = x.shape
        N, _ = weight.shape
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        
        # We pass weight.T because standard math is X @ W.T
        matmul_kernel_fwd[grid](
            x, weight, y, bias,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0), # Treats (N, K) as (K, N)
            y.stride(0), y.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, ADD_BIAS=bias is not None
        )
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        M, K = x.shape
        N, _ = weight.shape
        
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        db = torch.empty_like(bias) if bias is not None else None

        grid_dx = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(K, META['BLOCK_N']),)
        grid_dw = lambda META: (triton.cdiv(N, META['BLOCK_M']) * triton.cdiv(K, META['BLOCK_N']),)

        # 1. dx = dy @ weight -> (M, N) @ (N, K) = (M, K)
        matmul_kernel[grid_dx](
            dy, weight, dx,
            M, K, N,
            dy.stride(0), dy.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 2. dw = dy.T @ x -> (N, M) @ (M, K) = (N, K)
        matmul_kernel[grid_dw](
            dy, x, dw,
            N, K, M,
            dy.stride(1), dy.stride(0), # Transpose dy: (N, M)
            x.stride(0), x.stride(1),    # x: (M, K)
            dw.stride(0), dw.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 3. db = dy.sum(0)
        if bias is not None:
            grid_db = (triton.cdiv(N, 64),)
            linear_bw_db_kernel[grid_db](
                dy, db, M, N,
                dy.stride(0), dy.stride(1),
                BLOCK_M=128, BLOCK_N=64
            )

        return dx, dw, db

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight, self.bias)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # -------------------------
    # Model and data parameters
    # -------------------------
    batch_size = 32
    in_features = 128
    out_features = 64

    # Input tensor
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_()

    # -------------------------
    # Triton model
    # -------------------------
    triton_model = TritonLinear(in_features, out_features).to(device).to(dtype)

    # -------------------------
    # PyTorch reference model
    # -------------------------
    torch_model = nn.Linear(in_features, out_features, bias=True).to(device).to(dtype)

    # Copy weights and bias for exact match
    with torch.no_grad():
        torch_model.weight.copy_(triton_model.weight)
        torch_model.bias.copy_(triton_model.bias)

    # -------------------------
    # Forward pass
    # -------------------------
    y_triton = triton_model(x)
    y_torch = torch_model(x_ref)

    # -------------------------
    # Loss
    # -------------------------
    target = torch.randn(batch_size, out_features, device=device, dtype=dtype)
    loss_triton = F.mse_loss(y_triton, target)
    loss_torch = F.mse_loss(y_torch, target)

    # -------------------------
    # Backward pass
    # -------------------------
    loss_triton.backward()
    loss_torch.backward()

    # -------------------------
    # Compare results
    # -------------------------
    def compare(name, a, b, atol=1e-2, rtol=1e-2):
        max_diff = (a - b).abs().max().item()
        rel_diff = max_diff / (b.abs().max().item() + 1e-6)
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        print(f"{name:12s} | ok={ok} | max={max_diff:.3e} | rel={rel_diff:.3e}")

    print("\n=== Forward ===")
    compare("output", y_triton, y_torch)

    print("\n=== Backward ===")
    compare("dX", x.grad, x_ref.grad)
    compare("dW", triton_model.weight.grad, torch_model.weight.grad)
    compare("dBias", triton_model.bias.grad, torch_model.bias.grad)

    print("\n Comparison complete")
