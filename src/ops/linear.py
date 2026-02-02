import triton
import triton.language as tl
import torch
from torch import nn
import torch.nn.functional as F

@triton.jit
def linear_kernel_fwd(
    A,
    B,
    C,
    Bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # CTA reordering
    if GROUP_M > 1:
        grid_m = tl.num_programs(0)
        grid_n = tl.num_programs(1)

        pid = pid_m + pid_n * grid_m
        num_cta_per_group = grid_n * GROUP_M

        group_id = pid // num_cta_per_group
        inner = pid % num_cta_per_group

        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > grid_m,
            grid_m % GROUP_M,
            GROUP_M,
        )

        pid_m = group_id * GROUP_M + inner % group_size
        pid_n = inner // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    num_iters = tl.cdiv(K, TILE_K)

    for _ in range(num_iters):
        if DIVISIBLE_K:
            mask_a = None if DIVISIBLE_M else mask_m[:, None]
            mask_b = None if DIVISIBLE_N else mask_n[None, :]
        else:
            mask_k = offs_k < K
            mask_a = mask_k[None, :] if DIVISIBLE_M else mask_m[:, None] & mask_k[None, :]
            mask_b = mask_k[:, None] if DIVISIBLE_N else mask_k[:, None] & mask_n[None, :]

        if mask_a is not None:
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        else:
            a = tl.load(a_ptrs)

        if mask_b is not None:
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        else:
            b = tl.load(b_ptrs)

        acc += tl.dot(a, b)

        offs_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    if ADD_BIAS:
        if not DIVISIBLE_N:
            bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        else:
            bias = tl.load(Bias_ptr + offs_n)

        acc += bias[None, :]

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M:
        mask_c = mask_n[None, :]
    elif DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]

    tl.store(c_ptrs, acc, mask=mask_c)

@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # CTA reordering 
    if GROUP_M > 1:
        grid_m = tl.num_programs(0)
        grid_n = tl.num_programs(1)

        pid = pid_m + pid_n * grid_m
        num_cta_per_group = grid_n * GROUP_M

        group_id = pid // num_cta_per_group
        inner = pid % num_cta_per_group

        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > grid_m,
            grid_m % GROUP_M,
            GROUP_M,
        )

        pid_m = group_id * GROUP_M + inner % group_size
        pid_n = inner // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    num_iters = tl.cdiv(K, TILE_K)

    for _ in range(num_iters):
        if DIVISIBLE_K:
            mask_a = None if DIVISIBLE_M else mask_m[:, None]
            mask_b = None if DIVISIBLE_N else mask_n[None, :]
        else:
            mask_k = offs_k < K
            mask_a = mask_k[None, :] if DIVISIBLE_M else mask_m[:, None] & mask_k[None, :]
            mask_b = mask_k[:, None] if DIVISIBLE_N else mask_k[:, None] & mask_n[None, :]

        if mask_a is not None:
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        else:
            a = tl.load(a_ptrs)

        if mask_b is not None:
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        else:
            b = tl.load(b_ptrs)

        acc += tl.dot(a, b)

        offs_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk


    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M:
        mask_c = mask_n[None, :]
    elif DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]

    tl.store(c_ptrs, acc, mask=mask_c)

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
        div_m = M % 64 == 0
        div_n = N % 64 == 0
        div_k = K % 32 == 0
        
        grid = lambda meta: (
        triton.cdiv(M, meta["TILE_M"]),
        triton.cdiv(N, meta["TILE_N"]),
    )
        
        # X @ W.T+b
        linear_kernel_fwd[grid](
            x, weight, y, bias,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0), # Treats (N, K) as (K, N)
            y.stride(0), y.stride(1),
            TILE_M=64,
            TILE_N=64,
            TILE_K=32,
            GROUP_M=8,
            DIVISIBLE_M=div_m,
            DIVISIBLE_N=div_n,
            DIVISIBLE_K=div_k,
            ADD_BIAS=bias is not None
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

        # Helper 
        def is_div(val, tile): return val % tile == 0

        # (M, N) @ (N, K) = (M, K)
        # (M/TILE_M, K/TILE_N)
        grid_dx = lambda meta: (
            triton.cdiv(M, meta["TILE_M"]),
            triton.cdiv(K, meta["TILE_N"]),
        )
        matmul_kernel[grid_dx](
            dy, weight, dx,
            M, K, N, # M, N, K for the kernel
            dy.stride(0), dy.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32,
            GROUP_M=8,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(K, 64), # Output N is K
            DIVISIBLE_K=is_div(N, 32)   # Inner K is N
        )

        #  (N, M) @ (M, K) = (N, K)
        grid_dw = lambda meta: (
            triton.cdiv(N, meta["TILE_M"]),
            triton.cdiv(K, meta["TILE_N"]),
        )
        matmul_kernel[grid_dw](
            dy, x, dw,
            N, K, M, # M, N, K for the kernel
            dy.stride(1), dy.stride(0), # Transpose dy
            x.stride(0), x.stride(1),
            dw.stride(0), dw.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32,
            GROUP_M=8,
            DIVISIBLE_M=is_div(N, 64),
            DIVISIBLE_N=is_div(K, 64),
            DIVISIBLE_K=is_div(M, 32)
        )

        db = dy.sum(0)

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
        orig_shape = x.shape
        if x.ndim > 2:
            x = x.view(-1, orig_shape[-1])
            
        y = TritonLinearFunction.apply(x, self.weight, self.bias)
        
        # Reshape back to (Batch, Seq, N)
        if len(orig_shape) > 2:
            y = y.view(*orig_shape[:-1], -1)
        return y
            

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

   
    batch_size = 512
    in_features = 512
    out_features = 512

    # Input tensor
    x = torch.randn(batch_size,512, in_features, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_()


    triton_model = TritonLinear(in_features, out_features).to(device).to(dtype)

   
    torch_model = nn.Linear(in_features, out_features, bias=True).to(device).to(dtype)

    # Copy weights and bias for exact match
    with torch.no_grad():
        torch_model.weight.copy_(triton_model.weight)
        torch_model.bias.copy_(triton_model.bias)

  
    y_triton = triton_model(x)
    y_torch = torch_model(x_ref)

   
    target = torch.randn(batch_size,512, out_features, device=device, dtype=dtype)
    loss_triton = F.mse_loss(y_triton, target)
    loss_torch = F.mse_loss(y_torch, target)

 
    loss_triton.backward()
    loss_torch.backward()

  
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
