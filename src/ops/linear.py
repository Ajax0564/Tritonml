import triton
import triton.language as tl
import torch
from torch import nn

@triton.jit
def _linear_kernel_fwd(
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
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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
def _matmul_kernel(
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
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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
def _linear_bw_db_kernel(
    dy_ptr, db_ptr, 
    M, N, 
    stride_dym, stride_dyn, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    mask_n = offs_n < N

    # Loop over M 
    for m in range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask = (offs_m[:, None] < M) & (mask_n[None, :])
        
        # Load a tile of dy
        dy = tl.load(dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn, 
                     mask=mask, other=0.0)
        
        
        acc += tl.sum(dy, axis=0)
    tl.store(db_ptr + offs_n, acc, mask=mask_n)
    
def is_div(val, tile): return val % tile == 0
    
class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: (M, K), w: (N, K) -> y: (M, N)
        M, K = x.shape
        N, _ = weight.shape
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (
        triton.cdiv(M, meta["TILE_M"]),
        triton.cdiv(N, meta["TILE_N"]),
    )
        
        # X @ W.T+b
        _linear_kernel_fwd[grid](
            x, weight, y, bias if bias is not None else x,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0), # (N, K) as (K, N)
            y.stride(0), y.stride(1),
            TILE_M=64,
            TILE_N=64,
            TILE_K=32,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(N, 64), # Output N is K
            DIVISIBLE_K=is_div(K, 32),   # Inner K is N
            ADD_BIAS=bias is not None,
            
        )
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        M, K = x.shape
        N, _ = weight.shape
        
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)

        # (M, N) @ (N, K) = (M, K)
        grid_dx = lambda meta: (
            triton.cdiv(M, meta["TILE_M"]),
            triton.cdiv(K, meta["TILE_N"]),
        )
        _matmul_kernel[grid_dx](
            dy, weight, dx,
            M, K, N, # M, N, K for the kernel
            dy.stride(0), dy.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
            TILE_M=64, TILE_N=32, TILE_K=64,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(K, 64), # Output N is K
            DIVISIBLE_K=is_div(N, 32), )

        #  (N, M) @ (M, K) = (N, K)
        grid_dw = lambda meta: (
            triton.cdiv(N, meta["TILE_M"]),
            triton.cdiv(K, meta["TILE_N"]),
        )
        _matmul_kernel[grid_dw](
            dy, x, dw,
            N, K, M, # M, N, K for the kernel
            dy.stride(1), dy.stride(0), # Transpose dy
            x.stride(0), x.stride(1),
            dw.stride(0), dw.stride(1),
            TILE_M=32, TILE_N=64, TILE_K=64,
            DIVISIBLE_M=is_div(N, 64),
            DIVISIBLE_N=is_div(K, 64), # Output N is K
            DIVISIBLE_K=is_div(N, 32), )

        
        if  not ctx.has_bias:
            db = None
        else:
            db =  torch.empty(N).to(x.device)
            grid = (triton.cdiv(N, 128),) 
            _linear_bw_db_kernel[grid](
                dy, db, M, N, 
                dy.stride(0), dy.stride(1),
                BLOCK_M=256, # Increase this to do more work per load
                BLOCK_N=128
            )

        return dx, dw, db
        
class TritonLinearLayer(nn.Module):
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