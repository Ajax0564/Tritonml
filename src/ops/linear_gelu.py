import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def _linear_layer_gelu_fwd(
    A,
    B,
    C,
    Bias_ptr,
    Z,
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
    z_ptrs = Z + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

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
    
    tl.store(z_ptrs,acc, mask=mask_c)
    
    acc = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    
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
def _gelu_backward_kernel(
    dz_ptr, dy_ptr, z_ptr,
    M, N,
    stride_dym, stride_dyn,
    stride_zm, stride_zn,
    stride_dzm, stride_dzn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (rows[:, None] < M) & (cols[None, :] < N)

    dy = tl.load(dy_ptr + rows[:, None] * stride_dym + cols[None, :] * stride_dyn, mask=mask, other=0.0)
    z  = tl.load(z_ptr  + rows[:, None] * stride_zm  + cols[None, :] * stride_zn,  mask=mask, other=0.0)

    z_f = z.to(tl.float32)
    s2i, s2pi = 0.707106781, 0.39894228
    cdf = 0.5 * (1 + tl.math.erf(z_f * s2i))
    pdf = s2pi * tl.exp(-0.5 * z_f * z_f)

    dz = dy * (cdf + z_f * pdf)

    tl.store(dz_ptr + rows[:, None] * stride_dzm + cols[None, :] * stride_dzn, dz, mask=mask)

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
    offs_m =  tl.arange(0, BLOCK_M)

    # Loop over M 
    for m in range(0, M, BLOCK_M):
        offs_m = m + offs_m
        mask = (offs_m[:, None] < M) & (mask_n[None, :])
        
        # Load a tile of dy
        dy = tl.load(dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn, 
                     mask=mask, other=0.0)
        
        
        acc += tl.sum(dy, axis=0)
    tl.store(db_ptr + offs_n, acc, mask=mask_n)
    
def is_div(val, tile): return val % tile == 0

class TritonLinearGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b=None):
        M, K = x.shape
        N, K2 = w.shape # w is (out_features, in_features)
        assert K == K2, f"Incompatible shapes: x({K}) and w({K2})"

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        z = torch.empty_like(y)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _linear_layer_gelu_fwd[grid](
            x, w, y, b, z,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(1), w.stride(0), # Treat W as (K, N)
            y.stride(0), y.stride(1),
            TILE_M=BLOCK_M, TILE_N=BLOCK_N, TILE_K=BLOCK_K,
            DIVISIBLE_M=is_div(M, BLOCK_M),
            DIVISIBLE_N=is_div(N, BLOCK_N),
            DIVISIBLE_K=is_div(K, BLOCK_K),
            ADD_BIAS=b is not None,
        )

        ctx.save_for_backward(x, w, z)
        ctx.has_bias = b is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, z = ctx.saved_tensors
        M, K = x.shape
        N, _ = w.shape

        dx = torch.empty_like(x)
        dw = torch.empty_like(w)
        dz = torch.empty_like(dy)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

        # dZ = dY * GELU'(Z)
        grid_dz = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _gelu_backward_kernel[grid_dz](
            dz, dy, z, M, N,
            dy.stride(0), dy.stride(1),
            z.stride(0), z.stride(1),
            dz.stride(0), dz.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # dX = dZ @ W  -> (M, N) @ (N, K) = (M, K)
        grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_dx](
            dz, w, dx,
            M, K, N, # M=M, N=K, K=N
            dz.stride(0), dz.stride(1),
            w.stride(0), w.stride(1), # Use W as (N, K)
            dx.stride(0), dx.stride(1),
            TILE_M=BLOCK_M, TILE_N=BLOCK_N, TILE_K=BLOCK_K,
            DIVISIBLE_M=is_div(M, BLOCK_M),
            DIVISIBLE_N=is_div(K, BLOCK_N),
            DIVISIBLE_K=is_div(N, BLOCK_K),
        )

        # dW = dZ^T @ X -> (N, M) @ (M, K) = (N, K)
        grid_dw = (triton.cdiv(N, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_dw](
            dz, x, dw,
            N, K, M, # M=N, N=K, K=M
            dz.stride(1), dz.stride(0), # dZ to (N, M)
            x.stride(0), x.stride(1),    # Use X as (M, K)
            dw.stride(0), dw.stride(1),
            TILE_M=BLOCK_M, TILE_N=BLOCK_N, TILE_K=BLOCK_K,
            DIVISIBLE_M=is_div(N, BLOCK_M),
            DIVISIBLE_N=is_div(K, BLOCK_N),
            DIVISIBLE_K=is_div(M, BLOCK_K),
        )
        if not ctx.has_bias:
            db = None
        else:
            db =  torch.empty(N).to(x.device)
            grid = (triton.cdiv(N, 128),) 
            _linear_bw_db_kernel[grid](
                dz, db, M, N, 
                dz.stride(0), dz.stride(1),
                BLOCK_M=256, # Increase load
                BLOCK_N=128
            )

        return dx, dw, db
        
class TritonLinearGeluLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)


        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Input x: [*, in_features]
        """
       
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, orig_shape[-1])

    
        out = TritonLinearGELU.apply(
            x, 
            self.weight, 
            self.bias )

        # Restore original shape
        if len(orig_shape) > 2:
            output_shape = list(orig_shape[:-1]) + [self.out_features]
            out = out.view(*output_shape)
            
        return out