import torch
import triton
import triton.language as tl
import math 

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
def _rms_norm_forward_kernel(
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
        c = k+ cols
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
def _rms_norm_backward_dx_kernel(
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

    # Row-wise dot product -> sum(dY * W * X * rstd)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0,N, BLOCK_N):
        c =  k + cols
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
def _rms_norm_backward_dw_kernel(
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
    rows = tl.arange(0, BLOCK_M)
    for m in range(0, M, BLOCK_M):
        rows = m + rows
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]

        dy = tl.load(dy_ptr + rows[:, None] * stride_dym + cols[None, :] * stride_dyn, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows * stride_rm, mask=row_mask, other=0.0)

        acc += tl.sum(dy * (x * rstd[:, None]), axis=0)

    tl.store(dw_ptr + cols, acc, mask=col_mask)


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

class TritonLinearRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, rms_weight, b=None, eps=1e-5):
        """
        x: [M, K]
        w: [N, K]  
        b: [N] or None
        rms_weight: [N] 
        """
        M, K = x.shape
        N, K2 = w.shape
        assert K == K2, f"Mismatch in K: x={K}, w={K2}"

        # x @ w.T
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = 64,64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        # w is [N, K]
        _linear_kernel_fwd[grid](
            x, w, y, b,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(1), w.stride(0),  # w.T
            y.stride(0), y.stride(1),
            TILE_M=BLOCK_M,
            TILE_N=BLOCK_N,
            TILE_K=BLOCK_K,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(N, 64), # Output N is K
            DIVISIBLE_K=is_div(K, 32),   # Inner K is N
            ADD_BIAS=b is not None,
        )

        # RMSNorm output
        z = torch.empty_like(y)
        rstd = torch.empty((M,), device=x.device, dtype=x.dtype)
        _rms_norm_forward_kernel[(triton.cdiv(M, BLOCK_M),)](
            y, z, rms_weight, rstd,
            y.stride(0), y.stride(1),
            z.stride(0), z.stride(1),
            rstd.stride(0),
            M, N, eps,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x, w, y, rstd, rms_weight)
        ctx.has_bias = b is not None
        ctx.eps = eps
        return z

    @staticmethod
    def backward(ctx, dz):
        x, w, y, rstd, rms_weight = ctx.saved_tensors
        M, K = x.shape
        N, _ = w.shape

        #  dY
        dy = torch.empty_like(y)
        BLOCK_M, BLOCK_N = 64, 64
        
        # 
        _rms_norm_backward_dx_kernel[(triton.cdiv(M, BLOCK_M),)](
            dz, y, rms_weight, rstd, dy,
            dz.stride(0), dz.stride(1),
            y.stride(0), y.stride(1),
            dy.stride(0), dy.stride(1),
            rstd.stride(0),
            M, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        #  [M, N] @ [N, K] -> [M, K])
        dx = torch.empty_like(x)
        grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_dx](
            dy, w, dx,
            M, K, N,
            dy.stride(0), dy.stride(1),
            w.stride(0), w.stride(1),
            dx.stride(0), dx.stride(1),
            TILE_M=BLOCK_M, TILE_N=BLOCK_N, TILE_K=32,
            DIVISIBLE_M=is_div(M, BLOCK_M),
            DIVISIBLE_N=is_div(K, BLOCK_N), # Output N is K
            DIVISIBLE_K=is_div(N, 32), 
        )

        #  [N, M] @ [M, K] -> [N, K])
        dw = torch.empty_like(w)
        grid_dw = (triton.cdiv(N, BLOCK_M), triton.cdiv(K, BLOCK_N))
        #MNK->NKM
        _matmul_kernel[grid_dw](
            dy, x, dw,
            N, K, M,
            dy.stride(1), dy.stride(0), # Transpose dy: [N, M]
            x.stride(0), x.stride(1),   # x: [M, K]
            dw.stride(0), dw.stride(1),
            TILE_M=BLOCK_N, TILE_N=BLOCK_N, TILE_K=32,
            DIVISIBLE_M=is_div(N, BLOCK_M),
            DIVISIBLE_N=is_div(K, BLOCK_N), 
            DIVISIBLE_K=is_div(M, 32), 
        )
        
        if not ctx.has_bias:
            db = None
        else:
            db =  torch.empty(N).to(x.device)
            grid = (triton.cdiv(N, 128),) 
            _linear_bw_db_kernel[grid](
                dy, db, M, N, 
                dy.stride(0), dy.stride(1),
                BLOCK_M=256, # Increase load
                BLOCK_N=128
            )
        drms = torch.empty_like(rms_weight)
        _rms_norm_backward_dw_kernel[(triton.cdiv(N, BLOCK_N),)](
            dz, y, rstd, drms,
            dz.stride(0), dz.stride(1),
            y.stride(0), y.stride(1),
            rstd.stride(0),
            M, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        return dx, dw, drms, db, None

class TritonLinearRMSNormLayer(torch.nn.Module):
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

        
        self.rms_weight = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        # Standard Kaiming initialization for Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize RMSNorm weight to ones
        torch.nn.init.ones_(self.rms_weight)

    def forward(self, x):
        """
        Input x: [*, in_features]
        """
       
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, orig_shape[-1])

    
        out = TritonLinearRMSNorm.apply(
            x, 
            self.weight, 
            self.rms_weight, 
            self.bias, 
            self.eps
        )

        # Restore original shape
        if len(orig_shape) > 2:
            output_shape = list(orig_shape[:-1]) + [self.out_features]
            out = out.view(*output_shape)
            
        return out