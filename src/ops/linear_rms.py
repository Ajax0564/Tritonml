import torch
import triton
import triton.language as tl
from torch.autograd import Function

def reference_linear_rmsnorm(x, w, gamma, bias=None, eps=1e-6):
    # Perform math in float32 to avoid float16 overflow/underflow
    x_f, w_f, g_f = x.float(), w.float(), gamma.float()
    y = torch.matmul(x_f, w_f.t())
    if bias is not None:
        y = y + bias.float()
    
    # RMSNorm logic: mean(y^2)
    rms = torch.sqrt(torch.mean(y**2, dim=-1, keepdim=True) + eps)
    z = (y / rms) * g_f
    return z.to(x.dtype)

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
        a = tl.load(a_ptr+offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=(offs_k[None, :]<K) & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptr+offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=(offs_k[:, None]<K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        

    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

@triton.jit
def rmsnorm_forward_kernel(
    Y, GAMMA, Z,
    stride_ym, stride_yn, stride_zm, stride_zn,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    # Load Y and compute sum of squares in float32
    # We loop if N > BLOCK_N
    y_ptr = Y + row_idx * stride_ym + offs_n * stride_yn
    sum_sq = 0.0
    
    # Pass 1: Sum of Squares
    for n_off in range(0, N, BLOCK_N):
        curr_offs = n_off + tl.arange(0, BLOCK_N)
        curr_y = tl.load(Y + row_idx * stride_ym + curr_offs * stride_yn, mask=curr_offs < N, other=0.0).to(tl.float32)
        sum_sq += tl.sum(curr_y * curr_y, axis=0)
    
    rms_inv = tl.rsqrt(sum_sq / N + eps)

    # Pass 2: Apply Norm
    for n_off in range(0, N, BLOCK_N):
        curr_offs = n_off + tl.arange(0, BLOCK_N)
        curr_mask = curr_offs < N
        curr_y = tl.load(Y + row_idx * stride_ym + curr_offs * stride_yn, mask=curr_mask, other=0.0)
        gamma = tl.load(GAMMA + curr_offs, mask=curr_mask, other=1.0)
        
        z = (curr_y.to(tl.float32) * rms_inv * gamma.to(tl.float32))
        tl.store(Z + row_idx * stride_zm + curr_offs * stride_zn, z.to(Z.dtype.element_ty), mask=curr_mask)

@triton.jit
def rmsnorm_backward_kernel(
    Y, dZ, GAMMA,
    dY, dGAMMA,
    stride_ym, stride_yn, stride_dzm, stride_dzn, stride_dym, stride_dyn,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    sum_sq = 0.0
    dot_dz_y = 0.0
    
    # Statistics
    for n_off in range(0, N, BLOCK_N):
        offs = n_off + tl.arange(0, BLOCK_N)
        mask = offs < N
        
        y = tl.load(Y + row_idx * stride_ym + offs * stride_yn, mask=mask, other=0.0).to(tl.float32)
        dz = tl.load(dZ + row_idx * stride_dzm + offs * stride_dzn, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(GAMMA + offs, mask=mask, other=0.0).to(tl.float32)
        
        sum_sq += tl.sum(y * y, axis=0)
        dot_dz_y += tl.sum(dz * gamma * y, axis=0)

    rms_inv = tl.rsqrt(sum_sq / N + eps)
    norm_scale = (rms_inv * rms_inv * rms_inv) / N

    # dY and dGamma
    for n_off in range(0, N, BLOCK_N):
        offs = n_off + tl.arange(0, BLOCK_N)
        mask = offs < N
        
        y = tl.load(Y + row_idx * stride_ym + offs * stride_yn, mask=mask, other=0.0).to(tl.float32)
        dz = tl.load(dZ + row_idx * stride_dzm + offs * stride_dzn, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(GAMMA + offs, mask=mask, other=0.0).to(tl.float32)
        
        dy = (gamma * rms_inv * dz) - (y * (dot_dz_y * norm_scale))
        tl.store(dY + row_idx * stride_dym + offs * stride_dyn, dy.to(dY.dtype.element_ty), mask=mask)
        
        # Fixed: Accumulate dGamma using the normalized y
        # We use float32 for atomic to minimize rounding errors
        dg = dz * (y * rms_inv)
        tl.atomic_add(dGAMMA + offs, dg, mask=mask)

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

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer arithmetic
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + tl.arange(0, BLOCK_K)) < K
        a = tl.load(a_ptrs, mask=k_mask[None, :] & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

   

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def rmsnorm_backward_kernel_1(
    Y, dZ, GAMMA,
    dY, dGAMMA,
    stride_ym, stride_yn, stride_dzm, stride_dzn, stride_dym, stride_dyn,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # We use float32 for all intermediate accumulations to prevent precision loss
    sum_sq = 0.0
    dot_dz_y = 0.0
    
    #  Compute row statistics
    for n_off in range(0, N, BLOCK_N):
        offs = n_off + tl.arange(0, BLOCK_N)
        mask = offs < N
        
        y = tl.load(Y + row_idx * stride_ym + offs * stride_yn, mask=mask, other=0.0).to(tl.float32)
        dz = tl.load(dZ + row_idx * stride_dzm + offs * stride_dzn, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(GAMMA + offs, mask=mask, other=0.0).to(tl.float32)
        
        sum_sq += tl.sum(y * y, axis=0)
        dot_dz_y += tl.sum(dz * gamma * y, axis=0)

    # Pre-compute scalars for the row
    rms_inv = tl.rsqrt(sum_sq / N + eps)
    # The RMSNorm gradient involves (rms^-3)/N
    norm_scale = (rms_inv * rms_inv * rms_inv) / N

    #  Compute dY and Atomic-Add dGamma
    for n_off in range(0, N, BLOCK_N):
        offs = n_off + tl.arange(0, BLOCK_N)
        mask = offs < N
        
        y = tl.load(Y + row_idx * stride_ym + offs * stride_yn, mask=mask, other=0.0).to(tl.float32)
        dz = tl.load(dZ + row_idx * stride_dzm + offs * stride_dzn, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(GAMMA + offs, mask=mask, other=0.0).to(tl.float32)
        
        # dY = (gamma * rms_inv) * dz - (gamma * y * norm_scale) * dot_dz_y
        # Rearranged for stability:
        dy = (gamma * rms_inv * dz) - (y * (dot_dz_y * norm_scale))
        tl.store(dY + row_idx * stride_dym + offs * stride_dyn, dy.to(dY.dtype.element_ty), mask=mask)
        
        # dGamma = dz * (y * rms_inv)
        dg = dz * (y * rms_inv)
        tl.atomic_add(dGAMMA + offs, dg, mask=mask)
        
class LinearRMSNormFn(Function):
    @staticmethod
    def forward(ctx, X, W, gamma, bias=None, eps=1e-6):
        X, W = X.contiguous(), W.contiguous()
        M, K = X.shape
        N,K = W.shape
        
        Y = torch.empty((M, N), device=X.device, dtype=X.dtype)
        Z = torch.empty_like(Y)
        grid = lambda META: (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
        
        # We pass weight.T because standard math is X @ W.T
        matmul_kernel_fwd[grid](
            X, W, Y, bias,
            M, N, K,
            X.stride(0), X.stride(1),
            W.stride(1), W.stride(0), # Treats (N, K) as (K, N)
            Y.stride(0), Y.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, ADD_BIAS=bias is not None
        )

       
        grid_norm = (M,)
        rmsnorm_forward_kernel[grid_norm](
            Y, gamma, Z,
            Y.stride(0), Y.stride(1), Z.stride(0), Z.stride(1),
            N, eps, BLOCK_N=64
        )

        ctx.save_for_backward(X, W, Y, gamma)
        ctx.eps, ctx.has_bias = eps, bias is not None
        return Z

    @staticmethod
    def backward(ctx, dZ):
        X, W, Y, gamma = ctx.saved_tensors
        M, N = Y.shape
        M, K = X.shape
        
        # Use float32 for the intermediate dY to keep the Matmuls stable
        dY = torch.empty_like(Y, dtype=torch.float32)
        dGamma = torch.zeros(N, device=gamma.device, dtype=torch.float32)

        grid_norm = (M,)
        rmsnorm_backward_kernel_1[grid_norm](
            Y, dZ, gamma, dY, dGamma,
            Y.stride(0), Y.stride(1), dZ.stride(0), dZ.stride(1), dY.stride(0), dY.stride(1),
            N, ctx.eps, BLOCK_N=64
        )
        dx = torch.empty_like(X)
        dw = torch.empty_like(W)
        # db = torch.empty_like(bias) if bias is not None else None

        grid_dx = lambda META: (triton.cdiv(M, 64) * triton.cdiv(K, 64),)
        grid_dw = lambda META: (triton.cdiv(N, 64) * triton.cdiv(K, 64),)

        # 1. dx = dy @ weight -> (M, N) @ (N, K) = (M, K)
        matmul_kernel[grid_dx](
            dY, W, dx,
            M, K, N,
            dY.stride(0), dY.stride(1),
            W.stride(0), W.stride(1),
            dx.stride(0), dx.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 2. dw = dy.T @ x -> (N, M) @ (M, K) = (N, K)
        matmul_kernel[grid_dw](
            dY, X, dw,
            N, K, M,
            dY.stride(1), dY.stride(0), # Transpose dy: (N, M)
            X.stride(0), X.stride(1),    # x: (M, K)
            dw.stride(0), dw.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        dBias = dY.sum(0) if ctx.has_bias else None
        
        return dx, dw, dGamma.to(gamma.dtype), dBias, None

def test_correctness():
    M, K, N = 128, 512, 1024
    device, dtype = "cuda", torch.float32
    torch.manual_seed(42)
    
    x = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(N, K, device=device, dtype=dtype, requires_grad=True)
    gamma = torch.randn(N, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(N, device=device, dtype=dtype, requires_grad=True)

    # --- Triton ---
    z_triton = LinearRMSNormFn.apply(x, w, gamma, bias)
    z_triton.backward(torch.ones_like(z_triton))
    grads_triton = [x.grad.clone(), w.grad.clone(), gamma.grad.clone(), bias.grad.clone()]
    
    x.grad, w.grad, gamma.grad, bias.grad = None, None, None, None
    
    # --- Reference ---
    z_ref = reference_linear_rmsnorm(x, w, gamma, bias)
    z_ref.backward(torch.ones_like(z_ref))
    grads_ref = [x.grad.clone(), w.grad.clone(), gamma.grad.clone(), bias.grad.clone()]
    
    # Check results
    print(f"Forward Match:  {torch.allclose(z_triton, z_ref, atol=1e-2)}")
    labels = ["X", "W", "G", "B"]
    for i, label in enumerate(labels):
        match = torch.allclose(grads_triton[i], grads_ref[i], atol=1e-2, rtol=1e-2)
        max_diff = (grads_triton[i] - grads_ref[i]).abs().max()
        print(f"Grad {label} Match:   {match} (Max Diff: {max_diff:.4f})")

if __name__ == "__main__":
    test_correctness()