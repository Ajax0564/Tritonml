import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def linear_layer_gelu_fwd(
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

    # CTA reordering (same idea as before)
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
def gelu_backward_kernel(
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

def is_div(val, tile): return val % tile == 0

class TritonLinearGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b=None):
        M, K = x.shape
        K2, N = w.shape
        assert K == K2

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        z = torch.empty_like(y)

        
        BLOCK_M, BLOCK_N, BLOCK_K = 64,64,32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        linear_layer_gelu_fwd[grid](
            x, w, y, b, z,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1),
            TILE_M=BLOCK_M,
            TILE_N=BLOCK_N,
            TILE_K=BLOCK_K,
            GROUP_M=8,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(N, 64),
            DIVISIBLE_K=is_div(K, 32),
            ADD_BIAS=b is not None,
        )

        ctx.save_for_backward(x, w, z)
        ctx.has_bias = b is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, z = ctx.saved_tensors
        M, K = x.shape
        _, N = w.shape

        dx = torch.empty_like(x)
        dw = torch.empty_like(w)
        dz = torch.empty_like(dy)

        # dZ = dY * GELU'(Z) 
        BLOCK_M, BLOCK_N = 64,64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        gelu_backward_kernel[grid](
            dz, dy, z,
            M, N,
            dy.stride(0), dy.stride(1),
            z.stride(0), z.stride(1),
            dz.stride(0), dz.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        # dX = dZ @ W^t
        grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        matmul_kernel[grid_dx](
            dz, w, dx,
            M, K, N,
            dz.stride(0), dz.stride(1),
            w.stride(1), w.stride(0),  # Wᵀ
            dx.stride(0), dx.stride(1),
            TILE_M=BLOCK_M,
            TILE_N=BLOCK_N,
            TILE_K=32,
            GROUP_M=8,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(N, 64),
            DIVISIBLE_K=is_div(K, 32),
        )

        #  dW = X^t @ dZ 
        grid_dw = (triton.cdiv(K, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid_dw](
            x, dz, dw,
            K, N, M,
            x.stride(1), x.stride(0),   
            dz.stride(0), dz.stride(1),
            dw.stride(0), dw.stride(1),
            TILE_M=BLOCK_M,
            TILE_N=BLOCK_N,
            TILE_K=32,
            GROUP_M=8,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(N, 64),
            DIVISIBLE_K=is_div(K, 32),
        )

       
        db = dz.sum(dim=0) if ctx.has_bias else None

        return dx, dw, db


def test_triton_linear_gelu_correctness():
    torch.manual_seed(0)
    device = "cuda"

    # Problem sizes (intentionally non-divisible to stress masks)
    M, K, N = 257, 513, 769
    dtype = torch.float32

    # Inputs
    x = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(K, N, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(N, device=device, dtype=dtype, requires_grad=True)

    dy = torch.randn(M, N, device=device, dtype=dtype)

    
    def torch_linear_gelu(x, w, b):
        z = x @ w + b
        return F.gelu(z)

    y_ref = torch_linear_gelu(x, w, b)
    y_ref.backward(dy)

    dx_ref = x.grad.detach().clone()
    dw_ref = w.grad.detach().clone()
    db_ref = b.grad.detach().clone()

    # Reset grads
    x.grad.zero_()
    w.grad.zero_()
    b.grad.zero_()

  
    y_tri = TritonLinearGELU.apply(x, w, b)
    y_tri.backward(dy)

    dx_tri = x.grad.detach()
    dw_tri = w.grad.detach()
    db_tri = b.grad.detach()

    
    atol = 1e-4
    rtol = 1e-4

    def report(name, tri, ref):
        max_diff = (tri - ref).abs().max().item()
        close = torch.allclose(tri, ref, atol=atol, rtol=rtol)
        print(f"{name:8s}: {'PASS' if close else 'FAIL'} | max diff = {max_diff:.3e}")
        return close

    print("\n--- Triton Linear + GELU Correctness ---")
    ok_fwd = report("Forward", y_tri, y_ref)
    ok_dx  = report("dX", dx_tri, dx_ref)
    ok_dw  = report("dW", dw_tri, dw_ref)
    ok_db  = report("dB", db_tri, db_ref)

    assert ok_fwd and ok_dx and ok_dw and ok_db, "Correctness test failed"
    print("All checks passed!")


if __name__ == "__main__":
    test_triton_linear_gelu_correctness()
