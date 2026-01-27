import torch
import triton
import triton.language as tl

@triton.jit
def gelu_grad_fn(z):
    k1 = 0.707106781  
    k2 = 0.39894228   
    # Use float32 for math functions to maintain stability in bfloat16
    z_f32 = z.to(tl.float32)
    cdf = 0.5 * (1 + tl.math.erf(k1 * z_f32))
    pdf = k2 * tl.exp(-0.5 * z_f32 * z_f32)
    return (cdf + z_f32 * pdf).to(z.dtype)

@triton.jit
def linear_gelu_fwd_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, z_ptr,
    B, M, K, N,
    stride_xb, stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    stride_zb, stride_zm, stride_zn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    ADD_BIAS: tl.constexpr
):
    pid_m, pid_n, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    curr_x_ptr = x_ptr + pid_b * stride_xb
    
    # Always accumulate in float32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(curr_x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, 
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn, 
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        # input_precision set to "ieee" or None based on your earlier debugging
        acc += tl.dot(x, w)

    if ADD_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :].to(tl.float32)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Store z (pre-activation) - cast back to input dtype
    tl.store(z_ptr + pid_b * stride_zb + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn, 
             acc.to(z_ptr.dtype.element_ty), mask=mask)
    
    # GELU calculation
    output = acc * 0.5 * (1 + tl.math.erf(0.707106781 * acc))
    
    # Store y (post-activation) - cast back to input dtype
    tl.store(y_ptr + pid_b * stride_yb + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, 
             output.to(y_ptr.dtype.element_ty), mask=mask)

@triton.jit
def linear_gelu_bwd_dx_kernel(
    dy_ptr, z_ptr, w_ptr, dx_ptr,
    B, M, K, N,
    stride_dyb, stride_dym, stride_dyn,
    stride_zb, stride_zm, stride_zn,
    stride_wn, stride_wk,
    stride_dxb, stride_dxm, stride_dxk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m, pid_k, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m, offs_k = pid_m * BLOCK_M + tl.arange(0, BLOCK_M), pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        
        dy = tl.load(dy_ptr + pid_b * stride_dyb + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn, mask=mask_mn, other=0.0)
        z = tl.load(z_ptr + pid_b * stride_zb + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn, mask=mask_mn, other=0.0)
        
        # Compute dz in high precision
        dz = (dy.to(tl.float32) * gelu_grad_fn(z).to(tl.float32)).to(w_ptr.dtype.element_ty)
        
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, 
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(dz, w)

    tl.store(dx_ptr + pid_b * stride_dxb + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk, 
             acc.to(dx_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

@triton.jit
def linear_gelu_bwd_dw_kernel(
    dy_ptr, z_ptr, x_ptr, dw_ptr,
    B, M, K, N,
    stride_dyb, stride_dym, stride_dyn,
    stride_zb, stride_zm, stride_zn,
    stride_xb, stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr
):
    pid_n, pid_k, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_n, offs_k = pid_n * BLOCK_N + tl.arange(0, BLOCK_N), pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    batch_start = pid_b * BLOCK_B
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for b in range(batch_start, min(batch_start + BLOCK_B, B)):
        for m in range(0, tl.cdiv(M, BLOCK_M)):
            offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_nm = (offs_n[:, None] < N) & (offs_m[None, :] < M)
            
            dy = tl.load(dy_ptr + b * stride_dyb + offs_m[None, :] * stride_dym + offs_n[:, None] * stride_dyn, mask=mask_nm, other=0.0)
            z = tl.load(z_ptr + b * stride_zb + offs_m[None, :] * stride_zm + offs_n[:, None] * stride_zn, mask=mask_nm, other=0.0)
            dz = (dy.to(tl.float32) * gelu_grad_fn(z).to(tl.float32)).to(x_ptr.dtype.element_ty)
            
            x = tl.load(x_ptr + b * stride_xb + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, 
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            acc += tl.dot(dz, x)

    # Atomic add must match the dtype of the pointer. 
    # If weight is bfloat16, atomic_add in Triton 3.0+ handles it, but float32 is safer for accumulation.
    tl.atomic_add(dw_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, 
                  acc.to(dw_ptr.dtype.element_ty), mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))

class TritonLinearGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        B, M, K = x.shape
        N, _ = weight.shape
        y = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
        z = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64), B)
        
        linear_gelu_fwd_kernel[grid](
            x, weight, bias, y, z,
            B, M, K, N,
            x.stride(0), x.stride(1), x.stride(2),
            weight.stride(1), weight.stride(0),
            y.stride(0), y.stride(1), y.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_M=64, BLOCK_K=32, BLOCK_N=64,
            ADD_BIAS=(bias is not None)
        )
        ctx.save_for_backward(x, weight, bias, z)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, z = ctx.saved_tensors
        B, M, K = x.shape
        N, _ = weight.shape

        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)
        db = None

        # 1. dx calculation
        grid_dx = (triton.cdiv(M, 64), triton.cdiv(K, 64), B)
        linear_gelu_bwd_dx_kernel[grid_dx](
            dy, z, weight, dx, B, M, K, N,
            dy.stride(0), dy.stride(1), dy.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1), dx.stride(2),
            BLOCK_M=64, BLOCK_N=32, BLOCK_K=64
        )

        # 2. dw calculation (Atomic)
        BB = 4
        grid_dw = (triton.cdiv(N, 64), triton.cdiv(K, 64), B)
        linear_gelu_bwd_dw_kernel[grid_dw](
            dy, z, x, dw, B, M, K, N,
            dy.stride(0), dy.stride(1), dy.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            weight.stride(0), weight.stride(1),
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, BLOCK_B=BB
        )

        if bias is not None:
            # Recompute GeLU grad for bias in PyTorch for simplicity
            k1, k2 = 0.707106781, 0.39894228
            cdf = 0.5 * (1 + torch.erf(k1 * z))
            pdf = k2 * torch.exp(-0.5 * z**2)
            dz = dy * (cdf + z * pdf)
            db = dz.sum(dim=(0, 1))

        return dx, dw, db

def test_correctness(B=4, M=128, K=256, N=512, dtype=torch.float32):
    device = "cuda"
    
    # High-precision reference for comparison
    x = torch.randn((B, M, K), device=device, dtype=dtype, requires_grad=True)
    w = torch.randn((N, K), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((N,), device=device, dtype=dtype, requires_grad=True)

    # Reference
    x_ref, w_ref, b_ref = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    y_ref = torch.nn.functional.gelu(torch.matmul(x_ref, w_ref.t()) + b_ref)

    # Triton
    y_triton = TritonLinearGelu.apply(x, w, b)

    # Tolerances for BF16/TF32
    # BF16 typically requires ~1e-2. 
    atol, rtol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)

    torch.testing.assert_close(y_triton, y_ref, atol=atol, rtol=rtol)
    print(f"Forward Pass ({dtype}): MATCH")

    dout = torch.randn_like(y_ref)
    y_ref.backward(dout)
    y_triton.backward(dout)

    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(w.grad, w_ref.grad, atol=atol, rtol=rtol)
    print(f"Backward Pass ({dtype}): MATCH")

if __name__ == "__main__":
    test_correctness()