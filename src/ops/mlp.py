import torch
import triton
import triton.language as tl

@triton.jit
def gelu_grad(z):
    k1 = 0.707106781
    k2 = 0.39894228
    cdf = 0.5 * (1 + tl.math.erf(k1 * z))
    pdf = k2 * tl.exp(-0.5 * z * z)
    return cdf + z * pdf

@triton.jit
def linear_kernel_fwd(
    x_ptr, w_ptr, b_ptr, y_ptr, z_ptr,
    B, M, K, N,
    stride_xb, stride_xm, stride_xk,
    stride_wk, stride_wn, # wk is the row stride, wn is the col stride
    stride_yb, stride_ym, stride_yn,
    stride_zb, stride_zm, stride_zn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    ADD_BIAS: tl.constexpr, SAVE_Z: tl.constexpr, APPLY_GELU: tl.constexpr
):
    pid_m, pid_n, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    curr_x_ptr = x_ptr + pid_b * stride_xb
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(curr_x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, 
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # Reads w using the provided strides
        b = tl.load(w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn, 
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b,input_precision="ieee")

    if ADD_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    if SAVE_Z:
        tl.store(z_ptr + pid_b * stride_zb + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn, 
                 acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    if APPLY_GELU:
        acc = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.707106781))
    tl.store(y_ptr + pid_b * stride_yb + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, 
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def bwd_dx_gelu_fused_kernel(
    dy_ptr, w_ptr, z_ptr, dz_ptr, 
    B, M, N, K, 
    stride_dyb, stride_dym, stride_dyn,
    stride_wn, stride_wk,
    stride_zb, stride_zm, stride_zn,
    stride_dzb, stride_dzm, stride_dzn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m, pid_k, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm, rk = pid_m * BLOCK_M + tl.arange(0, BLOCK_M), pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_N)):
        rn = n * BLOCK_N + tl.arange(0, BLOCK_N)
        dy = tl.load(dy_ptr + pid_b * stride_dyb + rm[:, None] * stride_dym + rn[None, :] * stride_dyn, 
                     mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk, 
                    mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        acc += tl.dot(dy, w,input_precision="ieee")
    z = tl.load(z_ptr + pid_b * stride_zb + rm[:, None] * stride_zm + rk[None, :] * stride_zn, 
                mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
    dz = acc * gelu_grad(z)
    tl.store(dz_ptr + pid_b * stride_dzb + rm[:, None] * stride_dzm + rk[None, :] * stride_dzn, 
             dz, mask=(rm[:, None] < M) & (rk[None, :] < K))

@triton.jit
def linear_backward_dw(
    dy_ptr, x_ptr, dw_ptr,
    B, M, N, K,
    stride_dyb, stride_dym, stride_dyn,
    stride_xb, stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr
):
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    pid_b = tl.program_id(2)
    rn, rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N), pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    batch_start = pid_b * BLOCK_B
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for b in range(batch_start, min(batch_start + BLOCK_B, B)):
        for m in range(0, tl.cdiv(M, BLOCK_M)):
            rm = m * BLOCK_M + tl.arange(0, BLOCK_M)
            dy = tl.load(dy_ptr + b * stride_dyb + rm[None, :] * stride_dym + rn[:, None] * stride_dyn,
                         mask=(rm[None, :] < M) & (rn[:, None] < N), other=0.0)
            x = tl.load(x_ptr + b * stride_xb + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
                        mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
            acc += tl.dot(dy, x,input_precision="ieee")
    tl.atomic_add(dw_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk, acc, 
             mask=(rn[:, None] < N) & (rk[None, :] < K))

class MlpGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        B, M, K = x.shape
        H, _ = w1.shape
        N, _ = w2.shape
        z = torch.empty((B, M, H), device=x.device, dtype=x.dtype)
        h = torch.empty((B, M, H), device=x.device, dtype=x.dtype)
        y = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

        # FWD 1: h = gelu(x @ w1.T + b1)
        # Instead of w1.t(), we swap stride_w1[0] and stride_w1[1]
        grid1 = (triton.cdiv(M, 64), triton.cdiv(H, 64), B)
        linear_kernel_fwd[grid1](
            x, w1, b1, h, z, B, M, K, H, 
            x.stride(0), x.stride(1), x.stride(2), 
            w1.stride(1), w1.stride(0), # <--- STRIDE SWAP (K, H) view
            h.stride(0), h.stride(1), h.stride(2), 
            z.stride(0), z.stride(1), z.stride(2), 
            64, 32, 64, True, True, True
        )

        # FWD 2: y = h @ w2.T + b2
        grid2 = (triton.cdiv(M, 64), triton.cdiv(N, 64), B)
        linear_kernel_fwd[grid2](
            h, w2, b2, y, h, B, M, H, N, #y value will not be used None can;t be used in triton it expect pointer 
            h.stride(0), h.stride(1), h.stride(2), 
            w2.stride(1), w2.stride(0), 
            y.stride(0), y.stride(1), y.stride(2), 
            h.stride(0), h.stride(1), h.stride(2), 
            64, 32, 64, True, False, False
        )
        
        ctx.save_for_backward(x, w1, w2, z, h)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2, z, h = ctx.saved_tensors
        B, M, K = x.shape; H, _ = w1.shape; N, _ = w2.shape
        dy = dy.contiguous()
        BK_N, BK_K, BK_M = 64, 64, 32

        # dz1 = (dy @ w2) * gelu_grad(z)
        # Ref: dy is (B, M, N), w2 is (N, H). Target: (B, M, H)
        dz1 = torch.empty((B, M, H), device=dy.device, dtype=dy.dtype)
        grid_fused = (triton.cdiv(M, 64), triton.cdiv(H, 64), B)
        # Passing w2 normally: w2 is N x H, kernel expects inner dimension N
        bwd_dx_gelu_fused_kernel[grid_fused](
            dy, w2, z, dz1, B, M, N, H, 
            *dy.stride(), 
            w2.stride(0), w2.stride(1), # Normal stride
            *z.stride(), *dz1.stride(), 64, 64, 32
        )

        # dx = dz1 @ w1
        # Ref: dz1 is (B, M, H), w1 is (H, K). Target: (B, M, K)
        dx = torch.empty_like(x)
        grid_dx = (triton.cdiv(M, 64), triton.cdiv(K, 64), B)
        linear_kernel_fwd[grid_dx](
            dz1, w1, None, dx, dz1, B, M, H, K, 
            *dz1.stride(), 
            w1.stride(0), w1.stride(1), # Normal stride
            *dx.stride(), *dz1.stride(), 64, 32, 64, False, False, False
        )

        # dw2 = dy^T @ h
        dw2 = torch.zeros_like(w2)
        grid_dw2 = (triton.cdiv(N, BK_N), triton.cdiv(H, BK_K),B)
        linear_backward_dw[grid_dw2](dy, h, dw2, B, M, N, H, *dy.stride(), *h.stride(), *dw2.stride(), BK_M, BK_N, BK_K,BLOCK_B=4)

        # dw1 = dz1^T @ x
        dw1 = torch.zeros_like(w1)
        grid_dw1 = (triton.cdiv(H, BK_N), triton.cdiv(K, BK_K),B)
        linear_backward_dw[grid_dw1](dz1, x, dw1, B, M, H, K, *dz1.stride(), *x.stride(), *dw1.stride(), BK_M, BK_N, BK_K,BLOCK_B=4)

        return dx, dw1, dz1.sum((0,1)), dw2, dy.sum((0,1))

def test_mlp():
    B, M, K, H, N = 4, 256, 128, 512, 128
    device = 'cuda'
    x = torch.randn(B, M, K, device=device, requires_grad=True)
    w1 = torch.randn(H, K, device=device, requires_grad=True)
    b1 = torch.randn(H, device=device, requires_grad=True)
    w2 = torch.randn(N, H, device=device, requires_grad=True)
    b2 = torch.randn(N, device=device, requires_grad=True)

    # Reference
    h_ref = torch.nn.functional.gelu(x @ w1.t() + b1)
    y_ref = h_ref @ w2.t() + b2
    y_ref.sum().backward()
    ref_grads = [x.grad.clone(), w1.grad.clone(), b1.grad.clone(), w2.grad.clone(), b2.grad.clone()]
    for p in [x, w1, b1, w2, b2]: p.grad = None

    # Triton
    y_tri = MlpGelu.apply(x, w1, b1, w2, b2)
    y_tri.sum().backward()
    tri_grads = [x.grad, w1.grad, b1.grad, w2.grad, b2.grad]

    print(f"Forward Match: {torch.allclose(y_ref, y_tri,rtol=1e-3, atol=1e-3)}")
    names = ["dx", "dw1", "db1", "dw2", "db2"]
    for name, r, t in zip(names, ref_grads, tri_grads):
        match = torch.allclose(r, t,rtol=1e-4, atol=1e-3)
        print(f"{name} Match: {match}")

if __name__ == "__main__":
    test_mlp()