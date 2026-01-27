import triton
import triton.language as tl
import torch

# Forward pass:
# z1 = x @ w1 + b1
# h  = GELU(z1)
# y  = h @ w2 + b2

@triton.jit
def linear_layer_gelu_fwd(x_ptr, w_ptr, b_ptr, y_ptr, z_ptr, M, K, N, 
                          stride_xm, stride_xk, stride_wk, stride_wn, stride_ym, stride_yn, 
                          block_m: tl.constexpr, block_k: tl.constexpr, block_n: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, block_n)
    pidm, pidn = pid // num_pid_n, pid % num_pid_n
    off_m = pidm * block_m + tl.arange(0, block_m)
    off_n = pidn * block_n + tl.arange(0, block_n)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, block_k)):
        off_k = k * block_k + tl.arange(0, block_k)
        x = tl.load(x_ptr + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk, 
                    mask=(off_m[:, None] < M) & (off_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + off_k[:, None] * stride_wk + off_n[None, :] * stride_wn, 
                    mask=(off_k[:, None] < K) & (off_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)

    if HAS_BIAS:
        acc += tl.load(b_ptr + off_n, mask=off_n < N, other=0.0)[None, :]
    
    tl.store(z_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, acc, mask=(off_m[:, None] < M) & (off_n[None, :] < N))
    acc = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.707106781))
    tl.store(y_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, acc, mask=(off_m[:, None] < M) & (off_n[None, :] < N))

@triton.jit
def linear_layer_fwd(x_ptr, w_ptr, b_ptr, y_ptr, M, K, N, 
                     stride_xm, stride_xk, stride_wk, stride_wn, stride_ym, stride_yn, 
                     block_m: tl.constexpr, block_k: tl.constexpr, block_n: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, block_n)
    pidm, pidn = pid // num_pid_n, pid % num_pid_n
    off_m = pidm * block_m + tl.arange(0, block_m)
    off_n = pidn * block_n + tl.arange(0, block_n)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, block_k)):
        off_k = k * block_k + tl.arange(0, block_k)
        x = tl.load(x_ptr + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk, 
                    mask=(off_m[:, None] < M) & (off_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + off_k[:, None] * stride_wk + off_n[None, :] * stride_wn, 
                    mask=(off_k[:, None] < K) & (off_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)

    if HAS_BIAS:
        acc += tl.load(b_ptr + off_n, mask=off_n < N, other=0.0)[None, :]
    
    tl.store(y_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, acc, mask=(off_m[:, None] < M) & (off_n[None, :] < N))


#  L = loss(y)

# Backward pass (chain rule):

# dL/dy
# dy = grad_output

# dL/dW2 = dL/dy * dy/dW2
# dy/dW2 = h
# => dW2 = h^T @ dy
# dw2 = hidden.T @ grad_output

# dL/db2 = sum(dL/dy)
# db2 = grad_output.sum(dim=0)

# dL/dh = dL/dy * dy/dh
# dy/dh = w2^T
# => dh = dy @ w2.T

# dL/dz1 = dL/dh * dh/dz1
# dh/dz1 = GELU'(z1)
# => dz1 = (dy @ w2.T) * GELU'(z1)

# where:
# GELU'(z) = Phi(z) + z * phi(z)
# Phi(z)  = 0.5 * (1 + erf(z / sqrt(2)))
# phi(z)  = (1 / sqrt(2*pi)) * exp(-0.5 * z^2)

# dL/dW1 = dL/dz1 * dz1/dW1
# dz1/dW1 = x
# => dW1 = x^T @ dz1
# dw1 = x.T @ dz1

# dL/db1 = sum(dL/dz1)
# db1 = dz1.sum(dim=0)

# dL/dx = dL/dz1 * dz1/dx
# dz1/dx = w1^T
# => dx = dz1 @ w1.T

@triton.jit
def bwd_dx_kernel(dz_ptr, w_ptr, dx_ptr,
                  M, H, K,
                  stride_dzm, stride_dzh,
                  stride_wh, stride_wk,
                  stride_dxm, stride_dxk,
                  BLOCK_M: tl.constexpr,
                  BLOCK_H: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for h in range(0, tl.cdiv(H, BLOCK_H)):
        rh = h * BLOCK_H + tl.arange(0, BLOCK_H)

        dz = tl.load(
            dz_ptr + rm[:, None] * stride_dzm + rh[None, :] * stride_dzh,
            mask=(rm[:, None] < M) & (rh[None, :] < H),
            other=0.0,
        )

        w = tl.load(
            w_ptr + rh[:, None] * stride_wh + rk[None, :] * stride_wk,
            mask=(rh[:, None] < H) & (rk[None, :] < K),
            other=0.0,
        )

        acc += tl.dot(dz, w)

    tl.store(
        dx_ptr + rm[:, None] * stride_dxm + rk[None, :] * stride_dxk,
        acc,
        mask=(rm[:, None] < M) & (rk[None, :] < K),
    )


@triton.jit
def bwd_dx_gelu_fused_kernel(dy_ptr, w_ptr, z_ptr, dz_ptr, M, N, K,
                              stride_dym, stride_dyn, stride_wk, stride_wn,
                              stride_zm, stride_zn, stride_dzm, stride_dzn,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    rm, rn = pid_m * BLOCK_M + tl.arange(0, BLOCK_M), pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
        dy = tl.load(dy_ptr + rm[:, None] * stride_dym + rk[None, :] * stride_dyn, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(dy, w)

    z1 = tl.load(z_ptr + rm[:, None] * stride_zm + rn[None, :] * stride_zn, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    s2i, s2pi = 0.707106781, 0.39894228
    cdf = 0.5 * (1 + tl.math.erf(z1 * s2i))
    pdf = s2pi * tl.exp(-0.5 * z1 * z1)
    dz1 = acc * (cdf + z1 * pdf)
    
    tl.store(dz_ptr + rm[:, None] * stride_dzm + rn[None, :] * stride_dzn, dz1, mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def bwd_dw_kernel(x_ptr, dz_ptr, dw_ptr, M, K, N, 
                  stride_xm, stride_xk, stride_dzm, stride_dzn, stride_dwk, stride_dwn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    rk, rn = pid_m * BLOCK_M + tl.arange(0, BLOCK_M), pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_K)):
        rm = m * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rk[:, None] * stride_xm + rm[None, :] * stride_xk, mask=(rk[:, None] < K) & (rm[None, :] < M), other=0.0)
        dz = tl.load(dz_ptr + rm[:, None] * stride_dzm + rn[None, :] * stride_dzn, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(x, dz)

    tl.store(dw_ptr + rk[:, None] * stride_dwk + rn[None, :] * stride_dwn, acc, mask=(rk[:, None] < K) & (rn[None, :] < N))

class TritonMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        M, K = x.shape
        H, _ = w1.shape
        N, _ = w2.shape
        z1 = torch.empty((M, H), device=x.device, dtype=x.dtype)
        hidden = torch.empty((M, H), device=x.device, dtype=x.dtype)
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid1 = (triton.cdiv(M, 32) * triton.cdiv(H, 32),)
        linear_layer_gelu_fwd[grid1](x, w1, b1, hidden, z1, M, K, H, x.stride(0), x.stride(1), w1.stride(1), w1.stride(0), hidden.stride(0), hidden.stride(1), 32, 32, 32, True)

        grid2 = (triton.cdiv(M, 32) * triton.cdiv(N, 32),)
        linear_layer_fwd[grid2](hidden, w2, b2, output, M, H, N, hidden.stride(0), hidden.stride(1), w2.stride(1), w2.stride(0), output.stride(0), output.stride(1), 32, 32, 32, True)

        ctx.save_for_backward(x, w1, b1, w2, b2, hidden, z1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, b1, w2, b2, hidden, z1 = ctx.saved_tensors
        M, K = x.shape; H, _ = w1.shape; N, _ = w2.shape

        dw2 = torch.empty_like(w2)
        grid_dw2 = (triton.cdiv(N, 32) * triton.cdiv(H, 32),)
        bwd_dw_kernel[grid_dw2](grad_output, hidden, dw2, M, N, H, grad_output.stride(1), grad_output.stride(0), hidden.stride(0), hidden.stride(1), dw2.stride(0), dw2.stride(1), 32, 32, 32)
        db2 = grad_output.sum(0)

        dz1 = torch.empty_like(z1)
        grid_dz1 = (triton.cdiv(M, 32) * triton.cdiv(H, 32),)
        bwd_dx_gelu_fused_kernel[grid_dz1](grad_output, w2, z1, dz1, M, H, N, grad_output.stride(0), grad_output.stride(1), w2.stride(0), w2.stride(1), z1.stride(0), z1.stride(1), dz1.stride(0), dz1.stride(1), 32, 32, 32)

        dw1 = torch.empty_like(w1)
        grid_dw1 = (triton.cdiv(H, 32) * triton.cdiv(K, 32),)
        bwd_dw_kernel[grid_dw1](dz1, x, dw1, M, H, K, dz1.stride(1), dz1.stride(0), x.stride(0), x.stride(1), dw1.stride(0), dw1.stride(1), 32, 32, 32)
        db1 = dz1.sum(0)

        dx = torch.empty_like(x)

        grid_dx = (triton.cdiv(M, 32) * triton.cdiv(K, 32),)

        bwd_dx_kernel[grid_dx](
            dz1, w1, dx,
            M, H, K,
            dz1.stride(0), dz1.stride(1),
            w1.stride(0), w1.stride(1),
            dx.stride(0), dx.stride(1),
            32, 32, 32
        )

        return dx, dw1, db1, dw2, db2


def verify_mlp():
    torch.manual_seed(42)
    M, K, H, N = 128, 64, 128, 32
    device = "cuda"
    
    x = torch.randn((M, K), device=device, requires_grad=True)
    w1 = torch.randn((H, K), device=device, requires_grad=True)
    b1 = torch.randn(H, device=device, requires_grad=True)
    w2 = torch.randn((N, H), device=device, requires_grad=True)
    b2 = torch.randn(N, device=device, requires_grad=True)

    # Reference PyTorch
    ref_z1 = torch.nn.functional.linear(x, w1, b1)
    ref_h = torch.nn.functional.gelu(ref_z1)
    ref_out = torch.nn.functional.linear(ref_h, w2, b2)
    ref_out.sum().backward()
    
    ref_grads = {"x": x.grad.clone(),
        "w1": w1.grad.clone(), 
        "b1": b1.grad.clone(), 
        "w2": w2.grad.clone(), 
        "b2": b2.grad.clone()
    }
    
    # Zero out grads
    x.grad, w1.grad, b1.grad, w2.grad, b2.grad = [None]*5

    # Triton
    tri_out = TritonMLPFunction.apply(x, w1, b1, w2, b2)
    tri_out.sum().backward()

    print(f"--- Verification Results ---")
    fwd_match = torch.allclose(ref_out, tri_out, atol=2e-4)
    print(f"Forward Match: {fwd_match} (Max Diff: {(ref_out - tri_out).abs().max():.6e})")
    
    results = {"X":(ref_grads["x"], x.grad),
        "W1": (ref_grads["w1"], w1.grad),
        "B1": (ref_grads["b1"], b1.grad),
        "W2": (ref_grads["w2"], w2.grad),
        "B2": (ref_grads["b2"], b2.grad),
    }
    
    for name, (ref, tri) in results.items():
        match = torch.allclose(ref, tri, atol=1e-4)
        diff = (ref - tri).abs().max()
        print(f"Gradient {name} Match: {match} (Max Diff: {diff:.6e})")

if __name__ == "__main__":
    verify_mlp()