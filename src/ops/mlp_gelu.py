import torch
import triton
import triton.language as tl


@triton.jit
def _linear_gelu_fwd_kernel(
    A, B, C, Bias, Z,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        # Recalculate mask inside loop to prevent illegal memory access
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk
        offs_k += TILE_K

    if ADD_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Store Z (pre-activation)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Z + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask_c)

    # GELU Activation
    acc = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.70710678118))
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask_c)

@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk
        offs_k += TILE_K

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask_c)

@triton.jit
def _bwd_gelu_kernel(
    d_out, W2, d_z1, Z1,
    M, N, K,
    stride_dom, stride_don,
    stride_w2k, stride_w2n,
    stride_dzm, stride_dzn,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    do_ptrs = d_out + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_don)
    w2_ptrs = W2 + (offs_k[:, None] * stride_w2k + offs_n[None, :] * stride_w2n)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        mask_do = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w2 = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        w2 = tl.load(w2_ptrs, mask=mask_w2, other=0.0)
        acc += tl.dot(do, w2)
        do_ptrs += TILE_K * stride_don
        w2_ptrs += TILE_K * stride_w2k
        offs_k += TILE_K

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    z1 = tl.load(Z1 + offs_m[:, None] * stride_dzm + offs_n[None, :] * stride_dzn, mask=mask_c)
    
    # Derivative of GELU: grad * (0.5 * (1 + erf(x/sqrt(2))) + (x/sqrt(2pi)) * exp(-x^2/2))
    s2i = 0.707106781
    s2pi = 0.39894228
    cdf = 0.5 * (1 + tl.math.erf(z1 * s2i))
    pdf = s2pi * tl.exp(-0.5 * z1 * z1)
    dz1 = acc * (cdf + z1 * pdf)

    tl.store(d_z1 + offs_m[:, None] * stride_dzm + offs_n[None, :] * stride_dzn, dz1, mask=mask_c)

class TritonMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        M, K = x.shape
        H, _ = w1.shape # w1 is (H, K)
        N, _ = w2.shape # w2 is (N, H)
        
        z1 = torch.empty((M, H), device=x.device, dtype=x.dtype)
        hidden = torch.empty((M, H), device=x.device, dtype=x.dtype)
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(H, meta["TILE_N"]))
        _linear_gelu_fwd_kernel[grid](
            x, w1, hidden, b1, z1,
            M, H, K,
            x.stride(0), x.stride(1),
            w1.stride(1), w1.stride(0), # w1 as (K, H)
            hidden.stride(0), hidden.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32, ADD_BIAS=True
        )

        grid2 = lambda meta: (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(N, meta["TILE_N"]))
        _matmul_kernel[grid2](
            hidden, w2, output,
            M, N, H,
            hidden.stride(0), hidden.stride(1),
            w2.stride(1), w2.stride(0), # w2 as (H, N)
            output.stride(0), output.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32
        )
        #  bias 
        output += b2

        ctx.save_for_backward(x, w1, b1, w2, b2, hidden, z1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, b1, w2, b2, hidden, z1 = ctx.saved_tensors
        M, K = x.shape
        H, _ = w1.shape
        N, _ = w2.shape

        db2 = grad_output.sum(0)
        
        # dw2 = grad_output^T @ hidden (N, M) @ (M, H) -> (N, H)
        dw2 = torch.empty((N, H), device=x.device, dtype=x.dtype)
        grid_dw2 = lambda meta: (triton.cdiv(N, meta["TILE_M"]), triton.cdiv(H, meta["TILE_N"]))
        _matmul_kernel[grid_dw2](
            grad_output, hidden, dw2,
            N, H, M,
            grad_output.stride(1), grad_output.stride(0),
            hidden.stride(0), hidden.stride(1),
            dw2.stride(0), dw2.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32
        )

        # dz1 = (grad_output @ w2) * gelu_grad(z1)
        dz1 = torch.empty_like(z1)
        grid_dz1 = lambda meta: (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(H, meta["TILE_N"]))
        _bwd_gelu_kernel[grid_dz1](
            grad_output, w2, dz1, z1,
            M, H, N,
            grad_output.stride(0), grad_output.stride(1),
            w2.stride(0), w2.stride(1), # w2 is (N, H)
            dz1.stride(0), dz1.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32
        )

        # dw1 = dz1^T @ x -> (H, M) @ (M, K) -> (H, K)
        dw1 = torch.empty_like(w1)
        grid_dw1 = lambda meta: (triton.cdiv(H, meta["TILE_M"]), triton.cdiv(K, meta["TILE_N"]))
        _matmul_kernel[grid_dw1](
            dz1, x, dw1,
            H, K, M,
            dz1.stride(1), dz1.stride(0),
            x.stride(0), x.stride(1),
            dw1.stride(0), dw1.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32
        )

        db1 = dz1.sum(0)
        
        # dx = dz1 @ w1 -> (M, H) @ (H, K) -> (M, K)
        dx = torch.empty_like(x)
        grid_dx = lambda meta: (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(K, meta["TILE_N"]))
        _matmul_kernel[grid_dx](
            dz1, w1, dx,
            M, K, H,
            dz1.stride(0), dz1.stride(1),
            w1.stride(0), w1.stride(1),
            dx.stride(0), dx.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32
        )

        return dx, dw1, db1, dw2, db2
    
class TritonGeluMlpLayer(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.empty(hidden_features, in_features))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_features))
        self.w2 = torch.nn.Parameter(torch.empty(out_features, hidden_features))
        self.b2 = torch.nn.Parameter(torch.zeros(out_features))

        torch.nn.init.kaiming_uniform_(self.w1, a=5 ** 0.5)
        torch.nn.init.kaiming_uniform_(self.w2, a=5 ** 0.5)

    def forward(self, x):
        orig_shape = x.shape
        if x.ndim > 2:
            x = x.view(-1, orig_shape[-1])
            
        y = TritonMLPFunction.apply(x, self.w1, self.b1, self.w2, self.b2)
        if len(orig_shape) > 2:
            y = y.view(*orig_shape[:-1], -1)
        return y
