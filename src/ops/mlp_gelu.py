import triton
import triton.language as tl
import torch
import torch.nn.functional as F
# Forward pass:
# z1 = x @ w1 + b1
# h  = GELU(z1)
# y  = h @ w2 + b2

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
def linear_layer_fwd(
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
def bwd_dx_gelu_fused_kernel(
    A,
    B,
    C,
    D,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dn,
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
    d_ptrs = D + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn

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

    z1 = tl.load(d_ptrs,mask = mask_c )
    s2i, s2pi = 0.707106781, 0.39894228
    cdf = 0.5 * (1 + tl.math.erf(z1 * s2i))
    pdf = s2pi * tl.exp(-0.5 * z1 * z1)
    dz1 = acc * (cdf + z1 * pdf)

    tl.store(c_ptrs, dz1, mask=mask_c)

class TritonMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        M, K = x.shape
        H, _ = w1.shape
        N, _ = w2.shape
        
        z1 = torch.empty((M, H), device=x.device, dtype=x.dtype)
        hidden = torch.empty((M, H), device=x.device, dtype=x.dtype)
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

       
        def is_div(val, tile): return val % tile == 0

        #(M, K) @ (K, H) -> (M, H)
        grid1 = lambda meta: (
            triton.cdiv(M, meta["TILE_M"]), 
            triton.cdiv(H, meta["TILE_N"])
        )

        linear_layer_gelu_fwd[grid1](
            x, w1, hidden, b1, z1,
            M, H, K,                 # N is H here
            x.stride(0), x.stride(1),
            w1.stride(1), w1.stride(0), 
            hidden.stride(0), hidden.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32,
            GROUP_M=4,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(H, 64),
            DIVISIBLE_K=is_div(K, 32),
            ADD_BIAS=True
        )

        #(M, H) @ (H, N) -> (M, N) 
        grid2 = lambda meta: (
            triton.cdiv(M, meta["TILE_M"]), 
            triton.cdiv(N, meta["TILE_N"])
        )

        linear_layer_fwd[grid2](
            hidden, w2, output, b2,
            M, N, H,                 # K is H here
            hidden.stride(0), hidden.stride(1),
            w2.stride(1), w2.stride(0),
            output.stride(0), output.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32,
            GROUP_M=4,
            DIVISIBLE_M=is_div(M, 64),
            DIVISIBLE_N=is_div(N, 64),
            DIVISIBLE_K=is_div(H, 32),
            ADD_BIAS=True
        )

        ctx.save_for_backward(x, w1, b1, w2, b2, hidden, z1)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w1, b1, w2, b2, hidden, z1 = ctx.saved_tensors
        M, K = x.shape
        H, _ = w1.shape
        N, _ = w2.shape

       
        # db2 = sum over M of grad_output
        db2 = torch.sum(grad_output, dim=0)
        
        # dw2 = hidden^T @ grad_output (Shape: H, N)
        dw2 = torch.empty((N, H), device=x.device, dtype=x.dtype)
        def is_div(val, tile): return val % tile == 0

        # we treat this as (N, M) @ (M, H) to get (N, H)
        grid_dw2 = lambda meta: (triton.cdiv(N, meta["TILE_M"]), triton.cdiv(H, meta["TILE_N"]))
        matmul_kernel[grid_dw2](
            grad_output, hidden, dw2,
            N,H,M,
            grad_output.stride(1), grad_output.stride(0), 
            hidden.stride(0), hidden.stride(1),
            dw2.stride(0), dw2.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32, GROUP_M=4,
            DIVISIBLE_M=is_div(M, 64), DIVISIBLE_N=is_div(N, 64), DIVISIBLE_K=is_div(H, tile=32)
        )
        dz1 = torch.empty_like(z1)
        grid_dz1 = grid_dw2 = lambda meta: (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(H, meta["TILE_N"]))
        bwd_dx_gelu_fused_kernel[grid_dz1](grad_output, w2, dz1, z1, M, H, N, grad_output.stride(0), grad_output.stride(1), w2.stride(0), w2.stride(1), z1.stride(0), z1.stride(1), dz1.stride(0), dz1.stride(1), TILE_M=64, TILE_N=64, TILE_K=32, GROUP_M=8,
            DIVISIBLE_M=is_div(M, 64), DIVISIBLE_N=is_div(H, 64), DIVISIBLE_K=is_div(N, tile=32))
        
        dw1 = torch.empty_like(w1)
        grid_dw1 = lambda meta: (triton.cdiv(H, meta["TILE_M"]), triton.cdiv(K, meta["TILE_N"]))
        matmul_kernel[grid_dw1](
            dz1, x, dw1,
            H,K,M,
            dz1.stride(1), dz1.stride(0),
            x.stride(0), x.stride(1),
            dw1.stride(0), dw1.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32, GROUP_M=4,
            DIVISIBLE_M=is_div(H, 64), DIVISIBLE_N=is_div(K, 64), DIVISIBLE_K=is_div(M, tile=32)
        )

        db1 = dz1.sum(0)
        dx = torch.empty_like(x)

        grid_dx = lambda meta: (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(K, meta["TILE_N"]))

        matmul_kernel[grid_dx](
            dz1, w1, dx,
            M, K, H,
            dz1.stride(0), dz1.stride(1),
            w1.stride(0), w1.stride(1),
            dx.stride(0), dx.stride(1),
            TILE_M=64, TILE_N=64, TILE_K=32, GROUP_M=4,
            DIVISIBLE_M=is_div(M, 64), DIVISIBLE_N=is_div(K, 64), DIVISIBLE_K=is_div(H, tile=32)
        )


        return dx,dw1,db1, dw2, db2

class TritonMlp(torch.nn.Module):
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
    

def test_triton_mlp_bmk_correctness():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    B, M, K = 2, 257, 129
    H, N = 255, 127 

    # Create input with batch dimension
    x = torch.randn((B, M, K), device=device, dtype=dtype, requires_grad=True)

    # Create Triton MLP layer
    mlp = TritonMlp(in_features=K, hidden_features=H, out_features=N).to(device=device, dtype=dtype)

    # Clone weights for reference
    w1, b1 = mlp.w1.clone().detach().requires_grad_(True), mlp.b1.clone().detach().requires_grad_(True)
    w2, b2 = mlp.w2.clone().detach().requires_grad_(True), mlp.b2.clone().detach().requires_grad_(True)

    y_ref = F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)
    grad_output = torch.randn_like(y_ref)
    y_ref.backward(grad_output)

    dx_ref, dw1_ref, db1_ref, dw2_ref, db2_ref = x.grad.clone(), w1.grad.clone(), b1.grad.clone(), w2.grad.clone(), b2.grad.clone()

    x.grad = None
    mlp.w1.grad = mlp.b1.grad = mlp.w2.grad = mlp.b2.grad = None

    y_triton = mlp(x)
    y_triton.backward(grad_output)

    dx_triton, dw1_triton, db1_triton, dw2_triton, db2_triton = x.grad, mlp.w1.grad, mlp.b1.grad, mlp.w2.grad, mlp.b2.grad

   
    
    diff = (y_triton - y_ref).abs()
    print("max abs diff :", diff.max().item())
    print("mean abs diff:", diff.mean().item())
    torch.testing.assert_close(y_triton, y_ref, atol=1e-3, rtol=1e-3)

 
    def check(name, got, ref, atol=1e-3, rtol=1e-3):
        diff = (got - ref).abs()
        print("max abs diff :", diff.max().item())
        print("mean abs diff:", diff.mean().item())
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)

    check("dx",  dx_triton,  dx_ref)
    check("dw1", dw1_triton, dw1_ref)
    check("db1", db1_triton, db1_ref)
    check("dw2", dw2_triton, dw2_ref)
    check("db2", db2_triton, db2_ref)

    print("FULL forward + backward correctness (BMK) PASSED")


if __name__ == "__main__":
    test_triton_mlp_bmk_correctness()
