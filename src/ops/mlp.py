import triton
import triton.language as tl
import torch

#MK@KH -> MH
# MH@HN -> MN

@triton.jit
def fused_matmul_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    M, N, K, H,
    stride_am, stride_ak,
    stride_bk, stride_bh,
    stride_ch, stride_cn,
    stride_dm, stride_dn,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, 
    block_size_k: tl.constexpr, block_size_h: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, block_size_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * block_size_m + tl.arange(0, block_size_m))
    offs_dn = (pid_n * block_size_n + tl.arange(0, block_size_n))
    
    # output tile (M x N)
    final_accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)

    for h in range(0, tl.cdiv(H, block_size_h)):
        offs_h = h * block_size_h + tl.arange(0, block_size_h)
        
        #  tile for A @ B (M x H)
        inter_accumulator = tl.zeros((block_size_m, block_size_h), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, block_size_k)):
            offs_k = k * block_size_k + tl.arange(0, block_size_k)
            
            # Pointers for A (M x K) and B (K x H)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_h[None, :] * stride_bh)
            
            # CRITICAL: Mask both dimensions for every load
            mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K)
            mask_b = (offs_k[:, None] < K) & (offs_h[None, :] < H)
            
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            
            inter_accumulator += tl.dot(a, b)

        # We now have a tile of (A @ B). Now multiply by C (H x N)
        # Load C tile for the current h and the current pid_n
        c_ptrs = c_ptr + (offs_h[:, None] * stride_ch + offs_dn[None, :] * stride_cn)
        mask_c = (offs_h[:, None] < H) & (offs_dn[None, :] < N)
        c = tl.load(c_ptrs, mask=mask_c, other=0.0)

        # Accumulate: (A@B_tile) @ C_tile
        final_accumulator += tl.dot(inter_accumulator, c)

    # Store results
    d_ptrs = d_ptr + (offs_am[:, None] * stride_dm + offs_dn[None, :] * stride_dn)
    mask_d = (offs_am[:, None] < M) & (offs_dn[None, :] < N)
    tl.store(d_ptrs, final_accumulator, mask=mask_d)


# stacked mlp 
@triton.jit
def mlp_stacked_fwd(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, y_ptr,
    M, N, K, H,
    stride_xm, stride_xk,
    stride_w1k, stride_w1h,
    stride_w2h, stride_w2n,
    stride_ym, stride_yn,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, 
    block_size_k: tl.constexpr, block_size_h: tl.constexpr,
    has_bias1: tl.constexpr,
    has_bias2: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, block_size_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = (pid_m * block_size_m + tl.arange(0, block_size_m))
    offs_n = (pid_n * block_size_n + tl.arange(0, block_size_n))
    
    # output tile (M x N)
    final_accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)

    for h in range(0, tl.cdiv(H, block_size_h)):
        offs_h = h * block_size_h + tl.arange(0, block_size_h)
        
        #  tile for A @ B (M x H)
        inter_accumulator = tl.zeros((block_size_m, block_size_h), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, block_size_k)):
            offs_k = k * block_size_k + tl.arange(0, block_size_k)
            
            # Pointers for A (M x K) and B (K x H)
            x_ptr_grid = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
            w1_ptr_grid = w1_ptr + (offs_k[:, None] * stride_w1k + offs_h[None, :] * stride_w1h)
            
            # CRITICAL: Mask both dimensions for every load
            mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            mask_w1 = (offs_k[:, None] < K) & (offs_h[None, :] < H)
            
            x = tl.load(x_ptr_grid, mask=mask_x, other=0.0)
            w1 = tl.load(w1_ptr_grid, mask=mask_w1, other=0.0)
            
            inter_accumulator += tl.dot(x, w1)

        if has_bias1:
            bias1 = tl.load(b1_ptr + offs_h, mask=offs_h < H, other=0.0)
            inter_accumulator += bias1[None, :]
        
        # gelu activation
        cdf = 0.5 * (1 + tl.math.erf(0.707106781 * inter_accumulator))
        inter_accumulator = cdf * inter_accumulator
        

        # We now have a tile of (A @ B). Now multiply by C (H x N)
        # Load C tile for the current h and the current pid_n
        w2_ptr_grid = w2_ptr + (offs_h[:, None] * stride_w2h + offs_n[None, :] * stride_w2n)
        mask_w2 = (offs_h[:, None] < H) & (offs_n[None, :] < N)
        w2 = tl.load(w2_ptr_grid, mask=mask_w2, other=0.0)

        # Accumulate: (A@B_tile) @ C_tile
        final_accumulator += tl.dot(inter_accumulator, w2)

    if has_bias2:
        bias2 = tl.load(b2_ptr + offs_n, mask=offs_n < N, other=0.0)
        final_accumulator += bias2[None, :]
    # Store results
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, final_accumulator, mask=mask_y)




import torch
import triton
import triton.language as tl

# --- FORWARD KERNELS ---

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

# --- BACKWARD KERNELS ---

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

# --- AUTOGRAD ---

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

        return None, dw1, db1, dw2, db2

# --- VERIFICATION SCRIPT ---

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
    
    ref_grads = {
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
    
    results = {
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