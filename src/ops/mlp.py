import torch
import triton
import triton.language as tl

# --- 1. The Triton Kernel ---
@triton.jit
def fused_mlp_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, y_ptr,
    M, N, K, H,
    stride_xm, stride_xk,
    stride_w1k, stride_w1h,
    stride_w2h, stride_w2n,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    # Initialize Layer 2 accumulator in float32
    l2_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Outer loop over Hidden Dimension (H)
    for h_idx in range(0, tl.cdiv(H, BLOCK_SIZE_H)):
        offs_h = h_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        l1_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)
        
        # Inner loop over Input Dimension (K)
        for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
            w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_h[None, :] * stride_w1h)
            
            x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            w1_tile = tl.load(w1_ptrs, mask=(offs_k[:, None] < K) & (offs_h[None, :] < H), other=0.0)
            l1_acc = tl.dot(x_tile, w1_tile, l1_acc)

        # Layer 1 Bias + ReLU
        if b1_ptr is not None:
            bias1 = tl.load(b1_ptr + offs_h, mask=offs_h < H, other=0.0)
            l1_acc += bias1[None, :]
        l1_output = tl.where(l1_acc > 0, l1_acc, 0).to(tl.float16)

        # Layer 2 MatMul
        w2_ptrs = w2_ptr + (offs_h[:, None] * stride_w2h + offs_n[None, :] * stride_w2n)
        w2_tile = tl.load(w2_ptrs, mask=(offs_h[:, None] < H) & (offs_n[None, :] < N), other=0.0)
        l2_acc = tl.dot(l1_output, w2_tile, l2_acc)

    # Layer 2 Bias
    if b2_ptr is not None:
        bias2 = tl.load(b2_ptr + offs_n, mask=offs_n < N, other=0.0)
        l2_acc += bias2[None, :]

    # Store result
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, l2_acc.to(tl.float16), mask=mask)

# --- 2. The Python Wrapper ---
def triton_mlp(x, w1, b1, w2, b2):
    M, K = x.shape
    K_ref, H = w1.shape
    H_ref, N = w2.shape
    assert K == K_ref and H == H_ref, "Dimension mismatch"

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

    fused_mlp_kernel[grid](
        x, w1, b1, w2, b2, y,
        M, N, K, H,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, BLOCK_SIZE_H=32
    )
    return y

# --- 3. The Comparison Script ---
def test_mlp():
    torch.manual_seed(42)
    device = "cuda"
    
    # Dimensions
    M, K, H, N = 256, 512, 1024, 256
    
    # Inputs (Using float16)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    w1 = torch.randn((K, H), device=device, dtype=torch.float16)
    b1 = torch.randn((H,), device=device, dtype=torch.float16)
    w2 = torch.randn((H, N), device=device, dtype=torch.float16)
    b2 = torch.randn((N,), device=device, dtype=torch.float16)

    # PyTorch Reference
    # Note: We compute in FP32 then cast to FP16 to match Triton's internal high-precision accumulation
    ref_l1 = torch.matmul(x.to(torch.float32), w1.to(torch.float32)) + b1.to(torch.float32)
    ref_relu = torch.relu(ref_l1).to(torch.float16)
    ref_y = (torch.matmul(ref_relu.to(torch.float32), w2.to(torch.float32)) + b2.to(torch.float32)).to(torch.float16)

    # Triton Result
    tri_y = triton_mlp(x, w1, b1, w2, b2)

    # Verification
    max_diff = torch.max(torch.abs(ref_y - tri_y)).item()
    print(f"Max absolute difference: {max_diff:.4f}")

    if torch.allclose(ref_y, tri_y, atol=1e-2, rtol=1e-2):
        print("✅ Success: Triton and PyTorch results match!")
    else:
        print("❌ Failure: Significant numerical mismatch.")

if __name__ == "__main__":
    test_mlp()