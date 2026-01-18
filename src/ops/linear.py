import triton
import triton.language as tl
import torch
from torch import nn
# Standard Matrix Multiplication Kernel
@triton.jit
def matrix_multiply_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = k * BLOCK_K + offs_k
        
        a_mask = (offs_m[:, None] < M) & (k_remaining[None, :] < K)
        b_mask = (k_remaining[:, None] < K) & (offs_n[None, :] < N)
        
        # Standard indexing: row * row_stride + col * col_stride
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + k_remaining[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + k_remaining[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator, mask=c_mask)

# Specialized Forward Kernel with Bias
@triton.jit
def linear_fwd_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, K, N,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, 
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # Load W as transposed: W is  we load (K, N)
        w = tl.load(w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk, 
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(x, w)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, 
             accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Bias Gradient Kernel
@triton.jit
def linear_bw_db_kernel(dy_ptr, db_ptr, M, N, stride_dym, stride_dyn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m in range(0, tl.cdiv(M, BLOCK_M)):
        offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
        dy = tl.load(dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn, 
                     mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc += tl.sum(dy, axis=0)

    tl.store(db_ptr + offs_n, acc, mask=offs_n < N)

# --- Launchers ---

def launch_dx(dy, w, dx):
    M, N = dy.shape # dy: (Batch, Out)
    _, K = w.shape  # w: (Out, In)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(K, META['BLOCK_N']),)
    # dx = dy @ w -> (M, N) @ (N, K)
    matrix_multiply_kernel[grid](
        dy, w, dx, M, N, K,
        dy.stride(0), dy.stride(1),
        w.stride(1), w.stride(0), # Normal strides for w (N, K)
        dx.stride(0), dx.stride(1),
        BLOCK_M=64, BLOCK_K=32, BLOCK_N=64
    )

def launch_dw(x, dy, dw):
    M, K = x.shape  # x: (Batch, In)
    _, N = dy.shape # dy: (Batch, Out)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']) * triton.cdiv(K, META['BLOCK_N']),)
    # dw = dy.T @ x -> (Out, Batch) @ (Batch, In) -> result (Out, In)
    matrix_multiply_kernel[grid](
        dy, x, dw, N, M, K,
        dy.stride(1), dy.stride(0), # Swap dy strides to treat as (N, M)
        x.stride(0), x.stride(1),   # Normal x strides (M, K)
        dw.stride(0), dw.stride(1),
        BLOCK_M=64, BLOCK_K=32, BLOCK_N=64
    )

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        x = x.contiguous()
        M, K = x.shape
        N, K_w = weight.shape
        assert K == K_w, "Incompatible dimensions"
        
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

        linear_fwd_kernel[grid](
            x, weight, bias, y, M, K, N,
            x.stride(0), x.stride(1), weight.stride(1), weight.stride(0), y.stride(0), y.stride(1),
            HAS_BIAS=bias is not None, BLOCK_M=128, BLOCK_K=32, BLOCK_N=128
        )
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        grad_input = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(bias) if bias is not None else None

        launch_dx(grad_output, weight, grad_input)
        launch_dw(x, grad_output, grad_weight)
        
        if bias is not None:
            grid_db = (triton.cdiv(grad_output.shape[1], 64),)
            linear_bw_db_kernel[grid_db](
                grad_output, grad_bias, grad_output.shape[0], grad_output.shape[1],
                grad_output.stride(0), grad_output.stride(1), BLOCK_M=128, BLOCK_N=64
            )

        return grad_input, grad_weight, grad_bias

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features,in_features))
    
        self.bias = nn.Parameter(torch.empty(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Apply the autograd function using .apply()
        return TritonLinearFunction.apply(x, self.weight, self.bias)

if __name__=="__main__":

    in_features = 128
    out_features = 64
    model = TritonLinear(in_features, out_features).cuda()

    # 2. Prepare dummy input data
    # Batch size: 32
    batch_size = 32
    x = torch.randn(batch_size, in_features, device='cuda', requires_grad=True)

    # 3. Forward Pass
    # This calls TritonLinearFunction.forward via .apply()
    output = model(x)

    # 4. Define a Loss and Compute Gradients
    # We'll use a simple MSE loss for the example
    target = torch.randn(batch_size, out_features, device='cuda')
    loss = torch.nn.functional.mse_loss(output, target)

    # 5. Backward Pass
    # This triggers the TritonLinearFunction.backward logic (dx, dw, db)
    loss.backward()

    # 6. Check results
    print(f"Output shape: {output.shape}")
    print(f"Gradient for x exists: {x.grad is not None}")
    print(f"Gradient for weights exists: {model.weight.grad is not None}")