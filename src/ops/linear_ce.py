import torch
import triton
import triton.language as tl


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
        w = tl.load(w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk, 
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(x, w)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, 
             accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def fused_ce_kernel(
    logits_ptr, targets_ptr, loss_ptr,
    n_cols, ignore_index,
    TILE_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    target = tl.load(targets_ptr + row_idx)
    row_start_ptr = logits_ptr + row_idx * n_cols
    
    if target == ignore_index:
        tl.store(loss_ptr + row_idx, 0.0)
        return

    # Online LogSumExp with improved stability
    m_row = -float('inf')
    d_row = 0.0
    
    for start_col in range(0, n_cols, TILE_SIZE):
        cols = start_col + tl.arange(0, TILE_SIZE)
        mask = cols < n_cols
        tile = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        
        m_tile = tl.max(tile, axis=0)
        m_combined = tl.maximum(m_row, m_tile)
        
        # Stability: scale existing denominator and new tile
        d_row = d_row * tl.exp(m_row - m_combined) + tl.sum(tl.exp(tile - m_combined), axis=0)
        m_row = m_combined

    lse = tl.log(d_row) + m_row
    target_logit = tl.load(row_start_ptr + target)
    tl.store(loss_ptr + row_idx, lse - target_logit)

def triton_linear(x, w, b=None):
    M, K = x.shape
    N, _ = w.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    linear_fwd_kernel[grid](x, w, b, y, M, K, N, x.stride(0), x.stride(1), 
                            w.stride(0), w.stride(1), y.stride(0), y.stride(1),
                            HAS_BIAS=b is not None, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return y

def triton_fused_ce(inputs, weight, bias, targets, ignore_index=-100):
    logits = triton_linear(inputs.view(-1, inputs.shape[-1]), weight, bias)
    M, N = logits.shape
    losses = torch.empty(M, device=logits.device, dtype=torch.float32)
    fused_ce_kernel[(M,)](logits, targets.view(-1), losses, N, ignore_index, TILE_SIZE=1024)
    valid_mask = (targets.view(-1) != ignore_index)
    return losses[valid_mask].mean()

@triton.jit
def fused_ce_bwd_kernel(
    logits_ptr, targets_ptr, grad_logits_ptr,
    n_cols, ignore_index,
    TILE_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    target = tl.load(targets_ptr + row_idx)
    row_start_ptr = logits_ptr + row_idx * n_cols
    grad_start_ptr = grad_logits_ptr + row_idx * n_cols

    if target == ignore_index:
        # Fill row with zeros if ignored
        for start_col in range(0, n_cols, TILE_SIZE):
            cols = start_col + tl.arange(0, TILE_SIZE)
            tl.store(grad_start_ptr + cols, 0.0, mask=cols < n_cols)
        return

    # 1. Compute LogSumExp (similar to your forward kernel)
    m_row = -float('inf')
    d_row = 0.0
    for start_col in range(0, n_cols, TILE_SIZE):
        cols = start_col + tl.arange(0, TILE_SIZE)
        mask = cols < n_cols
        tile = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        m_tile = tl.max(tile, axis=0)
        m_new = tl.maximum(m_row, m_tile)
        d_row = d_row * tl.exp(m_row - m_new) + tl.sum(tl.exp(tile - m_new), axis=0)
        m_row = m_new
    
    lse = tl.log(d_row) + m_row

    # 2. Compute (Softmax - 1_target) / Batch_Size
    # We use exp(logit - lse) to get the softmax value
    for start_col in range(0, n_cols, TILE_SIZE):
        cols = start_col + tl.arange(0, TILE_SIZE)
        mask = cols < n_cols
        tile = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        
        softmax_tile = tl.exp(tile - lse)
        
        # Subtract 1 from the target index
        target_mask = (cols == target) & mask
        grad_tile = tl.where(target_mask, softmax_tile - 1.0, softmax_tile)
        
        # Normalize by number of valid rows (this requires passing total valid count)
        # For simplicity, here we assume total rows M
        tl.store(grad_start_ptr + cols, grad_tile, mask=mask)


def compute_gradients_triton(grad_logits, x, weight, bias):
    M, N = grad_logits.shape
    K = x.shape[1]
    
    # 1. grad_input (dX) = grad_logits @ weight
    # Dimensions: M=M, N=K, K=N
    grad_input = torch.empty_like(x)
    grid_dx = (triton.cdiv(M, 32) * triton.cdiv(K, 32),)
    linear_fwd_kernel[grid_dx](
        grad_logits, weight, None, grad_input,
        M, N, K, # Note: Inner dim is N
        grad_logits.stride(0), grad_logits.stride(1),
        weight.stride(0), weight.stride(1),
        grad_input.stride(0), grad_input.stride(1),
        HAS_BIAS=False, BLOCK_M=32, BLOCK_K=32, BLOCK_N=32
    )

    # 2. grad_weight (dW) = grad_logits.T @ x
    # Dimensions: M=N, N=K, K=M
    grad_weight = torch.empty_like(weight)
    grid_dw = (triton.cdiv(N, 32) * triton.cdiv(K, 32),)
    linear_fwd_kernel[grid_dw](
        grad_logits, x, None, grad_weight,
        N, M, K, # Note: Inner dim is M
        grad_logits.stride(1), grad_logits.stride(0), # Transpose G by swapping strides
        x.stride(0), x.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
        HAS_BIAS=False, BLOCK_M=32, BLOCK_K=32, BLOCK_N=32
    )

    return grad_input, grad_weight

class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, targets, ignore_index=-100):
        # 1. Prepare shapes
        orig_shape = inputs.shape
        x_flat = inputs.view(-1, orig_shape[-1])
        M, K = x_flat.shape
        N, _ = weight.shape
        
        # 2. Linear Forward: MatMul(x, weight^T) + bias
        # Using your triton_linear logic
        logits = torch.empty((M, N), device=inputs.device, dtype=inputs.dtype)
        grid_linear = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        linear_fwd_kernel[grid_linear](
            x_flat, weight, bias, logits, M, K, N, 
            x_flat.stride(0), x_flat.stride(1),
            weight.stride(0), weight.stride(1), 
            logits.stride(0), logits.stride(1),
            HAS_BIAS=bias is not None, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 3. Cross Entropy Forward
        losses = torch.empty(M, device=inputs.device, dtype=torch.float32)
        fused_ce_kernel[(M,)](logits, targets.view(-1), losses, N, ignore_index, TILE_SIZE=1024)
        
        # Calculate mean loss for valid targets
        valid_mask = (targets.view(-1) != ignore_index)
        num_valid = valid_mask.sum()
        loss = losses.sum() / num_valid

        # Save for backward
        ctx.save_for_backward(x_flat, weight, bias, logits, targets.view(-1), valid_mask)
        ctx.ignore_index = ignore_index
        ctx.num_valid = num_valid
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, logits, targets, valid_mask = ctx.saved_tensors
        M, K = x.shape
        N = weight.shape[0]
        
        # 1. Compute dLoss/dLogits
        grad_logits = torch.empty_like(logits)
        fused_ce_bwd_kernel[(M,)](
            logits, targets, grad_logits, 
            N, ctx.ignore_index, TILE_SIZE=1024
        )
        
        scale = grad_output / ctx.num_valid
        grad_logits *= scale

        # 2. Compute dX and dW using your provided MatMul adaptation
        grad_input, grad_weight = compute_gradients_triton(grad_logits, x, weight, bias)
        
        # 3. Compute dBias
        grad_bias = grad_logits.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None


class TritonLinearCrossEntropy(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, ignore_index=-100):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.ignore_index = ignore_index
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, targets):
        return LinearCrossEntropyFunction.apply(
            x, self.weight, self.bias, targets, self.ignore_index
        )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # Vocabulary Size
        x_vals=[1024 * i for i in [1, 2, 4, 8, 16, 32, 64]],  # Sweep from 1k to 64k
        line_arg='provider',  # The grouped values
        line_vals=['pytorch', 'triton'],
        line_names=['PyTorch', 'Triton (Fused)'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Execution Time (ms)',
        plot_name='Linear-CrossEntropy Performance (Batch=2048, Hidden=1024)',
        args={'M': 2048, 'K': 1024},  # Constants
    )
)
def benchmark(M, K, N, provider):
    device = "cuda"
    dtype = torch.float16
    x = torch.randn((M, K), device=device, dtype=dtype, requires_grad=True)
    w = torch.randn((N, K), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((N,), device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, N, (M,), device=device)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        def run():
            logits = F.linear(x, w, b)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            return loss
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)
    
    if provider == 'triton':
        triton_layer = LinearCrossEntropyFunction.apply
        def run():
            loss = triton_layer(x, w, b, targets)
            loss.backward()
            return loss
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)
    
    return ms, max_ms, min_ms

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path='.')