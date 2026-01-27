import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_forward(
    input_ptr,
    output_ptr,
    weight_ptr,
    rstd_ptr,
    row_stride,
    feature_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of input and output tensors to compute
    row_idx = tl.program_id(0)
    output_ptr += row_idx * row_stride
    input_ptr += row_idx * row_stride

    #  (mean of squared values for RMS)
    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_offset in range(0, tl.cdiv(BLOCK_SIZE, feature_dim)):
        col_indices = block_offset*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        input_values = tl.load(
            input_ptr + col_indices, mask=col_indices < feature_dim, other=0.0
        ).to(tl.float32)
        sum_of_squares += input_values * input_values

    variance = tl.sum(sum_of_squares, axis=0) / feature_dim
    reciprocal_std = 1 / tl.sqrt(variance + eps)

    # Store reciprocal standard deviation for backward pass
    tl.store(rstd_ptr + row_idx, reciprocal_std)

    # Normalize input and apply weight transformation
    for block_offset in range(0, tl.cdiv(BLOCK_SIZE, feature_dim)):
        col_indices = block_offset*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid_mask = col_indices < feature_dim

        weight_values = tl.load(weight_ptr + col_indices, mask=valid_mask)
        input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0.0).to(
            tl.float32
        )

        normalized_values = input_values * reciprocal_std
        output_values = normalized_values * weight_values

        # Write final output
        tl.store(output_ptr + col_indices, output_values, mask=valid_mask)


@triton.jit
def rms_norm_backward(
    d_out_ptr,      # Gradient of output
    input_ptr,      # Original input X
    weight_ptr,     # Weights W
    rstd_ptr,       # Saved reciprocal std
    dx_ptr,         # Output gradient for input X
    dw_ptr,         # Global gradient for weights W
    row_stride,
    feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Adjust pointers for this row
    d_out_ptr += row_idx * row_stride
    input_ptr += row_idx * row_stride
    dx_ptr += row_idx * row_stride

    # Load reciprocal std (scalar for the row)
    inv_std = tl.load(rstd_ptr + row_idx)

    # We assume BLOCK_SIZE >= feature_dim for the most optimized "Single Pass"
    # If feature_dim is larger, this can be wrapped in a loop.
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < feature_dim

    #into registers ONCE
    do = tl.load(d_out_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    #Compute intermediate values
    x_hat = x * inv_std
    w_do = do * w
    
    # Compute the row-sum for dx (m_sum)
    m_sum = tl.sum(w_do * x_hat, axis=0)

  
    dx = inv_std * (w_do - (x_hat * m_sum / feature_dim))
    tl.store(dx_ptr + cols, dx, mask=mask)

    
    # This avoids storing the full M x N matrix
    dw_local = do * x_hat
    tl.atomic_add(dw_ptr + cols, dw_local, mask=mask)



class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        M, N = x.shape
        BLOCK_SIZE = triton.next_power_of_2(N) # Simplest case
        if BLOCK_SIZE > 65536: BLOCK_SIZE = 4096 # Tiling for large dims

        y = torch.empty_like(x)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        rms_norm_forward[(M,)](
            x, y, weight, rstd,
            x.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.feature_dim = N
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        M, N = x.shape
        
        dx = torch.empty_like(x)
        # Allocate ONLY the final weight gradient size
        dw = torch.zeros_like(weight, dtype=torch.float32)

        grid = (M,)
        rms_norm_backward[grid](
            grad_output, x, weight, rstd,
            dx, dw,
            x.stride(0), N,
            BLOCK_SIZE=ctx.BLOCK_SIZE
        )

        return dx, dw.to(weight.dtype), None

if __name__ == "__main__":
    # 1. Correctness Check
    M, N = 128, 1024
    x = torch.randn((M, N), device='cuda', requires_grad=True)
    w = torch.randn((N,), device='cuda', requires_grad=True)
    
    # Triton result
    y_tri = TritonRMSNorm.apply(x, w)
    y_tri.backward(torch.ones_like(y_tri))
    dx_tri, dw_tri = x.grad.clone(), w.grad.clone()
    
    # Torch result
    x.grad, w.grad = None, None
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
    y_ref = (x / rms) * w
    y_ref.backward(torch.ones_like(y_ref))
    dx_ref, dw_ref = x.grad.clone(), w.grad.clone()

    print(f"Max Fwd Diff: {torch.max(torch.abs(y_tri - y_ref)):.2e}")
    print(f"Max Dx Diff: {torch.max(torch.abs(dx_tri - dx_ref)):.2e}")
    print(f"Max Dw Diff: {torch.max(torch.abs(dw_tri - dw_ref)):.2e}")