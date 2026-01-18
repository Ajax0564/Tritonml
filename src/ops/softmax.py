import torch
import triton
import triton.language as tl

@triton.jit
def online_softmax_kernel(
    output_ptr, input_ptr, 
    n_cols, 
    TILE_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * n_cols
    
    #  Compute Global Statistics (Online) ---
    m_row = -float('inf')
    d_row = 0.0
    
    # Tiled loop to find global max and sum without storing the whole row
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = col_offsets < n_cols
        tile = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        # Online Update Logic
        m_new = tl.max(tile, axis=0)
        alpha = tl.exp(m_row - m_new) # Scaling factor for numerical stability
        d_row = d_row * alpha + tl.sum(tl.exp(tile - m_new), axis=0)
        m_row = m_new

    #  Normalize
    out_row_ptr = output_ptr + row_idx * n_cols
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = col_offsets < n_cols
        tile = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        # Use global stats to normalize
        output = tl.exp(tile - m_row) / d_row
        tl.store(out_row_ptr + col_offsets, output, mask=mask)


@triton.jit
def softmax_backward_kernel(
    d_out_ptr, y_ptr, dx_ptr,
    n_cols,
    TILE_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Pointers for this row
    d_out_row_ptr = d_out_ptr + row_idx * n_cols
    y_row_ptr = y_ptr + row_idx * n_cols
    dx_row_ptr = dx_ptr + row_idx * n_cols

    # Compute sum(d_out * y) 
    sum_dy_y = 0.0
    for start_col in range(0, n_cols, TILE_SIZE):
        offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = offsets < n_cols
        
        dy_tile = tl.load(d_out_row_ptr + offsets, mask=mask, other=0.0)
        y_tile = tl.load(y_row_ptr + offsets, mask=mask, other=0.0)
        
        sum_dy_y += tl.sum(dy_tile * y_tile, axis=0)

    # dx = y * (dy - sum_dy_y) 
    for start_col in range(0, n_cols, TILE_SIZE):
        offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = offsets < n_cols
        
        dy_tile = tl.load(d_out_row_ptr + offsets, mask=mask, other=0.0)
        y_tile = tl.load(y_row_ptr + offsets, mask=mask, other=0.0)
        
        dx_tile = y_tile * (dy_tile - sum_dy_y)
        tl.store(dx_row_ptr + offsets, dx_tile, mask=mask)


class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        # Flatten to 2D if necessary
        x = x.view(-1, n_cols)
        output = torch.empty_like(x)
        TILE_SIZE = 1024 # Standard block size
        
        # We reuse your forward kernel provided in the prompt
        from __main__ import online_softmax_kernel 
        grid = (x.shape[0],)
        online_softmax_kernel[grid](
            output, x, n_cols, TILE_SIZE=TILE_SIZE
        )
        ctx.save_for_backward(output)
        ctx.n_cols = n_cols
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        n_cols = ctx.n_cols
        grad_input = torch.empty_like(grad_output)
        
        grid = (output.shape[0],)
        softmax_backward_kernel[grid](
            grad_output, output, grad_input,
            n_cols, TILE_SIZE=1024
        )
        return grad_input

def test_softmax_backward():
    N_ROWS, N_COLS = 4096*8, 4096
    x = torch.randn((N_ROWS, N_COLS), device='cuda', dtype=torch.float32, requires_grad=True)
    
    # 1. Triton Pass
    y_triton = TritonSoftmax.apply(x)
    grad_output = torch.randn_like(y_triton)
    y_triton.backward(grad_output)
    dx_triton = x.grad.clone()
    
    # 2. PyTorch Pass (Reference)
    x.grad.zero_()
    y_torch = torch.softmax(x, dim=-1)
    y_torch.backward(grad_output)
    dx_torch = x.grad
    
    # Verification
    forward_check = torch.allclose(y_triton, y_torch, atol=1e-6)
    backward_check = torch.allclose(dx_triton, dx_torch, atol=1e-6)
    
    print(f"Forward match: {forward_check}")
    print(f"Backward match: {backward_check}")
    print(f"Max grad diff: {(dx_triton - dx_torch).abs().max().item()}")

if __name__ == "__main__":
    test_softmax_backward()