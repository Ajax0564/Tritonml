import torch
import triton
import triton.language as tl

@triton.jit
def online_softmax_kernel(
    output_ptr, input_ptr, 
    n_rows, n_cols, 
    stride_row,
    TILE_SIZE: tl.constexpr
):
    # Each program handles one row (0 to M-1)
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    # Offset the pointers to the start of this specific row
    row_start_ptr = input_ptr + row_idx * stride_row
    
    m_row = -float('inf')
    d_row = 0.0
    
    # Pass 1: Online Max and Sum
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = col_offsets < n_cols
        tile = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        m_new = tl.max(tile, axis=0)
        # Numerical stability: update denominator based on new max
        alpha = tl.exp(m_row - tl.maximum(m_row, m_new)) 
        m_combined = tl.maximum(m_row, m_new)
        
        d_row = d_row * tl.exp(m_row - m_combined) + tl.sum(tl.exp(tile - m_combined), axis=0)
        m_row = m_combined

    # Pass 2: Write out normalized values
    out_row_ptr = output_ptr + row_idx * stride_row
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = col_offsets < n_cols
        tile = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        output = tl.exp(tile - m_row) / d_row
        tl.store(out_row_ptr + col_offsets, output, mask=mask)

@triton.jit
def softmax_backward_kernel(
    d_out_ptr, y_ptr, dx_ptr,
    n_rows, n_cols,
    stride_row,
    TILE_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    # Calculate row-specific pointers
    row_offset = row_idx * stride_row
    d_out_row_ptr = d_out_ptr + row_offset
    y_row_ptr = y_ptr + row_offset
    dx_row_ptr = dx_ptr + row_offset

    # Compute sum(d_out * y) for the row
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
        # x is M x N
        M, N = x.shape
        output = torch.empty_like(x)
        TILE_SIZE = 1024 
        
        # Grid is M (one program per row)
        grid = (M,)
        online_softmax_kernel[grid](
            output, x, 
            M, N, 
            x.stride(0), 
            TILE_SIZE=TILE_SIZE
        )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        M, N = output.shape
        grad_input = torch.empty_like(grad_output)
        
        grid = (M,)
        softmax_backward_kernel[grid](
            grad_output, output, grad_input,
            M, N,
            output.stride(0),
            TILE_SIZE=1024
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