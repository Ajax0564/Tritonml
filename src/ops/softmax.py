import torch
import triton
import triton.language as tl

@triton.jit
def online_softmax_kernel_3d(
    output_ptr, input_ptr, 
    n_rows, n_cols,
    stride_b, stride_r,  # Strides for input
    BLOCK_M: tl.constexpr, TILE_SIZE: tl.constexpr
):
    # Grid: (Batch, num_blocks_of_rows)
    # Mapping: pid_batch handles the B dimension, pid_row handles the M dimension
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)

    row_offsets = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < n_rows
    
    # Base pointers for this specific batch
    batch_input_ptr = input_ptr + pid_batch * stride_b
    batch_output_ptr = output_ptr + pid_batch * stride_b
    
    # Initialize stats for each row in the block
    m_row = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    d_row = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # 1. Online Update Logic (Pass 1: Find Max and SumExp)
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = row_mask[:, None] & (col_offsets[None, :] < n_cols)
        
        tile_ptr = batch_input_ptr + row_offsets[:, None] * stride_r + col_offsets[None, :]
        tile = tl.load(tile_ptr, mask=mask, other=-float('inf')).to(tl.float32)
        
        m_new = tl.max(tile, axis=1)
        # Numerical stability scaling for the online sum
        alpha = tl.exp(m_row - m_new)
        d_row = d_row * alpha + tl.sum(tl.exp(tile - m_new[:, None]), axis=1)
        m_row = m_new

    # 2. Normalize and Store (Pass 2)
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = row_mask[:, None] & (col_offsets[None, :] < n_cols)
        
        tile_ptr = batch_input_ptr + row_offsets[:, None] * stride_r + col_offsets[None, :]
        tile = tl.load(tile_ptr, mask=mask, other=-float('inf')).to(tl.float32)
        
        output = tl.exp(tile - m_row[:, None]) / d_row[:, None]
        
        out_ptr = batch_output_ptr + row_offsets[:, None] * stride_r + col_offsets[None, :]
        tl.store(out_ptr, output, mask=mask)


@triton.jit
def softmax_backward_kernel_3d(
    d_out_ptr, y_ptr, dx_ptr,
    n_rows, n_cols,
    stride_b, stride_r,
    BLOCK_M: tl.constexpr, TILE_SIZE: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)

    row_offsets = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < n_rows

    batch_do_ptr = d_out_ptr + pid_batch * stride_b
    batch_y_ptr = y_ptr + pid_batch * stride_b
    batch_dx_ptr = dx_ptr + pid_batch * stride_b

    # Compute sum(d_out * y) for each row in the block
    sum_dy_y = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = row_mask[:, None] & (col_offsets[None, :] < n_cols)
        
        do_tile_ptr = batch_do_ptr + row_offsets[:, None] * stride_r + col_offsets[None, :]
        y_tile_ptr = batch_y_ptr + row_offsets[:, None] * stride_r + col_offsets[None, :]
        
        dy_tile = tl.load(do_tile_ptr, mask=mask, other=0.0).to(tl.float32)
        y_tile = tl.load(y_tile_ptr, mask=mask, other=0.0).to(tl.float32)
        
        sum_dy_y += tl.sum(dy_tile * y_tile, axis=1)

    # dx = y * (dy - sum_dy_y)
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = row_mask[:, None] & (col_offsets[None, :] < n_cols)
        
        do_tile_ptr = batch_do_ptr + row_offsets[:, None] * stride_r + col_offsets[None, :]
        y_tile_ptr = batch_y_ptr + row_offsets[:, None] * stride_r + col_offsets[None, :]
        
        dy_tile = tl.load(do_tile_ptr, mask=mask, other=0.0).to(tl.float32)
        y_tile = tl.load(y_tile_ptr, mask=mask, other=0.0).to(tl.float32)
        
        dx_tile = y_tile * (dy_tile - sum_dy_y[:, None])
        
        dx_tile_ptr = batch_dx_ptr + row_offsets[:, None] * stride_r + col_offsets[None, :]
        tl.store(dx_tile_ptr, dx_tile, mask=mask)


class TritonSoftmax3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        batch, n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        BLOCK_M = 16 
        TILE_SIZE = 1024
        
        # 2D Grid: (Rows / BLOCK_M, Batch)
        grid = (triton.cdiv(n_rows, BLOCK_M), batch)
        
        online_softmax_kernel_3d[grid](
            output, x, n_rows, n_cols, 
            x.stride(0), x.stride(1),
            BLOCK_M=BLOCK_M, TILE_SIZE=TILE_SIZE,
            num_warps=8
        )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        batch, n_rows, n_cols = output.shape
        grad_input = torch.empty_like(grad_output)
        
        BLOCK_M = 16
        grid = (triton.cdiv(n_rows, BLOCK_M), batch)
        
        softmax_backward_kernel_3d[grid](
            grad_output, output, grad_input,
            n_rows, n_cols, 
            grad_output.stride(0), grad_output.stride(1),
            BLOCK_M=BLOCK_M, TILE_SIZE=1024, num_warps=8
        )
        return grad_input


def test_softmax_3d():
    BATCH, N_ROWS, N_COLS = 16, 2048,2048
    print(f"Testing with Shape: ({BATCH}, {N_ROWS}, {N_COLS})")
    
    x = torch.randn((BATCH, N_ROWS, N_COLS), device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Triton Path
    y_triton = TritonSoftmax3D.apply(x)
    grad_output = torch.randn_like(y_triton)
    y_triton.backward(grad_output)
    dx_triton = x.grad.clone()
    
    # Torch Path
    x.grad.zero_()
    y_torch = torch.softmax(x, dim=-1)
    y_torch.backward(grad_output)
    dx_torch = x.grad
    
    fwd_match = torch.allclose(y_triton, y_torch, atol=1e-6)
    bwd_match = torch.allclose(dx_triton, dx_torch, atol=1e-6)
    
    print(f"Forward match: {fwd_match}")
    print(f"Backward match: {bwd_match}")

if __name__ == "__main__":
    test_softmax_3d()