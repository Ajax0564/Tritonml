import torch
import triton
import triton.language as tl

@triton.jit
def online_softmax_kernel(
    output_ptr, input_ptr, 
    M,N,
    stride_m,stride_n, 
    BLOCK_M: tl.constexpr, TILE_SIZE: tl.constexpr
):
    
    pid_row = tl.program_id(0)

    row_offsets = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Initialize stats for each row in the block
    m_row = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    d_row = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Online way of  Max and SumExp
    for start_col in range(0, N, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = (row_offsets[:, None]<M) & (col_offsets[None, :] < N)
        
        tile_ptr = input_ptr + row_offsets[:, None] * stride_m + col_offsets[None, :]*stride_n
        tile = tl.load(tile_ptr, mask=mask, other=-float('inf')).to(tl.float32)
        
        m_new = tl.max(tile, axis=1)
        # Numerical stability scaling for the online sum
        alpha = tl.exp(m_row - m_new)
        d_row = d_row * alpha + tl.sum(tl.exp(tile - m_new[:, None]), axis=1)
        m_row = m_new

    # Normalize and Store
    for start_col in range(0, N, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = (row_offsets[:, None]<M) & (col_offsets[None, :] < N)
        
        tile_ptr = input_ptr + row_offsets[:, None] * stride_m + col_offsets[None, :]*stride_n
        tile = tl.load(tile_ptr, mask=mask, other=-float('inf')).to(tl.float32)
        
        output = tl.exp(tile - m_row[:, None]) / d_row[:, None]
        
        out_ptr = output_ptr + row_offsets[:, None] * stride_m + col_offsets[None, :]*stride_n
        tl.store(out_ptr, output, mask=mask)


@triton.jit
def softmax_backward_kernel(
    d_out_ptr, y_ptr, dx_ptr,
    M,N,
    stride_gr,stride_gc,
    stride_yr,stride_yc,
    stride_xr,stride_xc,
    BLOCK_M: tl.constexpr, TILE_SIZE: tl.constexpr
):
    pid_row = tl.program_id(0)
   

    row_offsets = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)

    # Compute sum(d_out * y) for each row in the block
    sum_dy_y = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_col in range(0, N, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = (row_offsets[:, None]<M) & (col_offsets[None, :] < N)
        
        do_tile_ptr = d_out_ptr + row_offsets[:, None] * stride_gr + col_offsets[None, :]*stride_gc
        y_tile_ptr = y_ptr + row_offsets[:, None] * stride_yr + col_offsets[None, :]*stride_yc
        
        dy_tile = tl.load(do_tile_ptr, mask=mask, other=0.0).to(tl.float32)
        y_tile = tl.load(y_tile_ptr, mask=mask, other=0.0).to(tl.float32)
        
        sum_dy_y += tl.sum(dy_tile * y_tile, axis=1)

    # dx = y * (dy - sum_dy_y)
    for start_col in range(0, N, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = (row_offsets[:, None]<M) & (col_offsets[None, :] < N)
        
        do_tile_ptr = d_out_ptr + row_offsets[:, None] * stride_gr + col_offsets[None, :]*stride_gc
        y_tile_ptr = y_ptr + row_offsets[:, None] * stride_yr + col_offsets[None, :]*stride_yc
        
        dy_tile = tl.load(do_tile_ptr, mask=mask, other=0.0).to(tl.float32)
        y_tile = tl.load(y_tile_ptr, mask=mask, other=0.0).to(tl.float32)
        
        dx_tile = y_tile * (dy_tile - sum_dy_y[:, None])
        
        dx_tile_ptr = dx_ptr + row_offsets[:, None] * stride_xr + col_offsets[None, :]*stride_xc
        tl.store(dx_tile_ptr, dx_tile, mask=mask)


class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        BLOCK_M = 16 
        TILE_SIZE = 1024
        
        # 2D Grid: (Rows / BLOCK_M, Batch)
        grid = (triton.cdiv(n_rows, BLOCK_M),)
        
        online_softmax_kernel[grid](
            output, x, n_rows, n_cols, 
            x.stride(0), x.stride(1),
            BLOCK_M=BLOCK_M, TILE_SIZE=TILE_SIZE,
            num_warps=4
        )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        n_rows, n_cols = output.shape
        grad_input = torch.zeros_like(grad_output)
        
        BLOCK_M = 16
        grid = (triton.cdiv(n_rows, BLOCK_M),)
        
        softmax_backward_kernel[grid](
            grad_output, output, grad_input,
            n_rows, n_cols, 
            grad_output.stride(0), grad_output.stride(1),
            output.stride(0), output.stride(1),
            grad_input.stride(0), grad_input.stride(1),
            BLOCK_M=BLOCK_M, TILE_SIZE=1024, num_warps=4
        )
        return grad_input


def test_softmax():
    N_ROWS, N_COLS =  2048,2048
    print(f"Testing with Shape: ({N_ROWS}, {N_COLS})")
    
    x = torch.randn((N_ROWS, N_COLS), device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Triton Path
    y_triton = TritonSoftmax.apply(x)
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
    test_softmax()
