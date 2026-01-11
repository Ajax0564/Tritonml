import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_base(
    output_ptr, input_ptr, 
    input_row_stride, output_row_stride, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate pointers for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load row data (masking for safety)
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Online Softmax Logic (Numerical Stability)
    # 1. Find Max
    row_max = tl.max(row, axis=0)
    # 2. Compute Exponentials and Sum
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    # 3. Normalize
    out = numerator / denominator
    
    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, out, mask=mask)


@triton.jit
def softmax_forward(
    input_ptr,             # pointer to [n_rows, n_cols]
    output_ptr,            # pointer to [n_rows, n_cols]
    n_rows: tl.constexpr,  # number of rows
    n_cols: tl.constexpr,  # number of columns (feature dim)
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)

    # Base pointers for this row
    in_row_ptr = input_ptr + row_id * n_cols
    out_row_ptr = output_ptr + row_id * n_cols

    # ---- Fast path: entire row fits in BLOCK_SIZE ----
    if n_cols <= BLOCK_SIZE:
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        vals = tl.load(in_row_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)

        row_max = tl.max(vals, axis=0)
        vals_stable = vals - row_max
        numer = tl.exp(vals_stable)
        denom = tl.sum(numer, axis=0)
        out = numer / denom

        tl.store(out_row_ptr + col_offsets, out, mask=mask)
        return

    # ---- Tiled path: handle rows larger than BLOCK_SIZE ----
    # ==== Reduction Pass ====
    # Pass 1: compute row max
    row_max = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(in_row_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    # ==== Reduction Pass ====
    # Pass 2: compute exp-sum
    row_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(in_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(tl.exp(vals - row_max), axis=0)

    # ==== Pointwise pass ====
    # Pass 3: normalize + write
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(in_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = tl.exp(vals - row_max) / row_sum
        tl.store(out_row_ptr + cols, out, mask=mask)

# --- TRITON KERNEL: TILED ONLINE SOFTMAX ---
@triton.jit
def online_softmax_kernel(
    output_ptr, input_ptr, 
    n_cols, 
    TILE_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * n_cols
    
    # --- PASS 1: Compute Global Statistics (Online) ---
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

    # --- PASS 2: Normalize and Write back ---
    out_row_ptr = output_ptr + row_idx * n_cols
    for start_col in range(0, n_cols, TILE_SIZE):
        col_offsets = start_col + tl.arange(0, TILE_SIZE)
        mask = col_offsets < n_cols
        tile = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        # Use global stats to normalize
        output = tl.exp(tile - m_row) / d_row
        tl.store(out_row_ptr + col_offsets, output, mask=mask)

def triton_softmax(x):
    rows, cols = x.shape
    # Block size must be >= cols and a power of 2
    BLOCK_SIZE = triton.next_power_of_2(cols)
    out = torch.empty_like(x)
    
    # Launch kernel
    grid = (rows,)
    softmax_kernel_base[grid](
        out, x, 
        x.stride(0), out.stride(0), 
        cols, 
        BLOCK_SIZE
    )
    return out

if __name__ == "__main__":
    M = 4096
    N = 1024
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    # 1. Accuracy Check
    y_triton = triton_softmax(x)
    y_torch = torch.softmax(x, dim=-1)
    
    assert torch.allclose(y_triton, y_torch, atol=1e-5), "Triton and PyTorch results do not match!"
    print("✅ Accuracy Check Passed!")