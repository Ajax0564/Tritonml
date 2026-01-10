import torch
import triton
import triton.language as tl



# Rms = (x/(sum(x)+e/n))*w
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

    # ==== REDUCTION PART ====
    # Compute variance (mean of squared values for RMS)
    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_values = tl.load(
            input_ptr + col_indices, mask=col_indices < feature_dim, other=0.0
        ).to(tl.float32)
        sum_of_squares += input_values * input_values
        #eq to a[i] += b[i] * b[i]   for all i

    variance = tl.sum(sum_of_squares, axis=0) / feature_dim
    reciprocal_std = 1 / tl.sqrt(variance + eps)

    # Store reciprocal standard deviation for backward pass
    tl.store(rstd_ptr + row_idx, reciprocal_std)

    # === POINTWISE OPS ====
    # Normalize input and apply weight transformation
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
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
def softmax_forward_1(
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