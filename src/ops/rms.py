import torch
import triton
import triton.language as tl



@triton.jit
def rms_norm_base(
    input_ptr, 
    input_row_stride,
    output_ptr,
    output_row_stride,
    W_ptr,
    W_row_stride,
    n_cols, 
    eps,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Calculate pointers for this row
    output_ptr += row_idx * output_row_stride
    input_ptr += row_idx * input_row_stride
    

    X_row = tl.load(input_ptr + col_offsets, mask=mask, other=0)
    
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    # tl.store(RSTD_ptr, rstd)

    X_row = X_row * rstd
    if elementwise_affine:
        Y_row = X_row * (offset + W_row)
    else:
        Y_row = X_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)



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


