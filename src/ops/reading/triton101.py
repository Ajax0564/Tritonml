import triton
import triton.language as tl
import torch 

# from triton examples 
# https://triton-lang.org/main/getting-started/tutorials/index.html
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def row_sum_kernel(
    m_ptr,          
    out_ptr,      
    N,
    stride_m,
    stride_n,
    BLOCK: tl.constexpr,
):
    "calculate row wise sum of a given 2d grid"

    pid = tl.program_id(axis=0)
    row_ptr = m_ptr + pid * stride_m # move to the correct row 

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for k in range(0, tl.cdiv(N, BLOCK)):
        off_k = k * BLOCK + tl.arange(0, BLOCK)
        mask = off_k < N
        acc += tl.load(row_ptr + off_k*stride_n, mask=mask, other=0.0)

    row_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, row_sum)

def calculate_row_sum(M, N):
    mat = torch.ones((M, N), device="cuda", dtype=torch.float32)
    out = torch.empty((M,), device="cuda", dtype=torch.float32)

    grid = (M,)
    row_sum_kernel[grid](
        mat,
        out,
        N,
        mat.stride(0),
        mat.stride(1),
        BLOCK=4
    )

    return out,mat


@triton.jit
def row_mean_kernel(
    m_ptr,          
    out_ptr,      
    N,
    stride_m,
    stride_n,
    BLOCK: tl.constexpr,
):
    "calculate row wise mean of a given 2d grid"

    pid = tl.program_id(axis=0)

    row_ptr = m_ptr + pid * stride_m # move to the correct row 

    acc = tl.zeros((BLOCK,), dtype=tl.float32) #it will store the sum of the blocks

    for k in range(0, tl.cdiv(N, BLOCK)):
        off_k = k * BLOCK + tl.arange(0, BLOCK)
        mask = off_k < N
        acc += tl.load(row_ptr + off_k*stride_n, mask=mask, other=0.0)

    row_mean = tl.sum(acc, axis=0)/N

    
    tl.store(out_ptr + pid, row_mean)

def calculate_row_mean(M, N):
    mat = torch.ones((M, N), device="cuda", dtype=torch.float32)
    out = torch.empty((M,), device="cuda", dtype=torch.float32)

    grid = (M,)
    row_mean_kernel[grid](
        mat,
        out,
        N,
        mat.stride(0),
        mat.stride(1),
        BLOCK=4
    )

    return out,mat

#loads each row 1 by 1 and then devide it into blocks to  calculates the sum and store it into  output row
@triton.jit
def row_elsum(
    M1_ptr, M2_ptr, out_ptr,
    N,
    stride_m1, stride_n1,
    stride_m2, stride_n2,
    stride_out1, stride_out2,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    row1_ptr = M1_ptr + pid * stride_m1
    row2_ptr = M2_ptr + pid * stride_m2
    row_out_ptr = out_ptr + pid * stride_out1

    for k in range(0, tl.cdiv(N, BLOCK)):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        x1 = tl.load(row1_ptr + offs * stride_n1, mask=mask, other=0.0)
        x2 = tl.load(row2_ptr + offs * stride_n2, mask=mask, other=0.0)

        tl.store(row_out_ptr + offs * stride_out2, x1 + x2, mask=mask)

    
def calculate_elsum(M,N):
    mat1 = torch.ones((M,N),device="cuda")
    mat2 = torch.ones((M,N),device="cuda")
    out = torch.empty((M,N),device="cuda")

    grid = (M,)
    row_elsum[grid](mat1,mat2,out,N,mat1.stride(0),mat1.stride(1),mat2.stride(0),mat2.stride(1),out.stride(0),out.stride(1),32)
    return out

#load grid by grid to  calculates the sum and store it into  output grid.
@triton.jit
def row_elsum_grid(M1_ptr,M2_ptr,out_ptr,M,N,stride_m1_m,stride_m1_n,stride_m2_m,stride_m2_n,stride_out_m,stride_out_n,BlockM: tl.constexpr,BlockN: tl.constexpr):
    pid = tl.program_id(axis = 0)
    npid = tl.cdiv(N,BlockN)
    pid_m = pid//npid
    pid_n = pid%npid
    offs_m = pid_m*BlockM+tl.arange(BlockM)
    offs_n = pid_n*BlockN+tl.arange(BlockN)

    d2_grid_offs_m1 = offs_m[:,None]*stride_m1_m+offs_n[None,:]*stride_m1_n
    d2_grid_offs_m2 = offs_m[:,None]*stride_m2_m+offs_n[None,:]*stride_m2_n
    d2_grid_offs_out = offs_m[:,None]*stride_out_m+offs_n[None,:]*stride_out_n

    m1 =  tl.load(M1_ptr+d2_grid_offs_m1,mask = (offs_m[:,None]<M) & (offs_n[None,:]<N),other = 0.0)
    m2 =  tl.load(M2_ptr+d2_grid_offs_m2,mask = (offs_m[:,None]<M) & (offs_n[None,:]<N),other = 0.0)
    
    out = m1+m2
    tl.store(out_ptr+d2_grid_offs_out,out,mask = (offs_m[:,None]<M) & (offs_n[None,:]<N))

    
def calculate_elsum_grid(M,N):
    mat1 = torch.rand((M,N),device="cuda")
    mat2 = torch.rand((M,N),device="cuda")
    out = torch.empty((M,N),device="cuda")
    BlockM,BlockN = 32,32
    grid = (tl.cdiv(BlockM,M)*tl.cdiv(BlockN,N),)
    row_elsum_grid[grid](mat1,mat2,out,M,N,mat1.stride(0),mat1.stride(1),mat2.stride(0),mat2.stride(1),out.stride(0),out.stride(1),BlockM,BlockN)


# simple matrix multiplication MK@KN -> MN
@triton.jit
def matrix_multiply_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0) #global programm id
    num_pid_n = tl.cdiv(N, BLOCK_N)# for 1d grid launch we need to find the column and rows for output matrix to strore the result in its correct grid
    pid_m = pid // num_pid_n # row program id
    pid_n = pid % num_pid_n # column program id

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) #get the indexes of columns of output grid
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) #get the indexes of rows of output grid

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        a_mask = (offs_m[:, None] < M) & (k_remaining[None, :] < K)
        b_mask = (k_remaining[:, None] < K) & (offs_n[None, :] < N)
        
        # Standard indexing: row * row_stride + col * col_stride
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + k_remaining[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + k_remaining[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator, mask=c_mask)


#BMK@BKN -> BMN
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, K, N,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0) #global programm id

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_per_batch = num_pid_m * num_pid_n

    pid_b = pid // num_pid_per_batch # current batch id
    pid_in = pid % num_pid_per_batch

    pid_m = pid_in // num_pid_n # current row id
    pid_n = pid_in % num_pid_n # current column id

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptr += pid_b * stride_ab #select current batch id grid
    b_ptr += pid_b * stride_bb
    c_ptr += pid_b * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        # load grid data for current batch id, row_id, column_id
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask, other=0.0
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask, other=0.0
        )

        acc += tl.dot(a, b)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc, mask=c_mask
    )

def triton_bmm(a, b, BLOCK=128):
    B, M, K = a.shape
    _, _, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    grid = (
        B * triton.cdiv(M, BLOCK) * triton.cdiv(N, BLOCK),
    )

    batched_matmul_kernel[grid](
        a, b, c,
        B, M, K, N,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M=BLOCK, BLOCK_K=BLOCK, BLOCK_N=BLOCK,
    )
    return c

# batch matrix multiplication
#BMK@KN -> BMN
@triton.jit
def bmk_kn_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, K, N,
    stride_ab, stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_per_batch = num_pid_m * num_pid_n

    pid_b = pid // num_pid_per_batch
    pid_in = pid % num_pid_per_batch

    pid_m = pid_in // num_pid_n
    pid_n = pid_in % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # batch offset ONLY for A and C
    a_ptr += pid_b * stride_ab
    c_ptr += pid_b * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask, other=0.0
        )

        # B is shared across all batches
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask, other=0.0
        )

        acc += tl.dot(a, b)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc, mask=c_mask
    )


def triton_bmk_kn(a, b, BLOCK=128):
    """
    a: [B, M, K]
    b: [K, N]  (broadcasted)
    returns: [B, M, N]
    """
    B, M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    grid = (
        B * triton.cdiv(M, BLOCK) * triton.cdiv(N, BLOCK),
    )

    bmk_kn_matmul_kernel[grid](
        a, b, c,
        B, M, K, N,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M=BLOCK, BLOCK_K=BLOCK, BLOCK_N=BLOCK,
    )
    return c

# double matrix multiplication
#MK@KH -> MH
# MH@HN -> MN
@triton.jit
def fused_matmul_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    M, N, K, H,
    stride_am, stride_ak,
    stride_bk, stride_bh,
    stride_ch, stride_cn,
    stride_dm, stride_dn,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, 
    block_size_k: tl.constexpr, block_size_h: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, block_size_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * block_size_m + tl.arange(0, block_size_m))
    offs_dn = (pid_n * block_size_n + tl.arange(0, block_size_n))
    
    # output tile (M x N)
    final_accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)

    for h in range(0, tl.cdiv(H, block_size_h)):
        offs_h = h * block_size_h + tl.arange(0, block_size_h)
        
        #  tile for A @ B (M x H)
        inter_accumulator = tl.zeros((block_size_m, block_size_h), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, block_size_k)):
            offs_k = k * block_size_k + tl.arange(0, block_size_k)
            
            #  A (M x K) and B (K x H)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_h[None, :] * stride_bh)
            
           
            mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K)
            mask_b = (offs_k[:, None] < K) & (offs_h[None, :] < H)
            
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            
            inter_accumulator += tl.dot(a, b)

        # tile of (A @ B) multiply by C (H x N)
        c_ptrs = c_ptr + (offs_h[:, None] * stride_ch + offs_dn[None, :] * stride_cn)
        mask_c = (offs_h[:, None] < H) & (offs_dn[None, :] < N)
        c = tl.load(c_ptrs, mask=mask_c, other=0.0)

        # (A@B_tile) @ C_tile
        final_accumulator += tl.dot(inter_accumulator, c)

    d_ptrs = d_ptr + (offs_am[:, None] * stride_dm + offs_dn[None, :] * stride_dn)
    mask_d = (offs_am[:, None] < M) & (offs_dn[None, :] < N)
    tl.store(d_ptrs, final_accumulator, mask=mask_d)
