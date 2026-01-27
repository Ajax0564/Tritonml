import torch
import triton
import triton.language as tl

@triton.jit
def gelu_forward_kernel(
    input_ptr, output_ptr,
    M, N, 
    stride_ib, stride_im, stride_in,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Map IDs to 3D coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Compute range of indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Identify the base pointer for the current batch
    curr_input_ptr = input_ptr + pid_b * stride_ib
    curr_output_ptr = output_ptr + pid_b * stride_ob

    # 2D Offsets and Mask
    offs_i = rm[:, None] * stride_im + rn[None, :] * stride_in
    offs_o = rm[:, None] * stride_om + rn[None, :] * stride_on
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Load, Compute, Store
    x = tl.load(curr_input_ptr + offs_i, mask=mask, other=0.0)
    
    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * x))
    output = x * cdf
    
    tl.store(curr_output_ptr + offs_o, output, mask=mask)

@triton.jit
def gelu_backward_kernel(
    grad_ptr, input_ptr, output_ptr,
    M, N,
    stride_gb, stride_gm, stride_gn, 
    stride_ib, stride_im, stride_in,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Batch pointers
    curr_grad_ptr = grad_ptr + pid_b * stride_gb
    curr_input_ptr = input_ptr + pid_b * stride_ib
    curr_output_ptr = output_ptr + pid_b * stride_ob

    offs_i = rm[:, None] * stride_im + rn[None, :] * stride_in
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    
    x = tl.load(curr_input_ptr + offs_i, mask=mask, other=0.0)
    grad_out = tl.load(curr_grad_ptr + (rm[:, None] * stride_gm + rn[None, :] * stride_gn), mask=mask, other=0.0)

    # Constants
    inv_sqrt2 = 0.707106781
    inv_sqrt2pi = 0.39894228
    
    cdf = 0.5 * (1 + tl.math.erf(x * inv_sqrt2))
    pdf = inv_sqrt2pi * tl.exp(-0.5 * x * x)
    
    grad_in = grad_out * (cdf + x * pdf)
    tl.store(curr_output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on), grad_in, mask=mask)


class TritonGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        B, M, N = x.shape
        output = torch.empty_like(x)
        BLOCK_M, BLOCK_N = 32, 32
        
        # Grid is now (Num_Blocks_M, Num_Blocks_N, Batch_Size)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B)
        
        gelu_forward_kernel[grid](
            input_ptr=x, output_ptr=output,
            M=M, N=N,
            stride_ib=x.stride(0), stride_im=x.stride(1), stride_in=x.stride(2),
            stride_ob=output.stride(0), stride_om=output.stride(1), stride_on=output.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        B, M, N = x.shape
        grad_input = torch.empty_like(x)
        BLOCK_M, BLOCK_N = 32, 32
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B)
        
        gelu_backward_kernel[grid](
            grad_ptr=grad_output,
            input_ptr=x,
            output_ptr=grad_input,
            M=M, N=N,
            stride_gb=grad_output.stride(0), stride_gm=grad_output.stride(1), stride_gn=grad_output.stride(2),
            stride_ib=x.stride(0), stride_im=x.stride(1), stride_in=x.stride(2),
            stride_ob=grad_input.stride(0), stride_om=grad_input.stride(1), stride_on=grad_input.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        return grad_input

def run_verification():
    print("Verifying Grid Correctness...")
    x = torch.randn(16, 1024, 1024, device='cuda', requires_grad=True)
    dy = torch.randn(16, 1024, 1024, device='cuda')

    # Triton path
    y_tri = TritonGELU.apply(x)
    y_tri.backward(dy)
    dx_tri = x.grad.clone()

    # Torch path
    x.grad = None
    y_ref = torch.nn.functional.gelu(x)
    y_ref.backward(dy)
    dx_ref = x.grad

    assert torch.allclose(y_tri, y_ref, atol=1e-6), "Forward fail"
    assert torch.allclose(dx_tri, dx_ref, atol=1e-6), "Backward fail"
    print("Verification Successful!")

if __name__ == "__main__":
    run_verification()