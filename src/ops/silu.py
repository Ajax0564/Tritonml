import torch
import triton
import triton.language as tl

@triton.jit
def _silu_forward_kernel(
    input_ptr, output_ptr,
    M, N, 
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Map IDs to 2D coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
   
    # Compute range of indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 2D Offsets and Mask
    offs_i = rm[:, None] * stride_im + rn[None, :] * stride_in
    offs_o = rm[:, None] * stride_om + rn[None, :] * stride_on
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Load, Compute, Store
    x = tl.load(input_ptr + offs_i, mask=mask, other=0.0)
    output = x * tl.sigmoid(x)
    
    tl.store(output_ptr + offs_o, output, mask=mask)

@triton.jit
def _silu_backward_kernel(
    grad_ptr, input_ptr, output_ptr,
    M, N,
    stride_gm, stride_gn, 
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
   

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_i = rm[:, None] * stride_im + rn[None, :] * stride_in
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    
    x = tl.load(input_ptr + offs_i, mask=mask, other=0.0)
    grad_out = tl.load(grad_ptr + (rm[:, None] * stride_gm + rn[None, :] * stride_gn), mask=mask, other=0.0)
    x_sigmoid = tl.sigmoid(x)
    dx_sigmoid = x_sigmoid*(1-x_sigmoid)
    
    grad_in = grad_out * (x_sigmoid+x*dx_sigmoid)
    tl.store(output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on), grad_in, mask=mask)


class TritonSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        M, N = x.shape
        output = torch.empty_like(x)
        BLOCK_M, BLOCK_N = 32, 32
        
        # Grid  (Num_Blocks_M, Num_Blocks_N)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        _silu_forward_kernel[grid](
            input_ptr=x, output_ptr=output,
            M=M, N=N,
            stride_im=x.stride(0), stride_in=x.stride(1),
            stride_om=output.stride(0), stride_on=output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        M, N = x.shape
        grad_input = torch.empty_like(x)
        BLOCK_M, BLOCK_N = 32, 32
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        _silu_backward_kernel[grid](
            grad_ptr=grad_output,
            input_ptr=x,
            output_ptr=grad_input,
            M=M, N=N,
            stride_gm=grad_output.stride(0), stride_gn=grad_output.stride(1),
            stride_im=x.stride(0), stride_in=x.stride(1),
            stride_om=grad_input.stride(0), stride_on=grad_input.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        return grad_input
    
class TritonSiluLayer(torch.nn.Module):
    """
    Triton SiLu activation.
    Supports 2D and 3D inputs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        orig_shape = x.shape
        if x.ndim > 2:
            x = x.view(-1, orig_shape[-1])
            
        y = TritonSiLU.apply(x)
        
        # Reshape back
        if len(orig_shape) > 2:
            y = y.view(*orig_shape[:-1], -1)
        return y