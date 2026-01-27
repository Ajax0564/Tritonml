import torch
import triton
import triton.language as tl

@triton.jit
def gelu_forward_kernel_2d(
    input_ptr, output_ptr,
    M, N, 
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 1D Grid -> 2D Mapping
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 2D Offsets and Mask
    offs = rm[:, None] * stride_am + rn[None, :] * stride_an
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    
    x = tl.load(input_ptr + offs, mask=mask)
    
    # GELU Math
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * x))
    output = x * cdf
    
    tl.store(output_ptr + offs, output, mask=mask)

@triton.jit
def gelu_backward_kernel_2d(
    grad_out_ptr, inp_ptr, grad_in_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 1D Grid -> 2D Mapping
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    offs = rm[:, None] * stride_am + rn[None, :] * stride_an
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    x = tl.load(inp_ptr + offs, mask=mask)
    grad_out = tl.load(grad_out_ptr + offs, mask=mask)

    # Derivative calculation:
    # d/dx [GELU(x)] = 0.5 * (1 + erf(x/sqrt(2))) + (x / sqrt(2*pi)) * exp(-x^2 / 2)
    inv_sqrt2 = 0.707106781
    inv_sqrt2pi = 0.39894228
    
    cdf = 0.5 * (1 + tl.math.erf(x * inv_sqrt2))
    pdf = inv_sqrt2pi * tl.exp(-0.5 * x * x)
    
    grad_in = grad_out * (cdf + x * pdf)
    tl.store(grad_in_ptr + offs, grad_in, mask=mask)


class TritonGELU2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        M, N = x.shape
        output = torch.empty_like(x)
        BLOCK_M, BLOCK_N = 32, 32
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        gelu_forward_kernel_2d[grid](
            input_ptr=x, output_ptr=output,
            M=M, N=N,
            stride_am=x.stride(0), stride_an=x.stride(1),
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
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        gelu_backward_kernel_2d[grid](
            grad_out_ptr=grad_output, inp_ptr=x, grad_in_ptr=grad_input,
            M=M, N=N,
            stride_am=x.stride(0), stride_an=x.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        return grad_input, None

def run_verification_1():
    print("Verifying correctness...")
    x = torch.randn(1024, 1024, device='cuda', requires_grad=True)
    dy = torch.randn(1024, 1024, device='cuda')

    # Triton path
    y_tri = TritonGELU2D.apply(x)
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

@triton.jit
def gelu_forward_kernel_2d(
    input_ptr, output_ptr,
    B,M, N, 
    stride_ib,stride_im, stride_in,
    stride_ob,stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 1D Grid -> 2D Mapping
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_per_batch = num_pid_m * num_pid_n

    pid_b = pid // num_pid_per_batch
    pid_in = pid % num_pid_per_batch
    pid_m = pid_in // num_pid_n
    pid_n = pid_in % num_pid_n
    if pid_b >= B:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 2D Offsets and Mask
    offs_i = rm[:, None] * stride_im + rn[None, :] * stride_in
    offs_o = rm[:, None] * stride_om + rn[None, :] * stride_on
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    curr_input_ptr = input_ptr+pid_b*stride_ib
    curr_output_ptr = output_ptr+pid_b*stride_ob
    x = tl.load(curr_input_ptr + offs_i, mask=mask,other=0.0)
    # GELU Math
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * x))
    output = x * cdf
    tl.store(curr_output_ptr + offs_o, output, mask=mask)

@triton.jit
def gelu_backward_kernel_2d(
    grad_ptr, input_ptr, output_ptr,
    B,M, N,
    stride_gb,stride_gm, stride_gn, 
    stride_ib,stride_im, stride_in,
    stride_ob,stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_per_batch = num_pid_m * num_pid_n

    pid_b = pid // num_pid_per_batch
    pid_in = pid % num_pid_per_batch
    pid_m = pid_in // num_pid_n
    pid_n = pid_in % num_pid_n
    if pid_b >= B:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    offs_g = rm[:, None] * stride_gm + rn[None, :] * stride_gn
    offs_i = rm[:, None] * stride_im + rn[None, :] * stride_in
    offs_o = rm[:, None] * stride_om + rn[None, :] * stride_on
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    curr_grad_ptr = grad_ptr+pid_b*stride_gb
    curr_input_ptr = input_ptr+pid_b*stride_ib
    curr_output_ptr = output_ptr+pid_b*stride_ob
    
    x = tl.load(curr_input_ptr + offs_i, mask=mask,other=0.0)
    grad_out = tl.load(curr_grad_ptr + offs_g, mask=mask,other=0.0)

    # Derivative calculation:
    # d/dx [GELU(x)] = 0.5 * (1 + erf(x/sqrt(2))) + (x / sqrt(2*pi)) * exp(-x^2 / 2)
    inv_sqrt2 = 0.707106781
    inv_sqrt2pi = 0.39894228
    
    cdf = 0.5 * (1 + tl.math.erf(x * inv_sqrt2))
    pdf = inv_sqrt2pi * tl.exp(-0.5 * x * x)
    
    grad_in = grad_out * (cdf + x * pdf)
    tl.store(curr_output_ptr + offs_o, grad_in, mask=mask)


class TritonGELU2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        B,M, N = x.shape
        output = torch.empty_like(x)
        BLOCK_M, BLOCK_N = 32, 32
        
        grid = lambda META: (B * triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gelu_forward_kernel_2d[grid](
            input_ptr=x, output_ptr=output,
            B=B,M=M, N=N,
            stride_ib=x.stride(0),
            stride_im=x.stride(1),
            stride_in=x.stride(2),
            stride_ob=output.stride(0),
            stride_om=output.stride(1),
            stride_on=output.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        B,M, N = x.shape
        grad_input = torch.empty_like(x)
        BLOCK_M, BLOCK_N = 32, 32
        
        grid = lambda META: (B * triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        
        gelu_backward_kernel_2d[grid](
        grad_ptr=grad_output,
        input_ptr=x,
        output_ptr=grad_input,
        B=B, M=M, N=N,
        stride_gb=grad_output.stride(0),
        stride_gm=grad_output.stride(1),
        stride_gn=grad_output.stride(2),
        stride_ib=x.stride(0),
        stride_im=x.stride(1),
        stride_in=x.stride(2),
        stride_ob=grad_input.stride(0),
        stride_om=grad_input.stride(1),
        stride_on=grad_input.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
        return grad_input

def run_verification():
    print("Verifying correctness...")
    x = torch.randn(16,1024, 1024, device='cuda', requires_grad=True)
    dy = torch.randn(16,1024, 1024, device='cuda')

    # Triton path
    y_tri = TritonGELU2D.apply(x)
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
