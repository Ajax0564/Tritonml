import torch
import triton
import triton.language as tl

@triton.jit
def gelu_forward_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    
    # Load data
    x = tl.load(input_ptr + offs, mask=mask)
    
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # 0.707106781 is 1/sqrt(2)
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * x))
    output = x * cdf
    
    tl.store(output_ptr + offs, output, mask=mask)

@triton.jit
def gelu_backward_kernel(
    grad_out_ptr,  # Incoming gradient from next layer
    inp_ptr,       # Original input x
    grad_in_ptr,   # Where to store the resulting gradient dx
    n_elements, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load data
    x = tl.load(inp_ptr + offs, mask=mask)
    grad_out = tl.load(grad_out_ptr + offs, mask=mask)

    # Constants
    sqrt2_inv = 0.707106781      # 1/sqrt(2)
    sqrt2pi_inv = 0.39894228     # 1/sqrt(2*pi)

    # Derivative calculation:
    # d/dx [GELU(x)] = 0.5 * (1 + erf(x/sqrt(2))) + (x / sqrt(2*pi)) * exp(-x^2 / 2)
    cdf = 0.5 * (1 + tl.math.erf(x * sqrt2_inv))
    pdf = sqrt2pi_inv * tl.exp(-0.5 * x * x)
    
    # Chain Rule: grad_in = grad_out * derivative
    grad_in = grad_out * (cdf + x * pdf)
    
    tl.store(grad_in_ptr + offs, grad_in, mask=mask)


class TritonGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_elements = x.numel()
        output = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # Using KEYWORD arguments to avoid "multiple values" errors
        gelu_forward_kernel[grid](
            input_ptr=x, 
            output_ptr=output, 
            n_elements=n_elements, 
            BLOCK=BLOCK_SIZE
        )
        
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n_elements = x.numel()
        grad_input = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # Using KEYWORD arguments to match kernel signature exactly
        gelu_backward_kernel[grid](
            grad_out_ptr=grad_output,
            inp_ptr=x,
            grad_in_ptr=grad_input,
            n_elements=n_elements,
            BLOCK=BLOCK_SIZE
        )
        return grad_input

# --- 3. VERIFICATION AND BENCHMARK ---

def run_verification():
    print("--- Running Correctness Check ---")
    device = 'cuda'
    x = torch.randn(2048, device=device, requires_grad=True)
    dy = torch.randn(2048, device=device)

    # Triton
    y_tri = TritonGELU.apply(x)
    y_tri.backward(dy)
    dx_tri = x.grad.clone()

    # PyTorch Reference
    x.grad = None
    y_ref = torch.nn.functional.gelu(x)
    y_ref.backward(dy)
    dx_ref = x.grad

    # Validation
    fwd_match = torch.allclose(y_tri, y_ref, atol=1e-6)
    bwd_match = torch.allclose(dx_tri, dx_ref, atol=1e-6)
    
    print(f"Forward Match: {fwd_match}")
    print(f"Backward Match: {bwd_match}")
    if not (fwd_match and bwd_match):
        print(f"Max Fwd Diff: {(y_tri - y_ref).abs().max()}")
        print(f"Max Bwd Diff: {(dx_tri - dx_ref).abs().max()}")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'], 
        x_vals=[2**i for i in range(12, 24)], 
        line_arg='provider', 
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch Native'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Execution Time (ms)',
        plot_name='gelu-performance-comparison',
        args={}
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device='cuda', requires_grad=True)
    dy = torch.randn(N, device='cuda')
    
    # Define quantiles to get median, min (0.2), and max (0.8)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        def torch_fn():
            y = torch.nn.functional.gelu(x)
            y.backward(dy, retain_graph=True)
        # Add quantiles=quantiles here
        ms, min_ms, max_ms = triton.testing.do_bench(torch_fn, quantiles=quantiles)
    else:
        def triton_fn():
            y = TritonGELU.apply(x)
            y.backward(dy, retain_graph=True)
        # Add quantiles=quantiles here
        ms, min_ms, max_ms = triton.testing.do_bench(triton_fn, quantiles=quantiles)
    
    return ms, max_ms, min_ms

if __name__ == "__main__":
    run_verification()
    benchmark.run(show_plots=True, print_data=True)