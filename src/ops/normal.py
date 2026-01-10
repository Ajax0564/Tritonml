import triton
import triton.language as tl
import torch 

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
def sub(x_ptr,y_ptr,out_ptr,n_elements,BlockSize: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid*BlockSize
    offsets = block_start+tl.arange(0,BlockSize)
    mask = offsets<n_elements
    x = tl.load(x_ptr+offsets, mask=mask)
    y = tl.load(y_ptr+offsets,mask=mask)
    output = x-y
    tl.store(out_ptr+offsets,output,mask=mask)


@triton.jit
def rgb2grey_k(x_ptr, out_ptr, h, w, bs0: tl.constexpr, bs1: tl.constexpr):
    """
    GPU kernel for converting RGB image to grayscale
    
    Args:
        x_ptr: Pointer to input RGB image data
        out_ptr: Pointer to output grayscale image data
        h: Image height
        w: Image width
        bs0: Block size for height dimension
        bs1: Block size for width dimension
    """
    # Get program IDs for parallel processing
    pid_0 = tl.program_id(0)  # Block ID in height dimension
    pid_1 = tl.program_id(1)  # Block ID in width dimension
    
    # Calculate offsets for this block
    offs_0 = pid_0 * bs0 + tl.arange(0, bs0)  # Offsets in height dimension
    offs_1 = pid_1 * bs1 + tl.arange(0, bs1)  # Offsets in width dimension
    
    # Calculate 2D offset matrix
    offs = w * offs_0[:,None] + offs_1[None, :]
    
    # Create masks to handle image boundaries
    mask_0 = offs_0 < h
    mask_1 = offs_1 < w
    mask = mask_0[:,None] & mask_1[None,:]
    
    # Load RGB channels
    r = tl.load(x_ptr + 0*h*w + offs, mask=mask)
    g = tl.load(x_ptr + 1*h*w + offs, mask=mask)
    b = tl.load(x_ptr + 2*h*w + offs, mask=mask)
    
    # Convert to grayscale using standard weights
    # These weights represent human perception of color:
    # Red: 29.89%, Green: 58.70%, Blue: 11.40%
    out = 0.2989*r + 0.5870*g + 0.1140*b
    
    # Store the result
    tl.store(out_ptr + offs, out, mask=mask)


def cdiv(n, d):
    """
    Compute ceiling division between two numbers.
    Args:
        n: Numerator
        d: Denominator
    Returns:
        Ceiling division result
    """
    return (n + d - 1) // d

def rgb2grey(x, bs):
    """
    Convert RGB image to grayscale using GPU acceleration

    Args:
        x: Input RGB image tensor (channels, height, width)
        bs: Tuple of block sizes (height, width) for GPU processing

    Returns:
        Grayscale image tensor (height, width)
    """
    c, h, w = x.shape
    # Create output tensor
    out = torch.empty((h,w), dtype=x.dtype, device=x.device)

    # Define processing grid based on block sizes
    grid = lambda meta: (cdiv(h, meta['bs0']), cdiv(w, meta['bs1']))

    # Launch GPU kernel
    rgb2grey_k[grid](x, out, h, w, bs0=bs[0], bs1=bs[1])
    return out.view(h,w)