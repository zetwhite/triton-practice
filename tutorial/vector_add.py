import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, # pointer to first input vector
    y_ptr, # pointer to second input vector
    output_ptr, # pointer to output vector
    n_elements, # size of the vector
    BLOCK_SIZE : tl.constexpr # number of elements each program should process    
    ):
    
    pid = tl.program_id(axis = 0) # we use 1d grid
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # offset is list of pointers

    # create mask guard to avoid out of bound accesss
    mask = offsets < n_elements

    # load x and y from DRAM
    # mask out any extra elements in case the input is not a multiple of block
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y

    # write x+y back to DRAM
    tl.store(output_ptr + offsets, output, mask = mask)

def add(x :torch.Tensor, y: torch.Tensor):
    # preallocate output
    output = torch.empty_like(x)
    # assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    # 'SMPD launch' grid denotes the number of kernel instances that run in parallel
    # It is analogous to CUDA launch grids
    # we use 1D grid where the size is the number of blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size)
y = torch.rand(size)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')