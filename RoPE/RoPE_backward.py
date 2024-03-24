import triton
import torch
import triton.language as tl

device = torch.device("cuda:0")

import RoPE_forward
@triton.jit
def RoPE_bwd_kernel(
    i_grad_ptr,         # in_grad tesnor ptr
    freq_ptr,           # position embedding ptr (m*theta)
    o_grad_ptr,         # position embedding ptr

    i_grad_row_str,     # row_stride
    i_grad_batch_str, 
    i_grad_col_str,
    freq_row_str,
    freq_col_str,
    o_grad_row_str,
    o_grad_batch_str,
    o_grad_col_str,
   
    i_grad_row_num,     # number of rows 
    i_grad_batch_num,   # number of batch
    i_grad_col_num,     # number of columns
    freq_row_num,
    freq_col_num,
    o_grad_row_num,
    o_grad_batch_num,
    o_grad_col_num, 
    BLOCK_SIZE : tl.constexpr, 
):
    # expect that in_grad and out_grad has same shape
    assert i_grad_row_num == o_grad_row_num 
    assert i_grad_col_num == o_grad_col_num
    assert i_grad_batch_num == o_grad_batch_num
    
    pid = tl.program_id(0)
    tile_row_offset = pid // i_grad_batch_num 
    tile_batch_offset = pid % i_grad_batch_num
    tile_col_offset0 = 0
    tile_col_offset1 = i_grad_col_num // 2

    in_first_ptr = tl.make_block_ptr(
        i_grad_ptr,
        shape = (i_grad_row_num, i_grad_batch_num, i_grad_col_num),
        strides = (i_grad_row_str, i_grad_batch_str, i_grad_col_str), 
        offsets = (tile_row_offset, tile_batch_offset, tile_col_offset0),
        block_shape = (1, 1, BLOCK_SIZE // 2),
        order = (2, 1, 0),
    )
 
    in_second_ptr = tl.make_block_ptr(
        i_grad_ptr,
        shape = (i_grad_row_num, i_grad_batch_num, i_grad_col_num),
        strides = (i_grad_row_str, i_grad_batch_str, i_grad_col_str), 
        offsets = (tile_row_offset, tile_batch_offset, tile_col_offset1),
        block_shape = (1, 1, BLOCK_SIZE // 2),
        order = (2, 1, 0),
    )
    
    freq_block_ptr = tl.make_block_ptr(
        freq_ptr,
        shape =  (freq_row_num, freq_col_num),
        strides = (freq_row_str, freq_col_str),
        offsets = (tile_row_offset, tile_col_offset0),
        block_shape= (1, BLOCK_SIZE // 2),
        order = (1, 0) 
    )
    
    in_first_half = tl.load(in_first_ptr, boundary_check=(0, 1))
    in_second_half = tl.load(in_second_ptr, boundary_check=(0, 1))
    freq = tl.load(freq_block_ptr, boundary_check=(0, 1))
    
    out_first_half = in_first_half * tl.cos(freq) + in_second_half * tl.sin(freq)
    out_second_half = -1 * in_first_half * tl.sin(freq) + in_second_half * tl.cos(freq)

    out_first_half = tl.reshape(out_first_half, (1, 1, BLOCK_SIZE // 2))
    out_second_half = tl.reshape(out_second_half, (1, 1, BLOCK_SIZE // 2))

    out_first_ptr = tl.make_block_ptr(
        o_grad_ptr, 
        shape = (o_grad_row_num, o_grad_batch_num,  o_grad_col_num), 
        strides = (o_grad_row_str, o_grad_batch_str, o_grad_col_str),
        offsets = (tile_row_offset, tile_batch_offset, tile_col_offset0), 
        block_shape=(1,1, BLOCK_SIZE // 2),
        order = (2, 1, 0)
    )
    
    out_second_ptr = tl.make_block_ptr(
        o_grad_ptr, 
        shape = (o_grad_row_num, o_grad_batch_num, o_grad_col_num), 
        strides = (o_grad_row_str, o_grad_batch_str, o_grad_col_str),
        offsets = (tile_row_offset, tile_batch_offset, tile_col_offset1), 
        block_shape=(1, 1, BLOCK_SIZE // 2),
        order = (2, 1, 0)
    )
    
    tl.store(out_first_ptr, out_first_half, boundary_check=(0, 1))
    tl.store(out_second_ptr, out_second_half, boundary_check=(0, 1))


def RoPE_bwd(in_grad : torch.tensor, freq : torch.tensor) -> torch.tensor:
    
    old_shape = in_grad.shape
    #prepare in_grad
    n_row = in_grad.shape[0]
    n_col = in_grad.shape[-1]
    in_grad = torch.reshape(in_grad, (n_row, -1, n_col)) # [seq, batch*head_num, head_dim]
    n_batch = in_grad.shape[1]
    
    BLOCK_SIZE = triton.next_power_of_2(n_col)
    out_grad = torch.empty_like(in_grad)

    RoPE_bwd_kernel[(n_row * n_batch, )](
        in_grad,
        freq,
        out_grad,
        in_grad.stride(0), in_grad.stride(1), in_grad.stride(2), 
        freq.stride(0), freq.stride(1),
        out_grad.stride(0), out_grad.stride(1), out_grad.stride(2),  
        in_grad.shape[0], in_grad.shape[1], in_grad.shape[2], 
        freq.shape[0], freq.shape[1],    
        out_grad.shape[0], out_grad.shape[1], out_grad.shape[2],
        BLOCK_SIZE
    )
    return out_grad.reshape(old_shape)   
    