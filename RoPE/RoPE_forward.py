import triton
import torch
import triton.language as tl
import transformer_engine.pytorch.attention as at

device = torch.device("cuda:0")

def create_freq(seq_len, dim):
    '''
    Create matrix for m * theta
    TODO: optimize it using @triton.jit
    '''
    dim_index = torch.arange(0, dim, 2, device=device).float() / dim
    theta = 1.0 / (10000 ** dim_index)
    seq = torch.arange(seq_len, device=device).float()

    freqs = torch.einsum('i, j -> i j', seq, theta)

    assert freqs.shape[0] == seq_len
    assert freqs.shape[1] == dim //2

    return torch.reshape(freqs, (seq_len, dim //2))
    
# freq = create_freq(seq_len = 512, dim=64)
# print(freq.size()) #(512, 1, 1, 32)
# print(freq)

@triton.jit
def RoPE_fwd_kernel(
    in_ptr,         #input tesnor ptr
    freq_ptr,       # position embedding ptr (m*theta)
    out_ptr,        # position embedding ptr

    in_row_str,     # row_stride
    in_batch_str, 
    in_col_str,
    freq_row_str,
    freq_col_str,
    out_row_str,
    out_batch_str,
    out_col_str,
   
    in_row_num,     # number of rows 
    in_batch_num,   # number of batch
    in_col_num,     # number of columns
    freq_row_num,
    freq_col_num,
    out_row_num,
    out_batch_num,
    out_col_num, 
    BLOCK_SIZE : tl.constexpr, 
):
    
    assert out_row_num == in_row_num
    assert out_col_num == in_col_num
    assert out_batch_num == in_batch_num
    
    pid = tl.program_id(0)
    tile_row_offset = pid // in_batch_num
    tile_batch_offset = pid % in_batch_num
    tile_col_offset0 = 0
    tile_col_offset1 = in_col_num // 2
    
    in_first_ptr = tl.make_block_ptr(
        in_ptr,
        shape = (in_row_num, in_batch_num, in_col_num),
        strides = (in_row_str, in_batch_str, in_col_str), 
        offsets = (tile_row_offset, tile_batch_offset, tile_col_offset0),
        block_shape = (1, 1, BLOCK_SIZE // 2),
        order = (2, 1, 0),
    )
 
    in_second_ptr = tl.make_block_ptr(
        in_ptr,
        shape = (in_row_num, in_batch_num, in_col_num),
        strides = (in_row_str, in_batch_str, in_col_str), 
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
    
    out_first_half = in_first_half * tl.cos(freq) - in_second_half * tl.sin(freq)
    out_second_half = in_second_half * tl.cos(freq) + in_first_half * tl.sin(freq)

    out_first_ptr = tl.make_block_ptr(
        out_ptr, 
        shape = (out_row_num, out_batch_num,  out_col_num), 
        strides = (out_row_str, out_batch_str, out_col_str),
        offsets = (tile_row_offset, tile_batch_offset, tile_col_offset0), 
        block_shape=(1,1, BLOCK_SIZE // 2),
        order = (2, 1, 0)
    )
    
    out_second_ptr = tl.make_block_ptr(
        out_ptr, 
        shape = (out_row_num, out_batch_num, out_col_num), 
        strides = (out_row_str, out_batch_str, out_col_str),
        offsets = (tile_row_offset, tile_batch_offset, tile_col_offset1), 
        block_shape=(1, 1, BLOCK_SIZE // 2),
        order = (2, 1, 0)
    )
    
    tl.store(out_first_ptr, out_first_half, boundary_check=(0,1))
    tl.store(out_second_ptr, out_second_half, boundary_check=(0, 1))
 

def RoPE_fwd(input : torch.tensor, freq : torch.tensor) -> torch.tensor: 
    old_shape = input.shape
    #prepare input
    n_row = input.shape[0]
    n_col = input.shape[-1]
    input = torch.reshape(input, (n_row, -1, n_col)) # [seq, batch*head_num, head_dim]
    n_batch = input.shape[1]
    
    BLOCK_SIZE = triton.next_power_of_2(n_col)
    output = torch.empty_like(input)

    RoPE_fwd_kernel[(n_row * n_batch, )](
        input,
        freq,
        output,
        input.stride(0), input.stride(1), input.stride(2), 
        freq.stride(0), freq.stride(1),
        output.stride(0), output.stride(1), output.stride(2),  
        input.shape[0], input.shape[1], input.shape[2], 
        freq.shape[0], freq.shape[1],    
        output.shape[0], output.shape[1], output.shape[2],
        BLOCK_SIZE
    )
    return output.reshape(old_shape)   
    

class TransformerEngineRoPE: 
    def __init__(self, hidden_size = 64, max_seq = 512, device = torch.device("cuda:0")):
        create_pos_emb = at.RotaryPositionEmbedding(hidden_size)
        self.pos_emb = create_pos_emb(max_seq).to(device)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor.requires_grad = True
        self.output_tensor = at.apply_rotary_pos_emb(input_tensor, self.pos_emb, tensor_format="sbhd")
        return self.output_tensor
    
    def backward(self, grad_tensor : torch.Tensor):
        # print(grad_tensor.shape)
        # print(self.output_tensor.shape)
        assert grad_tensor.shape == self.output_tensor.shape

        gradient_passer = torch.ones_like(self.output_tensor)
        loss = torch.mul(self.output_tensor, gradient_passer) #elementwise multiplication
        loss.backward(grad_tensor)
        return self.input_tensor.grad.detach().clone()


# input
input_tensor = torch.randn((512, 1, 8, 64), dtype=torch.float32, device=device)

# base implementation    
te_impl = TransformerEngineRoPE(hidden_size=64)
output_torch = te_impl.forward(input_tensor)

# my implemtation
freq = create_freq(512, 64).to("cuda:0")
output_triton = RoPE_fwd(input_tensor, freq)

# allow some absolute error
# sin, cos causes some value diff
def get_tol():
    return dict(atol = 1e-5)

print('IS SAME...?', torch.allclose(output_torch, output_triton, **get_tol()))
print('expected : ', output_torch)
print('implemented : ', output_triton)
