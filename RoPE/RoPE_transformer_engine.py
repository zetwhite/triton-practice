import torch
import transformer_engine.pytorch.attention as at

'''
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd"
    ) -> torch.Tensor:
    """
        Parameters
        ----------
        t: torch.Tensor
            input tensor on which rotary positional embedding will be applied
        freqs: torch.Tensor
            rotary positional embeding tensor `freqs` is of shape
            `[seq_length, ..., dim]`
        tensor_format: {'sbhd', 'bshd'}, default = 'sbhd'
            is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
            of shape `[seq, bs, ...]`.

    """
    assert tensor_format in ("sbhd", "bshd"),("Only formats `sbhd` or `bshd` "
                                              "are supported for input tensor "
                                              "`t`.")
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    if cur_seq_len > max_seq_len:
        raise Exception(f"Rotary Embeddings only supported upto {max_seq_len} "
                        "sequence length!")

    freqs = freqs[:cur_seq_len].to(t.dtype)
    if tensor_format == "bshd":
        freqs = freqs.transpose(0,1) # [seq, 1, 1, dim] -> [1, seq, 1, dim]

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)
''' 

class TransformerEngineRoPE: 
    '''
        Helper class to invoke Transformer engine's RoPE implementation
    '''
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

