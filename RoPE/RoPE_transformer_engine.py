import torch
import transformer_engine.pytorch.attention as at

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

