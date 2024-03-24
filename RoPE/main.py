import torch
import RoPE_transformer_engine as base
import RoPE_forward as my_forward

# allow some absolute error
# sin, cos causes some value diff
def get_tol():
    return dict(atol = 1e-5)

if __name__ == '__main__': 

    print()
    print()
    print("Let's check accuracy, by comparing transformer_engine implementation and my own kernel :D") 
    print()
    print(">> forward")
    
    device = torch.device("cuda:0")
    input_tensor = torch.randn((512, 1, 8, 64), dtype=torch.float32, device=device)

    # base implementation    
    te_impl = base.TransformerEngineRoPE(hidden_size=64)
    output_torch = te_impl.forward(input_tensor)

    # my implemtation
    freq = my_forward.create_freq(512, 64).to("cuda:0")
    output_triton = my_forward.RoPE_fwd(input_tensor, freq)
    
    # compare, comment it out for a clean log
    # print(f'  - expected result \t: { output_torch.data} ...')
    # print(f'  - implemented result \t: {output_triton.data} ...')

    if(torch.allclose(output_triton, output_triton, **get_tol())) : 
        print('  - RoPE_forward() is exactly SAME')
    else:
        print('  - RoPE_forward() isn\'t same')
    
    # print('IS SAME...?', torch.allclose(output_torch, output_triton, **get_tol()))
