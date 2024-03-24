import torch
import RoPE_transformer_engine as base
import RoPE_forward as my_forward
import RoPE_backward as my_backward

# allow some absolute error
# sin, cos causes some value diff
def get_tol():
    return dict(atol = 1e-5)

if __name__ == '__main__': 
    print()
    print("Let's check accuracy, by comparing transformer_engine implementation and my own kernel :D") 
    
    print()    
    print(">> forward")
   
    SEQ=512
    BATCH=1
    N_HEAD = 8
    DIM_HEAD = 64
    
    device = torch.device("cuda:0")
    input_tensor = torch.randn(
        (SEQ, BATCH, N_HEAD, DIM_HEAD), 
        dtype=torch.float32, 
        device=device)

    # base implementation    
    te_impl = base.TransformerEngineRoPE(max_seq=SEQ, hidden_size=DIM_HEAD)
    output_torch = te_impl.forward(input_tensor)

    # my implemtation
    freq = my_forward.create_freq(SEQ, DIM_HEAD).to("cuda:0")
    output_triton = my_forward.RoPE_fwd(input_tensor, freq)
    
    # compare, comment it out for a clean log
    # print(f'  - expected result \t: { output_torch.data} ...')
    # print(f'  - implemented result \t: {output_triton.data} ...')

    if(torch.allclose(output_triton, output_triton, **get_tol())) : 
        print('  - RoPE_forward() output is exactly SAME with TransformerEngine\'s')
    else:
        print('  - RoPE_forward() is NOT SAME with TransformerEngine\'s')
    

    print() 
    print(">> backward")

    grad_tensor = torch.randn(
        (SEQ, BATCH, N_HEAD, DIM_HEAD), 
        dtype=torch.float32, 
        device=device)

    # base implementation
    grad_output_torch = te_impl.backward(grad_tensor) 

    # my implementation 
    grad_output_triton = my_backward.RoPE_bwd(grad_tensor, freq)
 
    # compare, comment it out for a clean log
    # print(f'  - expected result \t: { grad_output_torch.data} ...')
    # print(f'  - implemented result \t: {grad_output_triton.data} ...')
    
    if(torch.allclose(grad_output_torch, grad_output_triton, **get_tol())) : 
        print('  - RoPE_backward() output is exactly SAME with TransformerEngine\'s')
    else:
        print('  - RoPE_backward() is NOT SAME with TransformerEngine\'s')

    print()
    print('>> Successfully Done!')