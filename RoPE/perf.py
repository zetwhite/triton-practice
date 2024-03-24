import torch
import triton

import RoPE_transformer_engine as base
import RoPE_forward as my_forward

import os

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[2**i for i in range(5, 20, 1)],
        x_log=True,                                 # logarithmic in x axis
        line_arg = 'provider', 
        line_vals = ['my', 'base'],
        line_names = ['my', 'transformer_engine'],
        styles=[('blue', '-'), ('green','-')],
        ylabel = 'GB/s',
        plot_name = 'RoPE forward() perf',
        args = {'batch':1, 'n_head':8, 'dim_head':128}, 
    )) 
def benchmark(seq_len, batch, n_head, dim_head, provider) :
    
    # prepare input
    size = (seq_len, batch, n_head, dim_head)
    input = torch.rand(size, device='cuda:0', dtype=torch.float32)
   
    # prepare base 
    max_seq = 2 ** 20 
    te_base = base.TransformerEngineRoPE(max_seq = max_seq, hidden_size=dim_head)
    
    # prepare my
    freq =my_forward.create_freq(max_seq, dim_head).to('cuda:0')
    
    quantiles = [0.2, 0.5, 0.8]
    if provider == 'base':
        min_ms, ms, max_ms = triton.testing.do_bench(lambda: te_base.forward(input), quantiles=quantiles)
    if provider == 'my':
        min_ms, ms, max_ms = triton.testing.do_bench(lambda: my_forward.RoPE_fwd(input, freq), quantiles=quantiles)
    
    gbps = lambda ms : torch.numel(input) * input.element_size() / ms * 1e-6
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

save_path = "perf"
os.makedirs(save_path, exist_ok=True)
benchmark.run(print_data=True, show_plots=True, save_path=save_path)