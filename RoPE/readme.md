

# Requirement
* docker engine
* nvidia gpu drivers
  * driver version 535.161.07 tested on my local env

# Env setting

* Build a docker image
  ```bash
  git clone https://github.com/zetwhite/triton-practice.git 
  cd triton-practice
  
  docker build -t nvidia_env:1.0 -f nvidia_env.dockerfile .
  ```

* Run a container
  ```bash
  cd triton-practice
  
  docker run \
    -v ./RoPE:/home/RoPE \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --rm -it nvidia_env:1.0
  ```

# Run
* After attaching the container, you could run the code in this directory.
   ```bash
   # inside the container, don't forget to move this directroy 
   cd /home/RoPE
   ```

* `value_check.py` 는 triton으로 직접 구현한 RoPE forward, backword 함수와 [Transformer Engine에 있는 함수](https://github.com/NVIDIA/TransformerEngine/blob/5b90b7f5ed67b373bc5f843d1ac3b7a8999df08e/transformer_engine/pytorch/attention.py#L1037-L1078)의 텐서값을 비교합니다. 
    ```bash
    python3 value_check.py

    # wait for a moment
    # expected log
    # Let's check accuracy, by comparing transformer_engine implementation and my own kernel :D
    #
    # >> forward
    #  - RoPE_forward() output is exactly SAME with TransformerEngine's
    #
    # >> backward
    #  - RoPE_backward() output is exactly SAME with TransformerEngine's
    # 
    ``` 

* `perf.py`는 직접 구현한 RoPE forward함수와 Transformer Engine의 함수의 프로파일링 결과를 비교합니다.
    * 다만, 여기서 비교되는 Transformer Engine의 RoPE함수(`apply_rotary_pos_embed`)는 [fused](https://github.com/NVIDIA/TransformerEngine/commit/6c1a8bb5dffbce386380f8e5a12c45f7032d9b76#diff-8215778f23231390f7e41e1339eed64843646d7aba265b8dbf3d68a76c1a647f)가 적용되기전의 함수 입니다.
    * <sub> 아쉽게도 로컬에서 메모리가 최신버전의 Transformer Engine을 빌드하지 못 했습니다. 😢 </sub>  
    ``` bash 
    python perf.py
    
    # this takes some times... 
    # you could check a simple result image(*.png) in perf
    ```

* seq_len에 따른 프로파일링 결과는 아래와 같습니다. 
  * <img src="https://github.com/zetwhite/triton-practice/assets/61981457/bb6b90b2-eb28-4328-a73e-72a32640d6b0" width="30%">


# Note 
forward와 backward 커널에 대한 pseudo code는 아래와 같습니다. 

#### formula  

<img src="https://github.com/zetwhite/triton-practice/assets/61981457/9ca4e6ae-7667-4b4c-9ccd-23b741860278" width="40%"> 
<sub> from ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING, https://arxiv.org/pdf/2104.09864.pdf </sub>

#### forward 
```python 
# 커널이 실행되기전에 다음과 같은 tensor들이 준비되어 있다고 가정 
# in_tesor   (shape = (seq_len, batch, n_head, dim_head))  
# freq       (shape = (seq, 1, 1, dim_head // 2)),        ; 미리 계산해둔 (m x theta) array 
# out_tensor (shape = (seq_len, batch, n_head, dim_head)) ; output tensor가 저장될 위치 

# 최대 a*b*c개 병렬 수행
a, b, c = select based on tl.program_id()

# load 
in_first_half = LOAD(in_tensor[a, b, c, :dim//2])
in_second_half = LOAD(in_tensor[a, b, c, dim//2:])
freq = LOAD(freq[a, 1, 1, :])

# compute
first_out = in_first_half * cos(freq) - in_second_half * sin(freq)
second_out = in_second_half * sin(freq) + in_second_half * cos(freq)

#store
STORE(first_out, out_tensor[a, b, c, :dim//2])
STORE(second_out, out_tensor[a, b, c, dim//2:])
```


#### backward
```python 
# 커널이 실행되기전에 다음과 같은 tensor들이 준비되어 있다고 가정
# in_grad_tesor   (shape = (seq_len, batch, n_head, dim_head))  
# freq            (shape = (seq, 1, 1, dim_head // 2)),        ; 미리 계산해둔 (m x theta) array 
# out_grad_tensor (shape = (seq_len, batch, n_head, dim_head)) ; output tensor가 저장될 위치 

# 최대 a*b*c개 병렬 수행 
a, b, c = select based on tl.program_id() 

# load 
in_first_half = LOAD(in_grad_tensor[a, b, c, :dim//2])
in_second_half = LOAD(in_grad_tensor[a, b, c, dim//2:])
freq = LOAD(freq[a, 1, 1, :])

# compute
out_grad_first = in_first_half * cos(freq) + in_second_half * sin(freq)
out_grad_second  = -in_first_half * sin(freq) + in_second_half * cos(freq)

#store
STORE(out_grad_first, out_grad_tensor[a, b, c, :dim//2])
STORE(out_grad_second, out_grad_tensor[a, b, c, dim//2:])
``` 
