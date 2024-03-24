

# Requirement
* docker engine
* nvidia gpu drivers
  * driver version 535.161.07 tested on my local env

# Env setting

* build docker image
  ```bash
  git clone https://github.com/zetwhite/triton-practice.git 
  cd triton-practice
  
  docker build -t nvidia_env:1.0 -f nvidia_env.dockerfile .
  ```

* run container and please give the code directory as a docker volume.
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
* after attaching a container, you could run the code in this directory. 

* `value_check.py` compares the output tensors from under two implements :
   * own implemented RoPE forward and backward function using triton
   * code implmented in Transformer Engine [:link:](https://github.com/NVIDIA/TransformerEngine/blob/5b90b7f5ed67b373bc5f843d1ac3b7a8999df08e/transformer_engine/pytorch/attention.py#L1037-L1078)

    ```bash
    python3 value_check.py
    ``` 

* `perf.py` benchmarks the forward function of RoPE.
    * sadly, i coudn't build the RoPE with cuda 
    ``` bash 
    python perf.py
    
    # you could check a simple result image(*.png) in perf
    ```


# Note 
 * ... 