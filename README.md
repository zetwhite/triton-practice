
# Requirement
* docker engine
* nvidia gpu drivers
  * driver version 535.161.07 tested on my local env

# Env setting

* first, clone this repo and build a docker file. 
  ```bash
  git clone https://github.com/zetwhite/triton-practice.git 
  cd triton-practice
  build docker -t nvidia_env:1.0 -f nvidia_env.dockerfile .
  ```

* run container and please give the code directory as a docker volume.
  ```bash
  cd triton-practice
  run docker -v ./RoPE:/home/RoPE --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it nvidia_env:1.0
  ```

# RoPE
 * Rotary Position Embedding 코드는 [여기](https://github.com/zetwhite/triton-practice/tree/master/RoPE)에 있습니다.
