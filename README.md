
# Requirement
* docker engine
* nvidia gpu drivers
  * driver version 535.161.07 tested on my local env

# Run

* first, clone this repo and build a docker file to use the prepared env.
```bash
git clone https://github.com/zetwhite/triton-practice.git 
cd triton-practice
build docker -t nvidia_env:1.0 -f nvidia_env.dockerfile .
```

* run container and please git the code directory as a docker volume.
```bash
cd triton-practice
run docker -v tutorial:/home/tutorial --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it nvidia_env:1.0
```
