## TRT Benchmark

### Download the necessary huggingface models

Create a models directory in your HOME directory.

```bash
mkdir ~/models
cd ~/models
python3 -m venv venv-downloader
source venv/bin/activate
pip3 install huggingface[cli]

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ./llama-8b --cache-dir ./llama-8b --ignore *.pth
```


### Build the Docker image
```bash
docker build -t trtllm_bench -f Dockerfile.trtllm --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
``` 

### Run the docker image

```bash
GPUS='"device=0"'
GPUS='"device=0,1,2,3"'
docker run --shm-size=2g  -it -d --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus $GPUS -e HF_TOKEN=$HF_TOKEN -v $(pwd)/models:/models -v $(pwd)/.buildkite:/home/docker-user/.buildkite -v $(pwd)/benchmarks:/home/docker-user/benchmarks  --name trtllm_bench trtllm_bench:latest
```

### Enter the docker shell

```bash
docker exec -it trtllm_bench /bin/bash
```

### Build and Launch TRT Engine

```bash
CONFIG_PATH=/home/docker-user/.buildkite/trt-configs/llama-8b.json

bash .buildkite/nightly-benchmarks/scripts/build-trt.sh $CONFIG_PATH
bash .buildkite/nightly-benchmarks/scripts/launch-trt.sh $CONFIG_PATH
```
