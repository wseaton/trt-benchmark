## TRT Benchmark

### Download the necessary huggingface models

Create a models directory in your HOME directory.

```bash
mkdir models && cd models
python3 -m venv venv
source venv/bin/activate
pip3 install huggingface_hub[cli]

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ./llama-8b --cache-dir ./llama-8b --exclude *.pth
```


### Build the Docker image
```bash
docker build -t trtllm_bench -f Dockerfile.trtllm --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
``` 

### Run the docker image

```bash
GPUS='"device=0"'
GPUS='"device=0,1,2,3"'
git clone https://github.com/vllm-project/vllm.git
docker run --shm-size=2g \-it -d --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus $GPUS -e HF_TOKEN=$HF_TOKEN \
    -v $(pwd)/models:/models \
    -v $(pwd)/scripts:/home/docker-user/scripts \
    -v $(pwd)/vllm/benchmarks:/home/docker-user/benchmarks \
    --name trtllm_bench trtllm_bench:latest
```

### Build and Launch TRT Engine

- Enter shell
```bash
docker exec -it trtllm_bench /bin/bash
```
- Build and launch
```bash
CONFIG_PATH=scripts/llama-8b.json

bash scripts/build-trt.sh $CONFIG_PATH
bash scripts/launch-trt.sh $CONFIG_PATH
```

- See https://www.bentoml.com/blog/tuning-tensor-rt-llm-for-optimal-serving-with-bentoml
- See https://github.com/NVIDIA/TensorRT-LLM/blob/v0.14.0/docs/source/performance/perf-best-practices.md

### Benchmark

- Enter shell
```bash
docker exec -it trtllm_bench /bin/bash
```
- Build and launch
```bash
# You will need to tweak this for the scenario you want
bash scripts/benchmark.sh $CONFIG_PATH
```


