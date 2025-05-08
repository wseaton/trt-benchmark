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
git clone https://github.com/vllm-project/vllm.git
# must be run from the directory you want these volume mounts mounted to! be warned!
podman run --rm -it -d  --security-opt=label=disable  \
  --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 \
  --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 --ulimit nproc=65535 \
  --shm-size=2g   --ulimit memlock=-1   --ulimit stack=67108864   \
  -e HF_TOKEN=$HF_TOKEN   -v $(pwd)/models:/models:Z   -v $(pwd)/scripts:/home/docker-user/scripts:Z   \
  --name trtllm_bench -v $(pwd)/vllm/benchmarks:/home/docker-user/benchmarks:Z   nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3

```
### Build and Launch TRT Engine

- Enter shell
```bash
podman exec -it trtllm_bench /bin/bash
```
- Build and launch

1) Ensure HOME is set properly, in our case it's: `export HOME=/home/docker-user`
2) Ensure that `tensorrtllm_backend` is cloned in the `${HOME}` directory, this will not persist across a container restart, so be careful.

```bash
CONFIG_PATH=scripts/llama-70b.json
bash scripts/build-trt.sh $CONFIG_PATH # rebuilt every time the container is restarted :(
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


Deps

```sh
 curl -LsSf https://astral.sh/uv/install.sh | sh
 source $HOME/.local/bin/env
 uv venv vllm-env
 source vllm-env/bin/activate
 uv pip install vllm pandas datasets
```

Download dataset 

```
cd vllm/benchmarks/
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
cd $HOME
```

```bash
# You will need to tweak this for the scenario you want
bash scripts/benchmark.sh $CONFIG_PATH
```







### debug

```
podman run --rm -it --security-opt=label=disable  \
  --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 \
  --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 \
  --shm-size=2g   --ulimit memlock=-1   --ulimit stack=67108864   \
  nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3 nvidia-smi
```