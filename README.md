## TRT Benchmark

### Download the necessary huggingface models

Create a models directory in your HOME directory.

```bash
mkdir models && cd models
python3 -m venv venv
source venv/bin/activate
pip3 install huggingface_hub[cli]

#Replace the model location with the desired model path
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ./llama-8b --cache-dir ./llama-8b --exclude *.pth
```

### Run the triton server image with podman


Clone necessary repositories 
```bash
git clone https://github.com/wseaton/trt-benchmark.git
cd trt-benchmark
git clone https://github.com/vllm-project/vllm.git #we'll need this later

GPU setup based on 8 GPU machine and targeting 4 GPUs (unused) for deployment. Please tweak depending on your setup.
```

```bash
# must be run from the directory you want these volume mounts mounted to! be warned!
podman run --rm -it -d  --security-opt=label=disable  \
  --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 \
  --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 --ulimit nproc=65535 \
  --shm-size=2g   --ulimit memlock=-1   --ulimit stack=67108864   \
  -e HF_TOKEN=$HF_TOKEN   -v $(pwd)/models:/models:Z   -v $(pwd)/scripts:/home/docker-user/scripts:Z   \
  --name trtllm_bench -v $(pwd)/vllm/benchmarks:/home/docker-user/benchmarks:Z   nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3
```
### Setup Environment

- Enter shell

```bash
podman exec -it trtllm_bench /bin/bash
```

```bash
export HOME=/home/docker-user
cd $HOME
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
```
### Build and Launch TRT Engine
```bash
CONFIG_PATH=scripts/llama-70b.json # utilize the right model script here
bash scripts/build-trt.sh $CONFIG_PATH # rebuilt every time the container is restarted :(
bash scripts/launch-trt.sh $CONFIG_PATH
```

### Install benchmark dependencies

```bash
 curl -LsSf https://astral.sh/uv/install.sh | sh
 source $HOME/.local/bin/env
 uv venv vllm-env
 source vllm-env/bin/activate
 uv pip install vllm pandas datasets
```

### Download dataset 

```bash
cd vllm/benchmarks/
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
cd $HOME
```

### Run benchmark script for TRT-LLM
```bash
# You will need to tweak this for the scenario you want
bash scripts/benchmark_trt.sh $CONFIG_PATH
```

## vLLM benchmark

### Run vLLM image with podman

GPU setup based on 8 GPU machine and targeting 4 GPUs (unused) for deployment. Please tweak depending on your setup.

```sh
  podman run --rm -it --entrypoint /bin/sh  --security-opt=label=disable  -v $(pwd)/models:/models:Z -v $(pwd)/scripts:/scripts:Z  -v $(pwd)/vllm:/vllm:Z --device nvidia.com/gpu=4  --device nvidia.com/gpu=5   --device nvidia.com/gpu=6   --device nvidia.com/gpu=7 -e HF_HUB_OFFLINE=1  --name vllm-serve -p 8000:8000/tcp  quay.io/vllm/vllm:0.8.5.0_cu128 -c 'vllm serve "/models/llama-70b" --gpu-memory-utilization=0.9 --tensor-parallel-size=4'
```

Then exec into it (get the container name from `podman ps`, or optionally set it via `--name` flag like above)

```sh
podman exec -it {container_name} /bin/bash
```

### Setup virtual environment to install necessary dependencies 
In the conatiner shell:

```sh
uv venv benchmark-env
source benchmark-env/bin/activate
uv pip install vllm pandas datasets numpy
```

### Run vLLM benchmark script
Run the  mounted [scripts/benchmark_vllm.sh](scripts/benchmark_vllm.sh) script, which uses vllm that is git cloned and mounted into the container (the pip installed version doesn't ship the benchmark harness), and the environment in the upstream `vllm` image is unfortunately immutable.

```
bash /scripts/benchmark_vllm.sh
```

### Save results onto machine
Copy the output files out of the container shell (we are using `/models` to smuggle them):
```
(benchmark-env) [vllm@de343a58d211 ~]$ ls
benchmark-env  results_vllm_25s.json  results_vllm_30s.json  results_vllm_35s.json
(benchmark-env) [vllm@de343a58d211 ~]$ mv results_vllm*.json /models/
```


