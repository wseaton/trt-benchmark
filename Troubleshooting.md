Resource limits with launch-trt.sh scripts

Failure with uv / too many packages



#### debug with `nvidia-smi`

```
podman run --rm -it --security-opt=label=disable  \
  --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 \
  --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 \
  --shm-size=2g   --ulimit memlock=-1   --ulimit stack=67108864   \
  nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3 nvidia-smi
```