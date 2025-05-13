#!/bin/bash
set -x

config=$1

if [[ -z "$config" ]]; then
  echo "Usage: $0 path_to_config.json"
  exit 1
fi

# Parse model settings from config file
model_path=$(jq -r '.model' "$config")
model_name="${model_path##*/}"
model_type=$(jq -r '.model_type' "$config")
model_dtype=$(jq -r '.model_dtype' "$config")
model_tp_size=$(jq -r '.tp' "$config")
max_batch_size=$(jq -r '.max_batch_size' "$config")
max_num_tokens=$(jq -r '.max_num_tokens' "$config")

echo "model_path=$model_path"
echo "model_name=$model_name"
echo "model_type=$model_type"
echo "model_dtype=$model_dtype"
echo "model_tp_size=$model_tp_size"
echo "max_batch_size=$max_batch_size"
echo "max_num_tokens=$max_num_tokens"

trt_engine_path="${HOME}/models/${model_name}-trt-engine"

# Prepare Triton model repository
cd "${HOME}/tensorrtllm_backend"
rm -rf triton_model_repo
mkdir -p triton_model_repo

# Copy inflight_batcher template
cp -r all_models/inflight_batcher_llm/* triton_model_repo/

# Copy engine files into Triton model path
rm -rf triton_model_repo/tensorrt_llm/1/*
cp -r "${trt_engine_path}"/* triton_model_repo/tensorrt_llm/1/



engine_dir="${HOME}/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1"
logits_datatype="TYPE_FP32"
encoder_input_features_data_type="TYPE_FP32"

python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt "triton_backend:tensorrtllm,engine_dir:${engine_dir},decoupled_mode:true,batching_strategy:inflight_fused_batching,batch_scheduler_policy:guaranteed_no_evict,exclude_input_in_output:true,triton_max_batch_size:${max_batch_size},max_queue_delay_microseconds:0,max_beam_width:1,max_queue_size:2048,enable_kv_cache_reuse:false,encoder_input_features_data_type:${encoder_input_features_data_type},logits_datatype:${logits_datatype}"

python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:${max_batch_size},tokenizer_dir:${model_path},preprocessing_instance_count:4"

python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:${max_batch_size},tokenizer_dir:${model_path},postprocessing_instance_count:4,skip_special_tokens:false"

python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:${max_batch_size},logits_datatype:${logits_datatype}"

python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm_bls/config.pbtxt "triton_max_batch_size:${max_batch_size},decoupled_mode:true,accumulate_tokens:false,bls_instance_count:1,logits_datatype:${logits_datatype},tensorrt_llm_model_name:tensorrt_llm"

# Launch Triton Server
python3 scripts/launch_triton_server.py \
 --world_size=${model_tp_size} \
 --model_repo=${HOME}/tensorrtllm_backend/triton_model_repo