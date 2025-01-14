set -x
config=$1

build_trt_engine() {

  model_path=$(cat $config | jq -r '.model')
  model_name="${model_path#*/}"
  model_type=$(cat $config | jq -r '.model_type')
  model_dtype=$(cat $config | jq -r '.model_dtype')
  model_tp_size=$(cat $config | jq -r '.tp')
  max_batch_size=$(cat $config | jq -r '.max_batch_size')
  max_num_tokens=$(cat $config | jq -r '.max_num_tokens')

  echo "model_path=$model_path"
  echo "model_name=$model_name"
  echo "model_type=$model_type"
  echo "model_dtype=$model_dtype"
  echo "model_tp_size=$model_tp_size"
  echo "max_batch_size=$max_batch_size"
  echo "max_num_tokens=$max_num_tokens"

  # create model caching directory
  cd ${HOME}
  mkdir -p models
  cd models
  models_dir=$(pwd)
  trt_model_path=${models_dir}/${model_name}-trt-ckpt
  trt_engine_path=${models_dir}/${model_name}-trt-engine

  rm -rf $trt_model_path
  rm -rf $trt_engine_path

  # build trtllm engine
  cd ${HOME}/tensorrtllm_backend
  cd ./tensorrt_llm/examples/${model_type}
  python3 convert_checkpoint.py \
    --model_dir ${model_path} \
    --dtype ${model_dtype} \
    --tp_size ${model_tp_size} \
    --output_dir ${trt_model_path}

  trtllm-build \
    --checkpoint_dir ${trt_model_path} \
    --use_fused_mlp enable \
    --reduce_fusion disable \
    --workers 8 \
    --gpt_attention_plugin ${model_dtype} \
    --gemm_plugin ${model_dtype} \
    --max_batch_size ${max_batch_size} \
    --max_num_tokens ${max_num_tokens} \
    --output_dir ${trt_engine_path} \
    --kv_cache_type paged \
    --use_paged_context_fmha enable \
    --multiple_profiles enable
}

build_trt_engine
