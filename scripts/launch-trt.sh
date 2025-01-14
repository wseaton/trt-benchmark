set -x 

config=$1

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

trt_engine_path=${HOME}/models/${model_name}-trt-engine

# handle triton protobuf files and launch triton server
cd ${HOME}/tensorrtllm_backend
mkdir triton_model_repo
cp -r all_models/inflight_batcher_llm/* triton_model_repo/
cd triton_model_repo
rm -rf ./tensorrt_llm/1/*
cp -r ${trt_engine_path}/* ./tensorrt_llm/1
python3 ../tools/fill_template.py -i tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,engine_dir:${HOME}/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1,decoupled_mode:true,batching_strategy:inflight_fused_batching,batch_scheduler_policy:guaranteed_no_evict,exclude_input_in_output:true,triton_max_batch_size:2048,max_queue_delay_microseconds:0,max_beam_width:1,max_queue_size:2048,enable_kv_cache_reuse:false
python3 ../tools/fill_template.py -i preprocessing/config.pbtxt triton_max_batch_size:2048,tokenizer_dir:$model_path,preprocessing_instance_count:5
python3 ../tools/fill_template.py -i postprocessing/config.pbtxt triton_max_batch_size:2048,tokenizer_dir:$model_path,postprocessing_instance_count:5,skip_special_tokens:false
python3 ../tools/fill_template.py -i ensemble/config.pbtxt triton_max_batch_size:$max_batch_size
python3 ../tools/fill_template.py -i tensorrt_llm_bls/config.pbtxt triton_max_batch_size:$max_batch_size,decoupled_mode:true,accumulate_tokens:"False",bls_instance_count:1,enable_chunked_context:true

cd ${HOME}/tensorrtllm_backend
python3 scripts/launch_triton_server.py \
--world_size=${model_tp_size} \
--model_repo=${HOME}/tensorrtllm_backend/triton_model_repo
