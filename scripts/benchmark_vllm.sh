#!/bin/bash
MODEL=/models/llama-70b
REQUEST_RATES=(1 10 25 30 35)
TOTAL_SECONDS=120

for REQUEST_RATE in "${REQUEST_RATES[@]}";
do
   NUM_PROMPTS=$((TOTAL_SECONDS * REQUEST_RATE))
   echo ""
   echo "===== VLLM: RUNNING $MODEL FOR $NUM_PROMPTS PROMPTS WITH $REQUEST_RATE QPS ====="
   echo ""
   RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python3 vllm/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name sharegpt \
        --dataset-path vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts $NUM_PROMPTS \
        --ignore-eos \
        --result-filename "results.json" \
        --host 127.0.0.1 \
        --port 8000 \
        --save-result
done