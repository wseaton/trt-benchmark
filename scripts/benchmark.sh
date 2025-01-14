MODEL=/models/llama-70b
REQUEST_RATES=(25 30 35)
TOTAL_SECONDS=120

for REQUEST_RATE in "${REQUEST_RATES[@]}";
do
    NUM_PROMPTS=$(($TOTAL_SECONDS * $REQUEST_RATE))
    
    echo ""
    echo "===== RUNNING $MODEL FOR $NUM_PROMPTS PROMPTS WITH $REQUEST_RATE QPS ====="
    echo ""

    python3 vllm/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name sharegpt \
        --dataset-path vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
        --ignore-eos \
        --num-prompts $NUM_PROMPTS \
        --request-rate $REQUEST_RATE \
        --backend tensorrt-llm \
        --endpoint /v2/models/ensemble/generate_stream

    # python3 vllm/benchmarks/benchmark_serving.py \
    #     --model $MODEL \
    #     --dataset-name sonnet \
    #     --dataset-path vllm/benchmarks/sonnet_4x.txt \
    #     --sonnet-input-len 1200 \
    #     --sonnet-output-len 128 \
    #     --ignore-eos \
    #     --num-prompts $NUM_PROMPTS \
    #     --request-rate $REQUEST_RATE \
    #     --backend tensorrt-llm \
    #     --endpoint /v2/models/ensemble/generate_stream

done

