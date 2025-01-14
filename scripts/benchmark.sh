MODEL=/models/llama-8b
REQUEST_RATES=(1 10 20 30 40 45 50 55)
TOTAL_SECONDS=120

for REQUEST_RATE in "${REQUEST_RATES[@]}";
do
    NUM_PROMPTS=$(($TOTAL_SECONDS * $REQUEST_RATE))
    
    echo ""
    echo "===== RUNNING $MODEL FOR $NUM_PROMPTS PROMPTS WITH $REQUEST_RATE QPS ====="
    echo ""

    python3 benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name sharegpt \
        --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
        --ignore-eos \
        --num-prompts $NUM_PROMPTS \
        --request-rate $REQUEST_RATE \
        --backend tensorrt-llm \
        --endpoint /v2/models/ensemble/generate_stream

done

