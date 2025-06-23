#!/bin/bash

# Check if base model path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <base_model_path>"
    echo "Example: $0 /path/to/your/model"
    exit 1
fi

BASE_MODEL_PATH=$1

# Define checkpoints and corresponding GPUs
CKPTS=(
    256
    512
    768
    1024
)  # Replace with your actual checkpoint numbers
DEVICES=(
    0
    1
    2
    3
)   # GPU devices to use

# Function to run evaluation for a single checkpoint
run_checkpoint() {
    local ckpt=$1
    local device=$2
    local base_path=$3
    
    model_path="${base_path}/checkpoint-${ckpt}"
    
    if [[ $model_path == *"phi-2"* || $model_path == *"Llama-2"* ]]; then
        dtype=float16
    else  # phi-2, zephyr-3b
        dtype=bfloat16
    fi
    
    # First set of datasets
    for dataset in 'gsm8k' 'math' 'numglue'
    do
        mkdir -p ${model_path}/${dataset}
        eval_command="python run_open.py \
            --model $model_path \
            --shots 0 \
            --stem_flan_type "pot_prompt" \
            --batch_size 8 \
            --dataset $dataset \
            --model_max_length 2048 \
            --cot_backup \
            --use_vllm \
            --dtype $dtype \
            --enable_lora \
            --cache_dir /data/hf_models \
            --print"
        echo "GPU $device - Checkpoint $ckpt - Dataset $dataset: $eval_command"
        CUDA_VISIBLE_DEVICES=$device $eval_command 2>&1 | tee ${model_path}/${dataset}/eval.log
    done
    
    # Second set of datasets
    for dataset in 'svamp' 'deepmind' 'simuleq'
    do
        mkdir -p ${model_path}/${dataset}
        eval_command="python run_open.py \
            --model $model_path \
            --shots 0 \
            --stem_flan_type "pot_prompt" \
            --batch_size 8 \
            --dataset $dataset \
            --model_max_length 2048 \
            --cot_backup \
            --use_vllm \
            --dtype $dtype \
            --enable_lora \
            --cache_dir /data/hf_models \
            --print"
        echo "GPU $device - Checkpoint $ckpt - Dataset $dataset: $eval_command"
        CUDA_VISIBLE_DEVICES=$device $eval_command 2>&1 | tee ${model_path}/${dataset}/eval.log
    done
    
    echo "Completed checkpoint $ckpt on GPU $device"
}

# Export the function so it can be used with parallel execution
export -f run_checkpoint

echo "Starting evaluation for base model path: $BASE_MODEL_PATH"
echo "Checkpoints: ${CKPTS[*]}"
echo "GPUs: ${DEVICES[*]}"

# Run checkpoints in parallel
for i in "${!CKPTS[@]}"; do
    run_checkpoint "${CKPTS[$i]}" "${DEVICES[$i]}" "$BASE_MODEL_PATH" &
done

# Wait for all background processes to complete
wait

echo "All checkpoints completed!"