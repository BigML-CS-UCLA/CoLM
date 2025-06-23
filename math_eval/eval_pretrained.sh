#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
DEVICE=0
ORG_NAME=microsoft # meta-llama, microsoft, stabilityai
MODEL_NAME=phi-2 # Meta-Llama-3.1-8B-Instruct, phi-2, Phi-3-mini-4k-instruct, stablelm-zephyr-3b
model_path=${ORG_NAME}/${MODEL_NAME}
out_path="../out/${MODEL_NAME}"
mkdir -p $out_path
if [[ $model_path == *"phi-2"* || $model_path == *"Llama-2"* ]]; then
    dtype=float16
else  # phi-2, zephyr-3b
    dtype=bfloat16
fi

for dataset in 'gsm8k' 'math' 'numglue'
do
    mkdir -p "${out_path}/${dataset}"
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
        --cache_dir /data/hf_models \
        --print"
    echo $eval_command
    CUDA_VISIBLE_DEVICES=$DEVICE $eval_command 2>&1 | tee "${out_path}/${dataset}/eval.log"
done
for dataset in 'svamp' 'deepmind' 'simuleq'
do
    mkdir -p "${out_path}/${dataset}"
    eval_command="python run_open.py \
        --model $model_path \
        --shots 0 \
        --stem_flan_type "pot_prompt" \
        --batch_size 8 \
        --dataset $dataset \
        --model_max_length 2048 \
        --cot_backup \
        --print \
        --dtype $dtype \
        --cache_dir /data/hf_models \
        --use_vllm"
    echo $eval_command
    CUDA_VISIBLE_DEVICES=$DEVICE $eval_command 2>&1 | tee "${out_path}/${dataset}/eval.log"
done