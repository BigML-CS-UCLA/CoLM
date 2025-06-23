#!/bin/bash

source colm/scripts/train/base_training_args.sh

data_dir=$1
model_path=$2
percentage=$3
data_seed=$4
job_name=$5
gas=$6
rank=$7
alpha=$8
batch_ratio=${9}
selection_method=${10}
zo_dim=${11}
selection_unit=${12}
save_strategy=${13}
save_steps=${14}
max_steps=${15}
facility_similarity=${16}
mezo_selection=${17}
max_length=${18}
enable_dropout=${19}
mezo_topk=${20}
mezo_eps=${21}
mezo_optim=${22}
source_wise_selection=${23}
last_layers=${24}
mezo_transform=${25}
wandb_project=${26}
keep_sources=${27}
device_bs=${28}
efficient_mezo=${29}

output_dir=./out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

# Set LoRA modules
if [[ $model_path == "microsoft/phi-2" ]]; then
    lora_target_modules="q_proj k_proj v_proj fc1 fc2"
else # Llama-2, zephyr-3b
    lora_target_modules="q_proj k_proj v_proj o_proj"
fi

echo "Set LoRA for layers ${lora_target_modules} of ${model_path}"

# Set data type
if [[ $model_path == *"phi-2" || $model_path == *"Llama-2"* ]]; then
    fp16=True
    bf16=False
    torch_dtype=none
else # zephyr-3b
    fp16=False
    bf16=True
    torch_dtype=bfloat16
fi

echo "Set fp16 = ${fp16} and bf16 = ${bf16} and torch_dtype = ${torch_dtype} for ${model_path}"

task_list=("SST2" "CB" "MultiRC")

for train_task in "${task_list[@]}"
do
    output_dir=./out/${job_name}/${train_task}
    if [[ ! -d $output_dir ]]; then
        mkdir -p $output_dir
    fi

    training_args="$base_training_args \
    --model_name_or_path $model_path \
    --output_dir $output_dir \
    --percentage $percentage \
    --bf16 $bf16 \
    --fp16 $fp16 \
    --torch_dtype $torch_dtype \
    --seed $data_seed \
    --gradient_accumulation_steps $gas \
    --per_device_train_batch_size $device_bs \
    --lora True \
    --lora_r $rank \
    --lora_alpha $alpha \
    --lora_target_modules $lora_target_modules \
    --small_batch_ratio $batch_ratio \
    --data_selection_method $selection_method \
    --efficient_mezo $efficient_mezo \
    --zo_dim $zo_dim \
    --save_strategy $save_strategy \
    --save_steps $save_steps \
    --max_steps $max_steps \
    --mezo_transform $mezo_transform \
    --data_selection_unit $selection_unit \
    --last_layers $last_layers \
    --facility_similarity $facility_similarity \
    --mezo_selection $mezo_selection \
    --max_seq_length $max_length \
    --model_max_length $max_length \
    --enable_dropout $enable_dropout \
    --mezo_topk $mezo_topk \
    --mezo_eps $mezo_eps \
    --mezo_optim $mezo_optim \
    --source_wise_selection $source_wise_selection \
    --keep_sources=$keep_sources \
    --remove_unused_columns False \
    --wandb_project $wandb_project \
    --train_files load-superglue-${train_task[@]} 2>&1 | tee $output_dir/train.log"

    eval "$header" "$training_args"
done