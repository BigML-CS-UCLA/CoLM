#!/bin/bash

# Function to run a task on a specific GPU
run_task_on_gpu() {
    local task=$1
    local gpu_id=$2
    local model=$3
    local output_dir=$4

    model_name=$(basename $model)
    
    # Assign the GPU to this task
    echo "Starting task $task on GPU $gpu_id"

    start_time=$(date +%s)
    task_output_dir=${output_dir}/${task}
    echo "Writing outputs to $task_output_dir"
    mkdir -p $task_output_dir
    
    eval_command="python eval_superglue.py \
        --model $model \
        --task $task \
        --max_length 2048 \
        --cache_dir /data/hf_models \
        --output $task_output_dir"

    echo $eval_command

    # Run task
    CUDA_VISIBLE_DEVICES=${devices[$i]} $eval_command 2>&1 | tee ${task_output_dir}/${task}_eval.log

    # Extract accuracy from the log file
    if [[ $task == "SQuAD" || $task == "DROP" ]]; then
        accuracy=$(grep "f1:" "${task_output_dir}/${task}_eval.log" | awk '{print $NF}')
    else
        accuracy=$(grep "accuracy:" "${task_output_dir}/${task}_eval.log" | awk '{print $NF}')
    fi
    
    if [ -n "$accuracy" ]; then
        echo "${task} : ${accuracy}" >> ${task_output_dir}/summary.log
    else
        echo "${task} : No accuracy found" >> ${task_output_dir}/summary.log
    fi

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Task: $task, Elapsed time: ${elapsed} seconds" >> ${output_dir}/timing.log

    # Compile results for this model
    cat ${task_output_dir}/summary.log >> ${output_dir}/final_summary.log
    echo "" >> ${output_dir}/final_summary.log
    echo "Timing information:" >> ${output_dir}/final_summary.log
    cat ${task_output_dir}/timing.log >> ${output_dir}/final_summary.log
}

model=microsoft/phi-2
model_name=$(basename $model)
output_dir=../out/${model_name}/outputs
mkdir -p $output_dir
echo "Results for $model_name:" > ${output_dir}/final_summary.log

devices=(0 1 2 3)
task_list=("SST2" "CB" "MultiRC")
num_gpus=${#devices[@]}
num_tasks=${#task_list[@]}

# Initialize the task counter
task_counter=0

for task_id in $(seq 0 $((num_tasks-1))); do
    # Get the GPU index in a round-robin fashion
    gpu_id=$((task_counter % num_gpus))
    
    # Run the task in the background on the selected GPU
    run_task_on_gpu ${task_list[$task_id]} ${devices[$gpu_id]} $model $output_dir &

    ((task_counter++))

    # Check if we've launched a batch of 4 tasks
    if (( task_counter % num_gpus == 0 )); then
        echo "Waiting for batch of tasks to finish..."
        wait  # Wait for all background tasks to complete before launching the next batch
    fi
done