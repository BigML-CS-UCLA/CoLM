#!/bin/bash
# Weighted length sampling

DATA_DIR=./data
ORG_NAME=microsoft # meta-llama, microsoft, stabilityai
MODEL_NAME=phi-2 # phi-2, Phi-3-mini-4k-instruct, Meta-Llama-3.1-8B-Instruct, stablelm-zephyr-3b
MODEL_PATH="${ORG_NAME}/${MODEL_NAME}"
PERCENTAGE=1 # percentage of the full data to train, you can specify the training file you want to use in the script
for DATA_SEED in 0
do
    # Training
    GAS=32 # gradient accumulation steps
    DEVICE_BS=1  # batch size per device
    RANK=128
    ALPHA=512
    SAVE_STRATEGY=steps # epoch, steps
    SAVE_STEPS=256
    MAX_STEPS=1024 # total number of gradient updates
    MAX_LENGTH=512
    DROPOUT=True
    # Selection
    SELECTION_METHOD=submodlib  # submodlib, weightedsubmodlib, none
    BATCH_RATIO=0.5
    DATA_SELECTION=mezo  # rep, mezo, masked_grad, grad, proj_grad, mezo_rep, completion_length, length_loss_weighted
    FACILITY_SELECT=l1  # cosine, euclidean, l1
    SOURCE_WISE=proportional # none, proportional, balanced
    LAST_LAYERS=v_proj
    KEEP_SOURCES="2_9_14_15" # small source indices
    # MeZO
    EFF_MEZO=False
    ZO_DIM=2560
    MEZO_SELECTION=grad  # weight_grad, weight, grad
    MEZO_TOPK=largest # random, largest, smallest, sampling
    MEZO_EPS=1e-3
    MEZO_OPTIM=adam  # adam, sgd
    MEZO_TRANSFORM=none  # none, self_normalize, normalize, clip_full, clip_last
    if [ "$SELECTION_METHOD" = "none" ]; then
        JOB_NAME=${MODEL_NAME}-superglue-lora-gas${GAS}-bs${DEVICE_BS}-${MAX_STEPS}steps-seed${DATA_SEED}
    else
        JOB_NAME=${MODEL_NAME}-superglue-lora-gas${GAS}-bs${DEVICE_BS}-${DATA_SELECTION}-${LAST_LAYERS}-${ZO_DIM}_${MEZO_TOPK}_${MEZO_SELECTION}-${MAX_STEPS}steps-seed${DATA_SEED}
    fi
    
    WANDB_PROJECT=colm_glue_lora

    echo "Writing output to" $JOB_NAME
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./colm/scripts/train/lora_train_superglue.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$GAS" "$RANK" "$ALPHA" "$BATCH_RATIO" "$SELECTION_METHOD" "$ZO_DIM" "$DATA_SELECTION" "$SAVE_STRATEGY" "$SAVE_STEPS" "$MAX_STEPS" "$FACILITY_SELECT" "$MEZO_SELECTION" "$MAX_LENGTH" "$DROPOUT" "$MEZO_TOPK" "$MEZO_EPS" "$MEZO_OPTIM" "$SOURCE_WISE" "$LAST_LAYERS" "$MEZO_TRANSFORM" "$WANDB_PROJECT" "$KEEP_SOURCES" "$DEVICE_BS" "$EFF_MEZO"
done
