#!/bin/bash
FINETUNING_TYPE=lora
LR=1e-4
# p-tuning v2 param
PRE_SEQ_LEN=128
# lora param
RANK=8

CUDA_VISIBLE_DEVICES=0 python ../src/train_sft.py \
    --do_train \
    --dataset api_dataset_augmentation_v2.json \
    --dataset_dir ../data \
    --finetuning_type $FINETUNING_TYPE \
    --lora_rank $RANK \
    --output_dir ../results/api_dataset_augmentation-2w-lora-r8lr1em4 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 500 \
    --learning_rate $LR \
    --num_train_epochs 6.0 \
    --dev_ratio 0.01 \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 \
    --pre_seq_len $PRE_SEQ_LEN
