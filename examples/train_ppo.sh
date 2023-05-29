#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_ppo.py \
    --do_train \
    --dataset alpaca_data_zh_51k.json \
    --dataset_dir ../data \
    --finetuning_type lora \
    --checkpoint_dir ../results/faq-dataset-generated-1w-v5-r4lr5em4 \
    --reward_model ../results/reward-normal-zh \
    --output_dir path_to_ppo_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16 \
    --quantization_bit 8
