#!/bin/bash

python finetune_sdxl_flowers.py \
    --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir "./generator/sdxl-flowers102" \
    --resolution 1024 \
    --batch_size 1 \
    --num_epochs 20 \
    --learning_rate 1e-5 \
    --weight_decay 1e-3 \
    --max_grad_norm 0.5 \
    --save_steps 500 \
    --run_validation \
    --use_wandb
