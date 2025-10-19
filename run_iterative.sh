#!/bin/bash

# Example script to run iterative training with wandb logging
# Make sure to install wandb: pip install wandb
# Login to wandb: wandb login
CUDA_VISIBLE_DEVICES=1 python train_iterative.py \
    --model_class resnet50.a1_in1k \
    --pretrained_model vit_giant_patch14_dinov2.lvd142m \
    --dataset flowers \
    --num_iterations 10 \
    --steps 500 \
    --lr 1e-3 \
    --batch_size 128 \
    --optimizer adam \
    --edm_ckpt /NFS/workspaces/tg.ahn/Collab/edm/training-runs-flowers102/00001-flowers102-64x64-cond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-008132.pkl \
    --num_target_images 300 \
    --base_output_dir ./logs/iterative_training_10_500 \
    --use_wandb \
    --wandb_project "aft-flowers-iterative" \
    --wandb_name "resnet50_flowers_10_500" \
    --wandb_save_checkpoints \
    --use_all_synthetic \
    --seed 0 \
    --original_feature_path /NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/features/vit_giant_patch14_dinov2.lvd142m_flowers.pt