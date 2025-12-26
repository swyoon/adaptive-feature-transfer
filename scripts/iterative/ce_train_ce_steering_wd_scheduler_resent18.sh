#!/bin/bash
device=3
num_iterations=30
steps=1500
num_target_images=1000
aft_score=ce
dataset=flowers


CUDA_VISIBLE_DEVICES=${device} python train_iterative.py \
    --model_class resnet18.tv_in1k \
    --pretrained_model vit_giant_patch14_dinov2.lvd142m \
    --dataset ${dataset} \
    --num_iterations ${num_iterations} \
    --steps ${steps} \
    --lr 1e-3 \
    --batch_size 128 \
    --optimizer adam \
    --edm_ckpt /NFS/workspaces/tg.ahn/Collab/edm/training-runs-flowers102/00001-flowers102-64x64-cond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-008132.pkl \
    --num_target_images ${num_target_images} \
    --base_output_dir /NFS/database_personal/tg.ahn/Collab/adaptive-feature-transfer/logs/base/${dataset}_resnet18_${num_iterations}_iter_${steps}_steps_${num_target_images}_images_per_iter_${aft_score}_fk_steering_wd \
    --use_wandb \
    --wandb_project "aft-${dataset}-iterative" \
    --wandb_name "base/${dataset}_resnet18_${num_iterations}_iter_${steps}_steps_${num_target_images}_images_per_iter_${aft_score}_fk_steering_wd" \
    --wandb_save_checkpoints \
    --use_all_synthetic \
    --seed 0 \
    --original_feature_path /NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/features/vit_giant_patch14_dinov2.lvd142m_flowers.pt \
    --aft_score ${aft_score} \
    --scheduler warmup_stable_decay \
    --warmup_steps 150 \
    --stable_steps 0 \
    --decay_steps 1350 \
    --method init \
    --resample_frequency 2 \
    --resampling_t_start 3 \
    --resampling_t_end 14 \
    --time_steps 18
