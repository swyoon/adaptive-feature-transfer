#!/bin/bash
device=1
num_iterations=6
steps=3000
num_target_images=10000
aft_score=ce
dataset=flowers


CUDA_VISIBLE_DEVICES=${device} python train_iterative.py \
    --model_class resnet50.a1_in1k \
    --pretrained_model vit_giant_patch14_dinov2.lvd142m \
    --dataset ${dataset} \
    --num_iterations ${num_iterations} \
    --steps ${steps} \
    --lr 1e-3 \
    --batch_size 128 \
    --optimizer adam \
    --edm_ckpt /NFS/workspaces/tg.ahn/Collab/edm/training-runs-flowers102/00001-flowers102-64x64-cond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-008132.pkl \
    --num_target_images ${num_target_images} \
    --base_output_dir /NFS/database_personal/tg.ahn/Collab/adaptive-feature-transfer/logs/${dataset}_resnet50_${num_iterations}_iter_${steps}_steps_${num_target_images}_images_per_iter_${aft_score}_fk_steering_wd_with_init_syn \
    --use_wandb \
    --wandb_project "aft-${dataset}-iterative" \
    --wandb_name "${dataset}_resnet50_${num_iterations}_iter_${steps}_steps_${num_target_images}_images_per_iter_${aft_score}_fk_steering_wd_with_init_syn" \
    --wandb_save_checkpoints \
    --use_all_synthetic \
    --seed 0 \
    --original_feature_path /NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/features/vit_giant_patch14_dinov2.lvd142m_flowers.pt \
    --aft_score ${aft_score} \
    --scheduler warmup_stable_decay \
    --warmup_steps 300 \
    --stable_steps 0 \
    --decay_steps 2700 \
    --resample_frequency 2 \
    --resampling_t_start 3 \
    --resampling_t_end 14 \
    --time_steps 18 \
    --use_synthetic_from_beginning \
    --initial_synthetic_dir /NFS/database_personal/tg.ahn/Collab/adaptive-feature-transfer/artifacts/251109/baselin/edm_18_steps_1000000/flowers-0 \
    --initial_synthetic_feature_dir /NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/features/vit_giant_patch14_dinov2.lvd142m_flowers_no_steer_1000000.pt \
    --initial_synthetic_dataset_size 40000 \
