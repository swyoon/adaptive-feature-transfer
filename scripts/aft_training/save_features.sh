device=0
pretrained_model=vit_giant_patch14_dinov2.lvd142m
dataset=aircraft # aircraft or flowers
save_path="./features/${pretrained_model}_${dataset}.pt"
batch_size=64 # lowered due to memory issues

CUDA_VISIBLE_DEVICES=${device} python save_features.py \
    --model_class=${pretrained_model} \
    --dataset=${dataset} \
    --save_path=${save_path} \
    --batch_size=${batch_size}