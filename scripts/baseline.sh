device=0
num_steps=18

# almost fixed
seed=0
dataset_name=aircraft # aircraft or flowers
dataset_size=3334 # Train dataset size(used for epochs->steps conversion): 3334 for aircraft
prec=3 # 3 for aircraft, 10 for flowers
num_target_images_list=(0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000)
edm_ckpt="./ckpts/edm/aircraft/aircraft-64x64-cond-ddpmpp-edm-gpus4-batch256-fp32.pkl"
pretrained_model=vit_giant_patch14_dinov2.lvd142m

image_dir="artifacts/251029/no_steer/edm_${num_steps}_steps/${dataset_name}-${seed}"
class_file="./classes/${dataset_name}.txt"
feature_path="./features/${pretrained_model}_${dataset_name}_no_steer_${num_target_images_list[-1]}.pt"

CUDA_VISIBLE_DEVICES=${device} python generate_edm.py \
    --seed ${seed} \
    --dataset ${dataset_name} \
    --edm_ckpt ${edm_ckpt} \
    --num_target_images ${num_target_images_list[-1]} \
    --save_dir ${image_dir} \
    --num_steps ${num_steps}
    
CUDA_VISIBLE_DEVICES=${device} python save_auxiliary_data_features.py \
    --model_class=${pretrained_model} \
    --directory=${image_dir} \
    --class_file=${class_file} \
    --save_path=${feature_path} \
    --batch_size=64

# fixed
train_method=aft
model=resnet50.a1_in1k
lr=1e-3
epochs=100
learn_scales=True


for num_target_images in "${num_target_images_list[@]}"; do
steps=$(((dataset_size + num_target_images) * epochs / 128 ))

CUDA_VISIBLE_DEVICES=${device} python run.py \
    --seed=${seed} \
    --model_class=${model} \
    --dataset=${dataset_name} \
    --pretrained_models=${pretrained_model} \
    --train_frac=1 \
    --use_val=False \
    --method=${train_method} \
    --prec=${prec} \
    --learn_scales=${learn_scales} \
    --steps=${steps} \
    --eval_steps=500 \
    --optimizer=adam \
    --batch_size=128 \
    --lr=${lr} \
    --wd=0 \
    --no_augment=True \
    --ckpt_dir="./ckpts/${train_method}/${dataset_name}_with_no_steer/${model}_${pretrained_model}_ft_lr${lr}_seed${seed}/num_target_images_${num_target_images}" \
    --run_name="${train_method}_${dataset_name}_with_no_steer/${model}_${pretrained_model}_ft_lr${lr}_seed${seed}/num_target_images_${num_target_images}" \
    --use_wandb=True \
    --auxiliary_dataset=${dataset_name} \
    --directory=${image_dir} \
    --class_file=${class_file} \
    --feature_path_postfix="no_steer_${num_target_images_list[-1]}" \
    --pretrained=True \
    --num_images=${num_target_images} \
    --num_synthetic_images=${num_target_images}

done