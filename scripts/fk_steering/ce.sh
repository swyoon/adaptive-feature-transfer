device=1
aft_score="entropy"
use_downstream=True
learn_scales=True
num_steps=18
lmda=1.0
resample_frequency=3

# almost fixed
seed=0
dataset_name=flowers # aircraft or flowers
prec=10 # 3 for aircraft, 10 for flowers
num_target_images=1024 # ~3000 for aircraft, ~1000 for flowers
steps=5000 # 10000 for aircraft, 5000 for flowers
edm_ckpt="/NFS/workspaces/tg.ahn/Collab/edm/training-runs-flowers102/00001-flowers102-64x64-cond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-008132.pkl"
model_ckpt="/NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/ckpts/aft/flowers/resnet50.a1_in1k_vit_giant_patch14_dinov2.lvd142m_lr1e-3_seed0/model.pt"
prior_ckpt="/NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/ckpts/aft/flowers/resnet50.a1_in1k_vit_giant_patch14_dinov2.lvd142m_lr1e-3_seed0/prior.pt"

# auto
if [ "$use_downstream" = "True" ]; then
    image_dir="./artifacts/251029/fk/edm_aft_${aft_score}_fk_steering_${num_steps}_steps_${lmda}_lambda_${resample_frequency}_resample_with_downstream/${dataset_name}-${seed}"
else
    image_dir="./artifacts/251029/fk/edm_aft_${aft_score}_fk_steering_${num_steps}_steps_${lmda}_lambda_${resample_frequency}_resample/${dataset_name}-${seed}"
fi

# fixed
pretrained_model=vit_giant_patch14_dinov2.lvd142m

# auto
if [ "$use_downstream" = "True" ]; then
    steer_method=edm_aft_${aft_score}_fk_steering_${num_steps}_steps_${lmda}_lambda_${resample_frequency}_resample_with_downstream
else
    steer_method=edm_aft_${aft_score}_fk_steering_${num_steps}_steps_${lmda}_lambda_${resample_frequency}_resample
fi

class_file="./classes/${dataset_name}.txt"
feature_path="./features/${pretrained_model}_${dataset_name}_${steer_method}_${num_target_images}.pt"

# fixed
train_method=aft
model=resnet50.a1_in1k
lr=1e-3

# auto
if [ "$learn_scales" = "True" ]; then
    prior_update=update
else
    prior_update=freeze
fi

CUDA_VISIBLE_DEVICES=${device} python generate_edm_fk_steering.py \
    --seed ${seed} \
    --dataset ${dataset_name} \
    --edm_ckpt ${edm_ckpt} \
    --model_ckpt ${model_ckpt} \
    --prior_ckpt ${prior_ckpt} \
    --prec ${prec} \
    --aft_score ${aft_score} \
    --num_target_images ${num_target_images} \
    --use_downstream ${use_downstream} \
    --save_dir ${image_dir} \
    --num_steps ${num_steps} \
    --lmda ${lmda} \
    --resample_frequency ${resample_frequency}

CUDA_VISIBLE_DEVICES=${device} python save_auxiliary_data_features.py \
    --model_class=${pretrained_model} \
    --directory=${image_dir} \
    --class_file=${class_file} \
    --save_path=${feature_path} \
    --batch_size=64

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
    --ckpt_dir="./ckpts/${train_method}/${dataset_name}_with_${steer_method}/${model}_${pretrained_model}_ft_lr${lr}_seed${seed}/prior_${prior_update}" \
    --run_name="${train_method}_${dataset_name}_with_${steer_method}/${model}_${pretrained_model}_ft_lr${lr}_seed${seed}/prior_${prior_update}" \
    --use_wandb=True \
    --auxiliary_dataset=${dataset_name} \
    --directory=${image_dir} \
    --class_file=${class_file} \
    --feature_path_postfix="${steer_method}_${num_target_images}" \
    --model_ckpt=${model_ckpt} \
    --prior_ckpt=${prior_ckpt}