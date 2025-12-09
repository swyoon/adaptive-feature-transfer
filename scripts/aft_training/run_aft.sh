# Script to run AFT training on Flowers and Aircraft datasets
# We used prec = 10 for flowers, and prec = 3 for aircraft
# We used steps = 5000 for flowers, 10000 for aircraft
device=1
method=aft
ds=aircraft
pretrained_models=vit_giant_patch14_dinov2.lvd142m
model=resnet50.a1_in1k
lr=1e-3
prec=3
steps=10000
seed=0
batch_size=128

run_name=aircraft_prec${prec}

CUDA_VISIBLE_DEVICES=${device} python run.py \
    --seed=${seed} \
    --model_class=${model} \
    --init_model=${model} \
    --dataset=${ds} \
    --pretrained_models=${pretrained_models} \
    --train_frac=1 \
    --use_val=False \
    --method=${method} \
    --prec=${prec} \
    --learn_scales=True \
    --steps=${steps} \
    --eval_steps=500 \
    --optimizer=adam \
    --batch_size=${batch_size} \
    --lr=${lr} \
    --wd=0 \
    --no_augment=True \
    --use_wandb=True \
    --run_name=${run_name} \
    --ckpt_dir="./ckpts/aft/${ds}_prec${prec}/" \