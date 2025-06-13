#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 \
torchrun \
    --nproc_per_node=1 \
    --master_port=10011 \
    --standalone \
    main.py \
    --lr 0.0001 \
    --backbone vgg16_bn \
    --ce_loss_coef 1.0 \
    --point_loss_coef 5.0 \
    --eos_coef 0.5 \
    --dec_layers 2 \
    --hidden_dim 256 \
    --dim_feedforward 512 \
    --nheads 8 \
    --dropout 0.0 \
    --epochs 1500 \
    --dataset_file SHA \
    --eval_freq 5 \
    --output_dir vgg_baseline_enpet \
    --loss_set_up None \
    --probloss_cal Linear

    # --resume /data/zlt/RSPET/PET/outputs/SHA/pet_model/best_checkpoint.pth

# nohup sh train.sh> output_nohup/vgg_baseline_enpet.log 2>&1 &

# loss_set_up None, f4x, probloss, mixed
# loss_set_up Linear, Psq, NLL, Squard, Focal

# running:
# 0//: 
# 1//: SHA vgg baseline, on env:dinov2 and env:pet
# 2//: SHA vgg f4x
# 3//: SHA vgg probloss