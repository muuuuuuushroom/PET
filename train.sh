#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 \
torchrun \
    --nproc_per_node=1 \
    --master_port=10004 \
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
    --output_dir vgg_lossmixed \
    --set_up mixed # 'None', 'f4x', 'probloss', 'mixed'
    # --resume='/data/zlt/RSPET/PET/outputs/SHA/pet_model/best_checkpoint.pth'

# nohup sh train.sh> output_nohup/vgg_lossmixed.log 2>&1 &