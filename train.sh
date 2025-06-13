CUDA_VISIBLE_DEVICES='1' \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=10001 \
    --use_env main.py \
    --lr=0.0001 \
    --backbone="vgg16_bn" \
    --ce_loss_coef=1.0 \
    --point_loss_coef=5.0 \
    --eos_coef=0.5 \
    --dec_layers=2 \
    --hidden_dim=256 \
    --dim_feedforward=512 \
    --nheads=8 \
    --dropout=0.0 \
    --epochs=1500 \
    --dataset_file="SHA" \
    --eval_freq=5 \
    --output_dir='vgg_baseline' \
    --set_up='None'
    # --resume='/data/zlt/RSPET/PET/outputs/SHA/pet_model/best_checkpoint.pth'

# nohup sh train.sh> output_nohup/vgg_baseline.log 2>&1 &