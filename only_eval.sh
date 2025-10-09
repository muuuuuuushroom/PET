CUDA_VISIBLE_DEVICES='5' \
python eval_only.py \
    --dataset_file="SOY_evon" \
    --backbone="vgg16_bn" \
    --resume="/data/zlt/RSPET/PET/outputs/SOY/vgg16_bn_enco/best_checkpoint.pth" \
    --vis_dir=None 
    # --pretrained_dinov3="pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    
    #"/root/PET/eval_data/vis/adapter_s"