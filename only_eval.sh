CUDA_VISIBLE_DEVICES='0' \
python eval_only.py \
    --dataset_file="SOY_evon" \
    --resume="/root/PET/outputs/SOY/dinov3_halffixedprob/best_checkpoint.pth" \
    --vis_dir=None
    
    #"/root/PET/eval_data/vis/adapter_s"