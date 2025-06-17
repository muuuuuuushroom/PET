import torch

# 1) 读取检查点
ckpt_path = "/data/zlt/RSPET/PET/outputs/SHA/vgg_probloss/checkpoint.pth"          # 根据需要改成自己的完整路径
ckpt = torch.load(ckpt_path, map_location="cpu")

# 2) 添加/更新 best_epoch
ckpt["best_epoch"] = 250

# 3) 保存（覆盖原文件；如担心安全，可先另存）
torch.save(ckpt, ckpt_path)
