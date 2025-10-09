import argparse
import random
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset
import util.misc as utils
from engine_evon import evaluate
from models import build_model

import json
import re
import sys
from math import sqrt
from typing import List, Tuple


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)
    # experiment set up
    parser.add_argument('--loss_set_up', default='None', type=str,
                        choices=('None', 'f4x', 'probloss', 'mixed'))
    parser.add_argument('--probloss_cal', default='Linear', type=str,
                        choices=('Linear', 'Psq', 'NLL', 'Squard', 'Focal'))

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vitadapter', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    
    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)       # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)    # regression loss coefficient
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default=None)
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params:', n_parameters/1e6)

    # build dataset
    val_image_set = 'val'
    dataset_val = build_dataset(image_set=val_image_set, args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # load pretrained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)        
        cur_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    
    # evaluation
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    os.makedirs(vis_dir, exist_ok=True)
    results = evaluate(model, data_loader_val, device, vis_dir=vis_dir)
    name = args.resume.split('/')[-2]
    with open(f"jsons/662_{name}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    eval_metrics(f"jsons/662_{name}_results.json")

RE_KEY = re.compile(r'^(\d+)_gt(\d+)$')

def parse_json(path: str) -> Tuple[List[float], List[float]]:
    """读取JSON并返回(y_true, y_pred)列表"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    y_true, y_pred = [], []
    for k, v in data.items():
        m = RE_KEY.match(str(k))
        if not m:
            continue
        gt = float(m.group(2))
        pred = float(v)
        y_true.append(gt)
        y_pred.append(pred)
    return y_true, y_pred


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mape(y_true: List[float], y_pred: List[float]) -> float:
    """
    Mean Absolute Percentage Error (%)
    忽略真实值为0的样本
    """
    values = [abs((a - b) / a) for a, b in zip(y_true, y_pred) if a != 0]
    return sum(values) / len(values) * 100 if values else float('nan')


def rmspe(y_true: List[float], y_pred: List[float]) -> float:
    """
    Root Mean Squared Percentage Error (%)
    忽略真实值为0的样本
    """
    values = [((a - b) / a) ** 2 for a, b in zip(y_true, y_pred) if a != 0]
    return sqrt(sum(values) / len(values)) * 100 if values else float('nan')


def r2_score_safe(y_true: List[float], y_pred: List[float]) -> float:
    """R² 安全实现（兼容常量情况）"""
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty y_true/y_pred.")
    y_mean = sum(y_true) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))

    if ss_tot == 0.0:
        all_equal = all(yp == y_true[0] for yp in y_pred)
        return 1.0 if all_equal else 0.0
    return 1.0 - ss_res / ss_tot


def eval_metrics(path):
    y_true, y_pred = parse_json(path)

    if len(y_true) == 0:
        print("未解析到任何 (gt, pred) 对，请检查键名是否形如 '29_gt145'.")
        sys.exit(2)

    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    rmspe_val = rmspe(y_true, y_pred)
    r2_val = r2_score_safe(y_true, y_pred)

    print(f"样本数 : {len(y_true)}")
    print(f"MAE    : {mae_val:.6f}")
    print(f"RMSE   : {rmse_val:.6f}")
    print(f"MAPE   : {mape_val:.6f} %")
    print(f"RMSPE  : {rmspe_val:.6f} %")
    print(f"R²     : {r2_val:.6f}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    
