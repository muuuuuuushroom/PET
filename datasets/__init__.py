import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .WuhanMetro import build_whm

data_path = {
    'SHA': 'data/Crowd_Counting/ShanghaiTech/part_A_final/',
    'SHB': 'data/Crowd_Counting/ShanghaiTech/part_B_final/',
    'WuhanMetro': 'data/WuhanMetroCount',
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    elif args.dataset_file == 'WuhanMetro':
        return build_whm(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
