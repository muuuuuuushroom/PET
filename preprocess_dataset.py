import cv2
import os
import json
import torch

from PIL import Image
import numpy as np
from sys import getsizeof
import torchvision.transforms as standard_transforms


def parse_json(gt_path):
        with open(gt_path, 'r') as f:
            tree = json.load(f)

        points = []
        for shape in tree['shapes']:
            points.append(shape['points'][0])
        points = np.array(points, dtype=np.float32)

        return points

basetransform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

root = '/root/PET/eval_data'
for mode in ['train_ran']: #, 'test_ran']:
    data_list_path = f'/root/PET/eval_data/images'
    # data_list = [name.split(' ') for name in open(data_list_path).read().splitlines()]
    data_list = [[os.path.join(dirpath, f)] for dirpath, _, files in os.walk(data_list_path) for f in files]
    # img_loaded = {}
    # point_loaded = {}
    img_loaded = []
    point_loaded = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    import tqdm as td
    for sample in td.tqdm(data_list):
        # if sample[0].split('/')[-1].split('.')[0]+'.npy' in os.listdir('/root/PET/eval_data/images_npy/'):
        #     continue
        img_path = os.path.join(root,'images',sample[0].split('/')[-1])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        for i in range(3):
            # img[i,:,:] = (img[i,:,:] - mean[i]) / std[i]  # wokao
            img[:,:,i] = (img[:,:,i] - (mean[i])) / (std[i])
        img = img.transpose(2,0,1)

        np.save('/root/PET/eval_data/images_npy/'+sample[0].split('/')[-1], img)