"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn

import sys
import os
import scipy.spatial
from scipy.spatial import KDTree
import numpy as np

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .backbones import *
from .transformer import *
from .position_encoding import build_position_encoding


class BasePETCount(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'
    
    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + stride//2 -1) // stride

        # generate point queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get point queries embedding
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]
        query_embed = query_embed.view(bs, c, h, w)

        # get point queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down,shift_x_down]
        query_feats = query_feats.view(bs, c, h, w)

        return query_embed, points_queries, query_feats
    
    def points_queris_embed_inference(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during inference
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + stride//2 -1) // stride

        # generate points queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get points queries embedding 
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]

        # get points queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        
        # window-rize
        query_embed = query_embed.reshape(bs, c, h, w)
        points_queries = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
        query_feats = query_feats.reshape(bs, c, h, w)

        dec_win_w, dec_win_h = kwargs['dec_win_size']
        query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
        query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)
        
        # dynamic point query generation
        div = kwargs['div']
        div_win = window_partition(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
        valid_div = (div_win > 0.5).sum(dim=0)[:,0] 
        v_idx = valid_div > 0
        query_embed_win = query_embed_win[:, v_idx]
        query_feats_win = query_feats_win[:, v_idx]
        points_queries_win = points_queries_win.cuda()
        points_queries_win = points_queries_win[:, v_idx].reshape(-1, 2)
    
        return query_embed_win, points_queries_win, query_feats_win, v_idx
    
    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # generate points queries and position embedding
        if 'train' in kwargs:
            query_embed, points_queries, query_feats = self.points_queris_embed(samples, self.pq_stride, src, **kwargs)
            query_embed = query_embed.flatten(2).permute(2,0,1) # NxCxHxW --> (HW)xNxC
            v_idx = None
        else:
            query_embed, points_queries, query_feats, v_idx = self.points_queris_embed_inference(samples, self.pq_stride, src, **kwargs)

        out = (query_embed, points_queries, query_feats, v_idx)
        return out
    
    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = self.class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().cuda()
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / 256)
            outputs_offsets[...,1] /= (img_w / 256)

        outputs_points = outputs_offsets[-1] + points_queries
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets[-1]}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        # get points queries for transformer
        pqs = self.get_point_query(samples, features, **kwargs)
        
        # point querying
        kwargs['pq_stride'] = self.pq_stride
        hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)

        # prediction
        points_queries = pqs[1]
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        return outputs
    

class PET(nn.Module):
    """ 
    Point quEry Transformer
    """
    def __init__(self, backbone, num_classes, args=None):
        super().__init__()
        self.backbone = backbone
        
        # positional embedding
        self.pos_embed = build_position_encoding(args)

        # feature projection
        hidden_dim = args.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            ]
        )

        # context encoder
        self.encode_feats = '8x'
        enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]  # encoder window size
        args.enc_layers = len(enc_win_list)
        self.context_encoder = build_encoder(args, enc_win_list=enc_win_list)

        # quadtree splitter
        context_patch = (128, 64)
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h ,context_w)),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        # point-query quadtree
        args.sparse_stride, args.dense_stride = 8, 4    # point-query stride
        transformer = build_decoder(args)
        self.quadtree_sparse = BasePETCount(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer)
        self.quadtree_dense = BasePETCount(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer)
        
        self.set_up = args.loss_set_up
        if self.set_up in ['f4x', 'mixed']:
            self.prob_conv = nn.Sequential(
                nn.Conv2d(in_channels=backbone.num_channels, out_channels=1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

    def compute_loss(self, outputs, criterion, targets, epoch, samples, prob=None):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = 5

        # compute loss
        if epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'], prob=prob)
            loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'], prob=prob)
        else:
            loss_dict_sparse = criterion(output_sparse, targets, prob=prob)
            loss_dict_dense = criterion(output_dense, targets, prob=prob)

        # sparse point queries loss
        loss_dict_sparse = {k+'_sp':v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k+'_sp':v for k,v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        loss_dict_dense = {k+'_ds':v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k+'_ds':v for k,v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
    
        # point queries loss
        losses = loss_pq_sparse + loss_pq_dense 

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        weight_dict.update(weight_dict_dense)

        # quadtree splitter loss
        den = torch.tensor([target['density'] for target in targets])   # crowd density
        bs = len(den)
        ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        ds_div = outputs['split_map_raw'][ds_idx]
        sp_div = 1 - outputs['split_map_raw']

        # constrain sparse regions
        loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()

        # constrain dense regions
        if sum(ds_idx) > 0:
            ds_num = ds_div.shape[0]
            loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        else:
            loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        # update quadtree splitter loss            
        loss_split = loss_split_sp + loss_split_ds
        weight_split = 0.1 if epoch >= warmup_ep else 0.0
        loss_dict['loss_split'] = loss_split
        weight_dict['loss_split'] = weight_split

        # final loss
        losses += loss_split * weight_split
        return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}

    def forward(self, samples: NestedTensor, **kwargs):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # backbone
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
        if 'test' in kwargs:
            flag = 0
        features, pos = self.backbone(samples)

        # generate probability map if set to 'f4x' or mixed
        if 'train' in kwargs and self.set_up in ['f4x', 'mixed']:
            prob_map = self.prob_conv(features['4x'].tensors)
            H, W = samples.tensors.shape[-2], samples.tensors.shape[-1]
            prob_map_up = F.interpolate(prob_map, size=(H, W), mode='bilinear', align_corners=False)
            kwargs['prob_map_up'] = prob_map_up
        
        # positional embedding
        dense_input_embed = self.pos_embed(samples)
        kwargs['dense_input_embed'] = dense_input_embed

        # feature projection
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        # forward
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)   
        return out

    def pet_forward(self, samples, features, pos, **kwargs):
        # context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None
        encode_src = self.context_encoder(src, src_pos_embed, mask)
        # encode_src = src
        context_info = (encode_src, src_pos_embed, mask)
        # context_info = (encode_src, src_pos_embed, mask)
        
        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = self.quadtree_splitter(encode_src)        
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)
        
        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size'] = [16, 8]
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        else:
            outputs_sparse = None
        
        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = [8, 4]
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
        else:
            outputs_dense = None
        
        # format outputs
        outputs = dict()
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs
    
    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        prob = kwargs['prob_map_up'] if self.set_up in ['f4x', 'mixed'] else None
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples, prob)
        return losses
    
    def test_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        out_dense, out_sparse = outputs['dense'], outputs['sparse']
        thrs = 0.5  # inference threshold        
        
        # process sparse point queries
        if outputs['sparse'] is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            valid_sparse = out_sparse_scores > thrs
            index_sparse = valid_sparse.cpu()
        else:
            index_sparse = None

        # process dense point queries
        if outputs['dense'] is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            valid_dense = out_dense_scores > thrs
            index_dense = valid_dense.cpu()
        else:
            index_dense = None

        # format output
        div_out = dict()
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat([out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
        div_out['split_map_raw'] = outputs['split_map_raw']
        return div_out

def build_density_map_from_points_with_kdtree(target_points, batch_indices, img_h, img_w, device='cuda', alpha=0.4):
    B = int(batch_indices.max().item()) + 1  # Calculate the batch size
    density = torch.zeros((B, 1, img_h, img_w), dtype=torch.float32, device=device)  # Initialize the density map

    for b in range(B):
        mask = batch_indices == b
        points_b = target_points[mask]  # [nb, 2], get points for batch b
        if points_b.numel() == 0:
            continue  # Skip empty batches

        # Unnormalize to pixel coordinates
        pts_np = (points_b * torch.tensor([img_w, img_h], device=device)).cpu().numpy()
        pts_np = np.round(pts_np).astype(np.int32)
        pts_np[:, 0] = np.clip(pts_np[:, 0], 0, img_w - 1)
        pts_np[:, 1] = np.clip(pts_np[:, 1], 0, img_h - 1)

        num_pts = len(pts_np)
        if num_pts == 0:
            continue

        # Build KDTree (k = min(4, n))
        k = min(4, num_pts)
        tree = scipy.spatial.KDTree(pts_np.copy(), leafsize=2048)
        distances, locations = tree.query(pts_np, k=k)

        for i, (x, y) in enumerate(pts_np):
            pt_map = torch.zeros((1, 1, img_h, img_w), device=device)
            pt_map[0, 0, y, x] = 1.0

            # Estimate sigma
            # if num_pts > 1 and distances[i].shape[0] >= 2 and np.isfinite(distances[i][1]):
            #     di = distances[i][1]
            #     neighbor_idx = locations[i][1:]
            #     neighbor_dists = []

            #     for j in neighbor_idx:
            #         if j < len(distances) and distances[j].shape[0] >= 2 and np.isfinite(distances[j][1]):
            #             neighbor_dists.append(distances[j][1])

            #     if len(neighbor_dists) > 0:
            #         d_mtop3 = np.mean(neighbor_dists)
            #         d = min(di, d_mtop3)
            #     else:
            #         d = di
            #     sigma = max(alpha * d, 1.0)
            # else:
            #     sigma = np.average([img_h, img_w]) / 4.0  # fallback sigma
                
            sigma = 15 # fixed setting

            # Generate Gaussian kernel
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

            kernel = generate_gaussian_kernel_prob(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0)
            response = F.conv2d(pt_map, kernel, padding=kernel_size // 2)
            peak = response[0, 0, y, x]

            if peak > 0:
                response = response / peak
                density[b, 0] = torch.maximum(density[b, 0], response[0, 0])

    return density  # shape [B, 1, H, W]

def generate_prob_map_from_points(targets, img_h, img_w, device='cuda', alpha=0.4):
    """
    Generate a probability map using points from the target dictionary list.
    
    Args:
        targets (list of dict): List of dictionaries, each containing 'points', 'labels', and 'density'.
        img_h (int): Height of the image.
        img_w (int): Width of the image.
        device (str): The device to run the computation on ('cuda' or 'cpu').
        alpha (float): A scaling factor for the sigma value.

    Returns:
        torch.Tensor: The generated probability map of shape (1, img_h, img_w).
    """
    # Collect points from all entries in the targets list
    all_points = []
    for target in targets:
        points = target['points'].cpu().numpy()
        all_points.append(points)
    
    # Convert all points to a numpy array
    all_points = np.vstack(all_points)

    # Ensure points are valid
    if len(all_points) == 0:
        return torch.zeros((1, img_h, img_w), dtype=torch.float32, device=device)

    # Create a KDTree to compute distances between points
    tree = KDTree(all_points)
    distances, locations = tree.query(all_points, k=4)

    # Initialize an empty density map
    density = torch.zeros((1, 1, img_h, img_w), dtype=torch.float32, device=device)

    # for i, pt in enumerate(all_points):
        # Convert point coordinates to integers
        # x, y = int(pt[0]), int(pt[1])
    for i, pt in enumerate(all_points):
        # floor-int-clamp
        x = int(np.floor(pt[0]))
        y = int(np.floor(pt[1]))
        if x < 0 or y < 0:
            continue
        x = min(x, img_w - 1)
        y = min(y, img_h - 1)

        # Create a 2D map with a single point set to 1
        pt2d = torch.zeros((1, 1, img_h, img_w), dtype=torch.float32, device=device)
        pt2d[0, 0, y, x] = 1.0

        # Dynamically calculate sigma based on neighbor distances
        # if len(distances[i]) >= 2 and np.isfinite(distances[i][1]):
        #     di = distances[i][1]
        #     neighbor_idx = locations[i][1:]
        #     neighbor_distances = []
        #     for idx in neighbor_idx:
        #         if np.isfinite(distances[idx][1]):
        #             neighbor_distances.append(distances[idx][1])

        #     if len(neighbor_distances) > 0:
        #         d_mtop3 = np.mean(neighbor_distances)
        #         d = min(di, d_mtop3)
        #     else:
        #         d = di  # fallback
            
        #     sigma = alpha * d
        # else:
        #     sigma = np.average(np.array([img_h, img_w])) / 4.0  # fallback
        
        sigma = 15 # fixed setting

        sigma = max(sigma, 1.0)

        # Generate a Gaussian kernel based on sigma
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1

        gaussian_kernel = generate_gaussian_kernel_prob(kernel_size, sigma, device)
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

        # Apply Gaussian filter to the point map
        filter = F.conv2d(pt2d, gaussian_kernel, padding=kernel_size // 2)

        # Normalize the filter to avoid numerical instability
        peak = filter[0, 0, y, x]
        if peak > 0:
            filter = filter / peak

        # Update the density map with the current filter
        density = torch.maximum(density, filter)

    return density.squeeze()

def generate_gaussian_kernel_prob(kernel_size, sigma, device='cuda'):
    """
    Generate a Gaussian kernel for convolution.
    
    Args:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian.
        device (str): The device to run the computation on ('cuda' or 'cpu').
        
    Returns:
        torch.Tensor: The generated Gaussian kernel.
    """
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    y = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def generate_gaussian_kernel(kernel_size, sigma, device):

    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    y = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def generate_density_map_torch(point_map, fixed_sigma=15.0, device='cuda'):

    H, W = point_map.shape
    gt_count = np.count_nonzero(point_map)
    if gt_count == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    point_map_tensor = torch.from_numpy(point_map).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    
    sigma = fixed_sigma
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1

    gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma, device)
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
    density_map_tensor = F.conv2d(point_map_tensor, gaussian_kernel, padding=kernel_size // 2)

    return density_map_tensor.squeeze().cpu().numpy()



class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, loss_set_up, probloss_cal):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef    # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        self.div_thrs_dict = {8: 0.0, 4:0.5}
        self.loss_set_up = loss_set_up
        if loss_set_up in ['probloss', 'mixed']:
            self.probloss_cal = probloss_cal
    
    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # compute classification loss
        if 'div' in kwargs:
            # get sparse / dense image index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            ds_idx = den_sort[:len(den_sort)//2]
            sp_idx = den_sort[len(den_sort)//2:]
            eps = 1e-5

            # raw cross-entropy loss
            weights = target_classes.clone().float()
            weights[weights==0] = self.empty_weight[0]
            weights[weights==1] = self.empty_weight[1]
            raw_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=-1, reduction='none')

            # binarize split map
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs

            # dual supervision for sparse/dense images
            loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
            loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
            loss_ce = loss_ce_sp + loss_ce_ds

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
            loss_ce = loss_ce + loss_ce_nondiv
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # get indices
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        batch_indices = []  # which image in batch this point came from
        for i, (src_idx, _) in enumerate(indices):
            batch_indices.extend([i] * len(src_idx))
        batch_indices = torch.tensor(batch_indices, device=src_points.device)
        
        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        
        if self.loss_set_up in ['probloss', 'mixed']:
            gt_prob_map = build_density_map_from_points_with_kdtree(
                target_points, batch_indices, img_h, img_w, device=src_points.device
            )
            x = (src_points[:, 0] * img_w).clamp(0, img_w - 1)
            y = (src_points[:, 1] * img_h).clamp(0, img_h - 1)
            x_norm = x / (img_w - 1) * 2 - 1
            y_norm = y / (img_h - 1) * 2 - 1
            grid = torch.stack([x_norm, y_norm], dim=1).unsqueeze(1).unsqueeze(1)  # [N,1,1,2]
            prob_vals = F.grid_sample(
                gt_prob_map[batch_indices],  # [N,1,H,W]
                grid, mode='bilinear', align_corners=True
            ).squeeze(-1).squeeze(-1).squeeze(1)  # [N]
            
            eps = 1e-6
            if self.probloss_cal == 'Linear': # Linear, y = 1 - p
                loss_points_raw = 1.0 - prob_vals
            elif self.probloss_cal == 'Psq': # P_squard, y = 1 - p^2
                loss_points_raw = 1.0 - prob_vals.pow(2)
            elif self.probloss_cal == 'NLL': # Negative Log Likelihood, y = - log(p)
                loss_points_raw = -torch.log(prob_vals.clamp_min(eps))
            elif self.probloss_cal == 'Squard': # Squard, y = (1 - p)^2
                loss_points_raw = ((1.0 - prob_vals).pow(2))
            elif self.probloss_cal == 'Focal': # Focal, y = (1 - p)^γ * log(p)
                gamma = 2.0
                loss_points_raw = - (1.0 - prob_vals).pow(gamma) * torch.log(prob_vals.clamp_min(eps))
            
            # scale factor & shape adjustment
            loss_points_raw = loss_points_raw.unsqueeze(1).expand(-1, 2) * 0.05
        else:
            loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if 'div' in kwargs:
            # get sparse / dense index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort)//2]
            img_sp_idx = den_sort[len(den_sort)//2:]
            pt_ds_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx])
            pt_sp_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx])

            # dual supervision for sparse/dense images
            eps = 1e-5
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs
            loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
            loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps)
            loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps)

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (non_div_mask[idx].sum() + eps)   

            # final point loss
            losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        else:
            losses['loss_points'] = loss_points_raw.sum() / num_points
        
        return losses

    def loss_probs(self, outputs, targets, indices, num_points, **kwargs):
        criterion = nn.MSELoss(reduction='mean').cuda()
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        gt_prob_map = generate_prob_map_from_points(targets, img_h, img_w)
        prob_est = kwargs['prob']
        prob_loss = criterion(gt_prob_map, prob_est)
        losses = {}
        losses['loss_probs'] = prob_loss
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            'probs': self.loss_probs,
        } if self.loss_set_up in ['f4x', 'mixed'] else {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses


class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x


def build_pet(args):
    device = torch.device(args.device)

    # build model
    num_classes = 1
    if args.backbone == 'vitadapter':
        backbone = build_backbone_vitadapter(args)
    elif args.backbone == 'vgg16_bn':
        backbone = build_backbone_vgg(args)
    elif args.backbone == 'vit':
        backbone = build_backbone_vitsmall(args)
    # elif args.backbone == 'vit_in_adapter':
    #     backbone = build_backbone_vit_in_adapter(args)
    # elif args.backbone == 'dinov3':
    #     backbone = build_backbone_dinov3(args)
    
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # build loss criterion
    matcher = build_matcher(args)
    if args.loss_set_up in ['f4x', 'mixed']:
        weight_dict = {'loss_ce': args.ce_loss_coef, 
                    'loss_points': args.point_loss_coef,
                    'loss_probs': args.prob_loss_coef}
        losses = ['labels', 'points', 'probs']
    else:
        weight_dict = {'loss_ce': args.ce_loss_coef, 
                    'loss_points': args.point_loss_coef}
        losses = ['labels', 'points']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, 
                             loss_set_up=args.loss_set_up, probloss_cal=args.probloss_cal)
    criterion.to(device)
    return model, criterion
