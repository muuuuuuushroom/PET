# backbone_vit.py

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import NestedTensor
from ..position_encoding import build_position_encoding


# -----------------------------
# Patch Embedding (Conv stem)
# -----------------------------
class PatchEmbed(nn.Module):
    """
    Conv2d patchify: Bx3xHxW -> BxCpx(H/P)x(W/P)
    """
    def __init__(self, in_chans=3, embed_dim=384, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W -> B, D, H', W'
        x = self.proj(x)
        return x


# -----------------------------
# ViT Encoder Blocks
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.drop_path2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x_res + self.drop_path1(attn_out)

        x_res = x
        x = self.norm2(x)
        x = x_res + self.drop_path2(self.mlp(x))
        return x


class ViTEncoder(nn.Module):
    """
    ViT-Small: dim=384, depth=12, heads=6
    """
    def __init__(self, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, attn_drop, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# -----------------------------
# ViT Small Backbone (feature -> 8x map -> upsample -> 4x map)
# -----------------------------
class BackboneBase_ViT(nn.Module):
    """
    Produce two feature maps:
        - '8x': stride 8 wrt input, 256 channels
        - '4x': stride 4 wrt input, 256 channels
    Interface and mask handling match BackboneBase_VGG.
    """
    def __init__(self,
                 patch_size: int = 8,
                 embed_dim: int = 384,
                 depth: int = 12,
                 num_heads: int = 6,
                 num_channels_out: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_channels = num_channels_out  # to align with VGG backbone

        # stem
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dim, patch_size=patch_size)

        # learnable 2D positional embedding (no class token; pure grid)
        self.pos_embed: Optional[nn.Parameter] = None  # lazily init when first size is known

        # transformer encoder
        self.encoder = ViTEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)

        # heads to 256 channels
        self.proj_8x = nn.Conv2d(embed_dim, num_channels_out, kernel_size=1)
        self.refine_4x = nn.Conv2d(num_channels_out, num_channels_out, kernel_size=3, padding=1)

    def _maybe_init_pos_embed(self, B: int, H8: int, W8: int, device, dtype):
        if self.pos_embed is None or self.pos_embed.shape[-2:] != (H8, W8):
            # shape: (1, C, H8, W8)
            pe = torch.zeros(1, self.embed_dim, H8, W8, device=device, dtype=dtype)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = nn.Parameter(pe)

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor with fields
                - tensors: Bx3xHxW
                - mask:    BxHxW (bool), optional but expected (as in VGG backbone)
        Returns:
            Dict[str, NestedTensor]: {'4x': ..., '8x': ...}
        """
        x = tensor_list.tensors         # B,3,H,W
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype

        # patchify -> B, D, H8, W8
        x = self.patch_embed(x)
        _, D, H8, W8 = x.shape
        self._maybe_init_pos_embed(B, H8, W8, device, dtype)

        # add pos and flatten to tokens
        x = x + self.pos_embed               # B, D, H8, W8
        x = x.flatten(2).transpose(1, 2)     # B, N, D (N = H8*W8)

        # transformer
        x = self.encoder(x)                   # B, N, D

        # reshape back to map
        x_map = x.transpose(1, 2).reshape(B, D, H8, W8)  # B, D, H8, W8

        # 8x feature map -> 256ch
        feat_8x = self.proj_8x(x_map)        # B, 256, H8, W8

        # 4x feature map: upsample 2x then 3x3 refine
        feat_4x = F.interpolate(feat_8x, scale_factor=2.0, mode='bilinear', align_corners=False)
        feat_4x = self.refine_4x(feat_4x)    # B, 256, H4, W4

        # masks
        m = tensor_list.mask
        assert m is not None, "Expect mask in NestedTensor (same as VGG backbone)."
        mask_8x = F.interpolate(m[None].float(), size=feat_8x.shape[-2:]).to(torch.bool)[0]
        mask_4x = F.interpolate(m[None].float(), size=feat_4x.shape[-2:]).to(torch.bool)[0]

        out: Dict[str, NestedTensor] = {}
        out['8x'] = NestedTensor(feat_8x, mask_8x)
        out['4x'] = NestedTensor(feat_4x, mask_4x)
        return out


class Backbone_ViT(BackboneBase_ViT):
    """
    ViT-S backbone aligned to the VGG backbone interface.
    """
    def __init__(self,
                 patch_size: int = 8,
                 embed_dim: int = 384,
                 depth: int = 12,
                 num_heads: int = 6,
                 num_channels_out: int = 256):
        super().__init__(patch_size, embed_dim, depth, num_heads, num_channels_out)


class Joiner(nn.Sequential):
    """
    Keep identical to backbone_vgg.Joiner:
    forward returns (out_dict, pos_dict), with pos encoding applied to feature maps.
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[str, NestedTensor] = {}
        pos: Dict[str, torch.Tensor] = {}
        for name, x in xs.items():
            out[name] = x
            pos[name] = self[1](x).to(x.tensors.dtype)
        return out, pos


def build_backbone_vitsmall(args):
    """
    Usage mirrors build_backbone_vgg(args).
    The resulting model has .num_channels = 256 like the VGG backbone.
    """
    position_embedding = build_position_encoding(args)
    backbone = Backbone_ViT(
        patch_size=8,      # ensures we naturally produce an 8x map
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_channels_out=256
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == '__main__':
    # simple smoke test
    B, C, H, W = 2, 3, 256, 256
    imgs = torch.randn(B, C, H, W)
    mask = torch.zeros(B, H, W, dtype=torch.bool)
    x = NestedTensor(imgs, mask)

    vit = Backbone_ViT()
    outs = vit(x)
    for k, v in outs.items():
        print(k, v.tensors.shape, v.mask.shape)
