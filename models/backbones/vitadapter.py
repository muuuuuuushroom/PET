# vitadapter.py
import os
from typing import Dict, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import vit_base_patch16_224

from util.misc import NestedTensor
from .backbone_vgg import FeatsFusion

from ..position_encoding import build_position_encoding

# ----------------------------
# helpers
# ----------------------------
def _resize_pos_embed(pos_embed: torch.Tensor, new_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Resize absolute pos embed (B=1, 1+N, C) to match new grid size (h,w).
    Keep cls token intact (first token).
    """
    assert pos_embed.ndim == 3 and pos_embed.shape[0] == 1, "expect [1, 1+N, C]"
    cls_tok, grid = pos_embed[:, :1, :], pos_embed[:, 1:, :]
    C = grid.shape[-1]
    old_hw = int(grid.shape[1] ** 0.5)
    grid = grid.reshape(1, old_hw, old_hw, C).permute(0, 3, 1, 2)          # [1,C,Ho,Wo]
    grid = F.interpolate(grid, size=new_hw, mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_hw[0] * new_hw[1], C)   # [1, H'W', C]
    return torch.cat([cls_tok, grid], dim=1)


def _to_2d(x_tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """
    x_tokens: [B, H'W', C] (no cls)
    return:   [B, C, H', W']
    """
    B, N, C = x_tokens.shape
    h, w = hw
    assert N == h * w, f"token count {N} != {h}*{w}"
    return x_tokens.transpose(1, 2).reshape(B, C, h, w)


# ----------------------------
# base ViTAdapter
# ----------------------------
class ViTAdapter(nn.Module):
    """
    ViT-B/16 backbone with optional in_channels != 3 and external checkpoint.
    Forward returns tokens (no cls) with shape [B, H'W', C].
    """
    def __init__(self, img_size, in_channels=1, pretrained_path=None):
        super().__init__()
        self.vit = vit_base_patch16_224(pretrained=False)
        self.img_size = img_size
        # patch embed proj to support non-RGB
        if in_channels != 3:
            self.vit.patch_embed.proj = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16)

        # optional ckpt
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # adapt first conv if needed
            if 'patch_embed.proj.weight' in state_dict and state_dict['patch_embed.proj.weight'].shape[1] != in_channels:
                w = state_dict['patch_embed.proj.weight']
                state_dict['patch_embed.proj.weight'] = w.sum(dim=1, keepdim=True)
            self.vit.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return tokens without cls: [B, H'W', C]
        """
        B = x.shape[0]
        x = self.vit.patch_embed(x)                      # [B, C, H', W']
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)                 # [B, H'W', C]
        cls_token = self.vit.cls_token.expand(B, -1, -1) # [B, 1, C]
        # make pos embed match this H',W'
        pos_embed = self.vit.pos_embed
        if pos_embed.shape[1] != 1 + H * W:
            pos_embed = _resize_pos_embed(pos_embed, (H, W))
        x = torch.cat((cls_token, x), dim=1)             # [B, 1+H'W', C]
        x = self.vit.pos_drop(x + pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        x = x[:, 1:, :]                                  # drop cls → [B, H'W', C]
        return x


# ----------------------------
# multi-scale wrapper
# ----------------------------
class ViTAdapterMS(nn.Module):
    """
    Produce hierarchical 2D features from ViT by tapping intermediate blocks
    and building a light pyramid head to approximate C3(≈8x), C4(≈16x), C5(≈32x).

    Outputs:
        dict(c3: [B, C, H/8,  W/8 ],
             c4: [B, C, H/16, W/16],
             c5: [B, C, H/32, W/32])
    """
    def __init__(
        self,
        img_size,
        in_channels=1,
        pretrained_path=None,
        out_indices: Sequence[int] = (2, 5, 11),
        feat_dim: int = 768,
        neck_dim: int = 768
    ):
        super().__init__()
        self.core = ViTAdapter(img_size, in_channels, pretrained_path)
        self.out_indices = tuple(out_indices)
        assert all(0 <= i < len(self.core.vit.blocks) for i in self.out_indices)

        # simple 1x1 projectors
        self.proj = nn.ModuleList([nn.Conv2d(feat_dim, neck_dim, 1) for _ in range(3)])

        # up/down modules to obtain 8x/32x from 16x
        self.to_8x  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    nn.Conv2d(neck_dim, neck_dim, 3, padding=1))
        self.to_32x = nn.Conv2d(neck_dim, neck_dim, 3, stride=2, padding=1)
        
        self.num_channels = neck_dim
        self.patch_align = None

    @torch.no_grad()
    def _grid_hw(self, x: torch.Tensor) -> Tuple[int, int]:
        # recompute H',W' from patch embed result
        z = self.core.vit.patch_embed(x)
        return z.shape[-2], z.shape[-1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pe = self.core.vit.patch_embed
        H_in, W_in = x.shape[-2], x.shape[-1]
        if getattr(pe, "img_size", None) is None or pe.img_size != (H_in, W_in):
            ps = pe.proj.kernel_size[0]  # 16
            pe.img_size    = (H_in, W_in)
            pe.grid_size   = (H_in // ps, W_in // ps)
            pe.num_patches = pe.grid_size[0] * pe.grid_size[1]
            if hasattr(pe, "strict_img_size"):
                pe.strict_img_size = False

        z = pe(x)

        embed_dim = self.core.vit.cls_token.shape[-1]
        if z.dim() == 3:
            # NLC -> NCHW
            B, N, C = z.shape
            Hg, Wg = pe.grid_size
            assert N == Hg * Wg, f"N={N} != {Hg}*{Wg}"
            z = z.transpose(1, 2).reshape(B, C, Hg, Wg)  # [B,C,H',W']
        elif z.dim() == 4:
            if z.shape[-1] == embed_dim:
                z = z.permute(0, 3, 1, 2).contiguous()  # [B,C,H',W']
        else:
            raise RuntimeError(f"Unexpected patch_embed output shape: {z.shape}")

        B = x.shape[0]
        H, W = z.shape[-2:]



        Cz = z.shape[1]
        if Cz != embed_dim:
            if (self.patch_align is None
                or self.patch_align.in_channels != Cz
                or self.patch_align.out_channels != embed_dim):
                self.patch_align = nn.Conv2d(Cz, embed_dim, kernel_size=1, bias=False).to(z.device)
            z = self.patch_align(z) 



        pos_embed = self.core.vit.pos_embed
        if pos_embed.shape[1] != 1 + H * W:
            pos_embed = _resize_pos_embed(pos_embed, (H, W))
        pos_embed = pos_embed.to(z.dtype).to(z.device)

        tokens = z.flatten(2).transpose(1, 2)               # [B, H'W', embed_dim]
        cls_token = self.core.vit.cls_token.expand(B, -1, -1).to(z.dtype).to(z.device)
        x_tok = torch.cat((cls_token, tokens), dim=1)
        x_tok = self.core.vit.pos_drop(x_tok + pos_embed)



        grabbed = []
        for i, blk in enumerate(self.core.vit.blocks):
            x_tok = blk(x_tok)
            if i in self.out_indices:
                grabbed.append(x_tok[:, 1:, :])
        x_tok = self.core.vit.norm(x_tok)
        while len(grabbed) < 3:
            grabbed.append(x_tok[:, 1:, :])

        feats16 = [_to_2d(g, (H, W)) for g in grabbed]      # [B,C,H',W']
        f_shallow, f_mid, f_deep = feats16[0], feats16[1], feats16[2]

        c4 = self.proj[1](f_mid)                            # stride 16
        c3 = self.proj[0](f_shallow); c3 = self.to_8x(c3)   # stride  8
        c5 = self.proj[2](f_deep);    c5 = self.to_32x(c5)  # stride 32

        return {"c3": c3, "c4": c4, "c5": c5}




# ----------------------------
# builder
# ----------------------------
def build_vit_adapter(name: str = "vitadapter_b16",
                      img_size=None,
                      in_channels: int = 1,
                      pretrained_path: str = None,
                      out_indices: Sequence[int] = (2, 5, 11),
                      **kwargs) -> nn.Module:
    """
    forward(x: Tensor[B, C, H, W]) → dict(c3,c4,c5) 的多尺度特征。其中 C3≈8x、C4≈16x、C5≈32x
    """
    return ViTAdapterMS(
        img_size=img_size,
        in_channels=in_channels,
        pretrained_path=pretrained_path,
        out_indices=out_indices,
        feat_dim=768,
        neck_dim=kwargs.get("neck_dim", 768),
    )




class BackboneBase_ViTAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        self.body = backbone  # ViT-Adapter 主体
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

        self.fpn = FeatsFusion(
            C3_size=num_channels, C4_size=num_channels, C5_size=num_channels,
            hidden_size=num_channels, out_size=num_channels, out_kernel=3
        )

    def forward(self, tensor_list: NestedTensor):
        """
          - Input:  NestedTensor(tensors, mask)
          - Output: dict {'4x': NestedTensor, '8x': NestedTensor}
        """
        assert self.return_interm_layers, "ViTAdapter only works under return_interm_layers=True"

        xs = tensor_list.tensors
        # ViT-Adapter 主体应返回 {'c3','c4','c5'}
        feats = self.body(xs)
        assert isinstance(feats, dict) and all(k in feats for k in ("c3", "c4", "c5")), \
            "ViT-Adapter 需要返回 dict(c3,c4,c5)"

        P3, P4, P5 = self.fpn([feats["c3"], feats["c4"], feats["c5"]])

        features_8x = P3
        features_4x = F.interpolate(P3, scale_factor=2, mode="bilinear", align_corners=False)

        m = tensor_list.mask
        mask_8x = F.interpolate(m[None].float(), size=features_8x.shape[-2:], mode="nearest").to(torch.bool)[0]
        mask_4x = F.interpolate(m[None].float(), size=features_4x.shape[-2:], mode="nearest").to(torch.bool)[0]

        out = {
            "4x": NestedTensor(features_4x, mask_4x),  # 64x64
            "8x": NestedTensor(features_8x, mask_8x),  # 32x32
        }
        return out


class Backbone_ViTAdapter(BackboneBase_ViTAdapter):
    def __init__(self, name: str, return_interm_layers: bool, in_channels: int = 3):
        backbone = build_vit_adapter(name=name, neck_dim=256, in_channels=in_channels)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[NestedTensor] = {}
        pos = {}
        for name, x in xs.items():
            out[name] = x
            # position encoding（与你的 VGG Joiner 一致）
            pos[name] = self[1](x).to(x.tensors.dtype)
        return out, pos


def build_backbone_vitadapter(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_ViTAdapter(args.backbone, True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == "__main__":
    Backbone_ViTAdapter("vitadapter_b16", True)