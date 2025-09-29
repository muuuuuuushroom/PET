import os
import math
from typing import Dict, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import vit_base_patch16_224

from util.misc import NestedTensor
from .backbone_vgg import FeatsFusion
from ..position_encoding import build_position_encoding

# ==========================================================
# helpers
# ==========================================================

def _resize_pos_embed(pos_embed: torch.Tensor, new_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize absolute pos embed [1, 1+N, C] to match new grid size (h,w), keep CLS.
    仅用于具有绝对位置编码的 ViT；DINOv3 使用 RoPE 时不会调用。
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
    """tokens [B, H'W', C] (no cls) -> [B, C, H', W']"""
    B, N, C = x_tokens.shape
    h, w = hw
    assert N == h * w, f"token count {N} != {h}*{w}"
    return x_tokens.transpose(1, 2).reshape(B, C, h, w)


def _map_first_conv_weight(w: torch.Tensor, in_ch: int) -> torch.Tensor:
    """将预训练的 patch_embed.proj.weight (out, 3, k, k) 映射到新 in_ch。
    - in_ch == 1: 按通道求和 → [out,1,k,k]
    - in_ch > 3: 重复/平分复制到目标通道数（简单但有效）。
    - in_ch == 3: 原样返回。
    """
    out_ch, old_in, k, _ = w.shape
    if in_ch == old_in:
        return w
    if in_ch == 1:
        return w.sum(dim=1, keepdim=True)
    if in_ch > old_in:
        reps = (in_ch + old_in - 1) // old_in
        w_rep = w.repeat(1, reps, 1, 1)[:, :in_ch, :, :]
        # 归一化复制带来的能量放大（可选）：
        w_rep = w_rep * (old_in / in_ch)
        return w_rep
    # in_ch < old_in 且 !=1，例如 in_ch=2：取前 in_ch 并按比例缩放
    w_cut = w[:, :in_ch, :, :]
    w_cut = w_cut * (old_in / in_ch)
    return w_cut


# ==========================================================
# Adapter modules (Houlsby-style): down -> GELU -> up, zero-init
# ==========================================================
class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, drop: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=True)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=True)
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()
        # 初始化：保证 up 权重为 0，初始不扰动主干
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.up(self.act(self.down(x))))


def _wrap_block_with_adapters(blk: nn.Module, dim: int, reduction: int = 16, drop: float = 0.0):
    """
    在 timm/自定义 ViT Block 中注入两个 adapter，并兼容没有 drop_path 成员的实现。
    I/O 形状与语义保持不变。
    """
    bneck = max(1, dim // reduction)
    blk.attn_adapter = BottleneckAdapter(dim, bneck, drop)
    blk.mlp_adapter = BottleneckAdapter(dim, bneck, drop)

    # 可能存在的层缩放（timm 支持）
    ls1 = getattr(blk, 'ls1', None)
    ls2 = getattr(blk, 'ls2', None)

    # 兼容不同实现的随机深度命名
    dp_attn = getattr(blk, 'drop_path1', None) or getattr(blk, 'drop_path', None)
    dp_mlp  = getattr(blk, 'drop_path2', None) or getattr(blk, 'drop_path', None)
    if dp_attn is None:
        dp_attn = nn.Identity()
    if dp_mlp is None:
        dp_mlp = dp_attn  # 若只有一个 drop_path，就两处共用

    def forward_with_adapters(x: torch.Tensor) -> torch.Tensor:
        # attn 路径
        h = blk.norm1(x)
        attn_out = blk.attn(h)
        if ls1 is not None:
            attn_out = ls1(attn_out)
        x = x + dp_attn(attn_out)
        x = x + dp_attn(blk.attn_adapter(h))

        # mlp 路径
        h2 = blk.norm2(x)
        mlp_out = blk.mlp(h2)
        if ls2 is not None:
            mlp_out = ls2(mlp_out)
        x = x + dp_mlp(mlp_out)
        x = x + dp_mlp(blk.mlp_adapter(h2))
        return x

    blk.forward = forward_with_adapters


# ==========================================================
# ViTAdapter with per-block adapters + 可选 DINOv3 主干
# ==========================================================
class ViTAdapter(nn.Module):
    """ViT-B/16 主干（或 DINOv3 的 ViT-B/16），支持 in_channels!=3、外部 ckpt 加载，
    在每个 Transformer block 内插入 Houlsby-style adapter。forward 返回去掉 CLS 的 tokens：
    [B, H'W', C]。

    通过 use_dinov3 / dinov3_variant 切换 DINOv3 预训练权重（timm）。
    """
    def __init__(
        self,
        img_size,
        in_channels: int = 1,
        pretrained_path: str = None,
        adapter_reduction: int = 16,
        adapter_drop: float = 0.0,
        enable_adapters: bool = True,
        freeze_backbone: bool = False,
        train_norm: bool = True,
        use_dinov3: bool = False,
        dinov3_variant: str = 'vit_base_patch16_dinov3.lvd1689m',
        dinov3_pretrained: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.use_dinov3 = use_dinov3

        # 1) 构建骨干
        if use_dinov3:
            self.vit = timm.create_model(dinov3_variant, pretrained=True)
            print('dinov3 base loaded')
        else:
            self.vit = vit_base_patch16_224(pretrained=False)

        embed_dim = self.vit.cls_token.shape[-1]

        # 2) 适配非 3 通道输入（替换 patch_embed.proj）
        if in_channels != 3:
            old_proj = self.vit.patch_embed.proj
            new_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=old_proj.kernel_size,
                                 stride=old_proj.stride, padding=old_proj.padding, bias=old_proj.bias is not None)
        
            if isinstance(old_proj, nn.Conv2d) and old_proj.weight is not None:
                with torch.no_grad():
                    mapped = _map_first_conv_weight(old_proj.weight.data, in_channels)
                    new_proj.weight.copy_(mapped)
                    if old_proj.bias is not None and new_proj.bias is not None:
                        new_proj.bias.copy_(old_proj.bias.data)
            self.vit.patch_embed.proj = new_proj

        # 3) 外部 ckpt（优先级：用户传入路径）
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # 第一层卷积通道对齐
            key = 'patch_embed.proj.weight'
            if key in state_dict and state_dict[key].shape[1] != in_channels:
                state_dict[key] = _map_first_conv_weight(state_dict[key], in_channels)
            self.vit.load_state_dict(state_dict, strict=False)

        # 4) 注入 adapters
        if enable_adapters:
            for blk in self.vit.blocks:
                _wrap_block_with_adapters(blk, dim=embed_dim, reduction=adapter_reduction, drop=adapter_drop)

        # 5) 可选冻结主干，仅训练 adapter（和可选的 LayerNorm）
        if freeze_backbone:
            for n, p in self.vit.named_parameters():
                if ('attn_adapter' in n) or ('mlp_adapter' in n):
                    p.requires_grad = True
                elif train_norm and ('.norm' in n or n.endswith('bias')):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tokens without cls: [B, H'W', C]（接口与形状保持不变）。"""
        B = x.shape[0]
        x = self.vit.patch_embed(x)                      # [B, C, H', W'] or [B, N, C]
        # 统一为 [B, H'W', C]
        if x.dim() == 4:  # [B,C,H',W'] or [B,H',W',C]
            if x.shape[1] < x.shape[-1]:  # [B,H',W',C]
                x = x.permute(0, 3, 1, 2).contiguous()
            H, W = x.shape[-2:]
            x = x.flatten(2).transpose(1, 2)             # [B, H'W', C]
        elif x.dim() == 3:  # [B,N,C]
            H = W = int((x.shape[1]) ** 0.5)
        else:
            raise RuntimeError(f"Unexpected patch_embed output shape: {x.shape}")

        tokens = x
        embed_dim = self.vit.cls_token.shape[-1]

        # 处理 CLS 与位置编码：DINOv3 无绝对 pos_embed（RoPE 在 attn 内部），老 ViT 则需要加 absPE
        has_cls = hasattr(self.vit, 'cls_token') and (self.vit.cls_token is not None)
        has_abs_pos = hasattr(self.vit, 'pos_embed') and (self.vit.pos_embed is not None)

        if has_cls:
            cls_token = self.vit.cls_token.expand(B, -1, -1).to(tokens.dtype).to(tokens.device)
            x_tok = torch.cat((cls_token, tokens), dim=1)  # [B, 1+H'W', C]
        else:
            x_tok = tokens

        if has_abs_pos:
            pos_embed = self.vit.pos_embed
            if pos_embed.shape[1] != x_tok.shape[1]:
                pos_embed = _resize_pos_embed(pos_embed, (H, W))
            x_tok = (self.vit.pos_drop(x_tok + pos_embed)
                     if hasattr(self.vit, 'pos_drop') else (x_tok + pos_embed))
        else:
            # DINOv3 路径：RoPE 在 attn 内部，无需手动加 pos_embed
            x_tok = self.vit.pos_drop(x_tok) if hasattr(self.vit, 'pos_drop') else x_tok

        x_tok = self.vit.blocks(x_tok)
        x_tok = self.vit.norm(x_tok)
        if has_cls:
            x_tok = x_tok[:, 1:, :]  # 去 CLS
        return x_tok  # [B, H'W', C]


# ==========================================================
# multi-scale wrapper（保持原接口与输出）
# ==========================================================
class ViTAdapterMS(nn.Module):
    """
    从 ViT 抓取中间层，构建轻量金字塔，近似 C3(≈8x), C4(≈16x), C5(≈32x)。

    输出：
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
        neck_dim: int = 768,
        adapter_reduction: int = 16,
        adapter_drop: float = 0.0,
        enable_adapters: bool = True,
        freeze_backbone: bool = False,
        train_norm: bool = True,
        use_dinov3: bool = False,
        dinov3_variant: str = 'vit_base_patch16_dinov3.lvd1689m',
        dinov3_pretrained: bool = True,
    ):
        super().__init__()
        self.core = ViTAdapter(
            img_size,
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            adapter_reduction=adapter_reduction,
            adapter_drop=adapter_drop,
            enable_adapters=enable_adapters,
            freeze_backbone=freeze_backbone,
            train_norm=train_norm,
            use_dinov3=use_dinov3,
            dinov3_variant=dinov3_variant,
            dinov3_pretrained=dinov3_pretrained,
        )
        self.out_indices = tuple(out_indices)
        assert all(0 <= i < len(self.core.vit.blocks) for i in self.out_indices)

        # 1x1 投影
        self.proj = nn.ModuleList([nn.Conv2d(feat_dim, neck_dim, 1) for _ in range(3)])
        # 从 1/16 得到 1/8 与 1/32
        self.to_8x  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    nn.Conv2d(neck_dim, neck_dim, 3, padding=1))
        self.to_32x = nn.Conv2d(neck_dim, neck_dim, 3, stride=2, padding=1)

        self.num_channels = neck_dim
        self.patch_align = None

    @torch.no_grad()
    def _grid_hw(self, x: torch.Tensor) -> Tuple[int, int]:
        z = self.core.vit.patch_embed(x)
        if z.dim() == 4:
            return z.shape[-2], z.shape[-1]
        else:
            n = z.shape[1]
            s = int(n ** 0.5)
            return s, s

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pe = self.core.vit.patch_embed
        H_in, W_in = x.shape[-2], x.shape[-1]
        # 兼容 timm PatchEmbed 的动态尺寸
        if getattr(pe, "img_size", None) is None or pe.img_size != (H_in, W_in):
            ps = pe.proj.kernel_size[0]  # 一般为 16
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
            if z.shape[-1] == embed_dim:  # NHWC -> NCHW
                z = z.permute(0, 3, 1, 2).contiguous()
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

        # tokens + (可选)绝对位置编码
        tokens = z.flatten(2).transpose(1, 2)               # [B, H'W', embed_dim]
        has_cls = hasattr(self.core.vit, 'cls_token') and (self.core.vit.cls_token is not None)
        has_abs_pos = hasattr(self.core.vit, 'pos_embed') and (self.core.vit.pos_embed is not None)

        if has_cls:
            cls_token = self.core.vit.cls_token.expand(B, -1, -1).to(z.dtype).to(z.device)
            x_tok = torch.cat((cls_token, tokens), dim=1)
        else:
            x_tok = tokens

        if has_abs_pos:
            pos_embed = self.core.vit.pos_embed
            if pos_embed.shape[1] != x_tok.shape[1]:
                pos_embed = _resize_pos_embed(pos_embed, (H, W))
            x_tok = (self.core.vit.pos_drop(x_tok + pos_embed)
                     if hasattr(self.core.vit, 'pos_drop') else (x_tok + pos_embed))
        else:
            x_tok = self.core.vit.pos_drop(x_tok) if hasattr(self.core.vit, 'pos_drop') else x_tok

        grabbed = []
        for i, blk in enumerate(self.core.vit.blocks):
            x_tok = blk(x_tok)
            if i in self.out_indices:
                grabbed.append(x_tok[:, 1:, :] if has_cls else x_tok)
        x_tok = self.core.vit.norm(x_tok)
        while len(grabbed) < 3:
            grabbed.append(x_tok[:, 1:, :] if has_cls else x_tok)

        feats16 = [_to_2d(g, (H, W)) for g in grabbed]      # [B,C,H',W']
        f_shallow, f_mid, f_deep = feats16[0], feats16[1], feats16[2]

        c4 = self.proj[1](f_mid)                            # stride 16
        c3 = self.proj[0](f_shallow); c3 = self.to_8x(c3)   # stride  8
        c5 = self.proj[2](f_deep);    c5 = self.to_32x(c5)  # stride 32

        return {"c3": c3, "c4": c4, "c5": c5}


# ==========================================================
# builder & DETR-style backbone wrappers
# ==========================================================

def build_vit_adapter(
    name: str = "vitadapter_b16",
    img_size=None,
    in_channels: int = 1,
    pretrained_path: str = None,
    out_indices: Sequence[int] = (2, 5, 11),
    **kwargs
) -> nn.Module:
    """forward(x: Tensor[B, C, H, W]) → dict(c3,c4,c5) 的多尺度特征。
    关键 kwargs：
      - neck_dim: int = 768
      - adapter_reduction: int = 16
      - adapter_drop: float = 0.0
      - enable_adapters: bool = True
      - freeze_backbone: bool = False
      - train_norm: bool = True
      - use_dinov3: bool = False
      - dinov3_variant: str = 'vit_base_patch16_dinov3.lvd1689m'
      - dinov3_pretrained: bool = True
    """
    return ViTAdapterMS(
        img_size=img_size,
        in_channels=in_channels,
        pretrained_path=pretrained_path,
        out_indices=out_indices,
        feat_dim=kwargs.get("feat_dim", 768),
        neck_dim=kwargs.get("neck_dim", 768),
        adapter_reduction=kwargs.get("adapter_reduction", 16),
        adapter_drop=kwargs.get("adapter_drop", 0.0),
        enable_adapters=kwargs.get("enable_adapters", True),
        freeze_backbone=kwargs.get("freeze_backbone", False),
        train_norm=kwargs.get("train_norm", True),
        use_dinov3=kwargs.get("use_dinov3", False),
        dinov3_variant=kwargs.get("dinov3_variant", 'vit_base_patch16_dinov3.lvd1689m'),
        dinov3_pretrained=kwargs.get("dinov3_pretrained", True),
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
        Input:  NestedTensor(tensors, mask)
        Output: dict {'4x': NestedTensor, '8x': NestedTensor}
        """
        assert self.return_interm_layers, "ViTAdapter only works under return_interm_layers=True"

        xs = tensor_list.tensors
        feats = self.body(xs)  # {'c3','c4','c5'}
        assert isinstance(feats, dict) and all(k in feats for k in ("c3", "c4", "c5")), \
            "ViT-Adapter 需要返回 dict(c3,c4,c5)"

        P3, P4, P5 = self.fpn([feats["c3"], feats["c4"], feats["c5"]])

        features_8x = P3
        features_4x = F.interpolate(P3, scale_factor=2, mode="bilinear", align_corners=False)

        m = tensor_list.mask
        mask_8x = F.interpolate(m[None].float(), size=features_8x.shape[-2:], mode="nearest").to(torch.bool)[0]
        mask_4x = F.interpolate(m[None].float(), size=features_4x.shape[-2:], mode="nearest").to(torch.bool)[0]

        out = {
            "4x": NestedTensor(features_4x, mask_4x),
            "8x": NestedTensor(features_8x, mask_8x),
        }
        return out


class Backbone_ViTAdapter(BackboneBase_ViTAdapter):
    def __init__(self, name: str, return_interm_layers: bool, in_channels: int = 3,
                 use_dinov3: bool = False, dinov3_variant: str = 'vit_base_patch16_dinov3.lvd1689m'):
        backbone = build_vit_adapter(name=name, neck_dim=256, in_channels=in_channels,
                                     use_dinov3=use_dinov3, dinov3_variant=dinov3_variant,
                                     dinov3_pretrained=True)
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
            pos[name] = self[1](x).to(x.tensors.dtype)
        return out, pos


def build_backbone_dinov3(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_ViTAdapter(args.backbone, True,
                                   in_channels=getattr(args, 'in_channels', 3),
                                   use_dinov3=True,
                                   dinov3_variant='vit_base_patch16_dinov3.lvd1689m')
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == "__main__":
    _ = Backbone_ViTAdapter("dinov3", True, in_channels=3,
                             use_dinov3=True, dinov3_variant='vit_base_patch16_dinov3.lvd1689m')
