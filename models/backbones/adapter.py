"""
Backbone: ViT-Adapter (VIT-Small) in VGG-style Wrapper

- 输入: NestedTensor(tensors, mask)
- 输出: Dict[str, NestedTensor]，键为 '4x' 与 '8x'
- num_channels: 统一映射为 256（与 vgg16_bn 的 FPN 输出通道数保持一致）

依赖:
- util.misc.NestedTensor
- position_encoding.build_position_encoding
- .vit_adapter.ViTAdapter
"""

from typing import Dict
import math
import torch
import torch.nn.functional as F
from torch import nn

from typing import Optional, List
from torch import Tensor
from util.misc import NestedTensor

try:
    from util.misc import nested_tensor_from_tensor_list
except Exception:
    nested_tensor_from_tensor_list = None

# class NestedTensor(object):
#     def __init__(self, tensors, mask: Optional[Tensor]):
#         self.tensors = tensors
#         self.mask = mask

#     def to(self, device):
#         # # type: (Device) -> NestedTensor # noqa
#         cast_tensor = self.tensors.to(device)
#         mask = self.mask
#         if mask is not None:
#             assert mask is not None
#             cast_mask = mask.to(device)
#         else:
#             cast_mask = None
#         return NestedTensor(cast_tensor, cast_mask)

#     def decompose(self):
#         return self.tensors, self.mask

#     def __repr__(self):
#         return str(self.tensors)

from ..position_encoding import build_position_encoding
# from ..position_encoding import build_position_encoding

# 直接复用你项目中的 ViT-Adapter 主体实现
from .vit_adapter import ViTAdapter  # f1,f2,f3,f4 -> strides 4/8/16/32，通道为 embed_dim


class _Conv1x1(nn.Module):
    """简洁的 1x1 映射层，做通道对齐到指定维度。"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        # Kaiming 初始化，与项目中 conv 初始化风格一致
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        return self.proj(x)


class BackboneBase_ViTAdapter(nn.Module):
    """
    参考 BackboneBase_VGG 的组织方式：
      - 内部持有一个具体骨干（这里是 ViTAdapter）
      - forward 输入 NestedTensor，输出与 VGG 版保持相同的 dict（'4x'、'8x'）
      - 做好 mask 的双线性插值对齐
    """
    def __init__(
        self,
        vit: nn.Module,
        num_channels: int = 256,      # 与 backbone_vgg.py 的 FPN 输出对齐
        return_interm_layers: bool = True
    ):
        super().__init__()
        self.body = vit
        self.return_interm_layers = return_interm_layers

        # ViT-Adapter 的多尺度输出通道均为 embed_dim（ViT-S: 384）
        embed_dim = getattr(vit, "embed_dim", None)
        if embed_dim is None:
            raise AttributeError("ViTAdapter must have attribute 'embed_dim'.")

        # 为 4x/8x 两个尺度做 1x1 投影到 256 通道，以完全对齐 VGG FPN 输出规格
        self.proj_4x = _Conv1x1(embed_dim, num_channels)
        self.proj_8x = _Conv1x1(embed_dim, num_channels)

        self.num_channels = num_channels

    @torch.no_grad()
    def _resize_mask(self, mask: torch.Tensor, size_hw):
        """将 mask 插值到目标特征分辨率，保证 NestedTensor 一致性。"""
        # 输入 mask: [B, H, W] -> 先扩维后插值
        m = F.interpolate(mask[None].float(), size=size_hw, mode="nearest").to(torch.bool)[0]
        return m

    def forward(self, tensor_list):
        """接受 NestedTensor / Tensor / list[Tensor]，统一转成 NestedTensor 再前向。"""

        # ---- 新增：输入适配，替换你原来的 assert ----
        if not isinstance(tensor_list, NestedTensor):
            import torch
            # 若上层传来的是 list/tuple（可选依赖，如果项目里提供了该函数就用它）
            if ('nested_tensor_from_tensor_list' in globals()
                    and nested_tensor_from_tensor_list is not None
                    and isinstance(tensor_list, (list, tuple))):
                tensor_list = nested_tensor_from_tensor_list(tensor_list)
            # 若是 batched Tensor: [B, 3, H, W]
            elif torch.is_tensor(tensor_list):
                B, _, H, W = tensor_list.shape
                mask = torch.zeros(B, H, W, dtype=torch.bool, device=tensor_list.device)
                tensor_list = NestedTensor(tensor_list, mask)
            else:
                raise TypeError(f"Expect Tensor/NestedTensor, got: {type(tensor_list)}")
        # ---- 输入适配结束 ----

        xs = tensor_list.tensors  # [B, 3, H, W]

        # ViT-Adapter 原生输出: [f1, f2, f3, f4]  -> 4x / 8x / 16x / 32x
        f1, f2, f3, f4 = self.body(xs)

        # 对齐到与 VGG-FPN 相同的两个尺度输出（4x 与 8x），并统一通道到 256
        y4 = self.proj_4x(f1)  # stride 4
        y8 = self.proj_8x(f2)  # stride 8

        # 根据特征图空间大小，对 mask 做最近邻插值到对应分辨率
        m = tensor_list.mask
        mask_4x = self._resize_mask(m, y4.shape[-2:])
        mask_8x = self._resize_mask(m, y8.shape[-2:])

        out = {}
        out['4x'] = NestedTensor(y4, mask_4x)
        out['8x'] = NestedTensor(y8, mask_8x)
        return out



class Backbone_ViTAdapter(BackboneBase_ViTAdapter):

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,      # ViT-S
        depth: int = 12,           # ViT-S
        num_heads: int = 6,        # ViT-S
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        # Adapter 相关默认参数（与原实现一致/合理）
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: int = 6,
        init_values: float = 0.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 1.0,
        add_vit_feature: bool = True,
        use_extra_extractor: bool = True,
        # 交互块索引（将 12 层 Transformer 均分为 3 组）
        interaction_indexes = ((0, 3), (4, 7), (8, 11)),
        # 输出通道对齐
        num_channels: int = 256,
        return_interm_layers: bool = True,
    ):
        # 将 (s,e) 索引形式统一转成 List[List[int]]，与原实现匹配
        inter_idx = []
        for s, e in interaction_indexes:
            inter_idx.append(list(range(s, e + 1)))

        vit = ViTAdapter(
            pretrain_size=img_size,
            num_heads=num_heads,                 # 传入父类 VIT 的 heads
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            init_values=init_values,
            interaction_indexes=inter_idx,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_extractor=use_extra_extractor,
            # ---- 下面这些传给 TIMMVisionTransformer (父类) 的常规 ViT 参数 ----
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,             # 若父类签名使用 num_heads，则前面已用；若使用自定义名则兜底
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

        super().__init__(vit=vit, num_channels=num_channels, return_interm_layers=return_interm_layers)


class Joiner(nn.Sequential):
    """
    与 VGG 版本相同：级联 backbone 与 position embedding，并返回 (out_dict, pos_dict)
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # Dict[str, NestedTensor]
        out: Dict[str, NestedTensor] = {}
        pos = {}
        for name, x in xs.items():
            out[name] = x
            pos[name] = self[1](x).to(x.tensors.dtype)  
        return out, pos


def _interpolate_pos_embed(pos_embed, old_size_hw, new_size_hw):
    """
    pos_embed: [1, N+cls, C] 或 [1, N, C]
    old_size_hw/new_size_hw: (H, W) 的格点数，H*W=N
    返回与 new_size_hw 匹配的 pos_embed，若存在 cls_token 则原样保留。
    """
    # 拆 cls_token（若存在）
    has_cls = pos_embed.shape[1] != (old_size_hw[0] * old_size_hw[1])
    if has_cls:
        cls_tok, pos = pos_embed[:, :1], pos_embed[:, 1:]
    else:
        pos = pos_embed
        cls_tok = None

    C = pos.shape[-1]
    pos = pos.reshape(1, old_size_hw[0], old_size_hw[1], C).permute(0, 3, 1, 2)  # [1,C,H,W]
    pos = F.interpolate(pos, size=new_size_hw, mode='bicubic', align_corners=False)
    pos = pos.permute(0, 2, 3, 1).reshape(1, new_size_hw[0] * new_size_hw[1], C)

    if has_cls and cls_tok is not None:
        pos = torch.cat([cls_tok, pos], dim=1)
    return pos


def load_dinov3_into_vit_adapter(vit, ckpt_path, img_size=224, patch_size=16, verbose=True):
    """
    将 DINOv3 的 ViT-S/16 权重加载到 ViT-Adapter 的 ViT 主干（vit）里。
    - vit: 传入 backbone.body（ViTAdapter 实例）
    - ckpt_path: DINOv3 权重路径
    - img_size/patch_size: 当前模型的输入格点大小，用于 pos_embed 插值
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 兼容常见字段：'model' / 'teacher' / 'student' / 直接就是 state_dict
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("model") or
            ckpt.get("teacher") or
            ckpt.get("student") or
            ckpt
        )
    else:
        state_dict = ckpt

    # 只取 ViT 主干相关权重；DINOv3 常见命名已与 timm 一致（patch_embed/blocks/norm 等）
    new_sd = {}
    for k, v in state_dict.items():
        # 1) 过滤掉不相关/形状不匹配的键（如 head/分类器、optimizer 状态等）
        if any(x in k for x in ["head", "fc", "classifier", "pred"]):
            continue

        # 2) 常见前缀归一：有些权重会有 'backbone.'、'module.'、'encoder.' 前缀
        nk = k
        for pref in ["backbone.", "module.", "encoder."]:
            if nk.startswith(pref):
                nk = nk[len(pref):]

        # 3) ViT-Adapter 里 ViT 主干通常沿用 timm 命名；保留原名放进去
        new_sd[nk] = v

    # === 位置编码插值：若 pos_embed 存在且网格不匹配，做插值 ===
    if "pos_embed" in new_sd and isinstance(new_sd["pos_embed"], torch.Tensor):
        pos = new_sd["pos_embed"]
        # 旧网格：根据长度推断；考虑是否带 cls_token
        num_tokens = pos.shape[1]
        C = pos.shape[-1]
        # 当前图像网格（不含 cls）：H*W = (img_size/patch_size)^2
        new_hw = (img_size // patch_size, img_size // patch_size)

        # 旧网格估计
        has_cls = (num_tokens - 1) in [x * x for x in range(1, 1000)]
        if has_cls:
            N = num_tokens - 1
        else:
            N = num_tokens
        old_h = int(math.sqrt(N))
        old_w = N // old_h
        old_hw = (old_h, old_w)

        if old_hw != new_hw:
            with torch.no_grad():
                new_sd["pos_embed"] = _interpolate_pos_embed(pos, old_hw, new_hw)

    # === 严格程度：对不上（例如 Adapter 的额外模块）直接忽略，让它们随机初始化 ===
    missing, unexpected = vit.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f"[DINOv3] load completed. missing={len(missing)}, unexpected={len(unexpected)}")
        # 你也可以打印前若干项，帮助核对键名
        # print("  missing(excerpt):", missing[:10])
        # print("  unexpected(excerpt):", unexpected[:10])


def build_backbone_vitadapter(args):
    """
    Input:
      - img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias
      - conv_inplane, n_points, deform_num_heads, init_values, with_cffn, cffn_ratio,
        deform_ratio, add_vit_feature, use_extra_extractor, interaction_indexes
      - num_channels

    Return: 
      model: Joiner(backbone, position_embedding)
      model.num_channels = backbone.num_channels (= 256)
    """
    if not hasattr(args, "hidden_dim"):
        setattr(args, "hidden_dim", 256)
    if not hasattr(args, "position_embedding"):
        setattr(args, "position_embedding", "sine")
    position_embedding = build_position_encoding(args)

    kw = dict(
        img_size=getattr(args, "img_size", 224),
        patch_size=16, #getattr(args, "patch_size", 16),
        embed_dim=getattr(args, "embed_dim", 384),
        depth=getattr(args, "depth", 12),
        num_heads=getattr(args, "vit_num_heads", 6),
        mlp_ratio=getattr(args, "mlp_ratio", 4.0),
        qkv_bias=getattr(args, "qkv_bias", True),
        conv_inplane=getattr(args, "conv_inplane", 64),
        n_points=getattr(args, "n_points", 4),
        deform_num_heads=getattr(args, "deform_num_heads", 6),
        init_values=getattr(args, "init_values", 0.0),
        with_cffn=getattr(args, "with_cffn", True),
        cffn_ratio=getattr(args, "cffn_ratio", 0.25),
        deform_ratio=getattr(args, "deform_ratio", 1.0),
        add_vit_feature=getattr(args, "add_vit_feature", True),
        use_extra_extractor=getattr(args, "use_extra_extractor", True),
        interaction_indexes=getattr(args, "interaction_indexes", ((0,3),(4,7),(8,11))),
        num_channels=getattr(args, "num_channels", 256),
        return_interm_layers=True,
    )

    backbone = Backbone_ViTAdapter(**kw)
    
    ckpt_path = getattr(args, "pretrained_dinov3", None)
    if ckpt_path:
        print('loading dinov3_vits, pth:', ckpt_path)
        load_dinov3_into_vit_adapter(
            vit=backbone.body,           # 关键：ViT-Adapter 主体在 backbone.body 里
            ckpt_path=ckpt_path,
            img_size=getattr(args, "img_size", 224),
            patch_size=getattr(args, "vit_patch_size", 16),  # 确保是 16
            verbose=True
        )
        
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model




if __name__ == "__main__":

    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W)
    mask = torch.zeros(B, H, W, dtype=torch.bool)
    nt = NestedTensor(x, mask)

    class _Args: pass
    args = _Args()
    m = build_backbone_vitadapter(args)
    (outs, pos) = m(nt)
    print([k for k in outs.keys()], outs['4x'].tensors.shape, outs['8x'].tensors.shape)
