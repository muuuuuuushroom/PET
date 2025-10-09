from .backbone_vgg import build_backbone_vgg
# from .vitadapter import build_backbone_vitadapter
# from .vit_in_adapter import build_backbone_vit_in_adapter
# from .DINOv3 import build_backbone_dinov3
from .adapter import build_backbone_vitadapter
from .vit import build_backbone_vitsmall

__all__ = [
    'build_backbone_vgg',
    'build_backbone_vitadapter',
    'build_backbone_vitsmall',
    # 'build_backbone_vit_in_adapter',
    # 'build_backbone_dinov3',
]