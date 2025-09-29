from .backbone_vgg import build_backbone_vgg
from .vitadapter import build_backbone_vitadapter

__all__ = [
    'build_backbone_vgg',
    'build_backbone_vitadapter',
]