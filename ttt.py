from pathlib import Path
import re

p = Path("/root/PET/models/backbones/adapter.py")
src = p.read_text(encoding="utf-8")

# 在文件顶端导入 nested_tensor_from_tensor_list（若不存在也不报错）
if "nested_tensor_from_tensor_list" not in src:
    src = src.replace(
        "from util.misc import NestedTensor",
        "from util.misc import NestedTensor\ntry:\n    from util.misc import nested_tensor_from_tensor_list\nexcept Exception:\n    nested_tensor_from_tensor_list = None"
    )

# 用更健壮的 forward 实现替换 BackboneBase_ViTAdapter.forward
pattern = r"(class\s+BackboneBase_ViTAdapter[^\n]*:\n(?:.*\n)*?)\n\s*def\s+forward\(\s*self\s*,\s*tensor_list[^\)]*\)\s*:\n(?:.*\n)*?return\s+out\n"
m = re.search(pattern, src, flags=re.DOTALL)
assert m, "Could not locate BackboneBase_ViTAdapter.forward to patch."

prefix = m.group(1)
new_forward = f"""{prefix}
    def forward(self, tensor_list):
        \"\"\"接受 NestedTensor / Tensor / list[tensor]，统一转成 NestedTensor 再前向。\"\"\"
        # 1) 统一成 NestedTensor
        if not isinstance(tensor_list, NestedTensor):
            import torch
            if nested_tensor_from_tensor_list is not None and isinstance(tensor_list, (list, tuple)):
                tensor_list = nested_tensor_from_tensor_list(tensor_list)
            elif torch.is_tensor(tensor_list):
                # 假设输入 [B,3,H,W]
                B, _, H, W = tensor_list.shape
                mask = torch.zeros(B, H, W, dtype=torch.bool, device=tensor_list.device)
                tensor_list = NestedTensor(tensor_list, mask)
            else:
                raise TypeError(f"Expect Tensor/NestedTensor, got: {{type(tensor_list)}}")

        xs = tensor_list.tensors  # [B, 3, H, W]

        # ViT-Adapter 原生输出: [f1, f2, f3, f4] -> 4x / 8x / 16x / 32x
        f1, f2, f3, f4 = self.body(xs)

        # 对齐到与 VGG-FPN 相同的两个尺度输出（4x 与 8x），并统一通道到 256
        y4 = self.proj_4x(f1)  # stride 4
        y8 = self.proj_8x(f2)  # stride 8

        # mask 最近邻插值到对应分辨率
        m = tensor_list.mask
        assert m is not None, "NestedTensor.mask must not be None."
        import torch.nn.functional as F
        mask_4x = F.interpolate(m[None].float(), size=y4.shape[-2:], mode="nearest").to(torch.bool)[0]
        mask_8x = F.interpolate(m[None].float(), size=y8.shape[-2:], mode="nearest").to(torch.bool)[0]

        out = {{}}
        from util.misc import NestedTensor as _NT
        out['4x'] = _NT(y4, mask_4x)
        out['8x'] = _NT(y8, mask_8x)
        return out
"""

src = src[:m.start()] + new_forward + src[m.end():]
p.write_text(src, encoding="utf-8")
print("Patched:", p)