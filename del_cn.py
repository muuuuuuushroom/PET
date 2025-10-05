#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PIL import Image

# 替换规则
REPLACEMENTS = {
    '资': 'zi',
    '农': 'nong',
}

def apply_replacements(name: str) -> str:
    """替换文件名中的中文字符"""
    base, ext = os.path.splitext(name)
    for k, v in REPLACEMENTS.items():
        base = base.replace(k, v)
    return base + ext

def ensure_rgb(img: Image.Image) -> Image.Image:
    """确保图像为 RGB 格式"""
    return img.convert("RGB")

def convert_and_delete_tif(src_path: str):
    """将 tif 转为 jpg 并删除原文件"""
    dirpath, filename = os.path.split(src_path)
    base, _ = os.path.splitext(filename)
    jpg_path = os.path.join(dirpath, base + ".jpg")

    with Image.open(src_path) as im:
        im = ensure_rgb(im)
        im.save(jpg_path, format="JPEG", quality=95, optimize=True)

    os.remove(src_path)  # 删除原 tif
    print(f"[OK] Converted and deleted: {src_path} -> {jpg_path}")
    return jpg_path

def rename_if_needed(path: str) -> str:
    """按规则重命名文件（覆盖同名文件，不加括号）"""
    dirpath, filename = os.path.split(path)
    new_name = apply_replacements(filename)
    if new_name == filename:
        return path

    new_path = os.path.join(dirpath, new_name)
    if os.path.exists(new_path):
        os.remove(new_path)
    os.rename(path, new_path)
    print(f"[REN] {path} -> {new_path}")
    return new_path

def delete_parentheses_files(root: str):
    """检测并删除文件名中包含()的文件"""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if '(' in fn or ')' in fn:
                path = os.path.join(dirpath, fn)
                try:
                    os.remove(path)
                    print(f"[DEL] Removed file with parentheses: {path}")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {path}: {e}")

def process_folder(root: str):
    """主逻辑"""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            ext = os.path.splitext(fn)[1].lower()
            try:
                if ext in ['.tif', '.tiff']:
                    jpg_path = convert_and_delete_tif(path)
                    rename_if_needed(jpg_path)
                else:
                    rename_if_needed(path)
            except Exception as e:
                print(f"[ERROR] Failed for {path}: {e}")

    # 最后进行一次检测与清理
    delete_parentheses_files(root)
    print("✅ 检测完成，所有包含 () 的文件已删除。")

if __name__ == "__main__":
    root = r"/root/autodl-tmp/data"  # 修改为你的A文件夹路径
    process_folder(root)
    print("✅ 全部完成！")
