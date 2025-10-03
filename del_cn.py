import os
import re

def remove_chinese_chars_from_filenames(folder_path):
    """
    遍历指定文件夹下的所有文件，移除文件名中的中文字符，并重命名。

    :param folder_path: 目标文件夹的路径
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在或不是一个有效的目录。")
        return

    print(f"正在扫描文件夹: {folder_path}\n")

    # 定义匹配中文字符的正则表达式
    # \u4e00-\u9fff 是中日韩统一表意文字（CJK Unified Ideographs）的基本范围
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')

    # 遍历文件夹中的所有条目
    for filename in os.listdir(folder_path):
        # 构造文件的完整路径
        old_filepath = os.path.join(folder_path, filename)

        # 只处理文件，跳过子文件夹
        if not os.path.isfile(old_filepath):
            continue

        # 使用正则表达式移除文件名中的所有中文字符
        new_filename = re.sub(chinese_char_pattern, '', filename)

        # 检查文件名是否发生了变化
        if filename != new_filename:
            # 构造新的完整文件路径
            new_filepath = os.path.join(folder_path, new_filename)

            # 安全检查：如果新文件名已存在，则跳过以防覆盖
            if os.path.exists(new_filepath):
                print(f"跳过 '{filename}' -> '{new_filename}' (目标文件名已存在)")
                continue
            
            # 执行重命名
            try:
                os.rename(old_filepath, new_filepath)
                print(f"成功: '{filename}' -> '{new_filename}'")
            except Exception as e:
                print(f"错误: 重命名 '{filename}' 时出错: {e}")

    print("\n处理完成。")

# --- 主程序入口 ---
if __name__ == "__main__":
    # ##############################################
    # 请将这里替换为你的目标文件夹路径
    # ##############################################
    target_folder = '/root/PET/data/soybean/images' 
    
    # 示例:
    # Windows: 'C:\\path\\to\\your\\folder' 或 r'C:\path\to\your\folder'
    # macOS / Linux: '/path/to/your/folder'
    
    remove_chinese_chars_from_filenames(target_folder)