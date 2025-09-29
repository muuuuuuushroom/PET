# stat_code_lines.py
 
from glob import glob
import pandas as pd
import os
import numpy as np
 
# 设置项目根目录
project_root = "/data/zlt/RemoteSensePET"
# 设置项目所用的编程语言后缀名列表
source_file_suffix_list = [".py", ".cpp", ".h"]
 
source_file_path_list = []
code_lines_list = []
for root, _, _ in os.walk(project_root):
    for suffix in source_file_suffix_list:
        for source_file_path in glob(root + "/*" + suffix):
            source_file_path_list.append(source_file_path)
            with open(source_file_path, "r", encoding="utf-8") as fr:
                code_content = fr.readlines()
                code_lines_list.append(len(code_content))
 
source_file_path_list.append("总计")
code_lines_list.append(np.sum(code_lines_list))
 
source_file_with_code_lines = pd.DataFrame({"源文件":source_file_path_list,
                                            "代码行数":code_lines_list})
 
print(source_file_with_code_lines)