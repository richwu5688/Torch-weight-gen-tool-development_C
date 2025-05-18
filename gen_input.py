import argparse
import torch
import numpy as np
import os
import re
from quant_verify import save_array_to_txt


def save_array_to_txt(array, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if array is None:
        print(f"警告: 儲存 {filename} 時遇到空陣列")
        with open(filename, 'w') as f:
            array = np.zeros(1)
            array_str = np.array2string(
                array,
                separator=', ',    # 用逗號和空格分隔
                precision=8,       # 8位小數
                suppress_small=True,  # 抑制科學記號
                threshold=np.inf,   # 顯示所有元素
                max_line_width=np.inf  # 不換行
            )
            f.write(array_str)
            f.close()
        return
    if torch.is_tensor(array):
        if (array.dtype == torch.qint8) or (array.dtype == torch.quint8):
            # print(array)
            array = array.int_repr().numpy()
        else:
            array = array.detach().cpu().numpy()

        with open(filename, 'w') as f:
            array_str = np.array2string(
                array,
                separator=', ',    # 用逗號和空格分隔
                precision=20,       # 8位小數
                suppress_small=True,  # 抑制科學記號
                threshold=np.inf,   # 顯示所有元素
                max_line_width=np.inf  # 不換行
            )
            f.write(array_str)
            f.close()
    else:
        array = np.array(array)
        with open(filename, 'w') as f:
            array_str = np.array2string(
                array,
                separator=', ',    # 用逗號和空格分隔
                precision=20,       # 8位小數
                suppress_small=True,  # 抑制科學記號
                threshold=np.inf,   # 顯示所有元素
                max_line_width=np.inf  # 不換行
            )
            f.write(array_str)
            f.close()


def save_input():
    # (1,3,16,56,56)
    input = torch.randint(low=254, high=255, size=(1, 3, 16, 56, 56))
    torch.save(input, "./input/input.pt")
    save_array_to_txt(input, "./input/input.txt")
