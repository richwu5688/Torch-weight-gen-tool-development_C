import os
import re
import ast
import torch
import numpy as np

from torch._higher_order_ops.out_dtype import out_dtype


def read_txt(file_path):
    with open("weight/" + file_path, "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        f.close()
    return tensor_data


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

    # 設定資料夾路徑

# 遍歷資料夾內所有檔案


def params_process(folder_path="./params/", weight_path="./weight/"):
    layers = set()
    file = set()
    for filename in os.listdir(weight_path):
        match = re.match(r"^(.*?)\.", filename)
        file.add(filename)
        if match:
            layers.add(match.group(1))

    # 輸出所有已出現的名稱
    bias = -1
    scale = -1
    zero_point = -1
    weight_bias = -1
    weight_scale = -1
    weight_zero_point = -1
    pre_scale = -1
    if not os.path.exists(folder_path + "mult"):
        os.makedirs(folder_path + "mult")
    if not os.path.exists(folder_path + "bias"):
        os.makedirs(folder_path + "bias")
    layers = ['quant', 'conv1_dw', 'conv1_pw', 'conv2_dw', 'conv2_pw', 'conv3a_dw',
              'conv3a_pw', 'conv4a_dw', 'conv4a_pw', 'conv5a_dw', 'conv5a_pw', 'fc6', 'fc7', 'dequant']
    for name in layers:
        matched_files = []
        for filename in file:
            # 檢查是否有名稱是 filename 的開頭
            if filename.startswith(name) and filename.endswith(".pt"):
                matched_files.append(filename)
        for activate in matched_files:
            value = torch.load(weight_path+activate)
            if (activate.endswith("weight_bias.pt") or activate.endswith("_packed_params_bias.pt")):
                weight_bias = value
            elif (activate.endswith("bias.pt")):
                bias = value
            if (activate.endswith("_packed_params_scale.pt")):
                weight_scale = value
            elif (activate.endswith("weight_scale.pt")):
                weight_scale = value
            elif (activate.endswith("scale.pt")):
                scale = value
            if (activate.endswith("weight_zero_points.pt")):
                weight_zero_point = value
            elif (activate.endswith("zero_point.pt")):
                zero_point = value

        if name == 'quant':
            pre_scale = scale
        elif name == 'dequant':
            pass
        elif 'fc' in name:

            torch.save((pre_scale*weight_scale)/scale,
                       folder_path + f"mult/{name}_m0.pt")
            bias_scale = pre_scale*weight_scale
            bias_fp32 = weight_bias
            bias_i32 = out_dtype(torch.ops.aten.div.Tensor,
                                 torch.int32, bias_fp32, bias_scale)

            torch.save(bias_i32,
                       folder_path + f"bias/{name}_bias.pt")
            pre_scale = scale
        else:
            torch.save((pre_scale*weight_scale)/scale,
                       folder_path + f"mult/{name}_m0.pt")
            bias_scale = pre_scale*weight_scale
            bias_fp32 = bias
            bias_i32 = out_dtype(torch.ops.aten.div.Tensor,
                                 torch.int32, bias_fp32, bias_scale)
            torch.save(bias_i32,
                       folder_path + f"bias/{name}_bias.pt")

            pre_scale = scale
