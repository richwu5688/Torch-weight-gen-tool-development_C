import torch
import torch.ao.quantization
import torch.nn.functional as F
import sys
import numpy as np
import ast
import os
from torch.ao.nn.quantized import functional as qF
import torch.nn as nn
import timeit


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


def read_txt(file_path):
    tensor_data = torch.load("params/" + file_path)
    # with open("weight/" + file_path, "r") as f:
    #     content = f.read()
    #     data = ast.literal_eval(content)
    #     tensor_data = torch.tensor(data, dtype=torch.float32)
    #     f.close()
    return tensor_data


def mse_loss(data1, data2):
    squared_diff = (data1 - data2) ** 2
    return np.mean(squared_diff.numpy())


def linearqunat(input, M0: torch.Tensor, weight: torch.Tensor,  za_bias: torch.Tensor, zeropoint: torch.Tensor) -> torch.Tensor:
    mult = torch.matmul(input, weight.t())
    print(mult.shape)
    print(za_bias.shape)
    out = torch.round(M0 * (mult+za_bias)) + zeropoint
    out = torch.clamp(out, min=0, max=127)
    return out


def Conv3dquant(input, M0: torch.Tensor,  za_bias: torch.Tensor, zeropoint: torch.Tensor) -> torch.Tensor:
    # input_x_weight = quant_input * weight
    # shift = 2 ** (-32)
    b, c, t, h, w = input.shape
    output = torch.round(
        M0 * (input+za_bias.view(c, 1, 1, 1))) + zeropoint
    output = torch.clamp(output, min=0, max=127)
    return output


if __name__ == '__main__':
    # 读取参数D
    quant_scale = read_txt("quant.scale.pt")
    quant_zero_point = read_txt("quant.zero_point.pt")
    conv1_dw_M0 = read_txt("mult/conv1_dw_m0.pt")
    # conv1_dw_za_bias = read_txt("conv1_dw_za_bias.pt")
    conv1_dw_weight = read_txt("conv1_dw.weight.pt")
    conv1_dw_bias = read_txt("bias/conv1_dw_bias.pt")

    conv1_pw_M0 = read_txt("mult/conv1_pw_m0.pt")
    # conv1_pw_za_bias = read_txt("conv1_pw_za_bias.pt")
    conv1_pw_weight = read_txt("conv1_pw.weight.pt")
    conv1_pw_bias = read_txt("bias/conv1_pw_bias.pt")

    conv2_dw_M0 = read_txt("mult/conv2_dw_m0.pt")
    # conv2_dw_za_bias = read_txt("conv2_dw_za_bias.pt")
    conv2_dw_weight = read_txt("conv2_dw.weight.pt")
    conv2_dw_bias = read_txt("bias/conv2_dw_bias.pt")

    conv2_pw_M0 = read_txt("mult/conv2_pw_m0.pt")
    # conv2_pw_za_bias = read_txt("conv2_pw_za_bias.pt")
    conv2_pw_weight = read_txt("conv2_pw.weight.pt")
    conv2_pw_bias = read_txt("bias/conv2_pw_bias.pt")

    conv3a_dw_M0 = read_txt("mult/conv3a_dw_m0.pt")
    # conv3a_dw_za_bias = read_txt("conv3a_dw_za_bias.pt")
    conv3a_dw_weight = read_txt("conv3a_dw.weight.pt")
    conv3a_dw_bias = read_txt("bias/conv3a_dw_bias.pt")

    conv3a_pw_M0 = read_txt("mult/conv3a_pw_m0.pt")
    # conv3a_pw_za_bias = read_txt("conv3a_pw_za_bias.pt")
    conv3a_pw_weight = read_txt("conv3a_pw.weight.pt")
    conv3a_pw_bias = read_txt("bias/conv3a_pw_bias.pt")

    conv4a_dw_M0 = read_txt("mult/conv4a_dw_m0.pt")
    # conv4a_dw_za_bias = read_txt("conv4a_dw_za_bias.pt")
    conv4a_dw_weight = read_txt("conv4a_dw.weight.pt")
    conv4a_dw_bias = read_txt("bias/conv4a_dw_bias.pt")

    conv4a_pw_M0 = read_txt("mult/conv4a_pw_m0.pt")
    # conv4a_pw_za_bias = read_txt("conv4a_pw_za_bias.pt")
    conv4a_pw_weight = read_txt("conv4a_pw.weight.pt")
    conv4a_pw_bias = read_txt("bias/conv4a_pw_bias.pt")

    conv5a_dw_M0 = read_txt("mult/conv5a_dw_m0.pt")
    # conv5a_dw_za_bias = read_txt("conv5a_dw_za_bias.pt")
    conv5a_dw_weight = read_txt("conv5a_dw.weight.pt")
    conv5a_dw_bias = read_txt("bias/conv5a_dw_bias.pt")

    conv5a_pw_M0 = read_txt("mult/conv5a_pw_m0.pt")
    # conv5a_pw_za_bias = read_txt("conv5a_pw_za_bias.pt")
    conv5a_pw_weight = read_txt("conv5a_pw.weight.pt")
    conv5a_pw_bias = read_txt("bias/conv5a_pw_bias.pt")

    fc6_M0 = read_txt("mult/fc6_m0.pt")
    # fc6_za_bias = read_txt("fc6_za_bias.pt")
    fc6_weight = read_txt("fc6._packed_params._packed_params_weight.pt")
    fc6_bias = read_txt("bias/fc6_bias.pt")
    fc7_M0 = read_txt("mult/fc7_m0.pt")
    # fc7_za_bias = read_txt("fc7_za_bias.pt")
    fc7_weight = read_txt("fc7._packed_params._packed_params_weight.pt")
    fc7_bias = read_txt("bias/fc7_bias.pt")

    input = torch.load("./input/input.pt")

    start_time = timeit.default_timer()
    next = torch.clamp(
        torch.round(input/quant_scale+quant_zero_point), min=0, max=127)
    next = next.to(torch.int8)
    model_output = torch.load("./output/quant.pt")
    print(
        f"The final error of cal is {mse_loss(next, model_output.int_repr())}")
    x = next
    next = (next - quant_zero_point)
    print(next.to(torch.float))
    conv1_dw_weight = conv1_dw_weight.to(torch.float)
    print(conv1_dw_weight)
    # Conv1_dw
    next = F.conv3d(next.to(torch.float), conv1_dw_weight,
                    bias=conv1_dw_bias.to(torch.float), padding=(1, 1, 1), stride=(1, 1, 1), groups=3)

    next = torch.round(
        next * (conv1_dw_M0))

    next = F.relu(next)
    model_output = torch.load("./output/conv1_dw.pt")
    print(mse_loss(next, model_output.int_repr()))
    next = F.max_pool3d(next, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    # Conv1_pw
    next = F.conv3d(next.to(torch.float), conv1_pw_weight.to(torch.float),
                    bias=conv1_pw_bias.to(torch.float))

    next = torch.round(
        (next) * (conv1_pw_M0))

    next = F.relu(next)
    model_output = torch.load("./output/conv1_pw.pt")
    print(mse_loss(next, model_output.int_repr()))

    next = F.conv3d(next.to(torch.float), conv2_dw_weight.to(torch.float),
                    bias=conv2_dw_bias.to(torch.float), padding=(1, 1, 1), stride=1, groups=64)
    next = (next*conv2_dw_M0).round()
    next = F.relu(next)
    model_output = torch.load("./output/conv2_dw.pt")
    print(mse_loss(next, model_output.int_repr()))

    # Max_pool2d
    next = F.max_pool3d(next, kernel_size=(2, 2, 2), stride=(2, 2, 2))

    # Conv2_pw
    next = F.conv3d(next.to(torch.double),
                    conv2_pw_weight.to(torch.double), bias=conv2_pw_bias.to(torch.double))
    next = (next*conv2_pw_M0).round()
    next = torch.clamp(next, min=0, max=127)
    model_output = torch.load("./output/conv2_pw.pt")
    print(mse_loss(next, model_output.int_repr()))
    next = F.relu(next)
    # Conv3a_dw
    next = F.conv3d(next.to(torch.float), conv3a_dw_weight.to(torch.float),
                    bias=conv3a_dw_bias.to(torch.float), padding=(1, 1, 1), stride=1, groups=128)
    next = (next*conv3a_dw_M0).round()
    next = torch.clamp(next, min=0, max=127)

    model_output = torch.load("./output/conv3a_dw.pt")
    print(mse_loss(next, model_output.int_repr()))
    next = F.relu(next)
    # Max_pool2d
    next = F.max_pool3d(next, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    # Conv3a_pw
    next = F.conv3d(next.to(torch.float),
                    conv3a_pw_weight.to(torch.float), bias=None)
    next = Conv3dquant(next, M0=conv3a_pw_M0,
                       za_bias=conv3a_pw_bias, zeropoint=0)
    model_output = torch.load("./output/conv3a_pw.pt")
    print(mse_loss(next, model_output.int_repr()))
    next = F.relu(next)
    # Conv4a_dw
    next = F.conv3d(next.to(torch.float), conv4a_dw_weight.to(torch.float),
                    bias=None, padding=(1, 1, 1), stride=1, groups=256)
    next = Conv3dquant(next, M0=conv4a_dw_M0,
                       za_bias=conv4a_dw_bias, zeropoint=0)
    model_output = torch.load("./output/conv4a_dw.pt")
    print(mse_loss(next, model_output.int_repr()))
    next = F.relu(next)
    # Max_pool2d
    next = F.max_pool3d(next, kernel_size=(2, 2, 2),
                        stride=(2, 2, 2), padding=(0, 1, 1))
    # Conv4a_pw
    next = F.conv3d(next.to(torch.float),
                    conv4a_pw_weight.to(torch.float), bias=None)
    next = Conv3dquant(next, M0=conv4a_pw_M0,
                       za_bias=conv4a_pw_bias, zeropoint=0)
    model_output = torch.load("./output/conv4a_pw.pt")
    print(mse_loss(next, model_output.int_repr()))
    next = F.relu(next)
    # Conv5a_dw
    next = F.conv3d(next.to(torch.float), conv5a_dw_weight.to(torch.float),
                    bias=conv5a_dw_bias.to(torch.float), padding=(1, 1, 1), stride=1, groups=512)
    next = Conv3dquant(next, M0=conv5a_dw_M0,
                       za_bias=conv5a_dw_bias, zeropoint=0)
    model_output = torch.load("./output/conv5a_dw.pt")
    print(mse_loss(next, model_output.int_repr()))
    next = F.relu(next)
    # Max_pool2d
    next = F.max_pool3d(next, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    # Conv5a_pw
    next = F.conv3d(next.to(torch.float),
                    conv5a_pw_weight.to(torch.float), bias=None)
    next = Conv3dquant(next, M0=conv5a_pw_M0,
                       za_bias=conv5a_pw_bias, zeropoint=0)
    next = F.relu(next)
    # print(next)
    model_output = torch.load("./output/conv5a_pw.pt")
    print(mse_loss(next, model_output.int_repr()))
    # fc6

    next = next.view(-1, 2048)
    next = linearqunat(next, M0=fc6_M0, weight=fc6_weight.to(torch.float),
                       za_bias=fc6_bias, zeropoint=0)
    next = F.relu(next)
    model_output = torch.load("./output/fc6.pt")
    print(mse_loss(next, model_output.int_repr()))
    # fc7
    fc7_scale = torch.load("weight/fc7.scale.pt")
    fc7_zero_point = torch.load("weight/fc7.zero_point.pt")
    next = linearqunat(next, M0=fc7_M0, weight=fc7_weight.to(torch.float),
                       za_bias=fc7_bias, zeropoint=fc7_zero_point)
    model_output = torch.load("./output/fc7.pt")
    print(mse_loss(next, model_output.int_repr()))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")
    out = (next-fc7_zero_point)*fc7_scale
    stop_time = timeit.default_timer()
    model_output = torch.load("output/dequant.pt")
    print(mse_loss(out, model_output))
    print(nn.Softmax(dim=1)(out))
    print(repr(nn.Softmax(dim=1)(model_output)))
    print(torch.allclose(out, model_output.to(torch.float)))
    print(
        f"The final error of cal is {mse_loss(nn.Softmax(dim=1)(out), nn.Softmax(dim=1)(model_output))}")
