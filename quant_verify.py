from time import sleep
import torch
import torch.ao.quantization
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.ao.quantization import qconfig
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import numpy as np
from torchsummary import summary

from params_process import params_process
import ast
import shutil


###############################################################################################
# This function is used to save the data to txt file
# And read the data from txt file

torch.backends.quantized.engine = "fbgemm"


class qunat_model_tool():
    def __init__(self, Net, model_path, qconfig=None, backends="fbgemm", fuse=None, input_tensor=None):
        """
        This function is used to initialize the model and load the model

        :param Net: The model you want to load
        :param model_path: The path of the model you want to load
        :param qconfig: The quantization config you want to use
        :param backends: The backend you want to use default is fbgemm
        :param fuse: The fuse dict you want to use. And in the eager mode of pytorch, it should manually set the fuse.
        :param input_tensor: The input tensor you want to use. It should be the same shape as the model input shape
        """
        self.activation = {}
        self.device = self.__to_cpu()
        self.Net = Net
        if qconfig is None:
            # If your model is PTQ model,you can change this fuction to get_default_qconfig
            qconfig = torch.ao.quantization.get_default_qat_qconfig(backends)
        else:
            self.Net.qconfig = qconfig
        if fuse is None:
            raise ValueError(
                "fuse is None, In the eager mode of pytorch,it should manually set the fuse!")
        else:
            self.model_fuse = torch.ao.quantization.fuse_modules(
                self.Net, fuse)
        # If your model is PTQ model,you can change this function to prepare()
        self.model = torch.ao.quantization.prepare_qat(self.model_fuse)
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))
        self.model_int = torch.ao.quantization.convert(
            self.model)
        self.model_int.eval().to(self.device)
        self.input_tensor = input_tensor.to(self.device)

    def save_layer_output(self, out_path="./output/"):
        """
        Save the output of each layer to txt file
    
        :param out_path: The path to save the output of each layer
        """
        print("Saving the layer output...")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for name, layer in self.model_int.named_modules():
            layer.register_forward_hook(
                self.__get_activation(name))
        model_output = self.model_int(self.input_tensor)
        for layer, activation in self.activation.items():
            if not layer:
                pass
            else:
                # 将激活值保存为 txt 文件而不是 pt 文件
                save_array_to_txt(activation, out_path + layer + ".txt")
    
        print("Done!")

    def __delete_dump(self):

        shutil.rmtree('weight/')

    def save_params(self, keep_weight=False):
        """
        Save the parameters of the model to txt file

        :param keep_weight: If you want to keep the weight, you can set it to True otherwise it will be deleted
        """
        print("Saving the parameters...")
        if not os.path.exists("weight/"):
            os.makedirs("weight/")
        if not os.path.exists("params/"):
            os.makedirs("params/")
        for name in self.model_int.state_dict():
            if self.model_int.state_dict()[name] is None or isinstance(self.model_int.state_dict()[name], torch.dtype):
                pass
            elif "_packed_params._packed_params" in name:
                weight, bias = self.model_int.state_dict()[name]
                # 将参数保存为 txt 文件
                save_array_to_txt(weight.int_repr(), f"params/{name}_weight.txt")
                save_array_to_txt(weight.int_repr(), f"weight/{name}_weight.txt")
                save_array_to_txt(weight.q_scale(), f"weight/{name}_scale.txt")
                save_array_to_txt(weight.q_zero_point(), f"weight/{name}_zero_points.txt")
                save_array_to_txt(bias, f"weight/{name}_bias.txt")
                save_array_to_txt(bias, f"params/{name}_bias.txt")
            elif self.model_int.state_dict()[name].is_quantized:
                save_array_to_txt(self.model_int.state_dict()[name].q_scale(), f"weight/{name}_scale.txt")
                save_array_to_txt(self.model_int.state_dict()[name].q_zero_point(), f"weight/{name}_zero_points.txt")
                save_array_to_txt(self.model_int.state_dict()[name].int_repr(), f"weight/{name}.txt")
                save_array_to_txt(self.model_int.state_dict()[name].int_repr(), f"params/{name}.txt")
                weight = self.model_int.state_dict()[name].int_repr()
                save_array_to_txt(weight, f"weight/{name}_a.txt")
            else:
                save_array_to_txt(self.model_int.state_dict()[name], f"weight/{name}.txt")
        params_process()  # 确保处理更新为 txt 文件的参数
        if keep_weight:
            pass
        else:
            self.__delete_dump()  # 如果不保留权重，则删除
        print("Done!")

    def __to_cpu(self):
        device = torch.device('cpu')
        return device

    def __get_activation(self, name):

        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def __device_check(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def show(self):
        """
            Show the model summary
        """
        print(summary(self.model,
              input_size=self.input_tensor.shape[1:], device="cpu"))


def save_array_to_txt_(array, filename: str):
    """
    將陣列保存到文字檔，每行一個數字，沒有括號和其他符號
    
    Args:
        array: numpy陣列或torch張量
        filename: 輸出文件名
    """
    import os
    import numpy as np
    import torch
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 處理空陣列
    if array is None:
        print(f"警告: 儲存 {filename} 時遇到空陣列")
        with open(filename, 'w') as f:
            f.write("0\n")  # 寫入單個0並換行
        return
    
    # 將torch張量轉換為numpy陣列
    if torch.is_tensor(array):
        if (array.dtype == torch.qint8) or (array.dtype == torch.quint8):
            array = array.int_repr().numpy()
        else:
            array = array.detach().cpu().numpy()
    else:
        array = np.array(array)
    
    # 將陣列展平為一維
    flat_array = array.flatten()
    
    # 將數字寫入文件，每行一個
    with open(filename, 'w') as f:
        for value in flat_array:
            f.write(f"{value}\n")



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
    with open("weight/" + file_path, "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        f.close()
    return tensor_data
###############################################################################################


###############################################################################################
# You can modify the model here
# (e.g., add batch normalization, change the number of layers, etc.)
# Or just copy the model from the training code. : )


###############################################################################################

###############################################################################################
# This function is used to get the output of each layer in the model
#


###############################################################################################
