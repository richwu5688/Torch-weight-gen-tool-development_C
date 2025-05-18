from time import sleep
import torch


from params_process import params_process
import ast

from quant_verify import qunat_model_tool
from Network import Net
from gen_input import save_input
###############################################################################################
# The path of the QAT model and the int8 model
# You can modify the path here
QAT_float32_path = "./QAT.pth"
QAT_int8_path = "./QAT_int8.pth"


if __name__ == '__main__':
    ###############################################################################################
    #   In this case I use the QAT model convert to int8 model and extract the weight of each layer
    #   and the scale and zero_point of each layer
    ###############################################################################################
    #   At first we need to create the model and load the QAT model
    #   So you need to know the structure of the model

    model_fp32 = Net(num_classes=101)

    qat_config = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.MovingAverageMinMaxObserver,
            reduce_range=True,  # 設定量化範圍
            quant_min=0,       # 設定量化範圍 (0 ~ 255)
            quant_max=127,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        ),
        weight=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.MinMaxObserver,
            quant_min=-128,   # 設定權重範圍 (-128 ~ 127)
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
    )

    fuse = [['conv1_dw', 'relu1_dw'],
            ['conv1_pw', 'relu1_pw'],
            ['conv2_dw', 'relu2_dw'],
            ['conv2_pw', 'relu2_pw'],
            ['conv3a_dw', 'relu3a_dw'],
            ['conv3a_pw', 'relu3a_pw'],
            ['conv4a_dw', 'relu4a_dw'],
            ['conv4a_pw', 'relu4a_pw'],
            ['conv5a_dw', 'relu5a_dw'],
            ['conv5a_pw', 'relu5a_pw'],
            ['fc6', 'relu6']]
    #   The input shape is (1,3,16,56,56)
    save_input()
    with open("input/input.txt", "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        f.close()
    test_input = tensor_data
    model_tool = qunat_model_tool(
        model_fp32, QAT_float32_path, qconfig=qat_config, fuse=fuse, input_tensor=test_input)
    model_tool.show()
    model_tool.save_layer_output()
    model_tool.save_params(keep_weight=True)
