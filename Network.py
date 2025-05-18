
import torch.nn as nn
import torch


class Net(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(Net, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        # (3,16,56,56)
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv1_dw = nn.Conv3d(3, 3, kernel_size=(
            3, 3, 3), padding=(1, 1, 1), groups=3, bias=True)
        self.relu1_dw = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1_pw = nn.Conv3d(3, 64, kernel_size=(1, 1, 1), bias=True)
        self.relu1_pw = nn.ReLU()
        # (64,16,28,28)

        # self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2_dw = nn.Conv3d(64, 64, kernel_size=(
            3, 3, 3), padding=(1, 1, 1), groups=64, bias=True)
        self.relu2_dw = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2_pw = nn.Conv3d(64, 128, kernel_size=(1, 1, 1), bias=True)
        self.relu2_pw = nn.ReLU()
        # (128,8,14,14)

        # self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3a_dw = nn.Conv3d(128, 128, kernel_size=(
            3, 3, 3), padding=(1, 1, 1), groups=128, bias=True)
        self.relu3a_dw = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a_pw = nn.Conv3d(128, 256, kernel_size=(1, 1, 1), bias=True)
        self.relu3a_pw = nn.ReLU()
        # self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv3b_dw = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=256,bias=True)
        # self.relu3b_dw = nn.ReLU()
        # self.conv3b_pw = nn.Conv3d(256, 256, kernel_size=(1, 1, 1),bias=True)
        # self.relu3b_pw = nn.ReLU()
        # (256,4,7,7)

        # self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4a_dw = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), padding=(1, 1, 1), groups=256, bias=True)
        self.relu4a_dw = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.conv4a_pw = nn.Conv3d(256, 512, kernel_size=(1, 1, 1), bias=True)
        self.relu4a_pw = nn.ReLU()
        # self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b_dw = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=512,bias=True)
        # self.relu4b_dw = nn.ReLU()
        # self.conv4b_pw = nn.Conv3d(512, 512, kernel_size=(1, 1, 1),bias=True)
        # self.relu4b_pw = nn.ReLU()
        # (512,2,4,4)

        # self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5a_dw = nn.Conv3d(512, 512, kernel_size=(
            3, 3, 3), padding=(1, 1, 1), groups=512, bias=True)
        self.relu5a_dw = nn.ReLU()
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a_pw = nn.Conv3d(512, 512, kernel_size=(1, 1, 1), bias=True)
        self.relu5a_pw = nn.ReLU()

        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv5b_dw = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=512,bias=True)
        # self.relu5b_dw = nn.ReLU()
        # self.conv5b_pw = nn.Conv3d(512, 512, kernel_size=(1, 1, 1),bias=True)
        # self.relu5b_pw = nn.ReLU()

        self.fc6 = nn.Linear(2048, 2048)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(2048, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        # self.relu = nn.ReLU()
        self.dequant = torch.ao.quantization.DeQuantStub()

        # if pretrained:
        #    self.__load_pretrained_weights()

    def forward(self, x):
        # Stack 1
        x = self.quant(x)
        x = self.conv1_dw(x)
        x = self.relu1_dw(x)
        x = self.pool1(x)
        x = self.conv1_pw(x)
        x = self.relu1_pw(x)

        # Stack 2
        x = self.conv2_dw(x)
        x = self.relu2_dw(x)
        x = self.pool2(x)
        x = self.conv2_pw(x)
        x = self.relu2_pw(x)

        # Stack 3
        x = self.conv3a_dw(x)
        x = self.relu3a_dw(x)
        x = self.pool3(x)
        x = self.conv3a_pw(x)
        x = self.relu3a_pw(x)

        # Stack 4
        x = self.conv4a_dw(x)
        x = self.relu4a_dw(x)
        x = self.pool4(x)
        x = self.conv4a_pw(x)
        x = self.relu4a_pw(x)

        # Stack 5
        x = self.conv5a_dw(x)
        x = self.relu5a_dw(x)
        x = self.pool5(x)
        x = self.conv5a_pw(x)
        x = self.relu5a_pw(x)
        # Stack 7
        x = x.contiguous().view(-1, 2048)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout(x)

        # Stack 8
        x = self.fc7(x)
        logits = self.dequant(x)
        return logits
