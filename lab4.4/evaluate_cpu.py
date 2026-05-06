from torchvision.models import vgg19
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import cv2
import numpy
import time
# 导入自定义动态链接库
from op_mysigmoid import op_exp


class COCODataSet(Dataset):

    def __init__(self):
        super(COCODataSet, self).__init__()
        self.zip_files = ZipFile('../data/train2014_small.zip')
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)
        return image


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        return torch.nn.functional.relu(self.layer(x) + x)


class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            ###################下采样层################
            nn.Conv2d(3, 32, kernel_size=9, padding=4, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            ##################残差层##################
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),

            ################上采样层##################
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),

            ###############输出层#####################
            nn.Conv2d(32, 3, kernel_size=9, padding=4),
        )

    def forward(self, x):
        x = self.layer(x)
        # 调用自定义 mysigmoid 算子对 x 进行处理得到输出结果 out
        out = op_exp.mysigmoid_cpu(x)
        return out


if __name__ == '__main__':
    # 使用 cpu 生成图像转换网络模型并保存在 g_net 中
    g_net = TransNet().to('cpu')
    # 从 /models 文件夹下加载网络参数到 g_net 中
    g_net.load_state_dict(torch.load('../models/fst.pth', map_location='cpu'))
    print("g_net build PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    batch_size = 1
    data_group = DataLoader(data_set, batch_size, True, drop_last=True)

    for i, image in enumerate(data_group):
        image_c = image.cpu()
        start = time.time()
        # 计算 g_net, 得到 image_g
        image_g = g_net(image_c)
        end = time.time()
        delta_time = end - start
        print("Inference (CPU) processing time: %s" % delta_time)
        # 利用 save_image 函数将生成图像 image_g 与输入图像 image_c 左右拼接保存
        save_image(torch.cat((image_c, image_g), -1), f'../out/cpu/{i}.jpg')
        break
    print("TEST RESULT PASS!\n")
