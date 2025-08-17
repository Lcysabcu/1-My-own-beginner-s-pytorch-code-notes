# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 获取数据集
dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64)

# 定义神经网络类
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 定义二维卷积
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

writer = SummaryWriter("../logs")

tudui = Tudui()
step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # 查看原图片尺寸 torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)  # Tensorboard 查看图片

    output = tudui(imgs) # 对图片进行神经网络操作
    print(output.shape) # 查看卷积后图片尺寸 torch.Size([64, 6, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30)) # output 通道数为 6，不能直接打开。将张量 output 重新调整为四维形状 (batch_size, 3, 30, 30)，其中 -1 表示自动计算该维度的大小。
    writer.add_images("output", output, step) # Tensorboard 查看图片

    step = step + 1

writer.close()


