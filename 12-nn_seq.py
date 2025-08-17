# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential( # 顺序组合类
            Conv2d(3, 32, 5, padding=2), # 3*32*32 卷积：输入 2，输出 32，卷积核 5，扩充 2
            MaxPool2d(2), # 32*32*32 池化：池化核 2
            Conv2d(32, 32, 5, padding=2), # 32*16*16 卷积：输入 32，输出 32，卷积核 5，扩充 2
            MaxPool2d(2), # 32*16*16 池化：池化核 2
            Conv2d(32, 64, 5, padding=2), # 32*8*8 卷积：输入 32，输出 64，卷积核 5，扩充 2
            MaxPool2d(2), # 64*8*8 池化：池化核 2
            Flatten(), # 64*4*4 展平
            Linear(1024, 64),# 1024 线性层
            Linear(64, 10) # 64 线性层
        )

    def forward(self, x):
        x = self.model1(x)
        return x

tudui = Tudui()
print(tudui) # 输出网络相关参数

input = torch.ones((64, 3, 32, 32)) # 模拟对 CIFAR10 数据集操作
output = tudui(input)
print(output.shape)

writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input) # 形成导图
writer.close()
