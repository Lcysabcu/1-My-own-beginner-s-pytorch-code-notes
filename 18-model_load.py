# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from model_save import *
# 方式1-》保存方式1，加载模型
import torchvision
from torch import nn

# 方式1. 加载模型
model = torch.load("vgg16_method1.pth")
print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False) # 先加载模型结构
vgg16.load_state_dict(torch.load("vgg16_method2.pth")) # 向模型中加载状态字典
# model = torch.load("vgg16_method2.pth") # 显示为一堆字典
print(vgg16)

# 陷阱：方式1加载
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load('tudui_method1.pth') # 无法获取该类，需能访问定义内容，仅可省略实例化对象步骤
print(model)