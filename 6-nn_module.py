# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from torch import nn

# 基础神经网络搭建
class Tudui(nn.Module):
    def __init__(self): # 类的构造函数，用于初始化模型的层（Layers）和参数。
        super().__init__() # 调用父类 nn.Module 的 __init__() 方法，确保 PyTorch 正确初始化模型的基础功能（如参数注册）。

    def forward(self, input): # 是 PyTorch nn.Module 子类中必须实现的方法，用于定义数据如何通过模型。
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)