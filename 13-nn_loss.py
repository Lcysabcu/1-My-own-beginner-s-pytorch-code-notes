# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum') # 默认绝对值取平均，"sum"只相加
result = loss(inputs, targets)
print(result) # 输出 2.

loss_mse = nn.MSELoss() # 均方差
result_mse = loss_mse(inputs, targets)
print(result_mse) # 输出 1.3333


x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1]) # target 为第 1 类
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss() # 交叉熵：用于分类问题
result_cross = loss_cross(x, y)
print(result_cross) # 输出 1.1019