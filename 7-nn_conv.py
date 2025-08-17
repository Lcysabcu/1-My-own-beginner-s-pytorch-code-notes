# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torch
import torch.nn.functional as F

# 类似图像输入的矩阵
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
# 尺寸变换为四维张量：（batch 维度，通常表示 1 个样本），（通道维度，如单通道灰度图像），（高度），（宽度）
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1) # 步长1
print(output)

output2 = F.conv2d(input, kernel, stride=2) # 步长2
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1) # 步长1，四周填充一层0
print(output3)

