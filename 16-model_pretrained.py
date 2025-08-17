# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False) # 未训练的模型参数
vgg16_true = torchvision.models.vgg16(pretrained=True) # 已预训练的模型参数

print(vgg16_true) # 观察模型结构

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),download=True)
# 原数据集 ImageNet 分类数为 1000 个，当前数据集分类数为 10 个，所以增加一层线性层
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10)) # 在名为 classifier 的 seq 类里增加一层
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10) # 修改
print(vgg16_false)


