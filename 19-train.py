# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torchvision
from torch.utils.data import DataLoader
from model import * # Tudui 模型定义放在了 model 文件里
from torch import nn
from torch.utils.tensorboard import SummaryWriter


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),download=True)

# 数据集 length 长度
train_data_size = len(train_data) # 50000
test_data_size = len(test_data) # 10000
# format 用法：如果train_data_size=50000, 训练数据集的长度为：50000
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集：打包一次 64 张
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数：交叉熵
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2 # 学习率
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate) # 随机梯度下降

# 设置训练网络的一些参数
total_train_step = 0 # 记录训练的次数
total_test_step = 0 # 记录测试的次数
epoch = 10 # 训练的轮数

# 添加 tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1)) # 提示

    # 训练步骤开始
    tudui.train() # 进入训练模式（只改变特定层）
    for data in train_dataloader:
        imgs, targets = data # 取数据
        outputs = tudui(imgs) # 使用模型，输出
        loss = loss_fn(outputs, targets) # 计算损失函数

        # 优化器优化模型
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 优化器调优

        total_train_step = total_train_step + 1 # 训练次数 +1
        if total_train_step % 100 == 0: # 百倍轮次时输出，减少无用信息，方便观察关键信息
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item())) # .item() 将 tensor 数据类型转换为纯数字
            writer.add_scalar("train_loss", loss.item(), total_train_step) # tensorboard 显示

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    tudui.eval() # # 进入训练模式（只改变特定层，如关闭 Dropout、BatchNorm 的随机性）
    with torch.no_grad(): # 临时禁用梯度计算
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item() # 计算测试集整体 loss
            accuracy = (outputs.argmax(1) == targets).sum() # .argmax(1) 横向比较取最大值的索引。.sum() 将 bool 值转换为 01
            total_accuracy = total_accuracy + accuracy

    # 一轮测试完毕
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step) # tensorboard 显示
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step) # tensorboard 显示
    total_test_step = total_test_step + 1

    # 模型保存
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
