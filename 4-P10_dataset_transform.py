import torchvision
from torch.utils.tensorboard import SummaryWriter

# 转换 tensor 数据类型操作
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 训练集
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
# 测试集
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
"""
使用 PyTorch 的 torchvision.datasets.CIFAR10 加载 CIFAR-10 数据集（训练集）
并指定了数据存储路径、数据变换（transform）和自动下载选项。
"""

# 这部分和转换 tensor 操作不共存，用于查看原 PIL 图像
"""
# 打印测试集第一个样本（图像张量和标签）
print(test_set[0])
# 打印 CIFAR-10 的类别名称列表
print(test_set.classes)
# 解包第一个样本的图像和标签
img, target = test_set[0]
# 打印图像张量（经过 transform 处理后的）
print(img)
# 打印标签（整数）
print(target)
# 根据标签索引获取类别名称
print(test_set.classes[target])
# 显示图像（需确保 img 是 PIL 图像或张量已转为可显示格式）
img.show()
"""

# Tensorboard 显示
writer = SummaryWriter("p10")

for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()