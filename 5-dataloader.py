import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
# 使用 PyTorch 的 DataLoader 来加载测试集数据，并配置了批量处理、数据打乱、多线程等参数。
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

# tensorboard 显示
writer = SummaryWriter("dataloader")
for epoch in range(2): # 两轮取数据，"shuffle=True"时顺序会不一样。epoch：通常表示当前训练的轮次（整数，从 0 或 1 开始）。
    step = 0 # 第几次取图片，一次 64 张
    for data in test_loader:
        imgs, targets = data # 64 张图片和标签分别打包在一起
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step) # str.format() 方法：将变量按顺序填充到字符串的 {} 占位符中。
        step = step + 1

writer.close()


