from torch.utils.data import Dataset, DataLoader
"""
torch.utils.data 是 PyTorch 中用于数据加载和预处理的核心模块，提供了高效的数据集（Dataset）和数据加载器（DataLoader）的实现。
Dataset 类:表示数据集的抽象类，自定义数据集需要继承 torch.utils.data.Dataset 并实现以下方法：
    __len__()：返回数据集大小。
    __getitem__()：根据索引返回一个样本（如张量、图像、标签等）。
DataLoader 类:对 Dataset 进行包装，提供以下功能：
    批处理：通过 batch_size 参数将样本组合成批次。
    打乱数据：通过 shuffle=True 在每个 epoch 开始时随机打乱数据。
    多线程加载：通过 num_workers 参数使用多进程加速数据加载（避免主进程阻塞）。
    内存优化：通过 pin_memory=True 加速 GPU 数据传输（需配合 CUDA）。
"""
import numpy as np
from PIL import Image # Pillow 是一个广泛使用的图像处理库，提供了丰富的功能来打开、操作和保存多种图像格式（如 JPEG、PNG、BMP 等）。
import os # os 模块常用于文件管理、路径拼接、环境变量读取等任务。
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter("logs")

class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform):
        # 函数变量一般不通用，self. 相当于指定一个类中的全局变量
        self.root_dir = root_dir # 根目录：dataset 中的 trains："dataset/trains"

        self.image_dir = image_dir # 图片文件夹名称："ants_image"
        self.image_path = os.path.join(self.root_dir, self.image_dir) # 路径拼接："dataset/trains\\ants_image"
        self.image_list = os.listdir(self.image_path) # 返回目录下的文件和子目录列表，此处生成一个 str 列表（图片名称）

        self.label_dir = label_dir # 标签文件夹名称："ants_label"
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.label_list = os.listdir(self.label_path) # 同理得到标签文件名称的列表

        self.transform = transform

        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx): # idx 此处为序号
        img_name = self.image_list[idx] # 从 image_list 中获取第 idx 个图片的名称："XXX.jpg"
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name) # 路径拼接："dataset/trains\\ants_image\\XXX.jpg"
        img = Image.open(img_item_path)  # 按路径打开图片

        label_name = self.label_list[idx]  # 获取第 idx 个标签文件的名称
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        with open(label_item_path, 'r') as f: # 以只读模式（'r'）打开路径 label_item_path 指定的文件。with 语句：确保文件在使用后自动关闭。
            label = f.readline() #读取文件的第一行（包括换行符 \n，如果存在）。

        # img = np.array(img)
        img = self.transform(img)
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self): # 图片的数量
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # 根目录
    root_dir = "dataset/train"
    # ants 数据集
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    # bees 数据集
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    # 拼接两个数据集
    train_dataset = ants_dataset + bees_dataset

    # transforms = transforms.Compose([transforms.Resize(256, 256)])
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    writer.add_image('error', train_dataset[119]['img'])
    writer.close()
    # for i, j in enumerate(dataloader):
    #     # imgs, labels = j
    #     print(type(j))
    #     print(i, j['img'].shape)
    #     # writer.add_image("train_data_b2", make_grid(j['img']), i)
    #
    # writer.close()



