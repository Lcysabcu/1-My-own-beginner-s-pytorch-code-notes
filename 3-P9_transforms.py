from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
"""
是 PyTorch 中用于图像预处理和数据增强的核心模块。它提供了一系列常用的图像变换操作，可以将原始图像转换为适合深度学习模型输入的张量格式。
1.图像预处理（标准化、尺寸调整）
    transforms.Resize(size)：调整图像尺寸。
    transforms.CenterCrop(size)：中心裁剪。
    transforms.ToTensor()：将 PIL 图像或 NumPy 数组转为 PyTorch 张量（范围 [0, 1]）。
    transforms.Normalize(mean, std)：标准化张量（例如 mean=[0.5], std=[0.5] 将范围变为 [-1, 1]）。
2. 数据增强（训练时随机变换）
    transforms.RandomHorizontalFlip()：随机水平翻转。
    transforms.RandomRotation(degrees)：随机旋转。
    transforms.ColorJitter()：随机调整亮度、对比度等。
3. 组合变换
    transforms.Compose([...])：将多个变换按顺序组合。
"""

# 加载 PIL 图像
img_path = "dataset/train/ants/0013035.jpg" # 假设图像是 RGB 格式
img = Image.open(img_path)

# 加载 narrays 图像
cv_img = cv2.imread(img_path)

# 转换为张量 ToTensor
tensor_trans = transforms.ToTensor() # transforms.ToTensor 是一个类（Class），它的实例化对象才是可调用的（实现了 __call__ 方法）。
tensor_img = tensor_trans(img) # 输出形状: [3, H, W], 值范围 [0.0, 1.0]
print(tensor_img.shape)        # 示例输出: torch.Size([3, 224, 224])
print(tensor_img.dtype)        # 输出: torch.float32

#  对图像张量进行标准化 Normalization，将像素值从 [0, 1] 范围映射到 [-1, 1] 范围。计算公式 = output = (input − mean)/std
norm_trans = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
norm_img = norm_trans(tensor_img)
print(tensor_img[0][0][0])
print(norm_img[0][0][0]) # 此处公式为 2*i-1

# 路径终端下先后输入“conda activate d2l”、“tensorboard --logdir=logs”，
writer = SummaryWriter("logs")
writer.add_image("tensor_img",tensor_img)


# 调整尺寸 Resize，此处是 PIL 类型
print(img.size)
resize_trans = transforms.Resize((512,512))
resize_img = resize_trans(img)
print(resize_img)

# 多个变换按顺序组合 Compose：先调整尺寸，后转换类型
resize_trans2 = transforms.Resize((512))
compose_trans = transforms.Compose([resize_trans2,tensor_trans])
compose_img = compose_trans(img)

# 随机裁剪 RandomCrop
random_trans = transforms.RandomCrop(512)
compose_trans2 = transforms.Compose([random_trans,tensor_trans])
for i in range(10):
    crop_img = compose_trans2(img)
    writer.add_image("Crop_img", crop_img,i)

writer.close()