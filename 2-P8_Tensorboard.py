from torch.utils.tensorboard import SummaryWriter
"""
是 PyTorch 提供的用于可视化训练过程的模块，集成了 TensorBoard（一个由 TensorFlow 开发的强大可视化工具）。
通过 SummaryWriter，你可以将训练中的标量（如损失、准确率）、图像、直方图、模型结构等记录到日志文件，并在 TensorBoard 中实时查看。
"""
import numpy as np
from PIL import Image

writer = SummaryWriter("logs") # 指定日志保存目录（默认生成在 ./runs/ 下），终端命令行"tensorboard --logdir=logs --port=6007"打开

# writer.add_scalar 数值绘图
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i) # 记录标量：标题，y轴，x轴。数值更新前最好删去 logs 文件夹下文件，重新 run，再打开

# writer.add_image 显示图片
image_path = "data/train/ants_image/6240329_72c01e663e.jpg" # 定义相对路径
img_PIL = Image.open(image_path) # 以 PIL 的 image 类型打开图片
img_array = np.array(img_PIL) # 转换为 numpy.ndarray 型
print(type(img_array)) # 输出 img_array 的类型
print(img_array.shape) # 输出 img_array 的图片尺寸
writer.add_image("train", img_array, 1, dataformats='HWC') # 记录图像：标题，图片对象，全局步骤（同一个标题下的不同对象），更改尺寸适配（高度、宽度、通道）

writer.close() # 关闭 writer