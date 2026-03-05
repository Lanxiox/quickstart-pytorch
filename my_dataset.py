import os  # 导入操作系统模块，用于文件和目录操作
from PIL import Image  # 从Pillow库中导入Image类，用于图像读取和处理
from torch.utils.data import Dataset  # 从PyTorch中导入Dataset基类，用于自定义数据集

class MNISTImageDataset(Dataset):  # 定义一个继承自Dataset的自定义数据集类
    def __init__(self, root, transform=None):  # 构造函数，初始化数据集路径和预处理操作
        self.root = root  # 保存根目录路径
        self.transform = transform  # 保存图像预处理函数（如ToTensor等）
        self.samples = []  # 初始化一个列表，用于保存图像路径和对应标签

        for label in sorted(os.listdir(root)):  # root:mnist_images\train 遍历根目录下的子文件夹，子文件夹名代表类别标签（0~9）
            label_path = os.path.join(root, label)  # label_path: mnist_images\train\0构造每个子目录的完整路径
            if not os.path.isdir(label_path):  # 如果不是目录（可能是文件），就跳过
                continue
            for fname in os.listdir(label_path):  # 遍历该子目录下的所有文件 os.listdir(label_path)=[1.png,,21.png...]
                if fname.endswith('.png'):  # 如果文件是以 .png 结尾的图像
                    # os.path.join(label_path, fname): mnist_images\train\0 + 1.png = mnist_images\train\0\1.png
                    self.samples.append((os.path.join(label_path, fname), int(label)))  # 将图像路径和对应的标签（目录名转int）加入样本列表

    def __len__(self):  # 定义返回数据集中样本数量的方法
        return len(self.samples)  # 返回样本列表的长度

    def __getitem__(self, idx):  # 根据索引获取一个样本的方法
        img_path, label = self.samples[idx]  # 取出图像路径和对应标签
        image = Image.open(img_path).convert('L')  # 打开图像并转换为灰度图（MNIST 是单通道图像）

        if self.transform:  # 如果定义了图像预处理操作
            image = self.transform(image)  # 对图像进行预处理（如转Tensor、归一化等）
        return image, label  # 返回图像和对应的标签
