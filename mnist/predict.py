import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import LNnet
import matplotlib.pyplot as plt

# 1. 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 定义预处理操作（和训练时测试集的保持一致）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 3. 加载模型并加载训练好的权重
model = LNnet().to(device)
model.load_state_dict(torch.load('./models/best_model.pth', map_location=device))
model.eval()  # 设置为推理模式

# 4. 读取并处理单张图片
img_path = r'./mnist_images/test/9/16.png'  # 替换为要推理的图片路径
image = Image.open(img_path) 
image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度 C H W -> 1 C H W 

# 5. 模型推理
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# 6. 输出预测结果
print(f"预测结果类别为: {predicted.item()}")

# 展示图片与预测类别
plt.imshow(Image.open(img_path), cmap='gray')
plt.title(f"Predicted: {predicted.item()}")
plt.axis('off')
plt.show()
