from torchvision import datasets, transforms # 数据变换的库
from torch.utils.data import DataLoader # 数据加载器
import matplotlib.pyplot as plt # 绘图库
import torch.nn as nn # torch
import torch.optim as optim # 优化器
from tqdm import tqdm # 可视化进度条
from models import *
import torch
from torchvision.datasets import ImageFolder

# 对图像数据进行数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 图像转为灰度图
    transforms.ToTensor(), # 图像转换为张量 归一化
    transforms.Normalize([0.5],[0.5]) # 标准化 图像数据，对于灰度图像只需要一个通道标准化
])

# 使用torchvision里面的ImageFolder加载自定义格式的MNIST训练
train_dataset = ImageFolder(root='./mnist_images/train', transform=transform)
test_dataset = ImageFolder(root='./mnist_images/test', transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 检查是否有可用的 GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型，将模型移动到GPU
model = LNnet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() #交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器，学习率为0.001

# 保存训练过程中的损失和准确率
train_losses = []
train_accuracies = []
test_accuracies = []

# 训练模型
epochs = 10
best_accuracy = 0.0 # 记录验证集最佳准确率
best_model_path = "./models/best_model.pth"

for epoch in range(epochs):
    running_loss = 0.0
    correct_train = 0 # 正确预测的数量
    total_train = 0 # 样本总数
    # 训练过程
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"): # desc参数用来设置进度条前缀，显示当前是第几轮(epoch)的训练过程
        inputs, labels = inputs.to(device), labels.to(device) # 数据移动到GPU
        optimizer.zero_grad() # 梯度清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item() # 累加损失

        # 计算训练集上的准确率
        # torch.max(outputs, 1) 模型输出的一个二维张量，形状是 [batch_size, num_classes],取1代表类别维度
        _, predicted = torch.max(outputs, 1)  # 获取预测结果 【0，0.1，0.2，0，0，0，0.5，0.2，0，0】 6
        total_train += labels.size(0)  # 累加样本数量
        correct_train += (predicted == labels).sum().item()  # 累加正确预测的数量

    # 计算训练集上的准确率
    train_accuracy = correct_train / total_train
    train_losses.append(running_loss / len(train_loader))  # 记录每个epoch的平均损失,len(train_loader)为批次数，一个epoch结束后，running_loss包含了这个epoch中所有批次的总损失
    train_accuracies.append(train_accuracy)  # 记录每个epoch的训练集准确率
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2%}")

    # 在测试集上评估模型
    model.eval()  # 设定模型为评估模式
    correct = 0  # 正确的预测数量
    total = 0  # 样本总数
    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in tqdm(test_loader,desc=f"Epoch {epoch+1}/{epochs} - Testing"): # desc参数用来设置进度条前缀，显示当前是第几轮(epoch)的测试过程
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上
            outputs = model(inputs)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)  # 累加样本数量
            correct += (predicted == labels).sum().item()  # 累加正确预测的数量

    # 计算测试集上的准确率
    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)  # 记录每个 epoch 的测试集准确率
    print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_accuracy:.2%}")

    # 如果测试集准确率提高，保存当前模型的权重
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with accuracy: {best_accuracy:.2%}")

print(f"Best Accuracy on test set: {best_accuracy:.2%}")

# 绘制并保存损失和准确率曲线
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1) # 选择第一个子图
plt.plot(train_losses, label='Training Loss') # 传入数据、设置标签为Training Loss
plt.xlabel('Epoch') # x轴数据
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')  # 设置标签
plt.legend() # 添加图例
plt.grid(True) # 添加网格

# 绘制训练集和测试集准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)

# 保存图像
plt.tight_layout()
plt.savefig('loss_and_accuracy_curves_new.png')
plt.show()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_and_accuracy_curves_new.png')
plt.show()