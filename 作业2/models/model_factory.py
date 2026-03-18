"""
模型定义模块
定义各种神经网络模型
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class LNnet(nn.Module):
    """
    多层感知机(MLP)模型
    用于MNIST手写数字分类
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10):
        """
        初始化模型

        Args:
            input_size: 输入特征大小 (28*28=784)
            hidden_size: 隐藏层大小
            num_classes: 输出类别数
        """
        super(LNnet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)

        # 可选: 添加dropout层防止过拟合
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量, shape: [batch_size, 1, 28, 28]

        Returns:
            输出张量, shape: [batch_size, num_classes]
        """
        # 展平图像: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = torch.flatten(x, start_dim=1)

        # 全连接层 + ReLU激活
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        x = self.dropout(x)

        # 输出层 (不需要激活函数,CrossEntropyLoss会自动处理)
        x = self.fc4(x)

        return x


class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络模型
    用于MNIST手写数字分类
    """

    def __init__(self, num_classes: int = 10):
        """
        初始化CNN模型

        Args:
            num_classes: 输出类别数
        """
        super(SimpleCNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 经过两次池化: 28->14->7,第三次卷积后是7
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量, shape: [batch_size, 1, 28, 28]

        Returns:
            输出张量, shape: [batch_size, num_classes]
        """
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))

        # 展平
        x = torch.flatten(x, start_dim=1)

        # 全连接层
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建模型

    Args:
        model_config: 模型配置字典

    Returns:
        模型实例
    """
    model_name = model_config.get('name', 'LNnet')
    params = model_config.get('params', {})

    if model_name == 'LNnet':
        model = LNnet(**params)
    elif model_name == 'SimpleCNN':
        model = SimpleCNN(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
