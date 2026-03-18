"""
推理模块
负责模型的推理和预测
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Union, List, Optional
import os


class Inferencer:
    """推理器类"""

    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        transform: transforms.Compose,
        device: Optional[torch.device] = None
    ):
        """
        初始化推理器

        Args:
            model: 模型实例
            model_path: 模型权重路径
            transform: 数据变换
            device: 计算设备
        """
        self.model = model
        self.transform = transform

        # 设置设备
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.device = device
        self.model = self.model.to(self.device)

        # 加载模型权重
        self._load_model(model_path)

        # 设置为评估模式
        self.model.eval()

    def _load_model(self, model_path: str) -> None:
        """
        加载模型权重

        Args:
            model_path: 模型权重路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载: {model_path}")

        if 'epoch' in checkpoint:
            print(f"训练轮数: {checkpoint['epoch']}")

    def predict(self, image_path: str) -> dict:
        """
        预测单张图像

        Args:
            image_path: 图像路径

        Returns:
            预测结果字典
        """
        # 加载图像
        image = Image.open(image_path)

        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            'predicted_class': predicted.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0].tolist()
        }

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """
        批量预测图像

        Args:
            image_paths: 图像路径列表

        Returns:
            预测结果列表
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                result['success'] = True
            except Exception as e:
                result = {
                    'image_path': image_path,
                    'success': False,
                    'error': str(e)
                }
            results.append(result)

        return results

    def predict_tensor(self, input_tensor: torch.Tensor) -> dict:
        """
        预测张量

        Args:
            input_tensor: 输入张量

        Returns:
            预测结果字典
        """
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            'predicted_class': predicted.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0].tolist()
        }
