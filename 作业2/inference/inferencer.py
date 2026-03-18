"""
推理模块
负责模型的推理和预测 - 适配房价预测
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import os


class HousePriceInferencer:
    """房价预测推理器"""

    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        feature_processor,
        device: Optional[torch.device] = None
    ):
        """
        初始化推理器

        Args:
            model: 模型实例
            model_path: 模型权重路径
            feature_processor: 特征处理器
            device: 计算设备
        """
        self.model = model
        self.feature_processor = feature_processor

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

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载: {model_path}")

        if 'epoch' in checkpoint:
            print(f"训练轮数: {checkpoint['epoch']}")

    def predict(self, features: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]:
        """
        预测房价

        Args:
            features: 输入特征 (numpy数组或DataFrame)

        Returns:
            预测结果字典
        """
        self.model.eval()

        # 如果是DataFrame，转换为numpy
        if isinstance(features, pd.DataFrame):
            # 选择特征列
            feature_columns = (self.feature_processor.numeric_features +
                             self.feature_processor.categorical_features)
            features = features[feature_columns]
            features = self.feature_processor.transform(features)
        elif isinstance(features, np.ndarray):
            features = features
        else:
            raise ValueError("features must be numpy array or pandas DataFrame")

        # 转换为tensor
        input_tensor = torch.tensor(features, dtype=torch.float32)

        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.cpu().numpy().flatten()[0]

        # 逆变换目标变量（如果使用了log变换）
        if self.feature_processor is not None:
            prediction = self.feature_processor.transform_target(prediction, inverse=True)

        # 确保价格非负
        prediction = max(0, prediction)

        return {
            'predicted_price': prediction,
            'predicted_price_formatted': f"${prediction:,.2f}"
        }

    def predict_batch(self, features: Union[np.ndarray, pd.DataFrame]) -> List[Dict[str, float]]:
        """
        批量预测

        Args:
            features: 输入特征

        Returns:
            预测结果列表
        """
        self.model.eval()

        # 如果是DataFrame，转换为numpy
        if isinstance(features, pd.DataFrame):
            feature_columns = (self.feature_processor.numeric_features +
                             self.feature_processor.categorical_features)
            features = features[feature_columns]
            features = self.feature_processor.transform(features)

        # 转换为tensor
        input_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions = outputs.cpu().numpy().flatten()

        # 逆变换
        if self.feature_processor is not None:
            predictions = self.feature_processor.transform_target(predictions, inverse=True)

        # 确保非负
        predictions = np.maximum(predictions, 0)

        results = []
        for pred in predictions:
            results.append({
                'predicted_price': float(pred),
                'predicted_price_formatted': f"${pred:,.2f}"
            })

        return results


def load_test_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    加载测试数据

    Args:
        config: 配置字典

    Returns:
        测试数据
    """
    data_config = config.get('data', {})
    data_root = data_config.get('data_root', './data')
    test_file = data_config.get('test_file', 'test.csv')
    test_path = os.path.join(data_root, test_file)

    df = pd.read_csv(test_path)
    return df
