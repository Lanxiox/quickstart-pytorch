"""
数据处理模块
负责房价预测数据的加载、预处理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Dict, Any, Tuple
import numpy as np

from utils.feature_processor import FeatureProcessor, load_data


class HousePriceDataset(Dataset):
    """房价预测数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        """
        初始化数据集

        Args:
            X: 特征矩阵
            y: 目标变量
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class DataProcessor:
    """数据处理器类 - 适配房价预测"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config.get('data', {})

        # 创建特征处理器
        self.feature_processor = FeatureProcessor(config)

        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_and_process_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        加载并处理数据

        Returns:
            train_loader, val_loader
        """
        # 加载原始数据
        X_train, X_val, y_train, y_val = load_data(self.config)

        # 训练特征处理器
        self.feature_processor.fit(X_train, self.data_config)

        # 转换特征
        X_train_transformed = self.feature_processor.transform(X_train)
        X_val_transformed = self.feature_processor.transform(X_val)

        # 转换目标变量
        y_train_transformed = self.feature_processor.transform_target(y_train.values)
        y_val_transformed = self.feature_processor.transform_target(y_val.values)

        # 创建数据集
        self.train_dataset = HousePriceDataset(X_train_transformed, y_train_transformed)
        self.val_dataset = HousePriceDataset(X_val_transformed, y_val_transformed)

        # 创建数据加载器
        batch_size = self.data_config.get('batch_size', 32)
        num_workers = self.data_config.get('num_workers', 2)
        pin_memory = self.data_config.get('pin_memory', True)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        print(f"数据加载完成:")
        print(f"  - 训练批次数: {len(train_loader)}")
        print(f"  - 验证批次数: {len(val_loader)}")

        return train_loader, val_loader

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.feature_processor.get_feature_dim()

    def save_feature_processor(self, path: str) -> None:
        """保存特征处理器"""
        self.feature_processor.save(path)

    def load_feature_processor(self, path: str) -> None:
        """加载特征处理器"""
        self.feature_processor.load(path)


def create_data_processor(config: Dict[str, Any]) -> DataProcessor:
    """
    创建数据处理器

    Args:
        config: 配置字典

    Returns:
        数据处理器实例
    """
    return DataProcessor(config)
