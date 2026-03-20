"""
数据处理模块
负责数据的加载、预处理和数据增强
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from typing import Dict, Any, Optional
import os


class DataProcessor:
    """数据处理器类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.transforms_config = config.get('transforms', {})

    def get_transform(self, split: str = 'train') -> transforms.Compose:
        """
        获取数据变换

        Args:
            split: 数据集划分: train, test

        Returns:
            变换组合
        """
        transform_list = []

        # 根据配置构建变换
        transform_configs = self.transforms_config.get(split, [])

        for transform_config in transform_configs:
            name = transform_config.get('name')
            params = transform_config.get('params', {})

            if hasattr(transforms, name):
                transform_func = getattr(transforms, name)
                transform_list.append(transform_func(**params))
            else:
                raise ValueError(f"Unknown transform: {name}")

        return transforms.Compose(transform_list)

    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """
        创建数据加载器

        Returns:
            包含train和test dataloader的字典
        """
        data_root = self.data_config.get('data_root', './data')
        batch_size = self.data_config.get('batch_size', 32)
        num_workers = self.data_config.get('num_workers', 2)
        pin_memory = self.data_config.get('pin_memory', True)

        # 创建数据集
        train_dataset = self.create_dataset('train')
        test_dataset = self.create_dataset('test')

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return {
            'train': train_loader,
            'test': test_loader
        }

    def create_dataset(self, split: str = 'train') -> Dataset:
        """
        创建数据集

        Args:
            split: 数据集划分: train, test

        Returns:
            数据集实例
        """
        data_root = self.data_config.get('data_root', './data')
        split_dir = self.data_config.get(f'{split}_dir', split)

        data_path = os.path.join(data_root, split_dir)
        transform = self.get_transform(split)

        # 使用ImageFolder加载数据
        dataset = ImageFolder(root=data_path, transform=transform)

        print(f"{split.capitalize()} dataset loaded from {data_path}")
        print(f"  - Classes: {dataset.classes}")
        print(f"  - Number of samples: {len(dataset)}")

        return dataset


def create_data_processor(config: Dict[str, Any]) -> DataProcessor:
    """
    创建数据处理器

    Args:
        config: 配置字典

    Returns:
        数据处理器实例
    """
    return DataProcessor(config)
