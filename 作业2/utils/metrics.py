"""
指标追踪模块
"""

import matplotlib.pyplot as plt
import os
from typing import Dict, Any
import json


class MetricsTracker:
    """指标追踪器"""

    def __init__(self):
        """初始化指标追踪器"""
        self.history = {
            'train': {
                'loss': [],
                'accuracy': []
            },
            'val': {
                'loss': [],
                'accuracy': []
            }
        }

    def update(self, split: str, metrics: Dict[str, float], epoch: int) -> None:
        """
        更新指标

        Args:
            split: 数据集划分: train, val
            metrics: 指标字典
            epoch: 当前epoch
        """
        if split not in self.history:
            self.history[split] = {}

        for metric_name, value in metrics.items():
            if metric_name not in self.history[split]:
                self.history[split][metric_name] = []

            self.history[split][metric_name].append(value)

    def get_history(self) -> Dict[str, Any]:
        """
        获取训练历史

        Returns:
            历史指标字典
        """
        return self.history

    def save_plots(self, save_dir: str) -> None:
        """
        保存训练曲线图

        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 绘制损失曲线
        if 'train' in self.history and 'loss' in self.history['train']:
            self._plot_metric('loss', save_dir, 'Loss over Epochs')

        # 绘制准确率曲线
        if 'train' in self.history and 'accuracy' in self.history['train']:
            self._plot_metric('accuracy', save_dir, 'Accuracy over Epochs')

        # 保存指标到JSON文件
        self._save_json(save_dir)

    def _plot_metric(self, metric_name: str, save_dir: str, title: str) -> None:
        """
        绘制指标曲线

        Args:
            metric_name: 指标名称
            save_dir: 保存目录
            title: 图表标题
        """
        plt.figure(figsize=(12, 5))

        # 训练集
        if 'train' in self.history and metric_name in self.history['train']:
            plt.plot(
                self.history['train'][metric_name],
                label=f'Train {metric_name.capitalize()}',
                marker='o'
            )

        # 验证集
        if 'val' in self.history and metric_name in self.history['val']:
            plt.plot(
                self.history['val'][metric_name],
                label=f'Val {metric_name.capitalize()}',
                marker='s'
            )

        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.title(title)
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(save_dir, f'{metric_name}_curve.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"图表已保存: {save_path}")

    def _save_json(self, save_dir: str) -> None:
        """
        保存指标到JSON文件

        Args:
            save_dir: 保存目录
        """
        save_path = os.path.join(save_dir, 'metrics.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)

        print(f"指标已保存: {save_path}")
