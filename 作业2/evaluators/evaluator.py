"""
评估模块
负责模型的评估和指标计算 - 适配回归任务
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm


class Evaluator:
    """评估器类 - 适配回归任务"""

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        feature_processor=None
    ):
        """
        初始化评估器

        Args:
            model: 模型实例
            val_loader: 验证数据加载器
            device: 计算设备
            feature_processor: 特征处理器（用于目标变量逆变换）
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.feature_processor = feature_processor
        self.model = self.model.to(self.device)

    def evaluate(self) -> Dict[str, float]:
        """
        评估模型

        Returns:
            评估指标字典
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []

        criterion = nn.MSELoss()

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # 合并所有预测和目标
        all_predictions = torch.cat(all_predictions, dim=0).numpy().flatten()
        all_targets = torch.cat(all_targets, dim=0).numpy().flatten()

        # 如果有特征处理器，逆变换目标变量
        if self.feature_processor is not None:
            # 注意：这里评估的是变换后的值
            pass

        # 计算各项指标
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))

        # R² Score
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # MAPE (避免除零)
        mask = all_targets != 0
        if np.any(mask):
            mape = np.mean(np.abs((all_targets[mask] - all_predictions[mask]) / all_targets[mask])) * 100
        else:
            mape = 0.0

        avg_loss = total_loss / len(self.val_loader)

        metrics = {
            'loss': avg_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

        return metrics

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        """
        预测

        Args:
            inputs: 输入张量

        Returns:
            预测结果
        """
        self.model.eval()

        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)

        return outputs.cpu().numpy().flatten()

    def print_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """
        打印评估结果

        Args:
            metrics: 评估指标字典
        """
        print("\n" + "="*50)
        print("评估结果")
        print("="*50)
        print(f"均方损失 (MSE): {metrics['mse']:.4f}")
        print(f"均方根损失 (RMSE): {metrics['rmse']:.4f}")
        print(f"平均绝对损失 (MAE): {metrics['mae']:.4f}")
        print(f"R² 决定系数: {metrics['r2']:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {metrics['mape']:.2f}%")
        print("="*50 + "\n")
