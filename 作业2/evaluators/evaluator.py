"""
评估模块
负责模型的评估和指标计算
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm


class Evaluator:
    """评估器类"""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device
    ):
        """
        初始化评估器

        Args:
            model: 模型实例
            test_loader: 测试数据加载器
            device: 计算设备
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model = self.model.to(self.device)

    def evaluate(self) -> Dict[str, float]:
        """
        评估模型

        Returns:
            评估指标字典
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # 每个类别的统计
        class_correct = {}
        class_total = {}

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 每个类别的统计
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1

        # 计算总体指标
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total

        # 计算每个类别的准确率
        class_accuracies = {}
        for label in sorted(class_total.keys()):
            if class_total[label] > 0:
                class_accuracies[label] = class_correct[label] / class_total[label]

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies
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
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def print_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """
        打印评估结果

        Args:
            metrics: 评估指标字典
        """
        print("\n" + "="*50)
        print("评估结果")
        print("="*50)
        print(f"总准确率: {metrics['accuracy']:.2%}")
        print(f"平均损失: {metrics['loss']:.4f}")
        print("\n各类别准确率:")
        print("-"*50)

        for label, acc in sorted(metrics['class_accuracies'].items()):
            print(f"类别 {label}: {acc:.2%}")

        print("="*50 + "\n")
