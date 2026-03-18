"""
训练模块
负责模型的训练过程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, Optional
import os
from tqdm import tqdm
import time
from datetime import datetime

from ..utils.logger import Logger
from ..utils.metrics import MetricsTracker
from ..utils.checkpoint import CheckpointManager


class Trainer:
    """训练器类"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: Logger
    ):
        """
        初始化训练器

        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置字典
            logger: 日志记录器
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger

        # 获取设备
        self.device = self._get_device()
        self.model = self.model.to(self.device)

        # 训练配置
        self.training_config = config.get('training', {})
        self.output_config = config.get('output', {})

        # 创建输出目录
        self._create_output_dirs()

        # 初始化损失函数
        self.criterion = self._create_loss_function()

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()

        # 指标追踪器
        self.metrics_tracker = MetricsTracker()

        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.output_config.get('checkpoint_dir', './outputs/checkpoints'),
            logger=logger
        )

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0

        self.logger.info(f"训练器初始化完成, 使用设备: {self.device}")

    def _get_device(self) -> torch.device:
        """获取计算设备"""
        device_config = self.config.get('device', {})
        device_name = device_config.get('device', 'auto')

        if device_name == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_name)

        return device

    def _create_output_dirs(self) -> None:
        """创建输出目录"""
        dirs = [
            self.output_config.get('save_dir', './outputs'),
            self.output_config.get('checkpoint_dir', './outputs/checkpoints'),
            self.output_config.get('log_dir', './outputs/logs'),
            self.output_config.get('vis_dir', './outputs/visualizations')
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        loss_name = self.training_config.get('loss_function', 'CrossEntropyLoss')

        if hasattr(nn, loss_name):
            criterion = getattr(nn, loss_name)()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        return criterion

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_name = self.training_config.get('optimizer', 'Adam')
        learning_rate = self.training_config.get('learning_rate', 0.001)
        weight_decay = self.training_config.get('weight_decay', 0.0001)

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            momentum = self.training_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_config = self.training_config.get('scheduler')

        if not scheduler_config:
            return None

        scheduler_name = scheduler_config.get('name')
        params = scheduler_config.get('params', {})

        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **params
            )
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                **params
            )
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **params
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} - Training")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_clip = self.training_config.get('gradient_clip')
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # 参数更新
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        验证模型

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} - Validation")

            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': correct / total
        }

        return metrics

    def train(self) -> None:
        """训练模型"""
        epochs = self.training_config.get('epochs', 10)
        validation_config = self.config.get('validation', {})

        self.logger.info(f"开始训练, 共 {epochs} 个 epoch")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train - Loss: {train_metrics['loss']:.4f}, "
                f"Accuracy: {train_metrics['accuracy']:.2%}"
            )

            # 验证
            val_metrics = self.validate()
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - Val - Loss: {val_metrics['loss']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.2%}"
            )

            # 更新指标追踪器
            self.metrics_tracker.update('train', train_metrics, epoch)
            self.metrics_tracker.update('val', val_metrics, epoch)

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()

            # 保存模型
            self._save_model(val_metrics)

            # 保存检查点
            checkpoint_interval = self.output_config.get('save_interval', 1)
            if (epoch + 1) % checkpoint_interval == 0:
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.metrics_tracker,
                    is_best=False
                )

        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证准确率: {self.best_metric:.2%}")

        # 保存训练曲线
        self.metrics_tracker.save_plots(
            self.output_config.get('vis_dir', './outputs/visualizations')
        )

    def _save_model(self, val_metrics: Dict[str, float]) -> None:
        """
        保存模型

        Args:
            val_metrics: 验证指标
        """
        validation_config = self.config.get('validation', {})
        metric_name = validation_config.get('metric', 'accuracy')
        metric_value = val_metrics.get(metric_name, 0)

        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.checkpoint_manager.save(
                self.model,
                self.optimizer,
                self.current_epoch,
                self.metrics_tracker,
                is_best=True
            )
            self.logger.info(f"保存最佳模型, {metric_name}: {metric_value:.2%}")
