"""
训练模块
负责模型的训练过程 - 适配回归任务
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, Optional
from tqdm import tqdm
import numpy as np

from utils.logger import Logger
from utils.metrics import MetricsTracker
from utils.checkpoint import CheckpointManager


class Trainer:
    """训练器类 - 适配回归任务"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: Logger,
        feature_processor=None
    ):
        """
        初始化训练器

        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置字典
            logger: 日志记录器
            feature_processor: 特征处理器（用于目标变量逆变换）
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.feature_processor = feature_processor

        # 获取设备
        self.device = self._get_device()
        self.model = self.model.to(self.device)

        # 训练配置
        self.training_config = config.get('training', {})
        self.output_config = config.get('output', {})
        self.validation_config = config.get('validation', {})

        # 创建输出目录
        self._create_output_dirs()

        # 初始化损失函数
        self.criterion = self._create_loss_function()

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()

        # 早停
        self.early_stopping_config = self.training_config.get('early_stopping', {})
        self.early_stopping_patience = self.early_stopping_config.get('patience', 10)
        self.early_stopping_min_delta = self.early_stopping_config.get('min_delta', 0.001)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')

        # 指标追踪器
        self.metrics_tracker = MetricsTracker()

        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.output_config.get('checkpoint_dir', './outputs_house_price/checkpoints'),
            logger=logger
        )

        # 训练状态
        self.current_epoch = 0
        self.best_metric = float('inf')

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
            self.output_config.get('save_dir', './outputs_house_price'),
            self.output_config.get('checkpoint_dir', './outputs_house_price/checkpoints'),
            self.output_config.get('log_dir', './outputs_house_price/logs'),
            self.output_config.get('vis_dir', './outputs_house_price/visualizations')
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _create_loss_function(self) -> nn.Module:
        """创建损失函数 - 支持回归任务"""
        loss_name = self.training_config.get('loss_function', 'MSELoss')

        if loss_name == 'MSELoss':
            criterion = nn.MSELoss()
        elif loss_name == 'MAELoss':
            criterion = nn.L1Loss()
        elif loss_name == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss()
        elif loss_name == 'HuberLoss':
            criterion = nn.HuberLoss()
        elif hasattr(nn, loss_name):
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
            # 移除不兼容的参数
            params = {k: v for k, v in params.items() if k != 'verbose'}
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

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算回归指标

        Args:
            predictions: 预测值
            targets: 真实值

        Returns:
            指标字典
        """
        # 转换为numpy
        pred = predictions.cpu().numpy().flatten()
        true = targets.cpu().numpy().flatten()

        # MSE
        mse = np.mean((pred - true) ** 2)

        # RMSE
        rmse = np.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(pred - true))

        # R² Score
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

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
            all_predictions.append(outputs.detach())
            all_targets.append(targets.detach())

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

        # 合并所有batch的预测和目标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算指标
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(self.train_loader)

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        验证模型

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} - Validation")

            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                all_predictions.append(outputs)
                all_targets.append(targets)

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })

        # 合并所有batch的预测和目标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算指标
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(self.val_loader)

        return metrics

    def train(self) -> None:
        """训练模型"""
        epochs = self.training_config.get('epochs', 10)
        validation_config = self.config.get('validation', {})
        metric_name = validation_config.get('metric', 'rmse')
        mode = validation_config.get('mode', 'min')

        self.logger.info(f"开始训练, 共 {epochs} 个 epoch")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train - "
                f"Loss: {train_metrics['loss']:.4f}, "
                f"RMSE: {train_metrics['rmse']:.4f}, "
                f"MAE: {train_metrics['mae']:.4f}, "
                f"R²: {train_metrics['r2']:.4f}"
            )

            # 验证
            val_metrics = self.validate()
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - Val - "
                f"Loss: {val_metrics['loss']:.4f}, "
                f"RMSE: {val_metrics['rmse']:.4f}, "
                f"MAE: {val_metrics['mae']:.4f}, "
                f"R²: {val_metrics['r2']:.4f}"
            )

            # 更新指标追踪器
            self.metrics_tracker.update('train', train_metrics, epoch)
            self.metrics_tracker.update('val', val_metrics, epoch)

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[metric_name])
                else:
                    self.scheduler.step()

            # 保存模型
            self._save_model(val_metrics)

            # 早停检查
            val_loss = val_metrics.get(metric_name, val_metrics['loss'])
            if self._check_early_stopping(val_loss, mode):
                self.logger.info(f"早停触发，训练结束")
                break

            # 保存检查点
            checkpoint_interval = self.output_config.get('save_interval', 5)
            if (epoch + 1) % checkpoint_interval == 0:
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.metrics_tracker,
                    is_best=False
                )

        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证{metric_name}: {self.best_metric:.4f}")

        # 保存训练曲线
        self.metrics_tracker.save_plots(
            self.output_config.get('vis_dir', './outputs_house_price/visualizations')
        )

    def _check_early_stopping(self, val_metric: float, mode: str) -> bool:
        """
        检查是否触发早停

        Args:
            val_metric: 验证指标
            mode: 优化模式 (min 或 max)

        Returns:
            是否触发早停
        """
        if self.early_stopping_patience <= 0:
            return False

        if mode == 'min':
            if val_metric < self.best_metric - self.early_stopping_min_delta:
                self.best_metric = val_metric
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
        else:
            if val_metric > self.best_metric + self.early_stopping_min_delta:
                self.best_metric = val_metric
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.early_stopping_patience:
            return True

        return False

    def _save_model(self, val_metrics: Dict[str, float]) -> None:
        """
        保存模型

        Args:
            val_metrics: 验证指标
        """
        validation_config = self.config.get('validation', {})
        metric_name = validation_config.get('metric', 'rmse')
        mode = validation_config.get('mode', 'min')

        metric_value = val_metrics.get(metric_name, val_metrics.get('loss', float('inf')))

        # 判断是否是最佳模型
        is_best = False
        if mode == 'min':
            is_best = metric_value < self.best_metric
        else:
            is_best = metric_value > self.best_metric

        if is_best:
            self.best_metric = metric_value
            self.checkpoint_manager.save(
                self.model,
                self.optimizer,
                self.current_epoch,
                self.metrics_tracker,
                is_best=True
            )
            self.logger.info(f"保存最佳模型, {metric_name}: {metric_value:.4f}")
