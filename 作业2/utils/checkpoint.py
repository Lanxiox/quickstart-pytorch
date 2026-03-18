"""
检查点管理模块
"""

import torch
import os
from typing import Optional, Dict, Any


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, save_dir: str, logger):
        """
        初始化检查点管理器

        Args:
            save_dir: 保存目录
            logger: 日志记录器
        """
        self.save_dir = save_dir
        self.logger = logger
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics_tracker,
        is_best: bool = False,
        save_optimizer: bool = True
    ) -> None:
        """
        保存检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            metrics_tracker: 指标追踪器
            is_best: 是否是最佳模型
            save_optimizer: 是否保存优化器状态
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics_tracker.get_history()
        }

        if save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # 保存定期检查点
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存: {checkpoint_path}")

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"最佳模型已保存: {best_path}")

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            model: 模型
            optimizer: 优化器
            checkpoint_path: 检查点路径
            device: 设备

        Returns:
            检查点信息
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return {}

        if device is None:
            device = torch.device('cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"模型权重已从 {checkpoint_path} 加载")

        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("优化器状态已加载")

        return checkpoint
