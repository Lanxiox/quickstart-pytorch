"""
日志管理模块
"""

import logging
import os
from datetime import datetime
from typing import Optional


class Logger:
    """日志管理器"""

    def __init__(
        self,
        name: str = 'DeepLearningPipeline',
        log_dir: Optional[str] = None,
        level: str = 'INFO',
        save_to_file: bool = True,
        console_output: bool = True
    ):
        """
        初始化日志管理器

        Args:
            name: 日志名称
            log_dir: 日志目录
            level: 日志级别
            save_to_file: 是否保存到文件
            console_output: 是否输出到控制台
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()

        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # 文件输出
        if save_to_file and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'train_{timestamp}.log')

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"日志文件: {log_file}")

    def info(self, msg: str) -> None:
        """输出info级别日志"""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """输出warning级别日志"""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """输出error级别日志"""
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        """输出debug级别日志"""
        self.logger.debug(msg)
