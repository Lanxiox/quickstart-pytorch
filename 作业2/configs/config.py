"""
配置管理模块
使用YAML格式的配置文件来管理实验参数
"""

import yaml
import os
from typing import Any, Dict


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def save_config(self, config_path: str) -> None:
        """
        保存配置到文件

        Args:
            config_path: 保存路径
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        更新配置

        Args:
            config_dict: 配置字典
        """
        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = _update(self.config, config_dict)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            配置字典
        """
        return self.config
