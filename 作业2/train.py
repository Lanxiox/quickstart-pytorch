"""
房价预测训练脚本
使用配置文件进行模型训练
"""

import os
import sys
import argparse
import torch
import random
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import Config
from data import create_data_processor
from models import create_model
from trainers import Trainer
from utils import Logger


def resolve_path(path: str) -> str:
    """解析相对路径为绝对路径（相对于脚本所在目录）"""
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    设置随机种子

    Args:
        seed: 随机种子
        deterministic: 是否使用确定性算法
        benchmark: 是否启用cudnn benchmark
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='房价预测训练脚本')

    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(project_root, 'configs/house_price_config.yaml'),
        help='配置文件路径'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子'
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = Config(args.config)
    config_dict = config.to_dict()

    # 设置随机种子
    seed_config = config_dict.get('random_seed', {})
    seed = args.seed or seed_config.get('seed', 42)
    deterministic = seed_config.get('deterministic', True)
    benchmark = seed_config.get('benchmark', False)

    set_seed(seed, deterministic, benchmark)
    print(f"随机种子设置为: {seed}")

    # 创建日志记录器
    log_config = config_dict.get('logging', {})
    output_config = config_dict.get('output', {})

    logger = Logger(
        log_dir=resolve_path(output_config.get('log_dir', './outputs_house_price/logs')),
        level=log_config.get('level', 'INFO'),
        save_to_file=log_config.get('save_to_file', True),
        console_output=log_config.get('console_output', True)
    )

    # 打印实验信息
    experiment_config = config_dict.get('experiment', {})
    logger.info("="*60)
    logger.info(f"实验名称: {experiment_config.get('name', 'Unknown')}")
    logger.info(f"实验描述: {experiment_config.get('description', 'No description')}")
    logger.info(f"随机种子: {seed}")
    logger.info("="*60)

    # 创建数据加载器
    logger.info("创建数据加载器...")
    data_processor = create_data_processor(config_dict)
    train_loader, val_loader = data_processor.load_and_process_data()

    # 获取特征维度
    input_dim = data_processor.get_feature_dim()
    config_dict['model']['params']['input_dim'] = input_dim
    logger.info(f"特征维度: {input_dim}")

    # 保存特征处理器
    feature_processor_path = os.path.join(
        resolve_path(output_config.get('checkpoint_dir', './outputs_house_price/checkpoints')),
        'feature_processor.pkl'
    )
    data_processor.save_feature_processor(feature_processor_path)

    # 创建模型
    logger.info("创建模型...")
    model_config = config_dict.get('model', {})
    model = create_model(model_config)

    # 创建训练器
    logger.info("初始化训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_dict,
        logger=logger,
        feature_processor=data_processor.feature_processor
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 训练完成
    logger.info("训练完成!")


if __name__ == '__main__':
    main()
