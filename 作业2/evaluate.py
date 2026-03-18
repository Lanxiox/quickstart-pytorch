"""
房价预测评估脚本
评估训练好的模型
"""

import os
import sys
import argparse
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import Config
from data import create_data_processor
from models import create_model
from evaluators import Evaluator
from utils import Logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='房价预测评估脚本')

    parser.add_argument(
        '--config',
        type=str,
        default='./configs/house_price_config.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型检查点路径'
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

    # 创建日志记录器
    logger = Logger(
        level='INFO',
        save_to_file=False,
        console_output=True
    )

    # 获取设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logger.info(f"使用设备: {device}")

    # 创建数据加载器
    logger.info("创建数据加载器...")
    data_processor = create_data_processor(config_dict)
    train_loader, val_loader = data_processor.load_and_process_data()

    # 获取特征维度
    input_dim = data_processor.get_feature_dim()
    config_dict['model']['params']['input_dim'] = input_dim

    # 加载特征处理器
    output_config = config_dict.get('output', {})
    feature_processor_path = os.path.join(
        output_config.get('checkpoint_dir', './outputs_house_price/checkpoints'),
        'feature_processor.pkl'
    )
    data_processor.load_feature_processor(feature_processor_path)

    # 创建模型
    logger.info("创建模型...")
    model_config = config_dict.get('model', {})
    model = create_model(model_config)
    model = model.to(device)

    # 加载模型权重
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            output_config.get('checkpoint_dir', './outputs_house_price/checkpoints'),
            'best_model.pth'
        )

    if not os.path.exists(checkpoint_path):
        logger.error(f"模型文件不存在: {checkpoint_path}")
        return

    logger.info(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 创建评估器
    logger.info("开始评估...")
    evaluator = Evaluator(
        model,
        val_loader,
        device,
        feature_processor=data_processor.feature_processor
    )

    # 评估模型
    metrics = evaluator.evaluate()

    # 打印结果
    evaluator.print_evaluation_results(metrics)


if __name__ == '__main__':
    main()
