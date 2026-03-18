"""
房价预测推理脚本
对测试数据进行预测
"""

import os
import sys
import argparse
import torch
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import Config
from models import create_model
from inference import HousePriceInferencer, load_test_data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='房价预测推理脚本')

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

    parser.add_argument(
        '--output',
        type=str,
        default='./outputs_house_price/predictions.csv',
        help='预测结果输出路径'
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

    # 获取设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"使用设备: {device}")

    # 获取检查点路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        output_config = config_dict.get('output', {})
        checkpoint_path = os.path.join(
            output_config.get('checkpoint_dir', './outputs_house_price/checkpoints'),
            'best_model.pth'
        )

    if not os.path.exists(checkpoint_path):
        print(f"错误: 模型文件不存在: {checkpoint_path}")
        return

    # 先加载特征处理器获取input_dim
    from utils.feature_processor import FeatureProcessor
    output_config = config_dict.get('output', {})
    feature_processor_path = os.path.join(
        output_config.get('checkpoint_dir', './outputs_house_price/checkpoints'),
        'feature_processor.pkl'
    )

    feature_processor = FeatureProcessor(config_dict)
    feature_processor.load(feature_processor_path)

    # 获取特征维度
    input_dim = feature_processor.get_feature_dim()
    config_dict['model']['params']['input_dim'] = input_dim
    print(f"特征维度: {input_dim}")

    # 创建模型
    print("创建模型...")
    model_config = config_dict.get('model', {})
    model = create_model(model_config)

    # 创建推理器
    print(f"加载模型: {checkpoint_path}")
    inferencer = HousePriceInferencer(model, checkpoint_path, feature_processor, device)

    # 加载测试数据
    print("加载测试数据...")
    test_df = load_test_data(config_dict)

    # 获取ID列
    data_config = config_dict.get('data', {})
    id_column = data_config.get('id_column', 'Id')

    # 获取特征列
    numeric_features = data_config.get('numeric_features', [])
    categorical_features = data_config.get('categorical_features', [])
    feature_columns = numeric_features + categorical_features

    # 确保特征列存在
    test_features = test_df[[c for c in feature_columns if c in test_df.columns]]

    # 预测
    print("开始预测...")
    predictions = inferencer.predict_batch(test_features)

    # 保存结果
    results = []
    for i, pred in enumerate(predictions):
        results.append({
            id_column: test_df.iloc[i][id_column] if id_column in test_df.columns else i,
            'Predicted': pred['predicted_price']
        })

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print(f"\n预测完成!")
    print(f"预测结果已保存到: {args.output}")
    print(f"预测样本数: {len(results_df)}")
    print(f"\n预测价格统计:")
    print(f"  最低: ${results_df['Predicted'].min():,.2f}")
    print(f"  最高: ${results_df['Predicted'].max():,.2f}")
    print(f"  平均: ${results_df['Predicted'].mean():,.2f}")
    print(f"  中位数: ${results_df['Predicted'].median():,.2f}")


if __name__ == '__main__':
    main()
