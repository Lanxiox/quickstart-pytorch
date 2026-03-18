"""
推理脚本
对单个图像或批量图像进行推理
"""

import os
import sys
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import Config
from models import create_model
from inference import Inferencer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型推理脚本')

    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default_config.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型检查点路径'
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='推理图像路径'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='是否可视化结果'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='可视化输出路径'
    )

    return parser.parse_args()


def get_transform(config_dict: dict, split: str = 'test') -> transforms.Compose:
    """
    获取数据变换

    Args:
        config_dict: 配置字典
        split: 数据集划分

    Returns:
        变换组合
    """
    transform_list = []
    transforms_config = config_dict.get('transforms', {})
    transform_configs = transforms_config.get(split, [])

    for transform_config in transform_configs:
        name = transform_config.get('name')
        params = transform_config.get('params', {})

        if hasattr(transforms, name):
            transform_func = getattr(transforms, name)
            transform_list.append(transform_func(**params))

    return transforms.Compose(transform_list)


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

    # 创建模型
    print("创建模型...")
    model_config = config_dict.get('model', {})
    model = create_model(model_config)

    # 获取检查点路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        output_config = config_dict.get('output', {})
        checkpoint_path = os.path.join(
            output_config.get('checkpoint_dir', './outputs/checkpoints'),
            'best_model.pth'
        )

    if not os.path.exists(checkpoint_path):
        print(f"错误: 模型文件不存在: {checkpoint_path}")
        return

    # 获取数据变换
    transform = get_transform(config_dict, 'test')

    # 创建推理器
    print(f"加载模型: {checkpoint_path}")
    inferencer = Inferencer(model, checkpoint_path, transform, device)

    # 推理
    print(f"\n推理图像: {args.image}")
    result = inferencer.predict(args.image)

    # 打印结果
    print("\n" + "="*50)
    print("推理结果")
    print("="*50)
    print(f"预测类别: {result['predicted_class']}")
    print(f"置信度: {result['confidence']:.2%}")
    print("\n各类别概率:")
    print("-"*50)

    for i, prob in enumerate(result['probabilities']):
        print(f"类别 {i}: {prob:.2%}")

    print("="*50 + "\n")

    # 可视化
    if args.visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 显示原图
        image = Image.open(args.image)
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原图')
        axes[0].axis('off')

        # 显示预测结果
        classes = list(range(10))
        probabilities = result['probabilities']
        colors = ['green' if i == result['predicted_class'] else 'blue'
                  for i in range(len(classes))]

        bars = axes[1].bar(classes, probabilities, color=colors)
        axes[1].set_xlabel('类别')
        axes[1].set_ylabel('概率')
        axes[1].set_title(f'预测结果: {result["predicted_class"]} (置信度: {result["confidence"]:.2%})')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

        # 添加概率标签
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{prob:.2%}',
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if args.output:
            plt.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存: {args.output}")
        else:
            plt.show()


if __name__ == '__main__':
    main()
