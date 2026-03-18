"""
示例脚本: 如何使用深度学习Pipeline
"""

import torch
from torchvision import transforms

# 导入Pipeline组件
from configs.config import Config
from models import create_model
from data import create_data_processor
from trainers import Trainer
from evaluators import Evaluator
from inference import Inferencer
from utils import Logger


def example_basic_training():
    """示例1: 基本训练流程"""
    print("="*60)
    print("示例1: 基本训练流程")
    print("="*60)

    # 1. 加载配置
    config = Config('./configs/default_config.yaml')
    config_dict = config.to_dict()

    # 2. 创建日志记录器
    logger = Logger(
        log_dir='./outputs/logs',
        level='INFO',
        save_to_file=True,
        console_output=True
    )

    # 3. 创建数据加载器
    logger.info("创建数据加载器...")
    data_processor = create_data_processor(config_dict)
    dataloaders = data_processor.create_dataloaders()

    # 4. 创建模型
    logger.info("创建模型...")
    model = create_model(config_dict.get('model', {}))

    # 5. 创建训练器并训练
    logger.info("开始训练...")
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['test'],
        config=config_dict,
        logger=logger
    )
    trainer.train()

    logger.info("训练完成!")


def example_custom_model():
    """示例2: 使用自定义模型"""
    print("\n" + "="*60)
    print("示例2: 使用自定义模型")
    print("="*60)

    # 加载配置
    config = Config('./configs/default_config.yaml')

    # 修改模型配置
    config['model.name'] = 'SimpleCNN'
    config['model.params.num_classes'] = 10

    config_dict = config.to_dict()

    # 创建模型
    model = create_model(config_dict.get('model', {}))
    print(f"模型: {config['model.name']}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")


def example_evaluation():
    """示例3: 模型评估"""
    print("\n" + "="*60)
    print("示例3: 模型评估")
    print("="*60)

    # 加载配置
    config = Config('./configs/default_config.yaml')
    config_dict = config.to_dict()

    # 获取设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 创建模型和数据加载器
    model = create_model(config_dict.get('model', {}))
    data_processor = create_data_processor(config_dict)
    dataloaders = data_processor.create_dataloaders()

    # 加载训练好的模型
    checkpoint_path = './outputs/checkpoints/best_model.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {checkpoint_path}")
    except:
        print(f"模型文件不存在: {checkpoint_path}")
        return

    # 评估模型
    evaluator = Evaluator(model, dataloaders['test'], device)
    metrics = evaluator.evaluate()
    evaluator.print_evaluation_results(metrics)


def example_inference():
    """示例4: 单张图像推理"""
    print("\n" + "="*60)
    print("示例4: 单张图像推理")
    print("="*60)

    # 加载配置
    config = Config('./configs/default_config.yaml')
    config_dict = config.to_dict()

    # 获取设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 创建模型
    model = create_model(config_dict.get('model', {}))

    # 获取数据变换
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
    transform = transforms.Compose(transform_list)

    # 创建推理器
    checkpoint_path = './outputs/checkpoints/best_model.pth'
    try:
        inferencer = Inferencer(model, checkpoint_path, transform, device)
    except:
        print(f"模型文件不存在: {checkpoint_path}")
        return

    # 推理示例图像
    image_path = './mnist_images/test/0/1.png'
    try:
        result = inferencer.predict(image_path)
        print(f"图像: {image_path}")
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.2%}")
    except:
        print(f"图像文件不存在: {image_path}")


def example_custom_config():
    """示例5: 自定义配置"""
    print("\n" + "="*60)
    print("示例5: 自定义配置")
    print("="*60)

    # 加载默认配置
    config = Config('./configs/default_config.yaml')

    # 修改配置
    config['training.epochs'] = 5
    config['training.learning_rate'] = 0.0005
    config['data.batch_size'] = 32

    # 保存新配置
    config.save_config('./configs/custom_config.yaml')
    print("自定义配置已保存到: ./configs/custom_config.yaml")

    # 打印修改后的配置
    print("\n修改后的配置:")
    print(f"  - 训练轮数: {config['training.epochs']}")
    print(f"  - 学习率: {config['training.learning_rate']}")
    print(f"  - 批次大小: {config['data.batch_size']}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("深度学习Pipeline使用示例")
    print("="*60 + "\n")

    # 运行示例
    # example_basic_training()  # 需要先准备好数据
    example_custom_model()
    # example_evaluation()      # 需要先训练模型
    # example_inference()        # 需要先训练模型
    example_custom_config()

    print("\n" + "="*60)
    print("示例运行完成!")
    print("="*60 + "\n")
