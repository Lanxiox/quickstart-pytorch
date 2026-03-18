# 深度学习 Pipeline

一个可复用的深度学习训练、评估和推理流程框架，专为学习和实验设计。

## 项目结构

```
作业2/
├── configs/                    # 配置文件目录
│   ├── __init__.py
│   ├── config.py              # 配置管理类
│   └── default_config.yaml    # 默认配置文件
├── models/                     # 模型定义目录
│   ├── __init__.py
│   └── model_factory.py       # 模型工厂
├── data/                       # 数据处理目录
│   ├── __init__.py
│   └── data_loader.py         # 数据加载器
├── trainers/                   # 训练模块目录
│   ├── __init__.py
│   └── trainer.py             # 训练器
├── evaluators/                 # 评估模块目录
│   ├── __init__.py
│   └── evaluator.py           # 评估器
├── inference/                  # 推理模块目录
│   ├── __init__.py
│   └── inferencer.py          # 推理器
├── utils/                      # 工具函数目录
│   ├── __init__.py
│   ├── logger.py              # 日志管理
│   ├── metrics.py             # 指标追踪
│   └── checkpoint.py          # 检查点管理
├── outputs/                    # 输出目录
│   ├── checkpoints/           # 模型检查点
│   ├── logs/                  # 训练日志
│   └── visualizations/        # 训练曲线图
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── inference.py                # 推理脚本
└── README.md                   # 说明文档
```

## 特性

- **模块化设计**: 各功能模块独立，易于维护和扩展
- **配置驱动**: 使用YAML配置文件管理实验参数
- **完整流程**: 支持训练、评估、推理完整流程
- **自动保存**: 自动保存最佳模型、训练日志和可视化曲线
- **可复用性**: 可以方便地应用到其他深度学习项目

## 快速开始

### 1. 准备数据

将MNIST数据集放在项目根目录的`mnist_images`文件夹中：

```
mnist_images/
├── train/
│   ├── 0/
│   ├── 1/
│   └── ...
└── test/
    ├── 0/
    ├── 1/
    └── ...
```

### 2. 训练模型

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置训练
python train.py --config ./configs/default_config.yaml

# 设置随机种子
python train.py --seed 123

# 从检查点恢复训练
python train.py --resume ./outputs/checkpoints/checkpoint_epoch_5.pth
```

### 3. 评估模型

```bash
# 评估最佳模型
python evaluate.py

# 评估指定检查点
python evaluate.py --checkpoint ./outputs/checkpoints/best_model.pth
```

### 4. 推理预测

```bash
# 对单张图像进行推理
python inference.py --image ./mnist_images/test/0/1.png

# 推理并可视化结果
python inference.py --image ./mnist_images/test/0/1.png --visualize

# 保存可视化结果
python inference.py --image ./mnist_images/test/0/1.png --visualize --output ./result.png
```

## 配置说明

配置文件采用YAML格式，包含以下主要部分：

### 数据配置
```yaml
data:
  data_root: "./mnist_images"   # 数据根目录
  batch_size: 64                # 批次大小
  num_workers: 2                # 数据加载进程数
```

### 模型配置
```yaml
model:
  name: "LNnet"                 # 模型名称
  params:
    input_size: 784
    hidden_size: 256
    num_classes: 10
```

### 训练配置
```yaml
training:
  epochs: 10                    # 训练轮数
  learning_rate: 0.001          # 学习率
  optimizer: "Adam"             # 优化器
  loss_function: "CrossEntropyLoss"  # 损失函数
```

### 输出配置
```yaml
output:
  save_dir: "./outputs"         # 输出目录
  checkpoint_dir: "./outputs/checkpoints"
  log_dir: "./outputs/logs"
  vis_dir: "./outputs/visualizations"
```

## 扩展指南

### 添加新模型

1. 在`models/model_factory.py`中定义新模型类
2. 在`create_model`函数中添加模型创建逻辑
3. 在配置文件中指定模型名称和参数

示例：
```python
class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 定义你的模型结构
        ...

def create_model(model_config):
    model_name = model_config.get('name')
    if model_name == 'MyModel':
        model = MyModel(**model_config.get('params'))
    return model
```

### 自定义数据预处理

在配置文件的`transforms`部分添加或修改数据增强操作：

```yaml
transforms:
  train:
    - name: "RandomHorizontalFlip"
      params:
        p: 0.5
    - name: "RandomRotation"
      params:
        degrees: 15
```

### 添加新指标

1. 在`utils/metrics.py`中扩展`MetricsTracker`类
2. 在`Trainer`和`Evaluator`中计算新指标
3. 更新可视化逻辑

## 输出文件说明

训练完成后，在`outputs`目录会生成以下文件：

- **checkpoints/**: 模型检查点
  - `best_model.pth`: 最佳模型权重
  - `checkpoint_epoch_N.pth`: 各轮次检查点

- **logs/**: 训练日志
  - `train_YYYYMMDD_HHMMSS.log`: 训练过程日志

- **visualizations/**: 训练曲线
  - `loss_curve.png`: 损失曲线
  - `accuracy_curve.png`: 准确率曲线
  - `metrics.json`: 指标数据

## 常见问题

### Q: 如何更改数据集路径？
A: 修改配置文件中的`data.data_root`参数。

### Q: 如何使用GPU训练？
A: 修改配置文件中的`device.device`参数为"cuda"，或设置为"auto"自动选择。

### Q: 如何调整学习率？
A: 修改配置文件中的`training.learning_rate`参数。

### Q: 如何添加数据增强？
A: 在配置文件的`transforms.train`部分添加对应的transform配置。

## 依赖项

- PyTorch >= 1.10.0
- torchvision
- numpy
- matplotlib
- PyYAML
- tqdm

## 许可证

MIT License
