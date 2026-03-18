# California 房价预测 Pipeline

基于 Kaggle California House Prices 比赛的深度学习房价预测 Pipeline。

## 项目结构

```
作业2/
├── configs/
│   ├── config.py
│   └── house_price_config.yaml    # 房价预测配置文件
├── models/
│   ├── __init__.py
│   └── model_factory.py           # 模型定义（包含HousePriceMLP）
├── data/
│   ├── __init__.py
│   └── data_loader.py             # 数据处理
├── trainers/
│   ├── __init__.py
│   └── trainer.py                  # 训练器
├── evaluators/
│   ├── __init__.py
│   └── evaluator.py                # 评估器
├── inference/
│   ├── __init__.py
│   └── inferencer.py              # 推理器
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── metrics.py
│   ├── checkpoint.py
│   └── feature_processor.py        # 特征处理器
├── train.py                        # 训练脚本
├── evaluate.py                     # 评估脚本
├── inference.py                    # 推理脚本
└── README.md
```

## 数据说明

本项目使用 Kaggle California House Prices 比赛数据：
- 训练数据: `作业2 李沐房价预测/california-house-prices/train.csv`
- 测试数据: `作业2 李沐房价预测/california-house-prices/test.csv`

### 使用的特征

**数值特征 (18个)**:
- Year built, Lot, Bedrooms, Bathrooms
- Full bathrooms, Total interior livable area
- Total spaces, Garage spaces
- Elementary/Middle/High School Score
- Elementary/Middle/High School Distance
- Tax assessed value, Annual tax amount
- Listed Price, Last Sold Price

**类别特征 (4个)**:
- Type (房屋类型)
- Region (区域)
- City (城市)
- State (州)

### 目标变量
- Sold Price (销售价格)

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision numpy pandas matplotlib pyyaml tqdm scikit-learn
```

### 2. 训练模型

```bash
# 使用默认配置训练
python train.py

# 或指定配置文件
python train.py --config ./configs/house_price_config.yaml

# 设置随机种子
python train.py --seed 42
```

### 3. 评估模型

```bash
# 评估最佳模型
python evaluate.py

# 评估指定检查点
python evaluate.py --checkpoint ./outputs_house_price/checkpoints/best_model.pth
```

### 4. 推理预测

```bash
# 对测试集进行预测
python inference.py

# 指定输出路径
python inference.py --output ./outputs_house_price/my_predictions.csv
```

## 配置文件说明

主要配置项位于 `configs/house_price_config.yaml`:

```yaml
# 数据配置
data:
  data_root: "../作业2 李沐房价预测/california-house-prices"
  target_column: "Sold Price"
  batch_size: 128

# 特征工程
feature_engineering:
  normalize: true              # 标准化数值特征
  target_transform: "log1p"    # 对目标变量取对数

# 模型配置
model:
  name: "HousePriceMLP"
  params:
    input_dim: null            # 自动确定
    hidden_sizes: [256, 128, 64, 32]
    dropout: 0.3

# 训练配置
training:
  epochs: 100
  learning_rate: 0.001
  loss_function: "MSELoss"
  optimizer: "Adam"
```

## 输出文件

训练完成后，在 `outputs_house_price` 目录会生成：

- `checkpoints/best_model.pth` - 最佳模型权重
- `checkpoints/feature_processor.pkl` - 特征处理器（推理时需要）
- `checkpoints/checkpoint_epoch_*.pth` - 各轮次检查点
- `visualizations/` - 训练曲线图
  - `loss_curve.png`
  - `rmse_curve.png`
  - `mae_curve.png`
  - `r2_curve.png`
  - `metrics.json`

## 模型说明

### HousePriceMLP

专为房价预测设计的全连接神经网络：

```
Input(特征维度)
  -> Linear -> BatchNorm -> ReLU -> Dropout
  -> Linear -> BatchNorm -> ReLU -> Dropout
  -> Linear -> BatchNorm -> ReLU -> Dropout
  -> Linear -> BatchNorm -> ReLU -> Dropout
  -> Linear(1)  # 输出层
  -> Output(预测价格)
```

### 特征处理

1. **缺失值处理**: 数值特征用中位数填充，类别特征用众数填充
2. **数值标准化**: 使用 StandardScaler 标准化
3. **类别编码**: 使用 LabelEncoder 编码
4. **目标变量**: 使用 log1p 变换处理大数值

## 评估指标

- MSE (均方误差)
- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)
- MAPE (平均绝对百分比误差)

## 注意事项

1. 数据路径使用了相对路径，确保在正确的目录下运行
2. 特征处理器保存后，推理时需要加载以保持数据处理一致
3. 建议使用 GPU 加速训练
4. 可以通过修改配置文件调整超参数

## 扩展

如需添加新特征或修改模型，可参考以下文件：
- `utils/feature_processor.py` - 修改特征工程逻辑
- `models/model_factory.py` - 添加新模型
- `trainers/trainer.py` - 修改训练逻辑
