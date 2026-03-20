"""
特征处理模块
负责数据的特征工程：缺失值处理、标准化、类别编码等
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple


class FeatureProcessor:
    """特征处理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})

        # 存储处理器
        self.numeric_scaler = None
        self.label_encoders = {}
        self.onehot_encoder = None

        # 存储特征信息
        self.numeric_features = []
        self.categorical_features = []
        self.all_features = []

        # 填充值
        self.numeric_fill_value = None
        self.categorical_fill_value = None

    def fit(self, df: pd.DataFrame, data_config: Dict[str, Any]) -> 'FeatureProcessor':
        """
        拟合特征处理器

        Args:
            df: 训练数据
            data_config: 数据配置

        Returns:
            self
        """
        # 获取特征列配置
        self.numeric_features = data_config.get('numeric_features', [])
        self.categorical_features = data_config.get('categorical_features', [])
        self.all_features = self.numeric_features + self.categorical_features

        # 提取数值特征并转换为数值类型
        df_numeric = df[self.numeric_features].copy() if self.numeric_features else pd.DataFrame()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        # 提取类别特征
        df_categorical = df[self.categorical_features].copy() if self.categorical_features else pd.DataFrame()

        # 处理数值特征缺失值
        missing_config = self.feature_config.get('handle_missing', {})
        numeric_missing = missing_config.get('numeric', 'median')
        if numeric_missing == 'median':
            self.numeric_fill_value = df_numeric.median()
        elif numeric_missing == 'mean':
            self.numeric_fill_value = df_numeric.mean()
        else:
            self.numeric_fill_value = 0

        df_numeric = df_numeric.fillna(self.numeric_fill_value)

        # 标准化数值特征
        norm_method = self.feature_config.get('normalization_method', 'standard')
        if self.feature_config.get('normalize', True):
            if norm_method == 'standard':
                self.numeric_scaler = StandardScaler()
            else:
                self.numeric_scaler = MinMaxScaler()
            self.numeric_scaler.fit(df_numeric)
        else:
            self.numeric_scaler = None

        # 处理类别特征
        cat_missing = missing_config.get('categorical', 'mode')
        for col in self.categorical_features:
            if col in df_categorical.columns:
                if cat_missing == 'mode':
                    self.categorical_fill_value = df_categorical[col].mode()[0] if len(df_categorical[col].mode()) > 0 else 'Unknown'
                else:
                    self.categorical_fill_value = 'Unknown'
                df_categorical[col] = df_categorical[col].fillna(self.categorical_fill_value)

                # 标签编码
                le = LabelEncoder()
                df_categorical[col] = df_categorical[col].astype(str)
                le.fit(df_categorical[col])
                self.label_encoders[col] = le

        print(f"特征处理配置完成:")
        print(f"  - 数值特征数: {len(self.numeric_features)}")
        print(f"  - 类别特征数: {len(self.categorical_features)}")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        转换数据

        Args:
            df: 输入数据

        Returns:
            转换后的特征矩阵
        """
        # 提取特征并转换为数值类型
        df_numeric = df[self.numeric_features].copy() if self.numeric_features else pd.DataFrame()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        df_categorical = df[self.categorical_features].copy() if self.categorical_features else pd.DataFrame()

        # 处理数值特征缺失值
        if self.numeric_fill_value is not None:
            df_numeric = df_numeric.fillna(self.numeric_fill_value)

        # 标准化
        if self.numeric_scaler is not None:
            df_numeric = pd.DataFrame(
                self.numeric_scaler.transform(df_numeric),
                columns=self.numeric_features,
                index=df.index
            )

        # 处理类别特征
        for col in self.categorical_features:
            if col in df_categorical.columns:
                df_categorical[col] = df_categorical[col].fillna(self.categorical_fill_value)
                df_categorical[col] = df_categorical[col].astype(str)

                # 使用已编码的值，未知类别设为-1
                le = self.label_encoders.get(col)
                if le:
                    known_classes = set(le.classes_)
                    df_categorical[col] = df_categorical[col].apply(
                        lambda x: x if x in known_classes else le.classes_[0]
                    )
                    df_categorical[col] = le.transform(df_categorical[col])

        # 合并特征
        if not df_numeric.empty and not df_categorical.empty:
            df_transformed = pd.concat([df_numeric, df_categorical], axis=1)
        elif not df_numeric.empty:
            df_transformed = df_numeric
        else:
            df_transformed = df_categorical

        return df_transformed.values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, data_config: Dict[str, Any]) -> np.ndarray:
        """
        拟合并转换

        Args:
            df: 输入数据
            data_config: 数据配置

        Returns:
            转换后的特征矩阵
        """
        self.fit(df, data_config)
        return self.transform(df)

    def transform_target(self, y: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        转换目标变量

        Args:
            y: 目标变量
            inverse: 是否逆转换

        Returns:
            转换后的目标变量
        """
        transform = self.feature_config.get('target_transform', 'none')

        if inverse:
            if transform == 'log1p':
                return np.expm1(y)
            return y
        else:
            if transform == 'log1p':
                return np.log1p(y)
            return y

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return len(self.numeric_features) + len(self.categorical_features)

    def save(self, path: str) -> None:
        """
        保存特征处理器

        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        state = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'numeric_scaler': self.numeric_scaler,
            'label_encoders': self.label_encoders,
            'numeric_fill_value': self.numeric_fill_value,
            'categorical_fill_value': self.categorical_fill_value,
            'feature_config': self.feature_config
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        print(f"特征处理器已保存: {path}")

    def load(self, path: str) -> None:
        """
        加载特征处理器

        Args:
            path: 加载路径
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.numeric_features = state['numeric_features']
        self.categorical_features = state['categorical_features']
        self.numeric_scaler = state['numeric_scaler']
        self.label_encoders = state['label_encoders']
        self.numeric_fill_value = state['numeric_fill_value']
        self.categorical_fill_value = state['categorical_fill_value']
        self.feature_config = state['feature_config']

        print(f"特征处理器已加载: {path}")


def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    加载并分割数据

    Args:
        config: 配置字典

    Returns:
        X_train, X_val, y_train, y_val
    """
    data_config = config.get('data', {})

    # 读取训练数据
    data_root = data_config.get('data_root', './data')
    # 如果是相对路径，转换为相对于脚本所在目录的绝对路径
    if not os.path.isabs(data_root):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.join(script_dir, data_root)
    train_file = data_config.get('train_file', 'train.csv')
    train_path = os.path.join(data_root, train_file)

    print(f"加载数据: {train_path}")
    df = pd.read_csv(train_path)

    # 获取目标列
    target_column = data_config.get('target_column', 'Sold Price')
    id_column = data_config.get('id_column', 'Id')

    # 丢弃列
    drop_columns = data_config.get('drop_columns', [])
    drop_cols = [id_column, target_column] + drop_columns
    drop_cols = [c for c in drop_cols if c in df.columns]

    # 提取特征和目标
    y = df[target_column]
    X = df.drop(columns=drop_cols, errors='ignore')

    # 只保留配置中指定的特征列
    numeric_features = data_config.get('numeric_features', [])
    categorical_features = data_config.get('categorical_features', [])
    feature_columns = numeric_features + categorical_features

    # 过滤存在的列
    feature_columns = [c for c in feature_columns if c in X.columns]
    X = X[feature_columns]

    print(f"原始数据形状: {X.shape}")
    print(f"目标变量范围: {y.min():.2f} - {y.max():.2f}")

    # 分割数据
    test_size = data_config.get('test_size', 0.2)
    random_state = data_config.get('random_state', 42)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")

    return X_train, X_val, y_train, y_val
