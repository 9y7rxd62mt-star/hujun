# -*- coding: utf-8 -*-
"""
污染物去除率预测模型 - 改进版
增强列名匹配和错误输出
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
import difflib  # 用于模糊匹配列名
import joblib

warnings.filterwarnings('ignore')


def load_data():
    """加载数据文件"""
    COLUMN_CONFIG = {
        'pollutant': '污染物类型',  # 污染物类型列名
        'ph': 'ph',  # pH值列名
        'temperature': '温度',  # 温度列名
        'nutrient': '营养条件',  # 营养条件列名
        'target': '去除率'  # 目标变量列名
    }
    data = None
    source = ""

    # 尝试不同的文件格式
    file_paths = [
        ('Pollutant Experimental Data.xlsx', pd.read_excel),
        ('Pollutant Experimental Data', lambda f: pd.read_csv(f, encoding='utf-8')),
        ('Pollutant Experimental Data.csv', lambda f: pd.read_csv(f, encoding='gbk')),
        ('data.xlsx', pd.read_excel),
        ('data.csv', lambda f: pd.read_csv(f, encoding='utf-8')),
        ('污染物数据.xlsx', pd.read_excel)
    ]

    for file_path, reader in file_paths:
        if os.path.exists(file_path):
            try:
                data = reader(file_path)
                source = file_path
                print(f"✓ 成功从 '{file_path}' 导入数据")
                break
            except Exception as e:
                print(f"  - 尝试 '{file_path}' 失败: {e}")
                continue

    # 如果没有找到文件，生成示例数据
    if data is None:
        print("! 未找到数据文件，将生成示例数据")
        np.random.seed(42)
        n_samples = 100
        data = pd.DataFrame({
            COLUMN_CONFIG['pollutant']: np.random.choice(['重金属', '有机物', '氮磷营养物', '悬浮物'], n_samples),
            COLUMN_CONFIG['ph']: np.random.uniform(4, 9, n_samples),
            COLUMN_CONFIG['temperature']: np.random.uniform(15, 35, n_samples),
            COLUMN_CONFIG['nutrient']: np.random.choice(['富营养', '中营养', '贫营养'], n_samples),
            COLUMN_CONFIG['target']: np.random.uniform(30, 95, n_samples),
        })
        source = "示例数据"

    return data, source


def auto_match_columns(df, config):
    """
    自动匹配列名（增加模糊匹配输出更多诊断信息）
    """
    matched_config = config.copy()  # 保留原始配置
    for key, default_col in config.items():
        if default_col not in df.columns:
            print(f"! 列名'{default_col}'未找到！尝试匹配...")
            possible_matches = difflib.get_close_matches(default_col, df.columns, n=1, cutoff=0.5)
            if possible_matches:
                matched_col = possible_matches[0]
                matched_config[key] = matched_col
                print(f"  - 自动匹配: 原始为 '{default_col}', 匹配为 '{matched_col}'")
            else:
                raise KeyError(
                    f"列名 '{default_col}' 在数据中不存在！\n"
                    f"可用列: {list(df.columns)}\n"
                    f"请检查输入数据或配置。"
                )
    return matched_config

# ===================== 优化步骤 1: 引入新特征 =====================
def new_feature_engineering(df, config):
    """
    添加新特征：交互特征、多项式特征等
    """
    df['ph*temperature'] = df[config['ph']] * df[config['temperature']]
    df['temperature_squared'] = df[config['temperature']] ** 2
    df['ph_squared'] = df[config['ph']] ** 2
    return df


# ===================== 优化步骤 2: 使用LightGBM算法 =====================
def train_lightgbm(X_train, y_train):
    """
    使用网格搜索+LightGBM
    """
    model = LGBMRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 30]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print("\n最佳参数:", grid_search.best_params_)
    best_model = grid_search.best_estimator_  # 最优模型
    return best_model


def improved_pipeline():
    """
    改进的训练+测试+导出
    """
    COLUMN_CONFIG = {
        'pollutant': '污染物类型',
        'ph': 'ph',
        'temperature': '温度',
        'nutrient': '营养条件',
        'target': '去除率'
    }
    df, _ = load_data()  # 调用复制的load_data

    # 自动匹配列名
    COLUMN_CONFIG = auto_match_columns(df, COLUMN_CONFIG)

    # 特征工程
    df = new_feature_engineering(df, COLUMN_CONFIG)

    # 特征预处理
    le_pollutant = LabelEncoder()
    le_nutrient = LabelEncoder()
    df['污染物类型_编码'] = le_pollutant.fit_transform(df[COLUMN_CONFIG['pollutant']])
    df['营养条件_编码'] = le_nutrient.fit_transform(df[COLUMN_CONFIG['nutrient']])

    features = ['ph', 'temperature', 'ph*temperature', 'temperature_squared', 'ph_squared', '污染物类型_编码', '营养条件_编码']
    target = COLUMN_CONFIG['target']

    # 数据集划分
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    model = train_lightgbm(X_train_scaled, y_train)

    # 预测与评估
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n训练集R2:", train_r2)
    print("测试集R2:", test_r2)
    print("训练集RMSE:", train_rmse)
    print("测试集RMSE:", test_rmse)

    # 导出模型
    joblib.dump({'model': model, 'scaler': scaler, 'features': features}, '改进版预测模型.pkl')
    print("改进模型已保存为 '改进版预测模型.pkl'")

# 执行改进版管道
if __name__ == "__main__":
    improved_pipeline()