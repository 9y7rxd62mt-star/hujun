# -*- coding: utf-8 -*-
"""
污染物去除率预测模型 - 集成学习版（含模糊算法）
包含多种集成算法对比和模型融合
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import warnings
import joblib
import os

# 尝试导入额外的集成算法库
try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("提示: 未安装 xgboost，如需使用请执行: pip install xgboost")

try:
    from lightgbm import LGBMRegressor

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("提示: 未安装 lightgbm，如需使用请执行: pip install lightgbm")

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("提示: 未安装 catboost，如需使用请执行: pip install catboost")

# 尝试导入模糊逻辑库
try:
    import skfuzzy as fuzz

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("提示: 未安装 scikit-fuzzy，将使用原始数值特征。如需模糊算法请执行: pip install scikit-fuzzy")

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("   污染物去除率预测模型 - 集成学习版 (含模糊算法) v3.0")
print("=" * 70)

# ==================== 1. 数据导入 ====================
print("\n【第1步】数据导入")
print("-" * 50)

# 定义列名（请根据您的实际数据修改）
COLUMN_CONFIG = {
    'pollutant': '污染物类型',  # 污染物类型列名
    'ph': 'ph',  # pH值列名
    'temperature': '温度',  # 温度列名
    'nutrient': '营养条件',  # 营养条件列名
    'target': '去除率'  # 目标变量列名
}


def load_data():
    """加载数据文件"""
    data = None
    source = ""

    # 尝试不同的文件格式
    file_paths = [
        ('Pollutant Experimental Data.xlsx', pd.read_excel),
        ('Pollutant Experimental Data.csv', lambda f: pd.read_csv(f, encoding='utf-8')),
        ('Pollutant Experimental Data.csv', lambda f: pd.read_csv(f, encoding='gbk')),
        ('Pollutant Experimental Data.xlsx', pd.read_excel),
        ('Pollutant Experimental Data.csv', lambda f: pd.read_csv(f, encoding='utf-8')),
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
        data = generate_sample_data()
        source = "示例数据"

    return data, source


def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n_samples = 200  # 增加样本数以更好地训练集成模型

    data = {
        COLUMN_CONFIG['pollutant']: np.random.choice(['重金属', '有机物', '氮磷营养物', '悬浮物'], n_samples),
        COLUMN_CONFIG['ph']: np.random.uniform(4, 9, n_samples),
        COLUMN_CONFIG['temperature']: np.random.uniform(15, 35, n_samples),
        COLUMN_CONFIG['nutrient']: np.random.choice(['富营养', '中营养', '贫营养'], n_samples),
    }

    df = pd.DataFrame(data)

    # 生成去除率（基于特征的关系，增加非线性关系）
    removal_rate = []
    for i in range(n_samples):
        base_rate = 70

        # pH影响（非线性）
        ph_val = df[COLUMN_CONFIG['ph']][i]
        ph_effect = -10 * (ph_val - 7) ** 2 + 15  # 抛物线，最优在pH=7附近
        base_rate += ph_effect

        # 温度影响（非线性）
        temp_val = df[COLUMN_CONFIG['temperature']][i]
        temp_effect = -0.2 * (temp_val - 25) ** 2 + 10  # 抛物线，最优在25°C附近
        base_rate += temp_effect

        # 污染物类型影响
        if df[COLUMN_CONFIG['pollutant']][i] == '重金属':
            base_rate -= np.random.uniform(5, 15)
        elif df[COLUMN_CONFIG['pollutant']][i] == '有机物':
            base_rate += np.random.uniform(5, 20)
        elif df[COLUMN_CONFIG['pollutant']][i] == '氮磷营养物':
            base_rate += np.random.uniform(0, 10)

        # 营养条件影响
        if df[COLUMN_CONFIG['nutrient']][i] == '富营养':
            base_rate += np.random.uniform(5, 12)
        elif df[COLUMN_CONFIG['nutrient']][i] == '贫营养':
            base_rate -= np.random.uniform(0, 8)

        # 交互效应
        if 6.5 <= ph_val <= 7.5 and 20 <= temp_val <= 30:
            base_rate += np.random.uniform(3, 8)

        removal_rate.append(max(20, min(98, base_rate + np.random.normal(0, 3))))

    df[COLUMN_CONFIG['target']] = removal_rate
    return df


# 加载数据
df, data_source = load_data()

print(f"\n数据来源: {data_source}")
print(f"数据形状: {df.shape}")
print(f"数据列: {list(df.columns)}")

print("\n数据前5行:")
print(df.head())

print("\n数据基本信息:")
print(df.info())

# ==================== 2. 数据预处理 ====================
print("\n" + "=" * 70)
print("【第2步】数据预处理")
print("=" * 70)

# 处理缺失值
df = df.dropna()
print(f"删除缺失值后数据形状: {df.shape}")


# 自动匹配列名
def auto_match_columns(df, config):
    """自动匹配列名"""
    matched_config = config.copy()

    for key, default_col in config.items():
        if default_col not in df.columns:
            # 尝试查找相似的列名
            found = False
            for col in df.columns:
                col_lower = col.lower()
                if key in col_lower or default_col.lower() in col_lower:
                    matched_config[key] = col
                    print(f"✓ 自动匹配 {key} 列: '{col}'")
                    found = True
                    break
            if not found:
                print(f"! 未找到 {key} 列，将使用默认列名: '{default_col}'")
    return matched_config


COLUMN_CONFIG = auto_match_columns(df, COLUMN_CONFIG)

# 对分类变量进行编码
le_pollutant = LabelEncoder()
le_nutrient = LabelEncoder()

df['污染物类型_编码'] = le_pollutant.fit_transform(df[COLUMN_CONFIG['pollutant']])
df['营养条件_编码'] = le_nutrient.fit_transform(df[COLUMN_CONFIG['nutrient']])

print("\n污染物类型编码映射:")
for i, category in enumerate(le_pollutant.classes_):
    print(f"  {category} -> {i}")

print("\n营养条件编码映射:")
for i, category in enumerate(le_nutrient.classes_):
    print(f"  {category} -> {i}")

# ==================== 3. 模糊化处理 ====================
print("\n" + "=" * 70)
print("【第3步】模糊化处理 (将pH和温度转换为模糊隶属度)")
print("=" * 70)


def fuzzify_features(df, ph_col, temp_col):
    """对pH和温度进行模糊化，返回模糊化后的特征DataFrame"""
    if not FUZZY_AVAILABLE:
        print("! scikit-fuzzy 未安装，将使用原始数值特征")
        return df[[ph_col, temp_col]].copy()

    # 定义隶属函数
    ph_vals = df[ph_col].values
    temp_vals = df[temp_col].values

    # 使用三角形隶属函数
    ph_low_mf = fuzz.trimf(ph_vals, [3, 4.5, 6])  # 酸性
    ph_med_mf = fuzz.trimf(ph_vals, [5.5, 7, 8])  # 中性
    ph_high_mf = fuzz.trimf(ph_vals, [7, 8.5, 10])  # 碱性

    temp_low_mf = fuzz.trimf(temp_vals, [10, 17, 24])  # 低温
    temp_med_mf = fuzz.trimf(temp_vals, [18, 25, 32])  # 中温
    temp_high_mf = fuzz.trimf(temp_vals, [26, 33, 40])  # 高温

    # 构建模糊特征DataFrame
    fuzzy_df = pd.DataFrame({
        'pH_酸性': ph_low_mf,
        'pH_中性': ph_med_mf,
        'pH_碱性': ph_high_mf,
        '温度_低温': temp_low_mf,
        '温度_中温': temp_med_mf,
        '温度_高温': temp_high_mf
    })

    print("✓ 模糊化完成，生成6个模糊特征:")
    print("  - pH_酸性, pH_中性, pH_碱性")
    print("  - 温度_低温, 温度_中温, 温度_高温")
    return fuzzy_df


# 应用模糊化
fuzzy_features = fuzzify_features(df, COLUMN_CONFIG['ph'], COLUMN_CONFIG['temperature'])

# 原始分类特征
categorical_features = df[['污染物类型_编码', '营养条件_编码']].reset_index(drop=True)

# 合并所有特征
X = pd.concat([fuzzy_features, categorical_features], axis=1)
y = df[COLUMN_CONFIG['target']].reset_index(drop=True)

print(f"\n特征形状: {X.shape}")
print(f"特征列: {list(X.columns)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 4. 基础模型定义 ====================
print("\n" + "=" * 70)
print("【第4步】定义基础模型")
print("=" * 70)

# 定义基础模型
base_models = {}

# 随机森林
base_models['RandomForest'] = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)

# 梯度提升树
base_models['GradientBoosting'] = GradientBoostingRegressor(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    random_state=42
)

# XGBoost
if XGB_AVAILABLE:
    base_models['XGBoost'] = XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, verbosity=0
    )
else:
    print("! XGBoost不可用，将跳过")

# LightGBM
if LGBM_AVAILABLE:
    base_models['LightGBM'] = LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, verbose=-1
    )
else:
    print("! LightGBM不可用，将跳过")

# CatBoost
if CATBOOST_AVAILABLE:
    base_models['CatBoost'] = CatBoostRegressor(
        iterations=100, depth=6, learning_rate=0.1,
        random_seed=42, verbose=False
    )
else:
    print("! CatBoost不可用，将跳过")

print(f"\n可用模型数量: {len(base_models)}")

# ==================== 5. 模型训练和评估 ====================
print("\n" + "=" * 70)
print("【第5步】模型训练和评估")
print("=" * 70)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """评估单个模型"""
    # 训练
    model.fit(X_train, y_train)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    return {
        'model_name': model_name,
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


# 训练所有基础模型
results = []
trained_models = {}

print("\n开始训练基础模型...")
for name, model in base_models.items():
    print(f"\n训练 {name}...")
    result = evaluate_model(model, X_train_scaled, X_test_scaled,
                            y_train, y_test, name)
    results.append(result)
    trained_models[name] = result['model']

    print(f"  ✓ 测试集 R²: {result['test_r2']:.4f}")
    print(f"  ✓ 测试集 MAE: {result['test_mae']:.2f}%")
    print(f"  ✓ 交叉验证 R²: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")

# 转换为DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_r2', ascending=False)

print("\n" + "=" * 70)
print("基础模型性能对比:")
print("=" * 70)
print(results_df[['model_name', 'test_r2', 'test_mae', 'test_rmse', 'cv_mean']].to_string(index=False))

# ==================== 6. 集成学习 ====================
print("\n" + "=" * 70)
print("【第6步】集成学习")
print("=" * 70)

# 6.1 加权平均集成
print("\n6.1 加权平均集成")

# 根据R²分数计算权重
weights = results_df['test_r2'] / results_df['test_r2'].sum()
weighted_ensemble = VotingRegressor(
    estimators=[(name, trained_models[name]) for name in results_df['model_name']],
    weights=weights
)

weighted_ensemble.fit(X_train_scaled, y_train)
y_pred_weighted = weighted_ensemble.predict(X_test_scaled)

weighted_r2 = r2_score(y_test, y_pred_weighted)
weighted_mae = mean_absolute_error(y_test, y_pred_weighted)

print(f"✓ 加权平均集成 R²: {weighted_r2:.4f}")
print(f"✓ 加权平均集成 MAE: {weighted_mae:.2f}%")
print(f"  模型权重:")
for name, weight in zip(results_df['model_name'], weights):
    print(f"    - {name}: {weight:.3f}")

# 6.2 Stacking集成
print("\n6.2 Stacking集成")

# 选择前3个最好的模型作为基础模型
top_models = results_df.head(3)['model_name'].values
stacking_estimators = [(name, trained_models[name]) for name in top_models]

stacking_model = StackingRegressor(
    estimators=stacking_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

stacking_model.fit(X_train_scaled, y_train)
y_pred_stacking = stacking_model.predict(X_test_scaled)

stacking_r2 = r2_score(y_test, y_pred_stacking)
stacking_mae = mean_absolute_error(y_test, y_pred_stacking)

print(f"✓ Stacking集成 R²: {stacking_r2:.4f}")
print(f"✓ Stacking集成 MAE: {stacking_mae:.2f}%")
print(f"  基础模型: {', '.join(top_models)}")
print(f"  元模型: Ridge回归")

# ==================== 7. 模型优化 ====================
print("\n" + "=" * 70)
print("【第7步】最佳模型优化")
print("=" * 70)

# 选择表现最好的模型进行优化
best_model_name = results_df.iloc[0]['model_name']
best_base_model = trained_models[best_model_name]

print(f"最佳基础模型: {best_model_name}")

# 定义超参数网格
grid_search = None
if best_model_name == 'RandomForest':
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [8, 10, 12],
        'min_samples_split': [3, 5, 7],
        'min_samples_leaf': [1, 2, 3]
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, cv=5, scoring='r2', n_jobs=-1
    )
elif best_model_name == 'GradientBoosting':
    param_grid = {
        'n_estimators': [80, 100, 120],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'min_samples_split': [4, 5, 6]
    }
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid, cv=5, scoring='r2', n_jobs=-1
    )
elif best_model_name == 'XGBoost' and XGB_AVAILABLE:
    param_grid = {
        'n_estimators': [80, 100, 120],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid, cv=5, scoring='r2', n_jobs=-1
    )
elif best_model_name == 'LightGBM' and LGBM_AVAILABLE:
    param_grid = {
        'n_estimators': [80, 100, 120],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'num_leaves': [31, 50, 70]
    }
    grid_search = GridSearchCV(
        LGBMRegressor(random_state=42, verbose=-1),
        param_grid, cv=5, scoring='r2', n_jobs=-1
    )
else:
    print(f"! 不支持的模型类型或库未安装: {best_model_name}")
    optimized_model = best_base_model

if grid_search is not None:
    print(f"正在进行超参数优化...")
    grid_search.fit(X_train_scaled, y_train)
    optimized_model = grid_search.best_estimator_
    print(f"✓ 最佳参数: {grid_search.best_params_}")
    print(f"✓ 最佳交叉验证分数: {grid_search.best_score_:.4f}")
else:
    optimized_model = best_base_model
    print("! 跳过超参数优化")

# 评估优化后的模型
y_pred_optimized = optimized_model.predict(X_test_scaled)
optimized_r2 = r2_score(y_test, y_pred_optimized)
optimized_mae = mean_absolute_error(y_test, y_pred_optimized)

print(f"\n优化后模型性能:")
print(f"  R²: {optimized_r2:.4f} (提升: {(optimized_r2 - results_df.iloc[0]['test_r2']) * 100:.2f}%)")
print(f"  MAE: {optimized_mae:.2f}%")

# ==================== 8. 最终模型选择 ====================
print("\n" + "=" * 70)
print("【第8步】最终模型选择")
print("=" * 70)

# 比较所有模型
final_models = {
    '最佳基础模型': best_base_model,
    '加权平均集成': weighted_ensemble,
    'Stacking集成': stacking_model,
    '优化模型': optimized_model
}

final_scores = {
    '最佳基础模型': results_df.iloc[0]['test_r2'],
    '加权平均集成': weighted_r2,
    'Stacking集成': stacking_r2,
    '优化模型': optimized_r2
}

# 选择最终模型
best_final_model_name = max(final_scores, key=final_scores.get)
final_model = final_models[best_final_model_name]

print(f"\n模型性能对比:")
for name, score in final_scores.items():
    print(f"  {name}: R² = {score:.4f}")

print(f"\n✓ 选择最终模型: {best_final_model_name}")
print(f"  R² = {final_scores[best_final_model_name]:.4f}")

# ==================== 9. 可视化 ====================
print("\n" + "=" * 70)
print("【第9步】生成可视化图表")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('污染物去除率预测模型 - 集成学习评估', fontsize=16, fontweight='bold')

# 图1：模型性能对比
model_names = list(final_scores.keys())
scores = list(final_scores.values())
colors = ['lightblue' if name != best_final_model_name else 'lightgreen' for name in model_names]
axes[0, 0].bar(model_names, scores, color=colors)
axes[0, 0].set_ylabel('R² 分数')
axes[0, 0].set_title('模型性能对比')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 图2：最终模型预测vs实际
y_final_pred = final_model.predict(X_test_scaled)
axes[0, 1].scatter(y_test, y_final_pred, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('实际去除率 (%)')
axes[0, 1].set_ylabel('预测去除率 (%)')
axes[0, 1].set_title(f'最终模型: 预测值 vs 实际值 (R²={final_scores[best_final_model_name]:.3f})')
axes[0, 1].grid(True, alpha=0.3)

# 图3：基础模型R²对比
base_model_names = results_df['model_name'].values
base_scores = results_df['test_r2'].values
axes[0, 2].barh(base_model_names, base_scores, color='skyblue')
axes[0, 2].set_xlabel('R² 分数')
axes[0, 2].set_title('基础模型性能对比')
axes[0, 2].grid(True, alpha=0.3)

# 图4：残差分布
residuals = y_test - y_final_pred
axes[1, 0].hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('残差 (实际值 - 预测值)')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('最终模型残差分布')
axes[1, 0].axvline(x=0, color='r', linestyle='--')
axes[1, 0].grid(True, alpha=0.3)

# 图5：预测误差对比
mae_values = [results_df.iloc[0]['test_mae'], weighted_mae, stacking_mae, optimized_mae]
mae_names = ['最佳基础', '加权平均', 'Stacking', '优化模型']
axes[1, 1].bar(mae_names, mae_values, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
axes[1, 1].set_ylabel('平均绝对误差 (%)')
axes[1, 1].set_title('预测误差对比')
axes[1, 1].grid(True, alpha=0.3)

# 图6：交叉验证结果
cv_means = results_df['cv_mean'].values
cv_stds = results_df['cv_std'].values
axes[1, 2].bar(base_model_names, cv_means, yerr=cv_stds, capsize=5, color='lightblue')
axes[1, 2].set_ylabel('交叉验证 R² 分数')
axes[1, 2].set_title('5折交叉验证结果')
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('模型评估结果_集成学习.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图表已保存为 '模型评估结果_集成学习.png'")

# ==================== 10. 特征重要性 ====================
print("\n" + "=" * 70)
print("【第10步】特征重要性分析")
print("=" * 70)

# 对最终模型进行特征重要性分析
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': final_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print("\n最终模型特征重要性:")
    print(feature_importance.to_string(index=False))

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['特征'], feature_importance['重要性'], color='skyblue')
    plt.xlabel('重要性')
    plt.title('最终模型特征重要性分析')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('特征重要性_集成学习.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ 特征重要性图已保存为 '特征重要性_集成学习.png'")
elif hasattr(final_model, 'coef_'):
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '系数': final_model.coef_
    })
    print("\n最终模型特征系数:")
    print(feature_importance.to_string(index=False))

# ==================== 11. 保存模型 ====================
print("\n" + "=" * 70)
print("【第11步】保存模型")
print("=" * 70)

# 保存所有重要组件
model_components = {
    'final_model': final_model,
    'scaler': scaler,
    'le_pollutant': le_pollutant,
    'le_nutrient': le_nutrient,
    'features': list(X.columns),
    'column_config': COLUMN_CONFIG,
    'fuzzy_available': FUZZY_AVAILABLE,
    'model_results': results_df.to_dict(),
    'final_model_name': best_final_model_name,
    'all_models': trained_models,
    'ensemble_models': {
        'weighted_ensemble': weighted_ensemble,
        'stacking_ensemble': stacking_model
    }
}

joblib.dump(model_components, '污染物去除率预测模型_集成学习.pkl')
print("✓ 完整模型已保存为 '污染物去除率预测模型_集成学习.pkl'")

# ==================== 12. 预测函数 ====================
print("\n" + "=" * 70)
print("【第12步】预测功能")
print("=" * 70)


def predict_removal_rate(ph, temperature, pollutant_type, nutrient_condition, model_to_use='final'):
    """
    预测污染物去除率

    参数:
    ph: float - pH值
    temperature: float - 温度(°C)
    pollutant_type: str - 污染物类型
    nutrient_condition: str - 营养条件
    model_to_use: str - 使用的模型 ('final', 'weighted', 'stacking', 'best_base')

    返回:
    float: 预测去除率(%)
    """
    # 1. 模糊化处理
    if FUZZY_AVAILABLE:
        temp_df = pd.DataFrame({COLUMN_CONFIG['ph']: [ph], COLUMN_CONFIG['temperature']: [temperature]})
        fuzzy_input = fuzzify_features(temp_df, COLUMN_CONFIG['ph'], COLUMN_CONFIG['temperature'])
    else:
        fuzzy_input = pd.DataFrame({
            COLUMN_CONFIG['ph']: [ph],
            COLUMN_CONFIG['temperature']: [temperature]
        })

    # 2. 编码分类变量
    try:
        poll_code = le_pollutant.transform([pollutant_type])[0]
    except:
        print(f"警告: 未知的污染物类型 '{pollutant_type}'，使用默认编码0")
        poll_code = 0

    try:
        nut_code = le_nutrient.transform([nutrient_condition])[0]
    except:
        print(f"警告: 未知的营养条件 '{nutrient_condition}'，使用默认编码0")
        nut_code = 0

    categorical_input = pd.DataFrame({
        '污染物类型_编码': [poll_code],
        '营养条件_编码': [nut_code]
    })

    # 3. 合并特征
    X_new = pd.concat([fuzzy_input.reset_index(drop=True), categorical_input], axis=1)
    X_new = X_new[list(X.columns)]

    # 4. 标准化
    X_new_scaled = scaler.transform(X_new)

    # 5. 选择模型并预测
    if model_to_use == 'final':
        prediction = final_model.predict(X_new_scaled)[0]
    elif model_to_use == 'weighted':
        prediction = weighted_ensemble.predict(X_new_scaled)[0]
    elif model_to_use == 'stacking':
        prediction = stacking_model.predict(X_new_scaled)[0]
    elif model_to_use == 'best_base':
        prediction = best_base_model.predict(X_new_scaled)[0]
    else:
        prediction = final_model.predict(X_new_scaled)[0]

    return max(0, min(100, prediction))


# 交互式预测
print("\n" + "-" * 50)
print("交互式预测")
print("-" * 50)
print(f"可用的污染物类型: {list(le_pollutant.classes_)}")
print(f"可用的营养条件: {list(le_nutrient.classes_)}")
print(f"可用的模型: final, weighted, stacking, best_base")

try:
    use_interactive = input("\n是否进行交互式预测? (y/n, 默认y): ").lower() or 'y'

    if use_interactive == 'y':
        print("\n请输入预测参数:")
        ph_input = float(input("pH值 (例如 7.0): ") or "7.0")
        temp_input = float(input("温度 (°C) (例如 25): ") or "25")

        print(f"\n可选污染物类型: {list(le_pollutant.classes_)}")
        pollutant_input = input(f"污染物类型 (默认 {le_pollutant.classes_[0]}): ") or le_pollutant.classes_[0]

        print(f"\n可选营养条件: {list(le_nutrient.classes_)}")
        nutrient_input = input(f"营养条件 (默认 {le_nutrient.classes_[0]}): ") or le_nutrient.classes_[0]

        print(f"\n可选模型类型: final, weighted, stacking, best_base")
        model_choice = input(f"模型类型 (默认 final): ") or 'final'

        # 进行预测
        predicted_rate = predict_removal_rate(ph_input, temp_input, pollutant_input,
                                              nutrient_input, model_choice)

        print("\n" + "=" * 50)
        print("预测结果")
        print("=" * 50)
        print(f"pH值: {ph_input}")
        print(f"温度: {temp_input}°C")
        print(f"污染物类型: {pollutant_input}")
        print(f"营养条件: {nutrient_input}")
        print(f"使用模型: {model_choice}")
        print(f"预测去除率: {predicted_rate:.2f}%")
        print("=" * 50)
    else:
        print("\n示例预测:")
        examples = [
            (7.0, 25, le_pollutant.classes_[0], le_nutrient.classes_[0]),
            (5.5, 18, le_pollutant.classes_[1] if len(le_pollutant.classes_) > 1 else le_pollutant.classes_[0],
             le_nutrient.classes_[1] if len(le_nutrient.classes_) > 1 else le_nutrient.classes_[0]),
            (8.2, 32, le_pollutant.classes_[2] if len(le_pollutant.classes_) > 2 else le_pollutant.classes_[0],
             le_nutrient.classes_[2] if len(le_nutrient.classes_) > 2 else le_nutrient.classes_[0])
        ]

        for i, (ph, temp, pollutant, nutrient) in enumerate(examples, 1):
            pred = predict_removal_rate(ph, temp, pollutant, nutrient)
            print(f"\n示例 {i}:")
            print(f"  pH={ph}, 温度={temp}°C, 污染物={pollutant}, 营养={nutrient}")
            print(f"  预测去除率: {pred:.2f}%")

except Exception as e:
    print(f"\n! 预测过程出现错误: {e}")
    print("使用默认示例进行预测:")
    default_ph = 7.0
    default_temp = 25
    default_pollutant = le_pollutant.classes_[0]
    default_nutrient = le_nutrient.classes_[0]
    pred = predict_removal_rate(default_ph, default_temp, default_pollutant, default_nutrient)
    print(f"\n默认预测:")
    print(f"  pH={default_ph}, 温度={default_temp}°C, 污染物={default_pollutant}, 营养={default_nutrient}")
    print(f"  预测去除率: {pred:.2f}%")

print("\n" + "=" * 70)
print("                    程序执行完毕！")
print("=" * 70)
print("\n生成的文件:")
print("  1. 模型评估结果_集成学习.png - 可视化图表")
print("  2. 特征重要性_集成学习.png - 特征重要性图")
print("  3. 污染物去除率预测模型_集成学习.pkl - 训练好的模型")
print("\n" + "=" * 70)