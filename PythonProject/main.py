# -*- coding: utf-8 -*-
"""
污染物去除率预测 - 集成学习模型
直接使用预处理后的机器学习就绪数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
import joblib

# 尝试导入额外的集成库
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

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("污染物去除率预测 - 集成学习模型 (基于预处理数据)")
print("=" * 70)

# ==================== 1. 加载数据 ====================
print("\n【第1步】加载机器学习就绪数据")
print("-" * 50)

data_file = "污染物去除率_机器学习就绪数据.xlsx"
df = pd.read_excel(data_file)

print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print("\n前5行:")
print(df.head())

# 分离特征和目标变量
target_col = 'removal_rate'
if target_col not in df.columns:
    # 如果列名是中文，尝试匹配
    if '去除率' in df.columns:
        target_col = '去除率'
    else:
        raise KeyError(f"未找到目标列 {target_col}，请检查列名")

y = df[target_col]
X = df.drop(columns=[target_col])

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# ==================== 2. 划分训练集和测试集 ====================
print("\n【第2步】划分训练集和测试集")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")

# ==================== 3. 定义基础模型 ====================
print("\n【第3步】定义基础模型")
print("-" * 50)

base_models = {}

# 随机森林
base_models['RandomForest'] = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)

# 梯度提升树
base_models['GradientBoosting'] = HistGradientBoostingRegressor(
    max_iter=100, max_depth=5, learning_rate=0.1, random_state=42
)

# XGBoost
if XGB_AVAILABLE:
    base_models['XGBoost'] = XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, verbosity=0
    )

# LightGBM
if LGBM_AVAILABLE:
    base_models['LightGBM'] = LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, verbose=-1
    )

print(f"可用基础模型: {list(base_models.keys())}")

# ==================== 4. 训练并评估基础模型 ====================
print("\n【第4步】训练基础模型")
print("-" * 50)

results = []
trained_models = {}

for name, model in base_models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results.append({
        'model': name,
        'test_r2': r2,
        'test_mae': mae,
        'test_rmse': rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    })
    print(f"  ✓ 测试集 R²: {r2:.4f}, MAE: {mae:.2f}%, RMSE: {rmse:.2f}%")
    print(f"  ✓ 交叉验证 R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results).sort_values('test_r2', ascending=False)
print("\n基础模型性能排序:")
print(results_df[['model', 'test_r2', 'test_mae', 'cv_mean']].to_string(index=False))

# ==================== 5. 集成学习 ====================
print("\n【第5步】集成学习")
print("-" * 50)

# 5.1 加权平均集成
weights = results_df['test_r2'] / results_df['test_r2'].sum()
weighted_ensemble = VotingRegressor(
    estimators=[(name, trained_models[name]) for name in results_df['model']],
    weights=weights
)
weighted_ensemble.fit(X_train, y_train)
y_pred_weighted = weighted_ensemble.predict(X_test)
weighted_r2 = r2_score(y_test, y_pred_weighted)
weighted_mae = mean_absolute_error(y_test, y_pred_weighted)
print(f"\n✓ 加权平均集成 R²: {weighted_r2:.4f}, MAE: {weighted_mae:.2f}%")

# 5.2 Stacking 集成（使用前3个最优模型）
top_models = results_df.head(3)['model'].values
stacking_estimators = [(name, trained_models[name]) for name in top_models]
stacking_model = StackingRegressor(
    estimators=stacking_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)
stacking_r2 = r2_score(y_test, y_pred_stacking)
stacking_mae = mean_absolute_error(y_test, y_pred_stacking)
print(f"✓ Stacking集成 R²: {stacking_r2:.4f}, MAE: {stacking_mae:.2f}%")

# ==================== 6. 选择最终模型 ====================
print("\n【第6步】选择最终模型")
print("-" * 50)

final_models = {
    '最佳基础模型': trained_models[results_df.iloc[0]['model']],
    '加权平均集成': weighted_ensemble,
    'Stacking集成': stacking_model
}
final_scores = {
    '最佳基础模型': results_df.iloc[0]['test_r2'],
    '加权平均集成': weighted_r2,
    'Stacking集成': stacking_r2
}
best_name = max(final_scores, key=final_scores.get)
final_model = final_models[best_name]

print(f"最终选择: {best_name} (R² = {final_scores[best_name]:.4f})")

# ==================== 7. 可视化 ====================
print("\n【第7步】生成可视化图表")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('集成学习模型评估结果', fontsize=16, fontweight='bold')

# 图1：模型性能对比
model_names = list(final_scores.keys())
scores = list(final_scores.values())
colors = ['lightblue' if n != best_name else 'lightgreen' for n in model_names]
axes[0, 0].bar(model_names, scores, color=colors)
axes[0, 0].set_ylabel('R² 分数')
axes[0, 0].set_title('模型性能对比')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 图2：最终模型预测 vs 实际
y_final_pred = final_model.predict(X_test)
axes[0, 1].scatter(y_test, y_final_pred, alpha=0.6, color='green', edgecolors='black')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('实际去除率 (%)')
axes[0, 1].set_ylabel('预测去除率 (%)')
axes[0, 1].set_title(f'{best_name}: 预测 vs 实际 (R²={final_scores[best_name]:.3f})')
axes[0, 1].grid(True, alpha=0.3)

# 图3：基础模型 R² 对比
base_names = results_df['model'].values
base_r2 = results_df['test_r2'].values
axes[0, 2].barh(base_names, base_r2, color='skyblue')
axes[0, 2].set_xlabel('R² 分数')
axes[0, 2].set_title('基础模型 R² 对比')
axes[0, 2].grid(True, alpha=0.3)

# 图4：残差分布
residuals = y_test - y_final_pred
axes[1, 0].hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('残差 (实际 - 预测)')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('最终模型残差分布')
axes[1, 0].axvline(x=0, color='r', linestyle='--')
axes[1, 0].grid(True, alpha=0.3)

# 图5：预测误差对比
mae_values = [results_df.iloc[0]['test_mae'], weighted_mae, stacking_mae]
mae_names = ['最佳基础', '加权平均', 'Stacking']
axes[1, 1].bar(mae_names, mae_values, color=['lightblue', 'lightgreen', 'lightcoral'])
axes[1, 1].set_ylabel('平均绝对误差 (%)')
axes[1, 1].set_title('预测误差对比')
axes[1, 1].grid(True, alpha=0.3)

# 图6：交叉验证结果
cv_means = results_df['cv_mean'].values
cv_stds = results_df['cv_std'].values
axes[1, 2].bar(base_names, cv_means, yerr=cv_stds, capsize=5, color='lightblue')
axes[1, 2].set_ylabel('交叉验证 R²')
axes[1, 2].set_title('5折交叉验证结果')
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('集成模型评估结果.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图表已保存为 '集成模型评估结果.png'")

# ==================== 8. 特征重要性（如果有） ====================
if hasattr(final_model, 'feature_importances_'):
    print("\n【第8步】特征重要性分析")
    print("-" * 50)
    importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': final_model.feature_importances_
    }).sort_values('重要性', ascending=False)
    print(importance.head(10).to_string(index=False))

    plt.figure(figsize=(10, 8))
    plt.barh(importance['特征'][:15], importance['重要性'][:15], color='skyblue')
    plt.xlabel('重要性')
    plt.title('最终模型特征重要性 (Top 15)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('特征重要性.png', dpi=300)
    plt.show()
    print("✓ 特征重要性图已保存")

# ==================== 9. 保存模型 ====================
print("\n【第9步】保存模型")
print("-" * 50)

model_components = {
    'final_model': final_model,
    'weighted_ensemble': weighted_ensemble,
    'stacking_model': stacking_model,
    'trained_models': trained_models,
    'feature_columns': list(X.columns),
    'best_model_name': best_name,
    'results': results_df.to_dict()
}
joblib.dump(model_components, '污染物去除率_集成模型.pkl')
print("✓ 模型已保存为 '污染物去除率_集成模型.pkl'")

# ==================== 10. 预测示例 ====================
print("\n【第10步】预测示例")
print("-" * 50)

# 使用测试集的前5个样本进行预测演示
sample_X = X_test.iloc[:5]
sample_y_true = y_test.iloc[:5]
sample_y_pred = final_model.predict(sample_X)

print("预测结果对比:")
for i in range(len(sample_X)):
    print(
        f"  样本{i + 1}: 实际={sample_y_true.iloc[i]:.2f}%, 预测={sample_y_pred[i]:.2f}%, 误差={abs(sample_y_true.iloc[i] - sample_y_pred[i]):.2f}%")

print("\n" + "=" * 70)
print("                    模型训练与评估完成！")
print("=" * 70)
print("\n生成的文件:")
print("  1. 集成模型评估结果.png - 模型性能可视化")
print("  2. 特征重要性.png - 特征重要性图")
print("  3. 污染物去除率_集成模型.pkl - 训练好的模型文件")
print("\n使用方法:")
print("  import joblib")
print("  model = joblib.load('污染物去除率_集成模型.pkl')['final_model']")
print("  prediction = model.predict(新数据)")
print("=" * 70)