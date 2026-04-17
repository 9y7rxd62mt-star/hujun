import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取增强后的数据
# ==========================================
file_path = r'C:\Users\15787\PycharmProjects\PythonProject\预测结果_分层采样.xlsx'   # 请确认文件名正确
df = pd.read_excel(file_path)

print("原始数据形状:", df.shape)
print("列名:", df.columns.tolist())

# ==========================================
# 2. 定义特征列和目标列
# ==========================================
# 排除非特征列
exclude_cols = ['去除率', '预测值_训练集', '预测值_测试集', '文献名称', 'num', 'pollutant', 'nutrient', 'Unnamed: 7']
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
# 注意：二值特征（has_*）通常是 int64，也会被包含进来

print(f"使用的特征列 ({len(feature_cols)} 个): {feature_cols}")

X = df[feature_cols].copy()
y = df['去除率'].copy()

# 处理缺失值（用中位数填充）
for col in X.columns:
    if X[col].isna().any():
        X[col].fillna(X[col].median(), inplace=True)

# ==========================================
# 3. 按文献名称分层划分训练集和测试集
# ==========================================
# 如果您的数据中已经包含“文献名称”列
if '文献名称' in df.columns:
    sources = df['文献名称'].dropna().unique()
    train_sources, test_sources = train_test_split(sources, test_size=0.2, random_state=42)
    train_idx = df['文献名称'].isin(train_sources)
    test_idx = df['文献名称'].isin(test_sources)
    print(f"按文献分层：训练集 {train_idx.sum()} 条，测试集 {test_idx.sum()} 条")
else:
    # 如果没有文献名称，则随机划分
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    print("未找到文献名称列，使用随机划分")

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ==========================================
# 4. 训练 XGBoost 模型
# ==========================================
print("\n训练 XGBoost 模型...")
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 评估指标
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n训练集 R² = {train_r2:.4f}, MAE = {train_mae:.2f}")
print(f"测试集 R² = {test_r2:.4f}, MAE = {test_mae:.2f}")

# ==========================================
# 5. 生成可视化图表
# ==========================================
sns.set_style("whitegrid")
fig = plt.figure(figsize=(14, 10))

# 图1：实际值 vs 预测值（训练集）
ax1 = plt.subplot(2, 2, 1)
ax1.scatter(y_train, y_train_pred, alpha=0.6, edgecolors='k', c='royalblue')
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('实际去除率 (%)')
ax1.set_ylabel('预测去除率 (%)')
ax1.set_title(f'训练集 (R² = {train_r2:.3f})')
ax1.grid(True)

# 图2：实际值 vs 预测值（测试集）
ax2 = plt.subplot(2, 2, 2)
ax2.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', c='darkorange')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('实际去除率 (%)')
ax2.set_ylabel('预测去除率 (%)')
ax2.set_title(f'测试集 (R² = {test_r2:.3f})')
ax2.grid(True)

# 图3：残差分布直方图（训练+测试）
ax3 = plt.subplot(2, 2, 3)
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred
ax3.hist(residuals_train, bins=20, alpha=0.5, label=f'训练集 (MAE={train_mae:.1f})', color='royalblue')
ax3.hist(residuals_test, bins=20, alpha=0.5, label=f'测试集 (MAE={test_mae:.1f})', color='darkorange')
ax3.axvline(x=0, color='red', linestyle='--')
ax3.set_xlabel('残差 (实际 - 预测)')
ax3.set_ylabel('频数')
ax3.set_title('残差分布')
ax3.legend()

# 图4：特征重要性（Top 15）
ax4 = plt.subplot(2, 2, 4)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:15]
top_features = [feature_cols[i] for i in indices]
top_importance = importance[indices]
ax4.barh(top_features[::-1], top_importance[::-1], color='teal')
ax4.set_xlabel('特征重要性')
ax4.set_title('Top 15 特征重要性')
ax4.grid(True)

plt.tight_layout()
plt.savefig('model_evaluation_charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n图表已保存为 'model_evaluation_charts.png'")

# ==========================================
# 6. （可选）保存残差分析表
# ==========================================
result_df = df[train_idx | test_idx].copy()
result_df.loc[train_idx, '预测值'] = y_train_pred
result_df.loc[test_idx, '预测值'] = y_test_pred
result_df['残差'] = result_df['去除率'] - result_df['预测值']
result_df['数据集'] = '训练集'
result_df.loc[test_idx, '数据集'] = '测试集'
result_df.to_excel('预测结果_带残差.xlsx', index=False)
print("详细预测结果（含残差）已保存至 '预测结果_带残差.xlsx'")