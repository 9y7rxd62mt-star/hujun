# -*- coding: utf-8 -*-
"""
污染物去除率预测模型 - 融入模糊算法版
将pH和温度模糊化为隶属度，作为输入特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import joblib
import os

# 尝试导入模糊逻辑库
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("警告: 未安装 scikit-fuzzy，将使用原始数值特征。")
    print("如需使用模糊算法，请执行: pip install scikit-fuzzy")

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("   污染物去除率预测模型 (融入模糊算法) v2.0")
print("=" * 60)

# ==================== 1. 数据导入 ====================
print("\n【第1步】数据导入")
print("-" * 40)

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
        data = generate_sample_data()
        source = "示例数据"

    return data, source


def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n_samples = 100

    data = {
        COLUMN_CONFIG['pollutant']: np.random.choice(['重金属', '有机物', '氮磷营养物', '悬浮物'], n_samples),
        COLUMN_CONFIG['ph']: np.random.uniform(4, 9, n_samples),
        COLUMN_CONFIG['temperature']: np.random.uniform(15, 35, n_samples),
        COLUMN_CONFIG['nutrient']: np.random.choice(['富营养', '中营养', '贫营养'], n_samples),
    }

    df = pd.DataFrame(data)

    # 生成去除率（基于特征的关系）
    removal_rate = []
    for i in range(n_samples):
        base_rate = 70

        if 6.5 <= df[COLUMN_CONFIG['ph']][i] <= 7.5:
            base_rate += np.random.uniform(5, 15)
        else:
            base_rate -= np.random.uniform(0, 10)

        if 20 <= df[COLUMN_CONFIG['temperature']][i] <= 30:
            base_rate += np.random.uniform(5, 15)
        else:
            base_rate -= np.random.uniform(0, 10)

        if df[COLUMN_CONFIG['pollutant']][i] == '重金属':
            base_rate -= np.random.uniform(0, 10)
        elif df[COLUMN_CONFIG['pollutant']][i] == '有机物':
            base_rate += np.random.uniform(5, 15)

        if df[COLUMN_CONFIG['nutrient']][i] == '富营养':
            base_rate += np.random.uniform(5, 10)
        elif df[COLUMN_CONFIG['nutrient']][i] == '贫营养':
            base_rate -= np.random.uniform(0, 5)

        removal_rate.append(max(30, min(99, base_rate + np.random.normal(0, 5))))

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

print("\n数据统计描述:")
print(df.describe())

print("\n缺失值检查:")
print(df.isnull().sum())

# ==================== 2. 数据预处理 ====================
print("\n" + "=" * 60)
print("【第2步】数据预处理")
print("=" * 60)

# 处理缺失值
df = df.dropna()
print(f"删除缺失值后数据形状: {df.shape}")


# 自动匹配列名
def auto_match_columns(df, config):
    """自动匹配列名"""
    matched_config = config.copy()
    df_columns_lower = [col.lower() for col in df.columns]

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
print("\n" + "=" * 60)
print("【第3步】模糊化处理 (将pH和温度转换为模糊隶属度)")
print("=" * 60)


def fuzzify_features(df, ph_col, temp_col):
    """
    对pH和温度进行模糊化，返回模糊化后的特征DataFrame
    模糊集合:
      pH: 酸性(Low), 中性(Medium), 碱性(High)
      温度: 低温(Low), 中温(Medium), 高温(High)
    使用三角形隶属函数
    """
    if not FUZZY_AVAILABLE:
        print("! scikit-fuzzy 未安装，将使用原始数值特征")
        return df[[ph_col, temp_col]].copy()

    # 定义pH的隶属函数范围 (根据实际数据范围调整)
    # 酸性: 4-6.5, 中性: 5.5-8, 碱性: 7-9
    ph_low = np.linspace(3, 7, 100)  # 酸性
    ph_med = np.linspace(5, 8, 100)  # 中性
    ph_high = np.linspace(6.5, 10, 100)  # 碱性

    # 定义温度的隶属函数范围 (15-35℃)
    temp_low = np.linspace(10, 22, 100)  # 低温
    temp_med = np.linspace(18, 30, 100)  # 中温
    temp_high = np.linspace(26, 40, 100)  # 高温

    # 计算每个样本的隶属度
    ph_vals = df[ph_col].values
    temp_vals = df[temp_col].values

    # 使用三角形隶属函数
    ph_low_mf = fuzz.trimf(ph_vals, [3, 4.5, 6])  # 三角形顶点: 3, 4.5, 6
    ph_med_mf = fuzz.trimf(ph_vals, [5.5, 7, 8])  # 顶点: 5.5, 7, 8
    ph_high_mf = fuzz.trimf(ph_vals, [7, 8.5, 10])  # 顶点: 7, 8.5, 10

    temp_low_mf = fuzz.trimf(temp_vals, [10, 17, 24])  # 顶点: 10, 17, 24
    temp_med_mf = fuzz.trimf(temp_vals, [18, 25, 32])  # 顶点: 18, 25, 32
    temp_high_mf = fuzz.trimf(temp_vals, [26, 33, 40])  # 顶点: 26, 33, 40

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

# 特征标准化 (对模糊特征和编码特征都进行标准化)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 4. 模型训练 ====================
print("\n" + "=" * 60)
print("【第4步】模型训练 (随机森林回归)")
print("=" * 60)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✓ 模型训练完成！")

# ==================== 5. 模型评估 ====================
print("\n" + "=" * 60)
print("【第5步】模型评估")
print("=" * 60)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n训练集 R² 分数: {train_r2:.4f}")
print(f"测试集 R² 分数: {test_r2:.4f}")
print(f"训练集 RMSE: {train_rmse:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}")
print(f"训练集 MAE: {train_mae:.4f}")
print(f"测试集 MAE: {test_mae:.4f}")

# 交叉验证
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\n5折交叉验证 R² 分数: {cv_scores}")
print(f"交叉验证平均 R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ==================== 6. 特征重要性 ====================
print("\n" + "=" * 60)
print("【第6步】特征重要性分析")
print("=" * 60)

feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)

print("\n特征重要性排序:")
print(feature_importance.to_string(index=False))

# ==================== 7. 可视化 ====================
print("\n" + "=" * 60)
print("【第7步】生成可视化图表")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('污染物去除率预测模型评估 (融入模糊算法)', fontsize=16, fontweight='bold')

# 图1：训练集预测vs实际
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('实际去除率 (%)')
axes[0, 0].set_ylabel('预测去除率 (%)')
axes[0, 0].set_title(f'训练集: 预测值 vs 实际值 (R²={train_r2:.3f})')
axes[0, 0].grid(True, alpha=0.3)

# 图2：测试集预测vs实际
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('实际去除率 (%)')
axes[0, 1].set_ylabel('预测去除率 (%)')
axes[0, 1].set_title(f'测试集: 预测值 vs 实际值 (R²={test_r2:.3f})')
axes[0, 1].grid(True, alpha=0.3)

# 图3：特征重要性
axes[0, 2].barh(feature_importance['特征'], feature_importance['重要性'], color='skyblue')
axes[0, 2].set_xlabel('重要性')
axes[0, 2].set_title('特征重要性分析 (模糊特征)')
axes[0, 2].grid(True, alpha=0.3)

# 图4：残差分布
residuals = y_test - y_test_pred
axes[1, 0].hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('残差 (实际值 - 预测值)')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('测试集残差分布')
axes[1, 0].axvline(x=0, color='r', linestyle='--')
axes[1, 0].grid(True, alpha=0.3)

# 图5：去除率分布对比
axes[1, 1].hist(y_test, bins=15, alpha=0.5, label='实际值', edgecolor='black', color='blue')
axes[1, 1].hist(y_test_pred, bins=15, alpha=0.5, label='预测值', edgecolor='black', color='red')
axes[1, 1].set_xlabel('去除率 (%)')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title('去除率分布对比')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 图6：按污染物类型的误差
error_by_type = pd.DataFrame({
    '实际值': y_test.values,
    '预测值': y_test_pred,
    '污染物类型': df.loc[y_test.index, COLUMN_CONFIG['pollutant']].values
})
error_by_type['误差'] = abs(error_by_type['实际值'] - error_by_type['预测值'])
mean_error = error_by_type.groupby('污染物类型')['误差'].mean().sort_values()

axes[1, 2].bar(mean_error.index, mean_error.values, color='lightcoral')
axes[1, 2].set_xlabel('污染物类型')
axes[1, 2].set_ylabel('平均绝对误差 (%)')
axes[1, 2].set_title('不同污染物类型的预测误差')
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('模型评估结果_模糊算法.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图表已保存为 '模型评估结果_模糊算法.png'")

# ==================== 8. 保存模型 ====================
print("\n" + "=" * 60)
print("【第8步】保存模型")
print("=" * 60)

model_components = {
    'model': model,
    'scaler': scaler,
    'le_pollutant': le_pollutant,
    'le_nutrient': le_nutrient,
    'features': list(X.columns),
    'column_config': COLUMN_CONFIG,
    'fuzzy_available': FUZZY_AVAILABLE
}

joblib.dump(model_components, '污染物去除率预测模型_模糊算法.pkl')
print("✓ 模型已保存为 '污染物去除率预测模型_模糊算法.pkl'")

# ==================== 9. 预测函数 ====================
print("\n" + "=" * 60)
print("【第9步】预测功能 (含模糊化)")
print("=" * 60)


def predict_removal_rate(ph, temperature, pollutant_type, nutrient_condition):
    """
    预测污染物去除率 (包含模糊化预处理)

    参数:
    ph: float - pH值
    temperature: float - 温度(°C)
    pollutant_type: str - 污染物类型
    nutrient_condition: str - 营养条件

    返回:
    float: 预测去除率(%)
    """
    # 1. 模糊化处理
    if FUZZY_AVAILABLE:
        # 构建临时DataFrame用于模糊化
        temp_df = pd.DataFrame({COLUMN_CONFIG['ph']: [ph], COLUMN_CONFIG['temperature']: [temperature]})
        fuzzy_input = fuzzify_features(temp_df, COLUMN_CONFIG['ph'], COLUMN_CONFIG['temperature'])
    else:
        # 若模糊库不可用，直接使用原始数值
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

    # 确保列顺序与训练时一致
    X_new = X_new[list(X.columns)]

    # 4. 标准化
    X_new_scaled = scaler.transform(X_new)

    # 5. 预测
    prediction = model.predict(X_new_scaled)[0]

    return max(0, min(100, prediction))


# 交互式预测
print("\n" + "-" * 40)
print("交互式预测")
print("-" * 40)
print(f"可用的污染物类型: {list(le_pollutant.classes_)}")
print(f"可用的营养条件: {list(le_nutrient.classes_)}")

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

        predicted_rate = predict_removal_rate(ph_input, temp_input, pollutant_input, nutrient_input)

        print("\n" + "=" * 40)
        print("预测结果")
        print("=" * 40)
        print(f"pH值: {ph_input}")
        print(f"温度: {temp_input}°C")
        print(f"污染物类型: {pollutant_input}")
        print(f"营养条件: {nutrient_input}")
        print(f"预测去除率: {predicted_rate:.2f}%")
        print("=" * 40)
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

# ==================== 10. 导出预测结果 ====================
print("\n【第10步】导出预测结果")
print("-" * 40)

test_results = pd.DataFrame({
    '实际去除率': y_test.values,
    '预测去除率': y_test_pred,
    '绝对误差': abs(y_test.values - y_test_pred),
    '相对误差(%)': abs(y_test.values - y_test_pred) / y_test.values * 100
})

# 添加原始pH和温度（未模糊化的）
test_results['原始pH'] = df.loc[y_test.index, COLUMN_CONFIG['ph']].values
test_results['原始温度'] = df.loc[y_test.index, COLUMN_CONFIG['temperature']].values

# 添加模糊特征（可选）
for col in fuzzy_features.columns:
    test_results[col] = fuzzy_features.loc[y_test.index, col].values

# 添加分类变量
test_results[COLUMN_CONFIG['pollutant']] = df.loc[y_test.index, COLUMN_CONFIG['pollutant']].values
test_results[COLUMN_CONFIG['nutrient']] = df.loc[y_test.index, COLUMN_CONFIG['nutrient']].values

test_results.to_excel('预测结果对比_模糊算法.xlsx', index=False)
print("✓ 预测结果已保存为 '预测结果对比_模糊算法.xlsx'")

print("\n" + "=" * 60)
print("                    程序执行完毕！")
print("=" * 60)
print("\n生成的文件:")
print("  1. 模型评估结果_模糊算法.png - 可视化图表")
print("  2. 污染物去除率预测模型_模糊算法.pkl - 训练好的模型")
print("  3. 预测结果对比_模糊算法.xlsx - 测试集预测结果")
print("\n" + "=" * 60)