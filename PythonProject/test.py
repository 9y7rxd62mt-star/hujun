import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. 读取您的数据
# ==========================================
file_path = r'C:\Users\15787\PycharmProjects\PythonProject\Pollutant Experimental Data.xlsx'  # 请确保文件名正确
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 只保留我们需要的列，并重命名 'target' 为 '去除率'
df = df[['pollutant', 'ph', 'temperature', 'nutrient', 'target']].copy()
df.rename(columns={'target': '去除率'}, inplace=True)

# 去除去除率为空的行
df.dropna(subset=['去除率'], inplace=True)
print(f"原始数据量: {len(df)}")

# ==========================================
# 2. 从 'nutrient' 文本中提取关键特征 (核心优化)
# ==========================================
print("正在从 'nutrient' 列提取特征...")

# 定义要搜索的关键词及其对应的特征名
feature_keywords = {
    # === 氧化体系 ===
    'has_pms': 'PMS', 'has_pds': 'PDS', 'has_h2o2': 'H₂O₂',
    'has_fenton': ['Fenton', '芬顿'], 'has_uv': ['UV', '光'],
    'has_ozone': 'O₃', 'has_persulfate': ['过硫酸盐', 'persulfate'],

    # === 催化剂/添加剂 ===
    'has_fe_ion': ['Fe²⁺', 'Fe³⁺', 'Fe(II)', 'Fe(III)'],
    'has_mos2': 'MoS₂', 'has_tio2': 'TiO₂', 'has_cofe2o4': 'CoFe₂O₄',
    'has_biochar': '生物炭', 'has_surfactant': ['Tween', '表面活性剂'],
    'has_chelator': ['柠檬酸', '草酸', 'EDTA'], 'has_abts': 'ABTS',

    # === 生物体系 ===
    'is_biological': ['bio', '菌', '真菌', 'fungus', 'bacteria', 'BTF', '滴滤'],
    'is_anaerobic': '厌氧', 'is_aerobic': '好氧',

    # === 物理/放电 ===
    'has_dbd': 'DBD', 'has_plasma': '放电', 'has_adsorption': '吸附',
}

def extract_features(text):
    """从文本中提取二值特征"""
    if pd.isna(text):
        return {feat: 0 for feat in feature_keywords}
    text = str(text).lower()
    features = {}
    for feat, keywords in feature_keywords.items():
        if isinstance(keywords, list):
            features[feat] = 1 if any(kw.lower() in text for kw in keywords) else 0
        else:
            features[feat] = 1 if keywords.lower() in text else 0
    return features

# 应用提取函数
feature_dicts = df['nutrient'].apply(extract_features)
feature_df = pd.DataFrame(feature_dicts.tolist())

# 从 nutrient 中提取数值特征（如浓度、pH变化等）
def extract_numeric(text):
    """提取常见的数值参数"""
    if pd.isna(text):
        return {'tds_gL': np.nan, 'ec_uScm': np.nan, '氮浓度_mgL': np.nan}
    text = str(text)
    tds = re.search(r'总溶解固体\s*([\d\.]+)\s*g/L', text)
    ec = re.search(r'电导率\s*([\d\.]+)\s*μS/cm', text)
    n_conc = re.search(r'氮浓度\s*([\d\.]+)\s*mg/L', text)
    return {
        'tds_gL': float(tds.group(1)) if tds else np.nan,
        'ec_uScm': float(ec.group(1)) if ec else np.nan,
        '氮浓度_mgL': float(n_conc.group(1)) if n_conc else np.nan,
    }

numeric_dicts = df['nutrient'].apply(extract_numeric)
numeric_df = pd.DataFrame(numeric_dicts.tolist())

# 将所有新特征合并
df_features = pd.concat([df, feature_df, numeric_df], axis=1)

# ==========================================
# 3. 处理污染物名称 -> 简单编码（后续可替换为分子描述符）
# ==========================================
# 使用目标编码或频次编码，这里先用简单的频次编码（比One-Hot更稳定）
pollutant_freq = df_features['pollutant'].map(df_features['pollutant'].value_counts()) / len(df_features)
df_features['pollutant_freq'] = pollutant_freq

# 可选：对高基数的污染物做One-Hot（会增加维度，谨慎使用）
# ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# pollutant_ohe = ohe.fit_transform(df_features[['pollutant']])
# pollutant_ohe_df = pd.DataFrame(pollutant_ohe, columns=ohe.get_feature_names_out(['pollutant']))
# df_features = pd.concat([df_features, pollutant_ohe_df], axis=1)

# ==========================================
# 4. 初步清洗：去除明显矛盾的离群点
# ==========================================
print("正在清洗离群点...")
# 例如：去除去除率为0但体系明显高效的样本（可根据领域知识调整）
# 这里用一个简单规则：如果 has_pms + has_h2o2 + has_uv + has_fenton >= 2 且去除率 < 10，则视为可疑
advanced_cols = ['has_pms', 'has_h2o2', 'has_uv', 'has_fenton', 'has_pds']
df_features['advanced_score'] = df_features[advanced_cols].sum(axis=1)
mask_outlier = (df_features['advanced_score'] >= 2) & (df_features['去除率'] < 10)
print(f"去除可疑离群点数量: {mask_outlier.sum()}")
df_clean = df_features[~mask_outlier].copy()

# ==========================================
# 5. 准备最终特征矩阵和目标变量
# ==========================================
# 选择用于训练的特征列（排除文本列和ID列）
exclude_cols = ['pollutant', 'nutrient', '去除率', 'advanced_score']
feature_cols = [c for c in df_clean.columns if c not in exclude_cols]

# 处理缺失值：用中位数填充
for col in feature_cols:
    if df_clean[col].dtype in ['float64', 'int64']:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

X = df_clean[feature_cols]
y = df_clean['去除率']

print(f"清洗后数据量: {len(X)}")
print(f"最终特征列表: {feature_cols}")

# ==========================================
# 6. 快速验证模型提升效果
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n🎉 新特征工程后的 R² 分数: {r2:.4f}")

# ==========================================
# 7. 保存清洗后、特征丰富的数据集
# ==========================================
output_path = '../TestData_cleaned_with_features.xlsx'
df_clean.to_excel(output_path, index=False)
print(f"\n✅ 清洗后的新数据已保存至: {output_path}")
print("请使用这个新文件重新训练您的模型。")