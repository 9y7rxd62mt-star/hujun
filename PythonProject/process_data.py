import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==================== 1. 读取原始数据 ====================
file_path = "Pollutant Experimental Data.xlsx"  # 请修改为实际路径
df = pd.read_excel(file_path, sheet_name="Sheet1")

print("原始数据形状:", df.shape)
print("原始列名:", df.columns.tolist())
print(df.head(2))

# 重命名列（如果列名有空格或特殊字符，请根据实际调整）
df.columns = df.columns.str.strip()
df.rename(columns={
    "污染物类型": "pollutant",
    "ph": "pH",
    "温度": "temperature",
    "营养条件": "nutrient_condition",
    "去除率": "removal_rate"
}, inplace=True)


# ==================== 2. 提取营养条件中的数值特征 ====================
def extract_nutrient_features(text):
    """从营养条件文本中提取数值特征"""
    if pd.isna(text):
        return {}
    text = str(text)
    features = {}

    # 总溶解固体 (g/L)
    match = re.search(r'总溶解固体\s*([\d\.]+)\s*g/L', text)
    if match:
        features['TDS_gL'] = float(match.group(1))

    # 电导率 (μS/cm)
    match = re.search(r'电导率\s*([\d\.]+)\s*μS/cm', text)
    if match:
        features['conductivity_uS_cm'] = float(match.group(1))

    # 溶解氧 (mg/L)
    match = re.search(r'溶解氧\s*([\d\.]+)\s*mg/L', text)
    if match:
        features['DO_mgL'] = float(match.group(1))

    # 氮浓度 (mg/L)
    match = re.search(r'氮浓度\s*([\d\.]+)\s*mg/L', text)
    if match:
        features['nitrogen_mgL'] = float(match.group(1))

    # PMS 浓度 (可能单位 mM 或 g/L)
    match = re.search(r'PMS\s*([\d\.]+)\s*mM', text)
    if match:
        features['PMS_mM'] = float(match.group(1))
    match = re.search(r'PMS\s*([\d\.]+)\s*g/L', text)
    if match:
        features['PMS_gL'] = float(match.group(1))

    # Fe²⁺ 或 Fe³⁺ 浓度
    match = re.search(r'Fe²⁺\s*([\d\.]+)\s*g/L', text)
    if match:
        features['Fe2_gL'] = float(match.group(1))
    match = re.search(r'Fe³⁺\s*([\d\.]+)\s*g/L', text)
    if match:
        features['Fe3_gL'] = float(match.group(1))

    # 过氧化氢 H₂O₂ (mL/L 或 g/L)
    match = re.search(r'H₂O₂\s*([\d\.]+)\s*mL/L', text)
    if match:
        features['H2O2_mL_L'] = float(match.group(1))
    match = re.search(r'H₂O₂\s*([\d\.]+)\s*g/L', text)
    if match:
        features['H2O2_gL'] = float(match.group(1))

    return features


# 提取所有营养条件特征
nutrient_features_list = df['nutrient'].apply(extract_nutrient_features)
nutrient_df = pd.DataFrame(nutrient_features_list.tolist())
print("提取的营养特征列:", nutrient_df.columns.tolist())

# 将提取的特征合并到原DataFrame
df = pd.concat([df, nutrient_df], axis=1)


# ==================== 3. 污染物类型处理 ====================
# 方案一：基于化学类别的归类（推荐，可解释性强）
def classify_pollutant(name):
    name = str(name)
    if name in ['氯乙烯', '三氯乙烯', '四氯乙烯', '1,1-二氯乙烯', '顺-1,2-二氯乙烯', '反-1,2-二氯乙烯']:
        return 'chloroethylene'
    elif name in ['二氯甲烷', '氯仿', '四氯化碳', '1,1-二氯乙烷', '1,2-二氯乙烷', '1,1,1-三氯乙烷', '1,1,2-三氯乙烷',
                  '1,1,2,2-四氯乙烷']:
        return 'chloroalkane'
    elif name in ['苯', '甲苯', '乙苯', '二甲苯', '苯乙烯', '氯苯', '对二甲苯', '间二甲苯', '邻二甲苯']:
        return 'aromatic'
    elif name in ['乙酸乙酯', '甲基丙烯酸甲酯', '丙烯酸甲酯']:
        return 'ester'
    elif name in ['乙醇', '甲醇', '异丙醇']:
        return 'alcohol'
    elif name in ['正己烷', '己烷']:
        return 'alkane'
    elif name in ['苯酚', '苯胺', '对乙酰氨基酚']:
        return 'phenolic'
    elif '磷酸三丁酯' in name:
        return 'organophosphate'
    else:
        return 'other'


df['pollutant_category'] = df['pollutant'].apply(classify_pollutant)

# 对类别进行 LabelEncoder
le_cat = LabelEncoder()
df['pollutant_cat_encoded'] = le_cat.fit_transform(df['pollutant_category'])

# 方案二：目标编码（可选，需要小心过拟合，这里展示但不直接使用）
# 注意：目标编码应该使用训练集计算，测试集映射。这里仅作演示，实际使用时需要交叉验证。
# 为演示，先对整个数据集计算各污染物的平均去除率作为编码
# pollutant_target_enc = df.groupby('pollutant')['removal_rate'].mean().to_dict()
# df['pollutant_target_enc'] = df['pollutant'].map(pollutant_target_enc)

# ==================== 4. 检查并处理缺失值 ====================
# 数值特征中的缺失值用中位数填充
num_cols = ['pH', 'temperature', 'TDS_gL', 'conductivity_uS_cm', 'DO_mgL', 'nitrogen_mgL', 'PMS_mM', 'PMS_gL', 'Fe2_gL',
            'Fe3_gL', 'H2O2_mL_L', 'H2O2_gL']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ==================== 5. 选择最终用于建模的特征 ====================
# 保留原始标识列（可选），以及所有数值特征 + 污染物编码
feature_cols = ['pH', 'temperature', 'pollutant_cat_encoded']
# 添加提取出的数值特征（只要存在且非全空）
for col in ['TDS_gL', 'conductivity_uS_cm', 'DO_mgL', 'nitrogen_mgL', 'PMS_mM', 'PMS_gL', 'Fe2_gL', 'Fe3_gL',
            'H2O2_mL_L', 'H2O2_gL']:
    if col in df.columns and df[col].notna().any():
        feature_cols.append(col)

# 目标变量
target_col = 'target'

print("最终特征列:", feature_cols)

# 构建新DataFrame用于输出
new_df = df[['pollutant', 'nutrient'] + feature_cols + [target_col]].copy()
# 添加原始去除率供参考
new_df['original_target'] = df['target']

# ==================== 6. 输出新的Excel文件 ====================
output_path = "processed_data_for_ml.xlsx"
new_df.to_excel(output_path, index=False)
print(f"处理完成！新文件已保存为: {output_path}")
print(f"数据形状: {new_df.shape}")
print("\n新数据前5行:")
print(new_df.head())