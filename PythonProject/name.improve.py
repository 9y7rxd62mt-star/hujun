# -*- coding: utf-8 -*-
"""
污染物去除率预测模型 - 数据预处理与特征工程
针对实验数据中的污染物类型和营养条件进行特征提取和编码
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("污染物去除率数据预处理 - 特征工程")
print("=" * 70)

# ==================== 1. 读取原始数据 ====================
print("\n【第1步】读取原始数据")
print("-" * 50)

# 读取Excel文件
df = pd.read_excel('Pollutant_Experimental_Data_English.xlsx', sheet_name='Sheet1')

# 查看数据结构
print(f"原始数据形状: {df.shape}")
print(f"原始列名: {list(df.columns)}")
print("\n前5行数据:")
print(df.head())

# 重命名列（如果列名有变化）
df.columns = ['num', '污染物类型', 'ph', '温度', '营养条件', '去除率', '文献名称', '备注']

# ==================== 2. 污染物类型处理 ====================
print("\n" + "=" * 70)
print("【第2步】污染物类型处理")
print("=" * 70)

# 查看污染物类型分布
print("\n原始污染物类型分布:")
print(df['污染物类型'].value_counts())


# 2.1 污染物类型标准化（合并相似类型）
def standardize_pollutant(name):
    """标准化污染物类型名称"""
    name = str(name).strip()

    # 氯代烃类
    if '氯乙烯' in name:
        return '氯乙烯'
    elif '二氯乙烯' in name:
        return '二氯乙烯'
    elif '三氯乙烯' in name:
        return '三氯乙烯'
    elif '四氯乙烯' in name:
        return '四氯乙烯'
    elif '氯仿' in name or '三氯甲烷' in name:
        return '氯仿'
    elif '四氯化碳' in name:
        return '四氯化碳'
    elif '氯乙烷' in name:
        return '氯乙烷'
    elif '三氯乙烷' in name:
        return '三氯乙烷'
    elif '四氯乙烷' in name:
        return '四氯乙烷'

    # 苯系物
    elif '苯' in name and '氯苯' not in name:
        if '甲苯' in name:
            return '甲苯'
        elif '二甲苯' in name:
            return '二甲苯'
        elif '乙苯' in name:
            return '乙苯'
        elif '苯乙烯' in name:
            return '苯乙烯'
        else:
            return '苯'
    elif '氯苯' in name:
        return '氯苯'

    # 烷烃类
    elif '正己烷' in name:
        return '正己烷'
    elif '己烷' in name:
        return '己烷'

    # 酯类
    elif '乙酸乙酯' in name:
        return '乙酸乙酯'

    # 其他
    elif '二氯甲烷' in name:
        return '二氯甲烷'
    elif '苯胺' in name:
        return '苯胺'
    elif '苯酚' in name:
        return '苯酚'
    elif '磷酸三丁酯' in name:
        return '磷酸三丁酯'
    elif '对乙酰氨基酚' in name:
        return '对乙酰氨基酚'
    else:
        return name


df['污染物类型_标准化'] = df['污染物类型'].apply(standardize_pollutant)

print("\n标准化后污染物类型分布:")
print(df['污染物类型_标准化'].value_counts())

# 2.2 污染物类型编码（Label Encoding）
le_pollutant = LabelEncoder()
df['污染物类型编码'] = le_pollutant.fit_transform(df['污染物类型_标准化'])

print("\n污染物类型编码映射:")
for i, category in enumerate(le_pollutant.classes_):
    print(f"  {category}: {i}")

# 2.3 污染物类型独热编码（One-Hot Encoding，可选）
pollutant_dummies = pd.get_dummies(df['污染物类型_标准化'], prefix='污染物')
df = pd.concat([df, pollutant_dummies], axis=1)

# ==================== 3. 营养条件处理（关键步骤）====================
print("\n" + "=" * 70)
print("【第3步】营养条件特征提取")
print("=" * 70)

print("\n原始营养条件示例:")
print(df['营养条件'].head(10))


# 3.1 提取营养条件中的数值特征
def extract_nutrient_features(nutrient_str):
    """从营养条件文本中提取数值特征"""
    if pd.isna(nutrient_str):
        return {
            '总溶解固体_gL': np.nan,
            '电导率_uScm': np.nan,
            '溶解氧_mgL': np.nan,
            '氮浓度_mgL': np.nan,
            '营养类型': '未知'
        }

    nutrient_str = str(nutrient_str)
    features = {
        '总溶解固体_gL': np.nan,
        '电导率_uScm': np.nan,
        '溶解氧_mgL': np.nan,
        '氮浓度_mgL': np.nan,
        '营养类型': '其他'
    }

    # 提取总溶解固体 (TDS)
    tds_patterns = [
        r'总溶解固体\s*([\d.]+)\s*g/L',
        r'TDS\s*([\d.]+)',
        r'total dissolved solids\s*([\d.]+)'
    ]
    for pattern in tds_patterns:
        match = re.search(pattern, nutrient_str, re.IGNORECASE)
        if match:
            features['总溶解固体_gL'] = float(match.group(1))
            break

    # 提取电导率
    cond_patterns = [
        r'电导率\s*([\d.]+)\s*μS/cm',
        r'conductivity\s*([\d.]+)'
    ]
    for pattern in cond_patterns:
        match = re.search(pattern, nutrient_str, re.IGNORECASE)
        if match:
            features['电导率_uScm'] = float(match.group(1))
            break

    # 提取溶解氧
    do_patterns = [
        r'溶解氧\s*([\d.]+)\s*mg/L',
        r'DO\s*([\d.]+)'
    ]
    for pattern in do_patterns:
        match = re.search(pattern, nutrient_str, re.IGNORECASE)
        if match:
            features['溶解氧_mgL'] = float(match.group(1))
            break

    # 提取氮浓度
    n_patterns = [
        r'氮浓度\s*([\d.]+)\s*mg/L',
        r'nitrogen\s*([\d.]+)'
    ]
    for pattern in n_patterns:
        match = re.search(pattern, nutrient_str, re.IGNORECASE)
        if match:
            features['氮浓度_mgL'] = float(match.group(1))
            break

    # 识别营养类型
    if '富营养' in nutrient_str:
        features['营养类型'] = '富营养'
    elif '中营养' in nutrient_str:
        features['营养类型'] = '中营养'
    elif '贫营养' in nutrient_str:
        features['营养类型'] = '贫营养'
    elif '氮浓度' in nutrient_str:
        if features['氮浓度_mgL'] is not np.nan:
            if features['氮浓度_mgL'] > 500:
                features['营养类型'] = '高氮'
            elif features['氮浓度_mgL'] > 100:
                features['营养类型'] = '中氮'
            else:
                features['营养类型'] = '低氮'
    elif '总溶解固体' in nutrient_str:
        if features['总溶解固体_gL'] is not np.nan:
            if features['总溶解固体_gL'] > 3:
                features['营养类型'] = '高盐'
            elif features['总溶解固体_gL'] > 1:
                features['营养类型'] = '中盐'
            else:
                features['营养类型'] = '低盐'

    return features


# 应用特征提取
nutrient_features = df['营养条件'].apply(extract_nutrient_features)
nutrient_df = pd.DataFrame(nutrient_features.tolist())

# 合并特征
df = pd.concat([df, nutrient_df], axis=1)

print("\n提取的营养条件特征:")
print(nutrient_df.describe())

# 3.2 营养类型编码
le_nutrient_type = LabelEncoder()
df['营养类型编码'] = le_nutrient_type.fit_transform(df['营养类型'])

print("\n营养类型编码映射:")
for i, category in enumerate(le_nutrient_type.classes_):
    print(f"  {category}: {i}")

# 3.3 营养条件综合评分（根据去除率相关性构建）
print("\n计算营养条件综合评分...")

# 计算各营养特征与去除率的相关性
correlations = {}
for col in ['总溶解固体_gL', '电导率_uScm', '溶解氧_mgL', '氮浓度_mgL']:
    if df[col].notna().sum() > 10:  # 至少10个非空值
        corr = df[col].corr(df['去除率'])
        correlations[col] = corr
        print(f"  {col}与去除率相关性: {corr:.3f}")


# 构建营养条件综合评分（基于领域知识）
def calculate_nutrient_score(row):
    """计算营养条件评分（越高越有利于去除）"""
    score = 50  # 基础分

    # 总溶解固体影响（过高或过低都不利）
    if pd.notna(row['总溶解固体_gL']):
        if 1 < row['总溶解固体_gL'] < 3:
            score += 15
        elif row['总溶解固体_gL'] < 1:
            score -= 5
        elif row['总溶解固体_gL'] > 5:
            score -= 15

    # 溶解氧影响（高溶解氧有利于好氧生物降解）
    if pd.notna(row['溶解氧_mgL']):
        if row['溶解氧_mgL'] > 5:
            score += 10
        elif row['溶解氧_mgL'] > 3:
            score += 5
        elif row['溶解氧_mgL'] < 2:
            score -= 10

    # 氮浓度影响（适量氮促进微生物生长）
    if pd.notna(row['氮浓度_mgL']):
        if 100 < row['氮浓度_mgL'] < 600:
            score += 10
        elif row['氮浓度_mgL'] > 800:
            score -= 10

    # 营养类型影响
    if row['营养类型'] == '富营养':
        score += 10
    elif row['营养类型'] == '中营养':
        score += 5
    elif row['营养类型'] == '贫营养':
        score -= 5
    elif row['营养类型'] == '高氮':
        score += 8
    elif row['营养类型'] == '中氮':
        score += 5

    return max(0, min(100, score))


df['营养条件评分'] = df.apply(calculate_nutrient_score, axis=1)

# ==================== 4. 缺失值处理 ====================
print("\n" + "=" * 70)
print("【第4步】缺失值处理")
print("=" * 70)

print("\n缺失值统计:")
print(df[['ph', '温度', '去除率', '总溶解固体_gL', '电导率_uScm', '溶解氧_mgL', '氮浓度_mgL']].isnull().sum())

# 用中位数填充数值特征的缺失值
numeric_cols = ['总溶解固体_gL', '电导率_uScm', '溶解氧_mgL', '氮浓度_mgL']
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col}缺失值用中位数{median_val:.2f}填充")

# ==================== 5. 特征工程建议 ====================
print("\n" + "=" * 70)
print("【第5步】营养条件优化建议（提高预测准确率）")
print("=" * 70)

print("""
营养条件特征工程建议：

1. **提取关键数值特征**（已完成）：
   - 总溶解固体 (TDS)：影响微生物活性和传质效率
   - 电导率：反映离子强度，影响酶活性
   - 溶解氧 (DO)：好氧降解的关键因子
   - 氮浓度：微生物生长的限制性营养

2. **特征组合**（推荐尝试）：
   - TDS × 温度：高温高盐可能抑制微生物
   - DO × pH：影响自由基生成
   - 氮浓度 / TDS：营养盐与盐度比

3. **营养类型分类**（已完成）：
   - 富营养、中营养、贫营养
   - 高氮、中氮、低氮
   - 高盐、中盐、低盐

4. **营养条件评分**（已完成）：
   - 综合评分反映营养条件对去除的有利程度

5. **建议新增实验数据**：
   - 记录更详细的营养盐成分（N、P、微量元素）
   - 控制变量实验：固定其他条件，变化单一营养参数
   - 增加碳氮比(C/N)信息
""")

# ==================== 6. 生成处理后的数据表 ====================
print("\n" + "=" * 70)
print("【第6步】生成处理后的数据表")
print("=" * 70)

# 选择用于模型训练的特征列
feature_columns = [
    '污染物类型_标准化',  # 污染物类型（分类）
    '污染物类型编码',  # 污染物类型编码
    'ph',  # pH值
    '温度',  # 温度
    '总溶解固体_gL',  # 总溶解固体
    '电导率_uScm',  # 电导率
    '溶解氧_mgL',  # 溶解氧
    '氮浓度_mgL',  # 氮浓度
    '营养类型',  # 营养类型（分类）
    '营养类型编码',  # 营养类型编码
    '营养条件评分',  # 营养条件综合评分
    '去除率'  # 目标变量
]

# 添加独热编码的特征
onehot_cols = [col for col in df.columns if col.startswith('污染物_')]
feature_columns.extend(onehot_cols)

# 创建处理后的数据表
df_processed = df[feature_columns].copy()

# 重命名列名为英文（便于机器学习）
column_mapping = {
    '污染物类型_标准化': 'pollutant_type',
    '污染物类型编码': 'pollutant_code',
    'ph': 'ph',
    '温度': 'temperature',
    '总溶解固体_gL': 'tds_gL',
    '电导率_uScm': 'conductivity_uScm',
    '溶解氧_mgL': 'dissolved_oxygen_mgL',
    '氮浓度_mgL': 'nitrogen_concentration_mgL',
    '营养类型': 'nutrient_type',
    '营养类型编码': 'nutrient_type_code',
    '营养条件评分': 'nutrient_score',
    '去除率': 'removal_rate'
}

# 添加独热编码列映射
for col in onehot_cols:
    new_name = col.replace('污染物_', 'pollutant_')
    column_mapping[col] = new_name

df_processed = df_processed.rename(columns=column_mapping)

print(f"\n处理后数据形状: {df_processed.shape}")
print(f"处理后列名: {list(df_processed.columns)}")

print("\n处理后数据前10行:")
print(df_processed.head(10))

# 保存处理后的数据
output_file = '污染物去除率_处理后数据.xlsx'
df_processed.to_excel(output_file, index=False)
print(f"\n✓ 处理后数据已保存为: {output_file}")

# ==================== 7. 数据统计信息 ====================
print("\n" + "=" * 70)
print("【第7步】数据统计信息")
print("=" * 70)

print("\n数值特征统计:")
numeric_stats = df_processed[['ph', 'temperature', 'tds_gL', 'conductivity_uScm',
                              'dissolved_oxygen_mgL', 'nitrogen_concentration_mgL',
                              'nutrient_score', 'removal_rate']].describe()
print(numeric_stats)

print("\n分类特征分布:")
print(f"污染物类型分布:\n{df_processed['pollutant_type'].value_counts()}")
print(f"\n营养类型分布:\n{df_processed['nutrient_type'].value_counts()}")

# ==================== 8. 相关性分析 ====================
print("\n" + "=" * 70)
print("【第8步】特征与去除率相关性分析")
print("=" * 70)

# 计算各特征与去除率的相关系数
correlation_with_target = df_processed.corr(numeric_only=True)['removal_rate'].sort_values(ascending=False)

print("\n各特征与去除率的相关性（绝对值越大越重要）:")
for feature, corr in correlation_with_target.items():
    if feature != 'removal_rate':
        print(f"  {feature}: {corr:.4f}")

# ==================== 9. 机器学习数据准备 ====================
print("\n" + "=" * 70)
print("【第9步】机器学习数据准备")
print("=" * 70)

# 准备特征矩阵X和目标变量y
X = df_processed.drop('removal_rate', axis=1)
y = df_processed['removal_rate']

# 处理分类变量（使用独热编码）
categorical_cols = ['pollutant_type', 'nutrient_type']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 标准化数值特征
numeric_cols = ['ph', 'temperature', 'tds_gL', 'conductivity_uScm',
                'dissolved_oxygen_mgL', 'nitrogen_concentration_mgL',
                'nutrient_score', 'pollutant_code', 'nutrient_type_code']

# 只标准化存在的数值列
numeric_cols_existing = [col for col in numeric_cols if col in X_encoded.columns]

scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numeric_cols_existing] = scaler.fit_transform(X_encoded[numeric_cols_existing])

print(f"\n特征矩阵形状: {X_scaled.shape}")
print(f"目标变量形状: {y.shape}")

# 保存机器学习就绪的数据
ml_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
ml_data.to_excel('污染物去除率_机器学习就绪数据.xlsx', index=False)
print(f"✓ 机器学习就绪数据已保存为: 污染物去除率_机器学习就绪数据.xlsx")

# ==================== 10. 营养条件优化建议总结 ====================
print("\n" + "=" * 70)
print("【第10步】营养条件优化建议总结")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════╗
║                    营养条件优化建议                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. 【特征提取】                                                 ║
║     ✓ 从文本中提取了TDS、电导率、溶解氧、氮浓度等数值特征        ║
║     ✓ 创建了营养类型分类（富营养/中营养/贫营养/高氮/高盐等）     ║
║     ✓ 构建了营养条件综合评分                                     ║
║                                                                  ║
║  2. 【优化建议】                                                 ║
║     • 添加营养盐比例特征（C/N/P比）                              ║
║     • 记录微量元素浓度（Fe、Mn、Cu等）                           ║
║     • 增加pH缓冲能力指标                                         ║
║     • 考虑营养盐投加方式（批次/连续）                            ║
║                                                                  ║
║  3. 【预测准确率提升】                                           ║
║     • 使用提取的数值特征替代原始文本                             ║
║     • 营养条件评分可作为综合指标                                 ║
║     • 结合pH和温度构建交互特征                                   ║
║                                                                  ║
║  4. 【推荐特征组合】                                             ║
║     • TDS × 温度 → 盐度-温度交互效应                            ║
║     • DO / TDS → 溶解氧与盐度比                                  ║
║     • 氮浓度 × 污染物类型 → 营养-污染物特异性                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 70)
print("                    数据预处理完成！")
print("=" * 70)
print("\n生成的文件:")
print("  1. 污染物去除率_处理后数据.xlsx - 处理后的完整数据")
print("  2. 污染物去除率_机器学习就绪数据.xlsx - 可直接用于模型训练的数据")
print("\n" + "=" * 70)