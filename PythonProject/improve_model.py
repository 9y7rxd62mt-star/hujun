import os

import pandas as pd
import numpy as np
import re
import time
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 读取原始数据
# ==========================================
file_path = r'C:\Users\15787\PycharmProjects\PythonProject\Pollutant Experimental Data.xlsx'  # 请确保文件名正确
df_raw = pd.read_excel(file_path, sheet_name='Sheet1')
df_raw.rename(columns={'target': '去除率'}, inplace=True)
df_raw.dropna(subset=['去除率'], inplace=True)
print(f"原始数据量: {len(df_raw)}")

# ==========================================
# 2. 从 nutrient 文本中提取二值特征
# ==========================================
print("\n正在从 nutrient 列提取二值特征...")

feature_keywords = {
    'has_pms': 'PMS', 'has_pds': 'PDS', 'has_h2o2': 'H₂O₂',
    'has_fenton': ['Fenton', '芬顿'], 'has_uv': ['UV', '光'],
    'has_ozone': 'O₃', 'has_persulfate': ['过硫酸盐', 'persulfate'],
    'has_fe_ion': ['Fe²⁺', 'Fe³⁺', 'Fe(II)', 'Fe(III)'],
    'has_mos2': 'MoS₂', 'has_tio2': 'TiO₂', 'has_cofe2o4': 'CoFe₂O₄',
    'has_biochar': '生物炭', 'has_surfactant': ['Tween', '表面活性剂'],
    'has_chelator': ['柠檬酸', '草酸', 'EDTA'], 'has_abts': 'ABTS',
    'is_biological': ['bio', '菌', '真菌', 'fungus', 'bacteria', 'BTF', '滴滤'],
    'is_anaerobic': '厌氧', 'is_aerobic': '好氧',
    'has_dbd': 'DBD', 'has_plasma': '放电', 'has_adsorption': '吸附',
}

def extract_binary_features(text):
    if pd.isna(text):
        return {feat: 0 for feat in feature_keywords}
    text = str(text).lower()
    feats = {}
    for feat, kw in feature_keywords.items():
        if isinstance(kw, list):
            feats[feat] = 1 if any(k.lower() in text for k in kw) else 0
        else:
            feats[feat] = 1 if kw.lower() in text else 0
    return feats

binary_dicts = df_raw['nutrient'].apply(extract_binary_features)
binary_df = pd.DataFrame(binary_dicts.tolist())

# 从 nutrient 中提取数值特征（TDS、电导率、氮浓度）
def extract_numeric(text):
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

numeric_dicts = df_raw['nutrient'].apply(extract_numeric)
numeric_df = pd.DataFrame(numeric_dicts.tolist())

df = pd.concat([df_raw, binary_df, numeric_df], axis=1)

# ==========================================
# 3. 计算分子描述符（使用缓存，避免重复请求）
# ==========================================
print("\n正在获取污染物分子描述符...")

# 获取所有唯一污染物
pollutants = df['pollutant'].dropna().unique()
print(f"发现 {len(pollutants)} 种污染物")

# 尝试加载已有的SMILES缓存（如果存在）
cache_file = 'smiles_cache.csv'
if os.path.exists(cache_file):
    cache_df = pd.read_csv(cache_file)
    # 将 smilies 列中的 NaN 替换为 None，并确保是字符串类型
    cache_df['smiles'] = cache_df['smiles'].where(pd.notna(cache_df['smiles']), None)
    name_to_smiles = dict(zip(cache_df['pollutant'], cache_df['smiles']))
    print("已加载SMILES缓存")
else:
    name_to_smiles = {}

def get_smiles(name):
    if name in name_to_smiles:
        return name_to_smiles[name]
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            smiles = compounds[0].canonical_smiles
        else:
            smiles = None
        time.sleep(0.2)
        name_to_smiles[name] = smiles
        return smiles
    except:
        name_to_smiles[name] = None
        return None

for p in tqdm(pollutants):
    if p not in name_to_smiles:
        get_smiles(p)

# 保存缓存
pd.DataFrame(list(name_to_smiles.items()), columns=['pollutant', 'smiles']).to_csv(cache_file, index=False)

# 计算描述符
def compute_descriptors(smiles):
    # 增加类型检查：如果不是字符串或为 None，直接返回 NaN
    if not isinstance(smiles, str) or smiles is None or pd.isna(smiles):
        return {d: np.nan for d in ['MolWt','LogP','NumHDonors','NumHAcceptors',
                                    'NumRotatableBonds','TPSA','NumAromaticRings',
                                    'FractionCsp3','NumHalogens']}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {d: np.nan for d in ['MolWt','LogP','NumHDonors','NumHAcceptors',
                                    'NumRotatableBonds','TPSA','NumAromaticRings',
                                    'FractionCsp3','NumHalogens']}
    return {
        'MolWt': Descriptors.ExactMolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Lipinski.NumHDonors(mol),
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'FractionCsp3': Descriptors.FractionCsp3(mol),
        'NumHalogens': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9,17,35,53]),
    }

desc_data = []
for p in tqdm(pollutants):
    smiles = name_to_smiles.get(p)
    desc = compute_descriptors(smiles)
    desc['pollutant'] = p
    desc_data.append(desc)
desc_df = pd.DataFrame(desc_data)

# 合并到主数据框
df = df.merge(desc_df, on='pollutant', how='left')

# ==========================================
# 4. 准备特征矩阵和目标变量
# ==========================================
# 基础数值特征
base_features = ['ph', 'temperature', 'tds_gL', 'ec_uScm', '氮浓度_mgL']
# 分子描述符特征
desc_features = ['MolWt','LogP','NumHDonors','NumHAcceptors',
                 'NumRotatableBonds','TPSA','NumAromaticRings',
                 'FractionCsp3','NumHalogens']
# 二值特征（从nutrient提取）
binary_features = list(feature_keywords.keys())

all_features = base_features + desc_features + binary_features
# 删除全为NaN的列
all_features = [f for f in all_features if df[f].notna().any()]

X = df[all_features].copy()
y = df['去除率'].copy()

# 处理缺失值：用中位数填充
for col in X.columns:
    if X[col].dtype in ['float64','int64']:
        X[col].fillna(X[col].median(), inplace=True)

print(f"总特征数: {len(all_features)}")
print(f"特征列表: {all_features}")

# ==========================================
# 5. 分层采样（按文献名称）
# ==========================================
print("\n按文献名称进行分层采样...")
# 获取所有唯一的文献名称
sources = df['文献名称'].dropna().unique()
train_sources, test_sources = train_test_split(sources, test_size=0.2, random_state=42)

train_idx = df['文献名称'].isin(train_sources)
test_idx = df['文献名称'].isin(test_sources)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"训练集样本数: {len(X_train)}，来自 {len(train_sources)} 篇文献")
print(f"测试集样本数: {len(X_test)}，来自 {len(test_sources)} 篇文献")

# ==========================================
# 6. 特征重要性筛选（基于XGBoost）
# ==========================================
print("\n正在训练初始XGBoost模型以评估特征重要性...")
xgb_base = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_base.fit(X_train, y_train)

importances = xgb_base.feature_importances_
indices = np.argsort(importances)[::-1]

# 计算累积重要性，保留95%的特征
cumsum = np.cumsum(importances[indices])
n_selected = np.argmax(cumsum >= 0.95) + 1
selected_indices = indices[:n_selected]
selected_features = [all_features[i] for i in selected_indices]

print(f"选择 {n_selected} 个特征，累积重要性: {cumsum[n_selected-1]:.3f}")
print("选中的特征:", selected_features)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# ==========================================
# 7. 使用XGBoost（带正则化）最终训练
# ==========================================
print("\n最终训练模型...")
xgb_final = xgb.XGBRegressor(
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
xgb_final.fit(X_train_selected, y_train)

y_train_pred = xgb_final.predict(X_train_selected)
y_test_pred = xgb_final.predict(X_test_selected)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n========== 最终结果 ==========")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")
print(f"泛化差距: {train_r2 - test_r2:.4f}")

# ==========================================
# 8. 输出特征重要性（最终模型）
# ==========================================
final_importances = xgb_final.feature_importances_
final_indices = np.argsort(final_importances)[::-1]
print("\n最终模型特征重要性排序:")
for i in range(min(10, len(selected_features))):
    idx = final_indices[i]
    print(f"  {i+1}. {selected_features[idx]}: {final_importances[idx]:.4f}")

# ==========================================
# 9. 保存结果（可选）
# ==========================================
output_df = df[train_idx | test_idx].copy()
output_df['预测值_训练集'] = np.nan
output_df['预测值_测试集'] = np.nan
output_df.loc[train_idx, '预测值_训练集'] = y_train_pred
output_df.loc[test_idx, '预测值_测试集'] = y_test_pred
output_df.to_excel('预测结果_分层采样.xlsx', index=False)
print("\n预测结果已保存至 '预测结果_分层采样.xlsx'")
# 加载预测结果
result_df = pd.read_excel('预测结果_分层采样.xlsx')

# 分离训练集和测试集的预测结果
train_pred = result_df[result_df['预测值_训练集'].notna()].copy()
test_pred = result_df[result_df['预测值_测试集'].notna()].copy()

train_pred['残差'] = train_pred['去除率'] - train_pred['预测值_训练集']
test_pred['残差'] = test_pred['去除率'] - train_pred['预测值_测试集']  # 注意列名

# 输出残差统计
print("训练集残差统计：")
print(train_pred['残差'].describe())
print("\n测试集残差统计：")
print(test_pred['残差'].describe())

# 找出测试集中残差绝对值最大的10个样本
worst_samples = test_pred.nlargest(10, '残差', keep='all')
print("\n测试集中预测最差的10个样本（实际去除率 - 预测去除率）：")
print(worst_samples[['pollutant', '去除率', '预测值_测试集', '残差', '文献名称']])