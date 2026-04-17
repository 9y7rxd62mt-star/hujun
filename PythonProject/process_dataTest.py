import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. 读取原始数据并特征提取
# ==========================================
def extract_features_from_nutrient(text):
    """从 nutrient 文本中提取二值特征和数值特征"""
    if pd.isna(text):
        return {}
    text = str(text).lower()

    # 二值关键词特征
    keywords = {
        'has_pms': 'pms',
        'has_pds': 'pds',
        'has_h2o2': 'h2o2',
        'has_fenton': 'fenton',
        'has_uv': 'uv',
        'has_ozone': 'o3',
        'has_persulfate': '过硫酸盐',
        'has_fe_ion': r'fe[²³]?\+',
        'has_mos2': 'mos2',
        'has_tio2': 'tio2',
        'has_cofe2o4': 'cofe2o4',
        'has_biochar': '生物炭',
        'has_surfactant': 'tween',
        'has_chelator': '柠檬酸|草酸|edta',
        'has_abts': 'abts',
        'is_biological': 'bio|菌|真菌|bacteria|btf|滴滤',
        'is_anaerobic': '厌氧',
        'has_dbd': 'dbd',
        'has_plasma': '放电',
    }
    features = {}
    for name, pattern in keywords.items():
        features[name] = 1 if re.search(pattern, text) else 0

    # 数值特征提取
    tds = re.search(r'总溶解固体\s*([\d\.]+)\s*g/l', text)
    ec = re.search(r'电导率\s*([\d\.]+)\s*μs/cm', text)
    n_conc = re.search(r'氮浓度\s*([\d\.]+)\s*mg/l', text)
    features['tds_gL'] = float(tds.group(1)) if tds else np.nan
    features['ec_uScm'] = float(ec.group(1)) if ec else np.nan
    features['氮浓度_mgL'] = float(n_conc.group(1)) if n_conc else np.nan

    return features


def load_and_process_data(file_path):
    """加载原始 Excel 并生成特征矩阵"""
    df = pd.read_excel("processed_data_for_ml.xlsx", sheet_name='Sheet1')
    df = df[['pollutant', 'ph', 'temperature', 'nutrient', 'target']].copy()
    df.rename(columns={'target': '去除率'}, inplace=True)
    df.dropna(subset=['去除率'], inplace=True)

    # 提取特征
    feat_dicts = df['nutrient'].apply(extract_features_from_nutrient)
    feat_df = pd.DataFrame(feat_dicts.tolist())
    df = pd.concat([df, feat_df], axis=1)

    # 污染物频次编码（后续可替换为分子描述符）
    pollutant_freq = df['pollutant'].map(df['pollutant'].value_counts()) / len(df)
    df['pollutant_freq'] = pollutant_freq

    # 清洗离群点：高级氧化体系但去除率异常低
    advanced_cols = ['has_pms', 'has_h2o2', 'has_uv', 'has_fenton', 'has_pds']
    df['advanced_score'] = df[advanced_cols].sum(axis=1)
    mask = ~((df['advanced_score'] >= 2) & (df['去除率'] < 10))
    df_clean = df[mask].copy()

    # 特征列（排除文本和ID）
    exclude = ['pollutant', 'nutrient', '去除率', 'advanced_score']
    feature_cols = [c for c in df_clean.columns if c not in exclude]

    # 填充数值缺失值
    for col in feature_cols:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean, feature_cols


# ==========================================
# 2. 训练模型并评估
# ==========================================
def train_model(df, feature_cols, target='去除率'):
    X = df[feature_cols]
    y = df[target]

    # 划分训练集和测试集（按文献分层可选，这里随机）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 特征标准化（树模型不需要，但为了可比性）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 随机森林（可换成 XGBoost）
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 预测与评估
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Test R² Score: {r2:.4f}")
    print(f"Test RMSE: {rmse:.2f}")

    # 交叉验证
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([0, 100], [0, 100], 'r--')
    plt.xlabel('Actual Removal Rate (%)')
    plt.ylabel('Predicted Removal Rate (%)')
    plt.title(f'Random Forest Model (R² = {r2:.3f})')
    plt.grid(True)
    plt.show()

    return model, scaler, feature_cols


# ==========================================
# 3. 保存模型并演示预测新样本
# ==========================================
def predict_new_sample(model, scaler, feature_cols, new_sample_dict):
    """
    new_sample_dict: 包含 pollutant, ph, temperature, nutrient 的字典
    """
    # 构造一个临时 DataFrame
    temp_df = pd.DataFrame([new_sample_dict])
    # 提取特征（复用之前的函数）
    feat = extract_features_from_nutrient(temp_df.loc[0, 'nutrient'])
    for k, v in feat.items():
        temp_df[k] = v
    # 污染物频次编码（需要提前保存污染物的频率字典，这里简单用0填充，实际应使用训练集的频率）
    # 注意：生产环境应保存一个全局的 pollutant_freq_map
    temp_df['pollutant_freq'] = 0.0  # 示例，最好从训练集获取
    # 选择特征并标准化
    X_new = temp_df[feature_cols].fillna(0)
    X_new_scaled = scaler.transform(X_new)
    pred = model.predict(X_new_scaled)[0]
    return pred


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 加载并处理原始数据
    print("正在加载原始数据...")
    df_processed, feature_cols = load_and_process_data('processed_data_for_ml.xlsx')
    print(f"数据量: {len(df_processed)}, 特征数: {len(feature_cols)}")
    print("特征列表:", feature_cols)

    # 2. 训练模型
    print("\n开始训练模型...")
    model, scaler, features = train_model(df_processed, feature_cols)

    # 3. 保存模型和标准化器
    joblib.dump(model, 'voc_rf_model.pkl')
    joblib.dump(scaler, 'voc_scaler.pkl')
    print("\n模型已保存为 'voc_rf_model.pkl' 和 'voc_scaler.pkl'")

    # 4. 演示预测新样本
    new_sample = {
        'pollutant': '甲苯',
        'ph': 7.0,
        'temperature': 25,
        'nutrient': 'Fe³⁺/PMS/MoS₂体系，空气鼓泡'
    }
    pred_rate = predict_new_sample(model, scaler, feature_cols, new_sample)
    print(f"\n新样本预测去除率: {pred_rate:.2f}%")