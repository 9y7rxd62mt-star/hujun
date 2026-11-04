# -*- coding: utf-8 -*-
"""
以「拆分后的数据表」重新建模 —— 严格复刻 ml_pipeline 的完整特征工程
(文献级插补 + 缺失指示 + 完整二元/类别特征)，仅把数值特征来源换成拆分列。
配置A = 无营养条件3特征 | 配置B = 含营养条件3特征(完整基线)
"""
import sys, io, re, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

BASE = r"C:\Users\15787\WorkBuddy\2026-11-03-11-42-33\BTF_ML"
OUT = os.path.join(BASE, "output")
split = pd.read_csv(os.path.join(OUT, "BTF_数据表_参数拆分.csv")).reset_index(drop=True)
raw = pd.read_csv(os.path.join(OUT, "BTF_data_parsed.csv")).reset_index(drop=True)

CANON = {'n-己烷': '正己烷', '二甲基乙酰胺': 'N,N-二甲基乙酰胺', '己烷': '正己烷'}
poll = split['污染物名称'].astype(str).str.strip().replace(CANON)
CAT_MAP = {
    '甲苯': '芳香族', '苯': '芳香族', '间二甲苯': '芳香族', '对二甲苯': '芳香族', '二甲苯': '芳香族',
    '乙苯': '芳香族', '苯乙烯': '芳香族', '苯甲醇': '芳香族',
    '二氯甲烷': '氯代VOCs', '氯仿': '氯代VOCs', '氯苯': '氯代VOCs', '1,2-二氯苯': '氯代VOCs', '二氯溴甲烷': '氯代VOCs',
    '正己烷': '烷烃', '环己烷': '烷烃',
    '甲醇': '含氧VOCs', '乙醇': '含氧VOCs', '乙醛': '含氧VOCs', '丙酮': '含氧VOCs', '2-丁酮': '含氧VOCs',
    '乙酸乙酯': '含氧VOCs', '乙酸甲酯': '含氧VOCs', '乙酸丁酯': '含氧VOCs', '甲基丙烯酸正丁酯': '含氧VOCs', '丙烯酸乙酯': '含氧VOCs',
    '硫化氢': '硫化物', '乙硫醇': '硫化物', '丙硫醇': '硫化物', '甲硫醚': '硫化物', '二甲基硫': '硫化物',
    '丙烯腈': '含氮化合物', '三甲胺': '含氮化合物', '二甲胺': '含氮化合物',
    'N,N-二甲基乙酰胺': '含氮化合物', '氨': '含氮化合物', '磷酸三丁酯': '其他',
}
cat = poll.map(CAT_MAP).fillna('恶臭/其他')
hydro = cat.isin(['烷烃', '芳香族', '氯代VOCs']).astype(int)
pcnt = poll.value_counts()
common_poll = poll.where(poll.map(pcnt) >= 4, '其他')

SRC = split['数据来源'].fillna('').astype(str)
NUT = raw['营养条件'].fillna('').astype(str)
TXT_ALL = (NUT + ' ; ' + SRC)   # 完整文本(含营养条件) — 用于营养条件相关特征
NUM = r'(\d+(?:\.\d+)?)'
def find_val(t, unit_pat, range_first=True):
    if range_first:
        m = re.search(NUM + r'\s*[-–~]\s*' + NUM + r'\s*' + unit_pat, t)
        if m:
            try: return (float(m.group(1)) + float(m.group(2))) / 2
            except Exception: pass
    m = re.search(NUM + r'\s*' + unit_pat, t)
    if m:
        try: return float(m.group(1))
        except Exception: return np.nan
    return np.nan
def get_day(t):
    for pat in [r'第\s*' + NUM + r'\s*天', r'[Dd]ay\s*' + NUM, r'（\s*' + NUM + r'\s*[天d]']:
        m = re.search(pat, t)
        if m:
            try: return float(m.group(1))
            except Exception: pass
    return np.nan
def get_stage(t):
    if re.search(r'稳定|stable', t, re.I): return 3
    if '中期' in t: return 2
    if re.search(r'初期|启动|驯化|挂膜|start', t, re.I): return 1
    return np.nan
def flag_from(src, pat):
    return src.str.contains(pat, case=False, regex=True).astype(int)
def target_range_width(val):
    if not isinstance(val, str): return 0.0
    m = re.findall(r'\d+(?:\.\d+)?', val)
    if '-' in val and len(m) >= 2: return abs(float(m[1]) - float(m[0]))
    return 0.0

PACKS = {'填料_陶粒': r'陶粒', '填料_浮石': r'浮石|pumice', '填料_火山岩': r'火山岩|[Vv]olcanic',
         '填料_树皮': r'树皮|[Bb]ark', '填料_复合填料': r'复合填料', '填料_多面球': r'多面球',
         '填料_聚氨酯': r'聚氨酯|PU|[Pp]olyurethane', '填料_鲍尔环': r'鲍尔环|Pall',
         '填料_木屑堆肥': r'木屑|堆肥|compost|木屑和堆肥', '填料_丝瓜': r'丝瓜|loofah',
         '填料_活性炭': r'活性炭|activated carbon'}

# 数值原始列(拆分列) — 与 ml_pipeline 的 EBRT/浓度/负荷 一一对应
df = pd.DataFrame({
    '论文标题': split['论文标题'],
    'pH': split['ph_数值'], '温度': split['温度_数值'],
    'EBRT_s': split['空床停留时间(s)'],
    '进气浓度_mgm3': split['目标污染物入口浓度(mg/m³)'],
    '进气浓度_ppm': split['目标污染物入口浓度(ppm)'],
    '入口负荷_gmh': split['入口表面负荷(g/m³·h)'],
    '氮浓度_mgL': TXT_ALL.apply(lambda t: find_val(t.replace('含氮522', '含氮 522'), r'mg/L')),
    '运行天数': TXT_ALL.apply(get_day),
    '运行阶段': TXT_ALL.apply(get_stage),
    '疏水性污染物': hydro,
    '饥饿胁迫': flag_from(TXT_ALL, r'饥饿|starvation'),
    '重启恢复': flag_from(TXT_ALL, r'重启|restart|恢复|recover'),
    '等离子耦合': flag_from(TXT_ALL, r'DBD|放电|等离子|plasma|耦合'),
    '混合共降解': flag_from(TXT_ALL, r'共降解|共存|混合废气|多组分|双组分|coupled|dual|mixtur'),
    '有表面活性剂': flag_from(TXT_ALL, r'Tween|吐温|鼠李糖脂|rhamnolipid|表面活性|saponin|皂角|皂素|surfactant|Tomadol'),
    '有铁离子': flag_from(TXT_ALL, r'Fe³|Fe3|铁离子|ferric|亚铁'),
    '有紫外预处理': flag_from(TXT_ALL, r'UV|紫外'),
    '有营养液': flag_from(TXT_ALL, r'矿物|营养|培养基|氮磷|NH4|K2HPO4|KH2PO4|维生素|微量元素|MSM|mineral|nutrient'),
    '有额外碳源': flag_from(TXT_ALL, r'葡萄糖|乙酸钠|正丙醇|正丁醇|碳源'),
    '真菌主导': flag_from(TXT_ALL, r'真菌|fungal|Trametes|Paecilomyces'),
    '污染物类别': cat, '常见污染物': common_poll,
    '去除率': split['去除率_数值'].values,
    '_数据来源': split['数据来源'].values,
    '_污染物': split['污染物名称'].values,
    '_ph原始': split['ph'].values,
    '_温度原始': split['温度'].values,
    '_营养条件': raw['营养条件'].values,
    '_去除率原始': raw['去除率'].values,
    '_span': raw['去除率'].apply(target_range_width).values,
    '_shake': flag_from(TXT_ALL, r'摇瓶').values,
    '_mech': flag_from(TXT_ALL, r'灭活|非生物|抑制剂|介体|漆酶|abiotic|inactivated|质量平衡').values,
})
for k, v in PACKS.items():
    df[k] = flag_from(TXT_ALL, v)

# 清洗(与 ml_pipeline 完全一致)
df = df[df['去除率'].notna() & (df['去除率'] >= 0) & (df['去除率'] <= 100)]
df = df[df['_span'] <= 40]
df = df.drop_duplicates(subset=['论文标题', '_数据来源', '_污染物', '_ph原始', '_温度原始', '_营养条件', '_去除率原始'])
df = df[(df['_shake'] == 0) & (df['_mech'] == 0)].reset_index(drop=True)
df = df.drop(columns=['_span', '_shake', '_mech', '_数据来源', '_污染物', '_ph原始', '_温度原始', '_营养条件', '_去除率原始'])
print("清洗后样本数:", len(df), " (应≈336)")

# 文献级插补 + 缺失指示 (复刻 ml_pipeline)
IMP_MAP = {'pH': 'pH_imp', '温度': '温度_imp', 'EBRT_s': 'EBRT_imp',
           '进气浓度_mgm3': '进气浓度_imp', '进气浓度_ppm': 'ppm_imp',
           '入口负荷_gmh': '负荷_imp', '氮浓度_mgL': '氮浓度_imp',
           '运行天数': '天数_imp', '运行阶段': '阶段_imp'}
for src, dst in IMP_MAP.items():
    pm = df.groupby('论文标题')[src].transform('median')
    df[dst] = df[src].fillna(pm)
df['pH缺失'] = df['pH'].isna().astype(int)
df['温度缺失'] = df['温度'].isna().astype(int)
for src in ['EBRT_s', '进气浓度_mgm3', '运行阶段']:
    df[src + '原始缺失'] = df[src].isna().astype(int)

num_feats = list(IMP_MAP.values())
bin_all = (['pH缺失', '温度缺失', 'EBRT_s原始缺失', '进气浓度_mgm3原始缺失', '运行阶段原始缺失',
            '疏水性污染物', '饥饿胁迫', '重启恢复', '等离子耦合', '混合共降解', '有表面活性剂',
            '有铁离子', '有紫外预处理', '有营养液', '有额外碳源', '真菌主导'] + list(PACKS.keys()))
bin_nut = ['有表面活性剂', '有营养液', '有额外碳源']
bin_common = [b for b in bin_all if b not in bin_nut]
cat_feats = ['污染物类别', '常见污染物']

def run_config(name, use_nut):
    bin_feats = bin_all if use_nut else bin_common
    X = df[num_feats + bin_feats + cat_feats].copy()
    yv = df['去除率'].values
    groups = df['论文标题'].values
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, yv, groups, test_size=0.2, random_state=42)
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_feats),
        ('bin', Pipeline([('imp', SimpleImputer(strategy='most_frequent'))]), bin_feats),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_feats)])
    model = Pipeline([('pre', pre), ('m', RandomForestRegressor(n_estimators=300, max_depth=None,
                        max_features=0.4, min_samples_leaf=1, random_state=42, n_jobs=-1))])
    model.fit(Xtr, ytr)
    r2_tr = r2_score(ytr, model.predict(Xtr))
    r2_te = r2_score(yte, model.predict(Xte))
    rmse = np.sqrt(mean_squared_error(yte, model.predict(Xte)))
    cv = cross_val_score(model, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=42), scoring='r2')
    gcv = cross_val_score(model, X, yv, cv=GroupKFold(5), groups=groups, scoring='r2')
    print(f"\n[{name}] 特征: {len(num_feats)}数值 + {len(bin_feats)}二元 + {len(cat_feats)}类别")
    print(f"  训练R2={r2_tr:.4f} | 测试R2={r2_te:.4f} (RMSE={rmse:.2f})")
    print(f"  随机5折CV R2={cv.mean():.4f}±{cv.std():.4f}")
    print(f"  按文献GroupKFold R2={gcv.mean():.4f}±{gcv.std():.4f}")
    return {'train': r2_tr, 'test': r2_te, 'rmse': rmse, 'cv_mean': cv.mean(), 'cv_std': cv.std(),
            'gcv_mean': gcv.mean(), 'gcv_std': gcv.std()}

resA = run_config('配置A(拆分数据, 无营养条件3特征)', use_nut=False)
resB = run_config('配置B(拆分数据+营养条件3特征)', use_nut=True)

print("\n===== 对比汇总 =====")
print(f"{'指标':<22}{'A无营养特征':>12}{'B完整特征':>12}")
for k, label in [('train', '训练R2'), ('test', '测试R2'), ('rmse', '测试RMSE'),
                 ('cv_mean', '随机5折CV R2'), ('gcv_mean', '按文献GroupKFold')]:
    print(f"{label:<22}{resA[k]:>12.4f}{resB[k]:>12.4f}")
print("\n(原ml_pipeline最优模型: 测试R2=0.4662, RMSE=16.37, GroupKFold≈-0.018)")
print("DONE")
