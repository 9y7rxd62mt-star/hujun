# -*- coding: utf-8 -*-
"""
生成:
  fig14 训练集/测试集分布图 + 实际-预测线性拟合图 (2x2)
  fig15 特征重要性图 (置换重要性 Top20)
复用 ml_pipeline 特征工程与随机森林最优模型(seed42)
"""
import sys, io, re, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

BASE = r"C:\Users\15787\WorkBuddy\2026-11-03-11-42-33\BTF_ML"
OUT = os.path.join(BASE, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ============ 特征工程 (复现) ============
df = pd.read_csv(os.path.join(OUT, "BTF_data_parsed.csv"))
df = df[df['去除率_数值'].notna()]
df = df[(df['去除率_数值'] >= 0) & (df['去除率_数值'] <= 100)]
def target_range_width(val):
    if not isinstance(val, str): return 0.0
    m = re.findall(r'\d+(?:\.\d+)?', val)
    if '-' in val and len(m) >= 2: return abs(float(m[1]) - float(m[0]))
    return 0.0
df['目标跨度'] = df['去除率'].apply(target_range_width)
df = df[df['目标跨度'] <= 40].reset_index(drop=True)
df = df.drop_duplicates(subset=['论文标题', '数据来源', '污染物名称', 'ph', '温度', '营养条件', '去除率']).reset_index(drop=True)

CANON = {'n-己烷': '正己烷', '二甲基乙酰胺': 'N,N-二甲基乙酰胺', '己烷': '正己烷'}
df['污染物'] = df['污染物名称'].astype(str).str.strip().replace(CANON)
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
df['污染物类别'] = df['污染物'].map(CAT_MAP).fillna('恶臭/其他')
df['疏水性污染物'] = df['污染物类别'].isin(['烷烃', '芳香族', '氯代VOCs']).astype(int)
cnt = df['污染物'].value_counts()
df['常见污染物'] = df['污染物'].where(df['污染物'].map(cnt) >= 4, '其他')
TXT = (df['营养条件'].fillna('') + ' ; ' + df['数据来源'].fillna('')).astype(str)
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
def get_ebrt(t):
    for pat in [r'EBRT[^0-9\n]{0,14}?' + NUM + r'\s*[sS秒]', r'停留时间\s*[=为]?\s*' + NUM + r'\s*[sS秒]', r'进气口\s*[=为]?\s*' + NUM + r'\s*秒']:
        m = re.search(pat, t)
        if m:
            try: return float(m.group(1))
            except Exception: pass
    return np.nan
df['EBRT_s'] = TXT.apply(get_ebrt)
df['进气浓度_mgm3'] = TXT.apply(lambda t: find_val(t, r'mg/m'))
df['进气浓度_ppm'] = TXT.apply(lambda t: find_val(t, r'[pP][pP][mM]'))
df['入口负荷_gmh'] = TXT.apply(lambda t: find_val(t, r'g/m[³3]\s*·?\s*h'))
df['氮浓度_mgL'] = TXT.apply(lambda t: find_val(t.replace('含氮522', '含氮 522'), r'mg/L') if isinstance(t, str) else np.nan)
def get_day(t):
    for pat in [r'第\s*' + NUM + r'\s*天', r'[Dd]ay\s*' + NUM, r'（\s*' + NUM + r'\s*[天d]']:
        m = re.search(pat, t)
        if m:
            try: return float(m.group(1))
            except Exception: pass
    return np.nan
df['运行天数'] = TXT.apply(get_day)
def get_stage(t):
    if re.search(r'稳定|stable', t, re.I): return 3
    if '中期' in t: return 2
    if re.search(r'初期|启动|驯化|挂膜|start', t, re.I): return 1
    return np.nan
df['运行阶段'] = TXT.apply(get_stage)
def flag(pat): return TXT.str.contains(pat, case=False, regex=True).astype(int)
df['饥饿胁迫'] = flag(r'饥饿|starvation')
df['重启恢复'] = flag(r'重启|restart|恢复|recover')
df['摇瓶实验'] = flag(r'摇瓶')
df['对照机理实验'] = flag(r'灭活|非生物|抑制剂|介体|漆酶|abiotic|inactivated|质量平衡')
df['等离子耦合'] = flag(r'DBD|放电|等离子|plasma|耦合')
df['混合共降解'] = flag(r'共降解|共存|混合废气|多组分|双组分|coupled|dual|mixtur')
df['有表面活性剂'] = flag(r'Tween|吐温|鼠李糖脂|rhamnolipid|表面活性|saponin|皂角|皂素|surfactant|Tomadol')
df['有铁离子'] = flag(r'Fe³|Fe3|铁离子|ferric|亚铁')
df['有紫外预处理'] = flag(r'UV|紫外')
df['有营养液'] = flag(r'矿物|营养|培养基|氮磷|NH4|K2HPO4|KH2PO4|维生素|微量元素|MSM|mineral|nutrient')
df['有额外碳源'] = flag(r'葡萄糖|乙酸钠|正丙醇|正丁醇|碳源')
df['真菌主导'] = flag(r'真菌|fungal|Trametes|Paecilomyces')
PACKS = {'填料_陶粒': r'陶粒', '填料_浮石': r'浮石|pumice', '填料_火山岩': r'火山岩|[Vv]olcanic',
         '填料_树皮': r'树皮|[Bb]ark', '填料_复合填料': r'复合填料', '填料_多面球': r'多面球',
         '填料_聚氨酯': r'聚氨酯|PU|[Pp]olyurethane', '填料_鲍尔环': r'鲍尔环|Pall',
         '填料_木屑堆肥': r'木屑|堆肥|compost|木屑和堆肥', '填料_丝瓜': r'丝瓜|loofah',
         '填料_活性炭': r'活性炭|activated carbon'}
for name, pat in PACKS.items():
    df[name] = flag(pat)
df['pH'] = df['ph_数值']; df['温度'] = df['温度_数值']
df['pH缺失'] = df['pH'].isna().astype(int); df['温度缺失'] = df['温度'].isna().astype(int)
df = df[(df['摇瓶实验'] == 0) & (df['对照机理实验'] == 0)].reset_index(drop=True)
y = df['去除率_数值'].values
IMP_MAP = {'pH': 'pH_imp', '温度': '温度_imp', 'EBRT_s': 'EBRT_imp', '进气浓度_mgm3': '进气浓度_imp',
           '进气浓度_ppm': 'ppm_imp', '入口负荷_gmh': '负荷_imp', '氮浓度_mgL': '氮浓度_imp',
           '运行天数': '天数_imp', '运行阶段': '阶段_imp'}
for src, dst in IMP_MAP.items():
    pm = df.groupby('论文标题')[src].transform('median')
    df[dst] = df[src].fillna(pm)
for src in ['EBRT_s', '进气浓度_mgm3', '运行阶段']:
    df[src + '原始缺失'] = df[src].isna().astype(int)
num_feats = list(IMP_MAP.values())
bin_feats = (['pH缺失', '温度缺失', 'EBRT_s原始缺失', '进气浓度_mgm3原始缺失', '运行阶段原始缺失',
              '疏水性污染物', '饥饿胁迫', '重启恢复', '等离子耦合', '混合共降解', '有表面活性剂',
              '有铁离子', '有紫外预处理', '有营养液', '有额外碳源', '真菌主导'] + list(PACKS.keys()))
cat_feats = ['污染物类别', '常见污染物']
X = df[num_feats + bin_feats + cat_feats].copy()
groups = df['论文标题'].values

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X, y, groups, test_size=0.2, random_state=42)
numeric_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
binary_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent'))])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', numeric_pipe, num_feats), ('bin', binary_pipe, bin_feats),
                         ('cat', cat_pipe, cat_feats)])
model = Pipeline([('pre', pre), ('m', RandomForestRegressor(n_estimators=300, max_depth=None,
                  max_features=0.4, min_samples_leaf=1, random_state=42, n_jobs=-1))])
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
r2_tr = r2_score(y_train, pred_train)
r2_te = r2_score(y_test, pred_test)
print(f"训练R2={r2_tr:.4f} 测试R2={r2_te:.4f}")

C_TRAIN, C_TEST = '#4C72B0', '#C44E52'

# ============ 图14: 分布图 + 线性拟合图 ============
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (1) 训练集去除率分布
ax = axes[0, 0]
ax.hist(y_train, bins=30, color=C_TRAIN, edgecolor='white', alpha=0.85, density=True,
        label=f'训练集 (n={len(y_train)})')
ax.axvline(y_train.mean(), color=C_TRAIN, ls='--', lw=1.6, label=f'均值 {y_train.mean():.1f}%')
ax.set_xlabel('去除率 (%)'); ax.set_ylabel('密度')
ax.set_title('训练集去除率分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.25)

# (2) 测试集去除率分布
ax = axes[0, 1]
ax.hist(y_test, bins=20, color=C_TEST, edgecolor='white', alpha=0.85, density=True,
        label=f'测试集 (n={len(y_test)})')
ax.axvline(y_test.mean(), color=C_TEST, ls='--', lw=1.6, label=f'均值 {y_test.mean():.1f}%')
ax.set_xlabel('去除率 (%)'); ax.set_ylabel('密度')
ax.set_title('测试集去除率分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.25)

# (3) 训练集 实际 vs 预测 (线性拟合)
ax = axes[1, 0]
ax.scatter(y_train, pred_train, s=24, alpha=0.55, c=C_TRAIN, edgecolors='none')
z = np.polyfit(y_train, pred_train, 1)
xs = np.linspace(y_train.min(), y_train.max(), 50)
ax.plot(xs, np.polyval(z, xs), color='#D62728', lw=2, ls='--',
        label=f'线性拟合: y={z[0]:.2f}x{z[1]:+.1f}')
ax.plot([0, 105], [0, 105], 'k-', lw=1.2, alpha=0.6, label='1:1 参考线')
ax.set_xlabel('实际去除率 (%)'); ax.set_ylabel('预测去除率 (%)')
ax.set_title(f'训练集 实际-预测线性关系 (R²={r2_tr:.3f})', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.25)

# (4) 测试集 实际 vs 预测 (线性拟合)
ax = axes[1, 1]
ax.scatter(y_test, pred_test, s=42, alpha=0.85, c=C_TEST, marker='D', edgecolors='white', linewidths=0.6)
z = np.polyfit(y_test, pred_test, 1)
xs = np.linspace(y_test.min(), y_test.max(), 50)
ax.plot(xs, np.polyval(z, xs), color='#D62728', lw=2, ls='--',
        label=f'线性拟合: y={z[0]:.2f}x{z[1]:+.1f}')
ax.plot([0, 105], [0, 105], 'k-', lw=1.2, alpha=0.6, label='1:1 参考线')
ax.set_xlabel('实际去除率 (%)'); ax.set_ylabel('预测去除率 (%)')
ax.set_title(f'测试集 实际-预测线性关系 (R²={r2_te:.3f})', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.25)

fig.suptitle('BTF 去除率预测 — 训练/测试分布与线性拟合 (随机森林)',
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG, 'fig14_训练测试分布与线性拟合.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("已保存 fig14")

# ============ 图15: 特征重要性 (置换重要性 Top20) ============
pi = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
imp_std = pd.Series(pi.importances_std, index=X_test.columns).loc[imp.index]
CN = {'pH_imp': 'pH', '温度_imp': '温度', 'EBRT_imp': 'EBRT(空床停留)', '进气浓度_imp': '进气浓度',
      'ppm_imp': '进气浓度(ppm)', '负荷_imp': '入口负荷', '氮浓度_imp': '氮浓度',
      '天数_imp': '运行天数', '阶段_imp': '运行阶段', 'pH缺失': 'pH缺失', '温度缺失': '温度缺失',
      'EBRT_s原始缺失': 'EBRT缺失', '进气浓度_mgm3原始缺失': '进气浓度缺失',
      '运行阶段原始缺失': '阶段缺失', '疏水性污染物': '疏水性', '饥饿胁迫': '饥饿胁迫',
      '重启恢复': '重启恢复', '等离子耦合': '等离子耦合', '混合共降解': '共降解',
      '有表面活性剂': '表面活性剂', '有铁离子': '铁离子', '有紫外预处理': '紫外预处理',
      '有营养液': '营养液', '有额外碳源': '额外碳源', '真菌主导': '真菌主导',
      '污染物类别': '污染物类别', '常见污染物': '常见污染物'}
for k, v in list(PACKS.items()):
    CN[k] = v.replace('填料_', '填料:')
topn = imp.tail(20)
topn_labels = [CN.get(i, i) for i in topn.index]
topn_std = imp_std.tail(20).values

fig, ax = plt.subplots(figsize=(10, 8.5))
colors = ['#C44E52' if v > 0.05 else '#4C72B0' for v in topn.values]
ax.barh(range(len(topn)), topn.values, color=colors, edgecolor='white', height=0.66)
ax.errorbar(topn.values, range(len(topn)), xerr=topn_std, fmt='none', ecolor='#666', capsize=3, lw=1)
ax.set_yticks(range(len(topn)), topn_labels)
ax.set_xlabel('置换重要性（测试集 R² 下降幅度）')
ax.set_title('特征重要性排序 Top 20（随机森林，置换检验）', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(topn.values):
    ax.text(v + topn_std[i] + 0.004, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_xlim(0, topn.values.max() + topn_std.max() + 0.06)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig15_特征重要性.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("已保存 fig15")
print("DONE")
