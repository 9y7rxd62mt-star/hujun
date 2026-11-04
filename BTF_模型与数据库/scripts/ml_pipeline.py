# -*- coding: utf-8 -*-
"""
BTF (生物滴滤塔) 去除率预测 - 机器学习建模与可视化 (v2 强化特征版)
特征来源: 污染物(身份/类别/疏水性), pH, 温度, 营养条件+数据来源文本深度解析
  (EBRT/进气浓度/入口负荷/氮浓度/运行天数/运行阶段/饥饿重启/耦合工艺/填料/添加剂等)
目标: 去除率 (RE, %)
"""
import sys, io, re, time, warnings, traceback, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, learning_curve, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

T0 = time.time()
FIG_DIR = r"C:\Users\15787\WorkBuddy\2026-11-03-11-42-33\BTF_ML\output\figures"
OUT_DIR = r"C:\Users\15787\WorkBuddy\2026-11-03-11-42-33\BTF_ML\output"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}")
    sys.stdout.flush()

# ============================================================
# 1. 数据加载与清洗
# ============================================================
log("加载数据...")
df = pd.read_csv(os.path.join(OUT_DIR, "BTF_data_parsed.csv"))
log(f"原始BTF数据: {len(df)} 行")

# 目标有效性
df = df[df['去除率_数值'].notna()]
df = df[(df['去除率_数值'] >= 0) & (df['去除率_数值'] <= 100)]

# 目标不确定度清洗: 剔除跨度>40%的范围值目标 (如 '0-100', '5-75', '20-90' 均值代表性差)
def target_range_width(val):
    if not isinstance(val, str):
        return 0.0
    m = re.findall(r'\d+(?:\.\d+)?', val)
    if '-' in val and len(m) >= 2:
        return abs(float(m[1]) - float(m[0]))
    return 0.0

df['目标跨度'] = df['去除率'].apply(target_range_width)
n0 = len(df)
df = df[df['目标跨度'] <= 40].reset_index(drop=True)
log(f"剔除宽范围目标(跨度>40%): {n0} -> {len(df)} 行")

# 去除完全重复的行 (同文献同条件同结果)
n0 = len(df)
key_cols = ['论文标题', '数据来源', '污染物名称', 'ph', '温度', '营养条件', '去除率']
df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
log(f"去重: {n0} -> {len(df)} 行 (删除 {n0-len(df)} 行完全重复)")

y = df['去除率_数值'].values  # 初步统计用, 2.2b 剔除后会重新赋值
log(f"初步有效数据: {len(df)} 行, 去除率 {y.mean():.1f}±{y.std():.1f}")

# ============================================================
# 2. 特征工程
# ============================================================
log("特征工程...")

# ---- 2.1 污染物 ----
CANON = {'n-己烷': '正己烷', '二甲基乙酰胺': 'N,N-二甲基乙酰胺', '己烷': '正己烷'}
df['污染物'] = df['污染物名称'].astype(str).str.strip().replace(CANON)
CAT_MAP = {
    '甲苯': '芳香族', '苯': '芳香族', '间二甲苯': '芳香族', '对二甲苯': '芳香族', '二甲苯': '芳香族',
    '乙苯': '芳香族', '苯乙烯': '芳香族', '苯甲醇': '芳香族',
    '二氯甲烷': '氯代VOCs', '氯仿': '氯代VOCs', '氯苯': '氯代VOCs', '1,2-二氯苯': '氯代VOCs',
    '二氯溴甲烷': '氯代VOCs',
    '正己烷': '烷烃', '环己烷': '烷烃',
    '甲醇': '含氧VOCs', '乙醇': '含氧VOCs', '乙醛': '含氧VOCs', '丙酮': '含氧VOCs',
    '2-丁酮': '含氧VOCs', '乙酸乙酯': '含氧VOCs', '乙酸甲酯': '含氧VOCs', '乙酸丁酯': '含氧VOCs',
    '甲基丙烯酸正丁酯': '含氧VOCs', '丙烯酸乙酯': '含氧VOCs',
    '硫化氢': '硫化物', '乙硫醇': '硫化物', '丙硫醇': '硫化物', '甲硫醚': '硫化物', '二甲基硫': '硫化物',
    '丙烯腈': '含氮化合物', '三甲胺': '含氮化合物', '二甲胺': '含氮化合物',
    'N,N-二甲基乙酰胺': '含氮化合物', '氨': '含氮化合物',
    '磷酸三丁酯': '其他',
}
df['污染物类别'] = df['污染物'].map(CAT_MAP).fillna('恶臭/其他')
df['疏水性污染物'] = df['污染物类别'].isin(['烷烃', '芳香族', '氯代VOCs']).astype(int)
# 常见污染物 (出现>=4次) one-hot
cnt = df['污染物'].value_counts()
df['常见污染物'] = df['污染物'].where(df['污染物'].map(cnt) >= 4, '其他')

# ---- 2.2 组合文本 (营养条件 + 数据来源) ----
TXT = (df['营养条件'].fillna('') + ' ; ' + df['数据来源'].fillna('')).astype(str)
NUM = r'(\d+(?:\.\d+)?)'

def find_val(t, unit_pat, range_first=True):
    """提取数值, 范围取均值"""
    if range_first:
        m = re.search(NUM + r'\s*[-–~]\s*' + NUM + r'\s*' + unit_pat, t)
        if m:
            try:
                return (float(m.group(1)) + float(m.group(2))) / 2
            except Exception:
                pass
    m = re.search(NUM + r'\s*' + unit_pat, t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    return np.nan

# EBRT (秒)
def get_ebrt(t):
    for pat in [r'EBRT[^0-9\n]{0,14}?' + NUM + r'\s*[sS秒]',
                r'停留时间\s*[=为]?\s*' + NUM + r'\s*[sS秒]',
                r'进气口\s*[=为]?\s*' + NUM + r'\s*秒']:
        m = re.search(pat, t)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return np.nan

df['EBRT_s'] = TXT.apply(get_ebrt)
df['进气浓度_mgm3'] = TXT.apply(lambda t: find_val(t, r'mg/m'))
df['进气浓度_ppm'] = TXT.apply(lambda t: find_val(t, r'[pP][pP][mM]'))
df['入口负荷_gmh'] = TXT.apply(lambda t: find_val(t, r'g/m[³3]\s*·?\s*h'))
df['氮浓度_mgL'] = TXT.apply(lambda t: find_val(t.replace('含氮522', '含氮 522'), r'mg/L')
                             if isinstance(t, str) else np.nan)
# 运行天数
def get_day(t):
    for pat in [r'第\s*' + NUM + r'\s*天', r'[Dd]ay\s*' + NUM, r'（\s*' + NUM + r'\s*[天d]']:
        m = re.search(pat, t)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return np.nan
df['运行天数'] = TXT.apply(get_day)

# 运行阶段 (稳定=3 > 中期=2 > 初期/启动=1)
def get_stage(t):
    if re.search(r'稳定|stable', t, re.I):
        return 3
    if '中期' in t:
        return 2
    if re.search(r'初期|启动|驯化|挂膜|start', t, re.I):
        return 1
    return np.nan
df['运行阶段'] = TXT.apply(get_stage)

# 二元特征
def flag(pat):
    return TXT.str.contains(pat, case=False, regex=True).astype(int)

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

# 填料 one-hot
PACKS = {'填料_陶粒': r'陶粒', '填料_浮石': r'浮石|pumice', '填料_火山岩': r'火山岩|[Vv]olcanic',
         '填料_树皮': r'树皮|[Bb]ark', '填料_复合填料': r'复合填料',
         '填料_多面球': r'多面球', '填料_聚氨酯': r'聚氨酯|PU|[Pp]olyurethane',
         '填料_鲍尔环': r'鲍尔环|Pall', '填料_木屑堆肥': r'木屑|堆肥|compost|木屑和堆肥',
         '填料_丝瓜': r'丝瓜|loofah', '填料_活性炭': r'活性炭|activated carbon'}
for name, pat in PACKS.items():
    df[name] = flag(pat)

df['pH'] = df['ph_数值']
df['温度'] = df['温度_数值']
df['pH缺失'] = df['pH'].isna().astype(int)
df['温度缺失'] = df['温度'].isna().astype(int)

# ---- 2.2b 剔除非反应器运行数据 (摇瓶筛选实验 / 机理对照实验) ----
n0 = len(df)
df = df[(df['摇瓶实验'] == 0) & (df['对照机理实验'] == 0)].reset_index(drop=True)
log(f"剔除摇瓶筛选/机理对照实验行: {n0} -> {len(df)} 行")
y = df['去除率_数值'].values
log(f"最终建模数据: {len(df)} 行, 去除率 {y.mean():.1f}±{y.std():.1f}")

# ---- 2.2c 文献级中位数插补 (同一文献的EBRT/浓度/pH等操作参数通常恒定) ----
IMP_MAP = {'pH': 'pH_imp', '温度': '温度_imp', 'EBRT_s': 'EBRT_imp',
           '进气浓度_mgm3': '进气浓度_imp', '进气浓度_ppm': 'ppm_imp',
           '入口负荷_gmh': '负荷_imp', '氮浓度_mgL': '氮浓度_imp',
           '运行天数': '天数_imp', '运行阶段': '阶段_imp'}
for src, dst in IMP_MAP.items():
    pm = df.groupby('论文标题')[src].transform('median')
    df[dst] = df[src].fillna(pm)
log("文献级插补后覆盖率: " + str({dst: int(df[dst].notna().sum()) for dst in IMP_MAP.values()}))

# 缺失指示特征
for src in ['EBRT_s', '进气浓度_mgm3', '运行阶段']:
    df[src + '原始缺失'] = df[src].isna().astype(int)

# ---- 2.3 特征分组 ----
num_feats = list(IMP_MAP.values())
bin_feats = (['pH缺失', '温度缺失', 'EBRT_s原始缺失', '进气浓度_mgm3原始缺失', '运行阶段原始缺失',
              '疏水性污染物', '饥饿胁迫', '重启恢复',
              '等离子耦合', '混合共降解', '有表面活性剂', '有铁离子', '有紫外预处理', '有营养液',
              '有额外碳源', '真菌主导'] + list(PACKS.keys()))
cat_feats = ['污染物类别', '常见污染物']

log("特征覆盖率: " + str({c: int(df[c].notna().sum()) for c in num_feats}))
log(f"特征总数: {len(num_feats)}数值 + {len(bin_feats)}二元 + {len(cat_feats)}类别({df['常见污染物'].nunique()}+{df['污染物类别'].nunique()}个水平)")

# 保存特征工程结果
feat_save_cols = ['论文标题', '污染物', '污染物类别', 'pH', '温度', 'EBRT_s', '进气浓度_mgm3',
                  '入口负荷_gmh', '氮浓度_mgL', '运行天数', '运行阶段'] + \
                 [c for c in bin_feats if not c.endswith('缺失') and not c.endswith('原始缺失')] + ['去除率_数值']
df[feat_save_cols].to_csv(os.path.join(OUT_DIR, "BTF_ML_features.csv"),
                          index=False, encoding='utf-8-sig')
log("已保存特征工程数据集: output/BTF_ML_features.csv")

# ============================================================
# 3. 数据集划分
# ============================================================
X = df[num_feats + bin_feats + cat_feats].copy()
groups = df['论文标题'].values
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42)
log(f"训练集: {len(X_train)} 行, 测试集: {len(X_test)} 行")

# ============================================================
# 4. 模型
# ============================================================
numeric_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
binary_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent'))])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', numeric_pipe, num_feats),
                         ('bin', binary_pipe, bin_feats),
                         ('cat', cat_pipe, cat_feats)])

cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    '线性回归': (Pipeline([('pre', pre), ('m', LinearRegression())]), {}),
    'Ridge回归': (Pipeline([('pre', pre), ('m', Ridge())]),
                  {'m__alpha': [0.1, 1, 10, 100]}),
    'SVR': (Pipeline([('pre', pre), ('m', SVR())]),
            {'m__C': [1, 10, 100], 'm__gamma': ['scale', 0.01, 0.1]}),
    'KNN': (Pipeline([('pre', pre), ('m', KNeighborsRegressor())]),
            {'m__n_neighbors': [5, 10, 15, 20], 'm__weights': ['uniform', 'distance']}),
    '随机森林': (Pipeline([('pre', pre), ('m', RandomForestRegressor(random_state=42, n_jobs=-1))]),
                {'m__n_estimators': [300, 600], 'm__max_depth': [None, 8, 12, 16],
                 'm__min_samples_leaf': [1, 2, 4], 'm__max_features': ['sqrt', 0.4, 0.7]}),
    'ExtraTrees': (Pipeline([('pre', pre), ('m', ExtraTreesRegressor(random_state=42, n_jobs=-1))]),
                   {'m__n_estimators': [300, 600], 'm__min_samples_leaf': [1, 2, 4],
                    'm__max_features': ['sqrt', 0.4, 0.7]}),
    'HistGB梯度提升': (Pipeline([('pre', pre), ('m', HistGradientBoostingRegressor(random_state=42))]),
                      {'m__learning_rate': [0.03, 0.05, 0.1], 'm__max_iter': [200, 400],
                       'm__max_depth': [None, 3, 5], 'm__min_samples_leaf': [5, 10, 20],
                       'm__l2_regularization': [0, 1.0]}),
    'GBDT': (Pipeline([('pre', pre), ('m', GradientBoostingRegressor(random_state=42))]),
             {'m__n_estimators': [200, 400], 'm__learning_rate': [0.03, 0.05, 0.1],
              'm__max_depth': [2, 3, 4], 'm__min_samples_leaf': [3, 7]}),
    '神经网络MLP': (Pipeline([('pre', pre), ('m', MLPRegressor(random_state=42, max_iter=5000,
                                                   early_stopping=True, n_iter_no_change=40))]),
                   {'m__hidden_layer_sizes': [(32,), (64, 32), (128, 64, 32)],
                    'm__alpha': [0.01, 0.1, 1.0]}),
}

# ============================================================
# 5. 训练与调参
# ============================================================
results, fitted = {}, {}
for name, (pipe, grid) in models.items():
    log(f"训练: {name} ...")
    try:
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv5, scoring='r2', n_jobs=-1, refit=True)
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            log(f"   最优参数: {gs.best_params_}")
            cv_mean, cv_std = gs.best_score_, gs.cv_results_['std_test_score'][gs.best_index_]
        else:
            best_pipe = pipe.fit(X_train, y_train)
            s = cross_val_score(pipe, X_train, y_train, cv=cv5, scoring='r2', n_jobs=-1)
            cv_mean, cv_std = s.mean(), s.std()
        pred_test = best_pipe.predict(X_test)
        results[name] = {'CV_R2均值': cv_mean, 'CV_R2标准差': cv_std,
                         '测试R2': r2_score(y_test, pred_test),
                         '测试RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
                         '测试MAE': mean_absolute_error(y_test, pred_test),
                         '训练R2': r2_score(y_train, best_pipe.predict(X_train))}
        fitted[name] = best_pipe
        log(f"   CV R2={cv_mean:.4f}±{cv_std:.4f} | 测试 R2={results[name]['测试R2']:.4f}")
    except Exception as e:
        log(f"   失败: {e}\n{traceback.format_exc()}")

# 投票集成 (Top-3)
try:
    top3 = sorted(results, key=lambda k: results[k]['CV_R2均值'], reverse=True)[:3]
    log(f"投票集成成员: {top3}")
    voting = Pipeline([('pre', pre),
                       ('m', VotingRegressor([(n, fitted[n].named_steps['m']) for n in top3]))])
    s = cross_val_score(voting, X_train, y_train, cv=cv5, scoring='r2', n_jobs=-1)
    voting.fit(X_train, y_train)
    pred_test = voting.predict(X_test)
    results['投票集成'] = {'CV_R2均值': s.mean(), 'CV_R2标准差': s.std(),
                         '测试R2': r2_score(y_test, pred_test),
                         '测试RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
                         '测试MAE': mean_absolute_error(y_test, pred_test),
                         '训练R2': r2_score(y_train, voting.predict(X_train))}
    fitted['投票集成'] = voting
    log(f"投票集成: CV R2={s.mean():.4f}±{s.std():.4f} | 测试 R2={results['投票集成']['测试R2']:.4f}")
except Exception as e:
    log(f"投票集成失败: {e}")

res_df = pd.DataFrame(results).T.sort_values('CV_R2均值', ascending=False)
log("\n===== 模型对比汇总 =====")
print(res_df.to_string(float_format=lambda x: f"{x:.4f}"))
res_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), encoding='utf-8-sig')

# ============================================================
# 6. 最优模型评估
# ============================================================
BEST = res_df.index[0]
best_model = fitted[BEST]
log(f"\n最优模型: {BEST}")
best_pred_train = best_model.predict(X_train)
best_pred_test = best_model.predict(X_test)
r2_test = r2_score(y_test, best_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, best_pred_test))
mae_test = mean_absolute_error(y_test, best_pred_test)

gk = GroupKFold(n_splits=5)
g_scores = cross_val_score(best_model, X, y, cv=gk, groups=groups, scoring='r2')
log(f"按文献分组5折CV R2 = {g_scores.mean():.4f} ± {g_scores.std():.4f} (各折: {np.round(g_scores, 3)})")

pd.DataFrame({'实际去除率': y_test, '预测去除率': best_pred_test,
              '文献': g_test, '污染物': df.loc[X_test.index, '污染物']}).to_csv(
    os.path.join(OUT_DIR, "test_predictions.csv"), index=False, encoding='utf-8-sig')

# ============================================================
# 7. 可视化
# ============================================================
log("\n生成可视化图表...")
C_TRAIN, C_TEST = '#4C72B0', '#C44E52'

def boxplot_compat(ax, data, tick_labels, **kw):
    """兼容新旧 matplotlib 的 boxplot 接口 (labels -> tick_labels)"""
    try:
        return ax.boxplot(data, tick_labels=tick_labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=tick_labels, **kw)

def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches='tight', dpi=150)
    plt.close(fig)
    log(f"   已保存 {name}")

# ---- 图1 数据概览 ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].hist(y, bins=30, color='#4C72B0', edgecolor='white')
axes[0, 0].set(title=f'去除率分布 (n={len(y)})', xlabel='去除率 (%)', ylabel='频数')
axes[0, 1].boxplot(y, vert=True, widths=0.5)
axes[0, 1].set(title='去除率箱线图', ylabel='去除率 (%)')
vp = df['pH'].dropna()
axes[1, 0].hist(vp, bins=25, color='#55A868', edgecolor='white')
axes[1, 0].set(title=f'pH 分布 (有效{len(vp)}行, 缺失{df["pH缺失"].sum()}行)', xlabel='pH', ylabel='频数')
vt = df['温度'].dropna()
axes[1, 1].hist(vt, bins=25, color='#CCB974', edgecolor='white')
axes[1, 1].set(title=f'温度分布 (有效{len(vt)}行, 缺失{df["温度缺失"].sum()}行)', xlabel='温度 (°C)', ylabel='频数')
fig.suptitle('图1 生物滴滤塔(BTF)建模数据概览', fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'fig01_数据概览.png')

# ---- 图2 相关性热力图 ----
corr_cols = ['去除率_数值', 'pH', '温度', 'EBRT_s', '进气浓度_mgm3', '氮浓度_mgL',
             '运行天数', '运行阶段', '疏水性污染物', '有营养液', '有表面活性剂', '饥饿胁迫', '混合共降解']
corr_df = df[corr_cols].rename(columns={'去除率_数值': '去除率(%)', 'EBRT_s': 'EBRT(s)',
                                        '进气浓度_mgm3': '进气浓度(mg/m³)',
                                        '氮浓度_mgL': '氮浓度(mg/L)'})
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                color='white' if abs(corr.iloc[i, j]) > 0.55 else 'black', fontsize=9)
fig.colorbar(im, shrink=0.8, label='Pearson 相关系数')
ax.set_title('图2 特征与去除率的Pearson相关性热力图', fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'fig02_相关性热力图.png')
log("   与去除率相关性: " + str({k: round(v, 3) for k, v in
                              corr['去除率(%)'].drop('去除率(%)').sort_values().items()}))

# ---- 图3 污染物分析 ----
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
order = df.groupby('污染物类别')['去除率_数值'].median().sort_values().index.tolist()
data_by_cat = [df.loc[df['污染物类别'] == c, '去除率_数值'].values for c in order]
bp = boxplot_compat(axes[0], data_by_cat, order, patch_artist=True, showmeans=True)
for patch, c in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(order)))):
    patch.set_facecolor(c)
axes[0].set(title='不同污染物类别的去除率', xlabel='污染物类别', ylabel='去除率 (%)')
axes[0].tick_params(axis='x', rotation=15)
for i, c in enumerate(order):
    axes[0].text(i + 1, 103, f'n={len(data_by_cat[i])}', ha='center', fontsize=9, color='gray')
hyd = [df.loc[df['疏水性污染物'] == v, '去除率_数值'].values for v in [1, 0]]
bp2 = boxplot_compat(axes[1], hyd, ['疏水性污染物\n(烷烃/芳香族/氯代VOCs)', '亲水性污染物\n(含氧/含氮/硫化物等)'],
                     patch_artist=True, showmeans=True)
for patch, c in zip(bp2['boxes'], ['#DD8452', '#55A868']):
    patch.set_facecolor(c)
axes[1].set(title='疏水性 vs 亲水性污染物的去除率', ylabel='去除率 (%)')
for i, d in enumerate(hyd):
    axes[1].text(i + 1, 103, f'n={len(d)}', ha='center', fontsize=9, color='gray')
fig.suptitle('图3 污染物特性对去除率的影响', fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'fig03_污染物分析.png')

# ---- 图4 运行条件与去除率 ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
pairs = [('pH', 'pH', 'pH'), ('温度', '温度 (°C)', '温度'),
         ('EBRT_s', 'EBRT (s)', 'EBRT_s'), ('进气浓度_mgm3', '进气浓度 (mg/m³)', '进气浓度_mgm3')]
for ax, (col, xlabel, _) in zip(axes.flat, pairs):
    d = df[[col, '去除率_数值']].dropna()
    ax.scatter(d[col], d['去除率_数值'], s=22, alpha=0.55, c='#4C72B0', edgecolors='none')
    if len(d) > 5:
        z = np.polyfit(d[col], d['去除率_数值'], 1)
        xs = np.linspace(d[col].min(), d[col].max(), 50)
        ax.plot(xs, np.polyval(z, xs), 'r--', lw=1.5,
                label=f'趋势线 (r={d[col].corr(d["去除率_数值"]):.2f})')
        ax.legend(fontsize=9)
    ax.set(title=f'{xlabel} vs 去除率 (n={len(d)})', xlabel=xlabel, ylabel='去除率 (%)')
fig.suptitle('图4 运行条件对去除率的影响', fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'fig04_运行条件散点.png')

# ---- 图5 模型对比 ----
fig, ax = plt.subplots(figsize=(12, 6))
names = res_df.index.tolist()
x_pos = np.arange(len(names))
colors = ['#C44E52' if n == BEST else '#4C72B0' for n in names]
ax.bar(x_pos, res_df['CV_R2均值'], yerr=res_df['CV_R2标准差'], capsize=4,
       color=colors, edgecolor='white', width=0.62)
ax.plot(x_pos, res_df['测试R2'], 'D-', color='#55A868', ms=7, lw=1.2, label='独立测试集 R²')
for i in range(len(names)):
    ax.text(i, res_df['CV_R2均值'].iloc[i] + 0.035, f"{res_df['CV_R2均值'].iloc[i]:.3f}",
            ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x_pos, names, rotation=18, ha='right')
ax.set_ylabel('R² (决定系数)')
ax.set_ylim(0, 1.08)
ax.axhline(0.8, ls=':', color='gray', lw=1)
ax.text(len(names) - 0.4, 0.81, 'R²=0.8', fontsize=8, color='gray', ha='right')
ax.legend(loc='lower right', fontsize=10)
ax.set_title('图5 各机器学习模型预测性能对比 (5折交叉验证 + 独立测试集)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save(fig, 'fig05_模型对比.png')

# ---- 图6 预测 vs 实际 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
lims = [min(y.min(), best_pred_train.min()) - 3, 103]
axes[0].scatter(y_train, best_pred_train, s=26, alpha=0.6, c=C_TRAIN, edgecolors='none',
                label=f'训练集 (R²={r2_score(y_train, best_pred_train):.3f})')
axes[0].scatter(y_test, best_pred_test, s=42, alpha=0.85, c=C_TEST, marker='D', edgecolors='none',
                label=f'测试集 (R²={r2_test:.3f})')
axes[0].plot(lims, lims, 'k--', lw=1.2, label='1:1 线')
axes[0].set(xlabel='实际去除率 (%)', ylabel='预测去除率 (%)',
            title=f'{BEST} 模型预测效果', xlim=lims, ylim=lims)
axes[0].legend(loc='upper left', fontsize=9)
err = best_pred_test - y_test
axes[1].scatter(y_test, err, s=42, c=C_TEST, alpha=0.8, edgecolors='none')
axes[1].axhline(0, color='k', ls='--', lw=1)
axes[1].set(xlabel='实际去除率 (%)', ylabel='预测误差 (预测-实际, 百分点)', title='测试集预测误差')
fig.suptitle(f'图6 最优模型预测性能 (测试集 RMSE={rmse_test:.2f}, MAE={mae_test:.2f})',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'fig06_预测效果图.png')

# ---- 图7 特征重要性 ----
try:
    log("计算置换重要性...")
    pi = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
    topn = imp.tail(18)
    cols = list(X_test.columns)
    stds = [pi.importances_std[cols.index(i)] for i in topn.index]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(topn.index, topn.values,
            color=np.where(np.array(topn.values) > 0, '#4C72B0', '#AAAAAA'), edgecolor='white')
    ax.errorbar(topn.values, range(len(topn)), xerr=stds, fmt='none', ecolor='#444', capsize=3, lw=1)
    ax.set_xlabel('置换重要性 (R²下降幅度)')
    ax.set_title(f'图7 特征重要性排序 ({BEST}, 测试集置换检验)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save(fig, 'fig07_特征重要性.png')
    log("   Top重要特征: " + str(topn.tail(10).round(4).to_dict()))
except Exception as e:
    log(f"图7失败: {e}\n{traceback.format_exc()}")

# ---- 图8 残差分析 ----
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
resid_train = y_train - best_pred_train
resid_test = y_test - best_pred_test
axes[0].scatter(best_pred_train, resid_train, s=24, alpha=0.5, c=C_TRAIN, edgecolors='none', label='训练集')
axes[0].scatter(best_pred_test, resid_test, s=42, alpha=0.85, c=C_TEST, marker='D', edgecolors='none', label='测试集')
axes[0].axhline(0, color='k', ls='--', lw=1)
axes[0].set(xlabel='预测去除率 (%)', ylabel='残差 (实际-预测, 百分点)', title='残差 vs 预测值')
axes[0].legend(fontsize=9)
axes[1].hist(resid_train, bins=25, alpha=0.6, color=C_TRAIN, label='训练集', edgecolor='white')
axes[1].hist(resid_test, bins=25, alpha=0.6, color=C_TEST, label='测试集', edgecolor='white')
axes[1].axvline(0, color='k', ls='--', lw=1)
axes[1].set(xlabel='残差 (百分点)', ylabel='频数', title='残差分布')
axes[1].legend(fontsize=9)
fig.suptitle(f'图8 残差分析 ({BEST})', fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'fig08_残差分析.png')

# ---- 图9 学习曲线 ----
try:
    ts, tr_sc, te_sc = learning_curve(best_model, X, y, cv=5, scoring='r2', n_jobs=-1,
                                      train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ts, tr_sc.mean(1), 'o-', color=C_TRAIN, label='训练集 R²')
    ax.plot(ts, te_sc.mean(1), 's-', color=C_TEST, label='交叉验证 R²')
    ax.fill_between(ts, tr_sc.mean(1) - tr_sc.std(1), tr_sc.mean(1) + tr_sc.std(1), alpha=0.15, color=C_TRAIN)
    ax.fill_between(ts, te_sc.mean(1) - te_sc.std(1), te_sc.mean(1) + te_sc.std(1), alpha=0.15, color=C_TEST)
    ax.set(xlabel='训练样本数', ylabel='R²', title='图9 学习曲线 (数据量对模型性能的影响)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, 'fig09_学习曲线.png')
except Exception as e:
    log(f"图9失败: {e}")

# ---- 图10 部分依赖图 ----
try:
    pdp_feats = ['pH_imp', '温度_imp', 'EBRT_imp', '进气浓度_imp', '阶段_imp', '天数_imp']
    avail = [f for f in pdp_feats if X[f].notna().sum() >= 20]
    fig, ax = plt.subplots(figsize=(11, 8))
    PartialDependenceDisplay.from_estimator(best_model, X, avail, ax=ax, n_cols=2,
                                            line_kw={'color': '#C44E52', 'lw': 2})
    fig.suptitle('图10 关键因素对去除率的部分依赖图 (其他变量取平均)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig10_部分依赖图.png')
except Exception as e:
    log(f"图10失败: {e}\n{traceback.format_exc()}")

# ============================================================
# 8. 汇总
# ============================================================
log("\n" + "=" * 70)
log(f"最优模型: {BEST}")
log(f"  5折交叉验证 R² = {res_df.loc[BEST, 'CV_R2均值']:.4f} ± {res_df.loc[BEST, 'CV_R2标准差']:.4f}")
log(f"  独立测试集   R² = {r2_test:.4f}, RMSE = {rmse_test:.2f}, MAE = {mae_test:.2f}")
log(f"  按文献分组CV R² = {g_scores.mean():.4f} ± {g_scores.std():.4f} (稳健性检验)")
log(f"建模样本: {len(df)} 行 | 特征: {len(num_feats)}数值 + {len(bin_feats)}二元 + {len(cat_feats)}类别")
log("全部完成!")
