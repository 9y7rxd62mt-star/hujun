# -*- coding: utf-8 -*-
"""
BTF 数据库整理导出 + 模型预测可视化
1. 复现 ml_pipeline.py 的特征工程与最优模型(随机森林, 固定种子) -> 全量预测
2. 生成模型预测可视化图 (fig11)
3. 导出格式化 Excel 数据库
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

BASE = r"C:\Users\15787\WorkBuddy\2026-11-03-11-42-33\BTF_ML"
OUT = os.path.join(BASE, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ============ 1. 数据加载与清洗 (复现 ml_pipeline.py) ============
df = pd.read_csv(os.path.join(OUT, "BTF_data_parsed.csv"))
df = df[df['去除率_数值'].notna()]
df = df[(df['去除率_数值'] >= 0) & (df['去除率_数值'] <= 100)]

def target_range_width(val):
    if not isinstance(val, str):
        return 0.0
    m = re.findall(r'\d+(?:\.\d+)?', val)
    if '-' in val and len(m) >= 2:
        return abs(float(m[1]) - float(m[0]))
    return 0.0

df['目标跨度'] = df['去除率'].apply(target_range_width)
df = df[df['目标跨度'] <= 40].reset_index(drop=True)
key_cols = ['论文标题', '数据来源', '污染物名称', 'ph', '温度', '营养条件', '去除率']
df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)

# ============ 2. 特征工程 (复现) ============
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
cnt = df['污染物'].value_counts()
df['常见污染物'] = df['污染物'].where(df['污染物'].map(cnt) >= 4, '其他')

TXT = (df['营养条件'].fillna('') + ' ; ' + df['数据来源'].fillna('')).astype(str)
NUM = r'(\d+(?:\.\d+)?)'

def find_val(t, unit_pat, range_first=True):
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

def get_stage(t):
    if re.search(r'稳定|stable', t, re.I):
        return 3
    if '中期' in t:
        return 2
    if re.search(r'初期|启动|驯化|挂膜|start', t, re.I):
        return 1
    return np.nan
df['运行阶段'] = TXT.apply(get_stage)

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

df = df[(df['摇瓶实验'] == 0) & (df['对照机理实验'] == 0)].reset_index(drop=True)
y = df['去除率_数值'].values

IMP_MAP = {'pH': 'pH_imp', '温度': '温度_imp', 'EBRT_s': 'EBRT_imp',
           '进气浓度_mgm3': '进气浓度_imp', '进气浓度_ppm': 'ppm_imp',
           '入口负荷_gmh': '负荷_imp', '氮浓度_mgL': '氮浓度_imp',
           '运行天数': '天数_imp', '运行阶段': '阶段_imp'}
for src, dst in IMP_MAP.items():
    pm = df.groupby('论文标题')[src].transform('median')
    df[dst] = df[src].fillna(pm)
for src in ['EBRT_s', '进气浓度_mgm3', '运行阶段']:
    df[src + '原始缺失'] = df[src].isna().astype(int)

num_feats = list(IMP_MAP.values())
bin_feats = (['pH缺失', '温度缺失', 'EBRT_s原始缺失', '进气浓度_mgm3原始缺失', '运行阶段原始缺失',
              '疏水性污染物', '饥饿胁迫', '重启恢复',
              '等离子耦合', '混合共降解', '有表面活性剂', '有铁离子', '有紫外预处理', '有营养液',
              '有额外碳源', '真菌主导'] + list(PACKS.keys()))
cat_feats = ['污染物类别', '常见污染物']

X = df[num_feats + bin_feats + cat_feats].copy()
groups = df['论文标题'].values

# ============ 3. 划分 + 最优模型(随机森林) ============
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42)

numeric_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
binary_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent'))])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', numeric_pipe, num_feats),
                         ('bin', binary_pipe, bin_feats),
                         ('cat', cat_pipe, cat_feats)])

# 最优参数 (来自 ml_pipeline.py GridSearchCV)
best_model = Pipeline([('pre', pre),
                       ('m', RandomForestRegressor(n_estimators=300, max_depth=None,
                                                   max_features=0.4, min_samples_leaf=1,
                                                   random_state=42, n_jobs=-1))])
best_model.fit(X_train, y_train)

pred_train = best_model.predict(X_train)
pred_test = best_model.predict(X_test)
r2_tr = r2_score(y_train, pred_train)
r2_te = r2_score(y_test, pred_test)
rmse_te = np.sqrt(mean_squared_error(y_test, pred_test))
mae_te = mean_absolute_error(y_test, pred_test)

print(f"复现完成: 训练R2={r2_tr:.4f}, 测试R2={r2_te:.4f}, RMSE={rmse_te:.2f}, MAE={mae_te:.2f}")

# ============ 4. 模型预测可视化图 ============
C_TRAIN, C_TEST = '#4C72B0', '#C44E52'
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={'width_ratios': [1.15, 1]})

lims = [min(y.min(), pred_train.min(), pred_test.min()) - 3, 103]
# 左: 实际 vs 预测
axes[0].scatter(y_train, pred_train, s=26, alpha=0.55, c=C_TRAIN, edgecolors='none',
                label=f'训练集 (n={len(y_train)}, R²={r2_tr:.3f})')
axes[0].scatter(y_test, pred_test, s=48, alpha=0.9, c=C_TEST, marker='D', edgecolors='white',
                linewidths=0.6, label=f'测试集 (n={len(y_test)}, R²={r2_te:.3f})')
axes[0].plot(lims, lims, 'k--', lw=1.4, label='1:1 参考线')
axes[0].fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims],
                     color='gray', alpha=0.08)
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_xlabel('实际去除率 (%)', fontsize=12)
axes[0].set_ylabel('预测去除率 (%)', fontsize=12)
axes[0].set_title('随机森林模型预测效果\n(实际 vs 预测)', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(alpha=0.25)
axes[0].text(0.02, 0.97,
             f'测试集  R² = {r2_te:.3f}\nRMSE = {rmse_te:.2f}%   MAE = {mae_te:.2f}%',
             transform=axes[0].transAxes, fontsize=11, va='top',
             bbox=dict(boxstyle='round,pad=0.4', fc='#FFF7E6', ec='#E6A23C', alpha=0.95))

# 右: 误差分布
err_train = pred_train - y_train
err_test = pred_test - y_test
axes[1].hist(err_train, bins=30, alpha=0.55, color=C_TRAIN, edgecolor='white',
             label=f'训练集 (σ={err_train.std():.1f})')
axes[1].hist(err_test, bins=30, alpha=0.8, color=C_TEST, edgecolor='white',
             label=f'测试集 (σ={err_test.std():.1f})')
axes[1].axvline(0, color='k', ls='--', lw=1.4)
axes[1].axvline(err_test.mean(), color=C_TEST, ls=':', lw=1.6,
                label=f'测试均值偏差 ({err_test.mean():+.1f} 百分点)')
axes[1].set_xlabel('预测误差 (预测 − 实际, 百分点)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title('预测误差分布', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.25)

fig.suptitle('BTF 生物滴滤塔去除率预测 — 最优模型（随机森林）可视化',
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = os.path.join(FIG, 'fig11_模型预测可视化.png')
fig.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.close(fig)
print("已保存预测可视化图:", fig_path)

# ============ 5. 导出格式化 Excel ============
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = Workbook()

HDR_FILL = PatternFill('solid', fgColor='1565C0')
HDR_FONT = Font(name='微软雅黑', bold=True, color='FFFFFF', size=10)
BODY_FONT = Font(name='微软雅黑', size=10)
BOLD_FONT = Font(name='微软雅黑', bold=True, size=10)
THIN = Side(style='thin', color='D0D0D0')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
ALT_FILL = PatternFill('solid', fgColor='F5F7FA')

def write_sheet(ws, headers, rows, widths=None, center_cols=None):
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HDR_FILL; cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = BORDER
    for r, row in enumerate(rows, start=2):
        for c, v in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = BODY_FONT; cell.border = BORDER
            if center_cols and c in center_cols:
                cell.alignment = Alignment(horizontal='center', vertical='center')
            if r % 2 == 0:
                cell.fill = ALT_FILL
    if widths:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = 'A2'

# --- Sheet1 说明 ---
ws = wb.active
ws.title = '说明'
info = [
    ['BTF 生物滴滤塔去除率机器学习建模 — 整理数据库'],
    [''],
    ['数据概况', '374 条原始文献数据 → 清洗/去重后 336 条建模样本'],
    ['建模目标', '去除率 (RE, %)'],
    ['特征体系', '38 维：9 数值 + 27 二元工艺标签 + 2 类别（污染物类别/常见污染物）'],
    ['最优模型', '随机森林 (n_estimators=300, max_features=0.4, min_samples_leaf=1)'],
    ['模型表现', '5折CV R²=0.4397±0.1150；测试集 R²=0.4662, RMSE=16.37%, MAE=11.41%'],
    ['稳健性提示', '按文献分组CV R²≈-0.018：模型适合同文献体系内插值，跨文献预测能力有限'],
    [''],
    ['工作表说明'],
    ['建模数据库', '336 条样本的完整特征（数值+工艺标签+填料类型）与去除率'],
    ['模型预测结果', '训练集+测试集逐条预测值、误差（随机森林）'],
    ['模型对比', '10 个候选模型性能汇总'],
]
ws.append(['BTF 生物滴滤塔去除率机器学习建模 — 整理数据库'])
ws['A1'].font = Font(name='微软雅黑', bold=True, size=14, color='1565C0')
ws.merge_cells('A1:C1')
for row in info[1:]:
    ws.append(row)
    if row and row[0] in ('数据概况', '建模目标', '特征体系', '最优模型', '模型表现', '稳健性提示', '工作表说明'):
        ws.cell(row=ws.max_row, column=1).font = BOLD_FONT
ws.column_dimensions['A'].width = 16
ws.column_dimensions['B'].width = 70
ws.column_dimensions['C'].width = 20

# --- Sheet2 建模数据库 ---
db_cols = ['论文标题', '污染物', '污染物类别', '常见污染物', '去除率_数值',
           'pH', '温度', 'EBRT_s', '进气浓度_mgm3', '入口负荷_gmh', '氮浓度_mgL',
           '运行天数', '运行阶段',
           '疏水性污染物', '饥饿胁迫', '重启恢复', '等离子耦合', '混合共降解',
           '有表面活性剂', '有铁离子', '有紫外预处理', '有营养液', '有额外碳源', '真菌主导',
           '填料_陶粒', '填料_浮石', '填料_火山岩', '填料_树皮', '填料_复合填料',
           '填料_多面球', '填料_聚氨酯', '填料_鲍尔环', '填料_木屑堆肥', '填料_丝瓜', '填料_活性炭']
db = df[db_cols].copy()
db.columns = [c.replace('_数值', '').replace('_mgm3', '(mg/m³)').replace('_gmh', '(g/m³·h)')
              .replace('_mgL', '(mg/L)').replace('_s', '(s)')
              .replace('疏水性污染物', '疏水性(0/1)').replace('去除率', '去除率(%)') for c in db.columns]
num_cols = [5, 6, 7, 8, 9, 10, 11, 12]  # 数值列(1-based): 去除率/pH/温度/EBRT/进气/负荷/氮/天数/阶段
ws = wb.create_sheet('建模数据库')
headers = list(db.columns)
write_sheet(ws, headers, db.values.tolist(),
            widths=[40, 14, 12, 12, 11, 8, 9, 10, 13, 13, 11, 10, 9] + [10]*22,
            center_cols=set(range(5, 36)))
for r in range(2, len(db) + 2):
    for c in num_cols:
        ws.cell(row=r, column=c).number_format = '0.0'

# --- Sheet3 模型预测结果 ---
pred_rows = []
for i in range(len(X_train)):
    pred_rows.append(['训练集', g_train[i], df.loc[X_train.index[i], '污染物'],
                      round(float(y_train[i]), 1), round(float(pred_train[i]), 1),
                      round(float(pred_train[i] - y_train[i]), 1)])
for i in range(len(X_test)):
    pred_rows.append(['测试集', g_test[i], df.loc[X_test.index[i], '污染物'],
                      round(float(y_test[i]), 1), round(float(pred_test[i]), 1),
                      round(float(pred_test[i] - y_test[i]), 1)])
ws = wb.create_sheet('模型预测结果')
write_sheet(ws, ['数据子集', '文献', '污染物', '实际去除率(%)', '预测去除率(%)', '误差(百分点)'],
            pred_rows, widths=[10, 52, 14, 14, 14, 14],
            center_cols={1, 3, 4, 5, 6})

# --- Sheet4 模型对比 ---
mc = pd.read_csv(os.path.join(OUT, 'model_comparison.csv'))
mc.columns = ['模型', 'CV_R2均值', 'CV_R2标准差', '测试R2', '测试RMSE', '测试MAE', '训练R2']
ws = wb.create_sheet('模型对比')
write_sheet(ws, list(mc.columns),
            [[r[0]] + [round(float(x), 4) for x in r[1:]] for r in mc.values.tolist()],
            widths=[16, 12, 12, 10, 10, 10, 10],
            center_cols={2, 3, 4, 5, 6, 7})

xlsx_path = os.path.join(OUT, 'BTF_建模数据库_整理.xlsx')
wb.save(xlsx_path)
print("已保存Excel数据库:", xlsx_path)
print("DONE")
