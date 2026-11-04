# -*- coding: utf-8 -*-
"""
把混合的操作参数从文本中拆分为独立列：
  空床停留时间 / 入口表面负荷 / 进气流量 / 目标污染物入口浓度
并删除「营养条件」列。
操作对象: BTF_data_parsed.csv (374 行原始解析数据)
"""
import sys, io, re, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE = r"C:\Users\15787\WorkBuddy\2026-11-03-11-42-33\BTF_ML"
OUT = os.path.join(BASE, "output")

df = pd.read_csv(os.path.join(OUT, "BTF_data_parsed.csv"))
print("原始行数:", len(df))

# 操作参数信息主要藏在 数据来源 + 营养条件 文本里
TXT = (df['数据来源'].fillna('').astype(str) + ' ; ' + df['营养条件'].fillna('').astype(str))
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

# 1. 空床停留时间 (EBRT, 秒)
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

df['空床停留时间(s)'] = TXT.apply(get_ebrt)

# 2. 入口表面负荷 (g/m³·h)
df['入口表面负荷(g/m³·h)'] = TXT.apply(lambda t: find_val(t, r'g/m[³3]\s*·?\s*h'))

# 3. 目标污染物入口浓度 (mg/m³ 与 ppm 分列)
df['目标污染物入口浓度(mg/m³)'] = TXT.apply(lambda t: find_val(t, r'mg/m'))
df['目标污染物入口浓度(ppm)'] = TXT.apply(lambda t: find_val(t, r'[pP][pP][mM]'))

# 4. 进气流量 (气体体积流量, 数据里罕见; 尝试多种表达)
def get_gasflow(t):
    for pat in [r'气(?:体)?流量\s*[=为]?\s*' + NUM + r'\s*(?:m[³3]/h|L/min|L·min)',
                r'gas\s*flow\s*(?:rate)?\s*[=:]?\s*' + NUM + r'\s*(?:m[³3]/h|L/min)',
                r'air\s*flow\s*[=:]?\s*' + NUM + r'\s*(?:m[³3]/h|L/min)',
                r'进气(?:流)?量\s*[=为]?\s*' + NUM + r'\s*(?:m[³3]/h|L/min)']:
        m = re.search(pat, t, re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return np.nan

df['进气流量'] = TXT.apply(get_gasflow)

# 统计提取覆盖率
for c in ['空床停留时间(s)', '入口表面负荷(g/m³·h)', '目标污染物入口浓度(mg/m³)',
          '目标污染物入口浓度(ppm)', '进气流量']:
    print(f"  {c}: 提取 {int(df[c].notna().sum())} 条 ({df[c].notna().mean()*100:.1f}%)")

# 构建最终表: 删除 营养条件 列, 保留其他 + 4参数
drop_cols = ['营养条件']
keep_cols = [c for c in df.columns if c not in drop_cols]
out = df[keep_cols].copy()

# 列顺序: 把4参数列放到温度之后、去除率之前
base_order = ['论文标题', '数据来源', '污染物名称', 'ph', '温度',
              '空床停留时间(s)', '入口表面负荷(g/m³·h)', '进气流量',
              '目标污染物入口浓度(mg/m³)', '目标污染物入口浓度(ppm)',
              '去除率', 'ph_数值', '温度_数值', '去除率_数值']
existing = [c for c in base_order if c in out.columns]
rest = [c for c in out.columns if c not in existing]
out = out[existing + rest]

# 导出 CSV
csv_path = os.path.join(OUT, "BTF_数据表_参数拆分.csv")
out.to_csv(csv_path, index=False, encoding='utf-8-sig')
print("已保存 CSV:", csv_path)

# 导出格式化 Excel
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = Workbook()
ws = wb.active
ws.title = '数据表'

HDR_FILL = PatternFill('solid', fgColor='1565C0')
HDR_FONT = Font(name='微软雅黑', bold=True, color='FFFFFF', size=10)
BODY_FONT = Font(name='微软雅黑', size=10)
THIN = Side(style='thin', color='D0D0D0')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
ALT_FILL = PatternFill('solid', fgColor='F5F7FA')

headers = list(out.columns)
ws.append(headers)
for c in range(1, len(headers) + 1):
    cell = ws.cell(row=1, column=c)
    cell.fill = HDR_FILL; cell.font = HDR_FONT
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    cell.border = BORDER
for r, row in enumerate(out.values.tolist(), start=2):
    for c, v in enumerate(row, start=1):
        cell = ws.cell(row=r, column=c, value=(None if (isinstance(v, float) and np.isnan(v)) else v))
        cell.font = BODY_FONT; cell.border = BORDER
        if r % 2 == 0:
            cell.fill = ALT_FILL

# 数值列格式 + 列宽
num_col_idx = {headers.index(h) + 1 for h in headers if any(k in h for k in ['空床停留', '负荷', '流量', '浓度', '去除率_数值', 'ph_数值', '温度_数值'])}
for r in range(2, len(out) + 2):
    for c in num_col_idx:
        ws.cell(row=r, column=c).number_format = '0.0'
widths = {'论文标题': 40, '数据来源': 45, '污染物名称': 14, 'ph': 8, '温度': 8,
          '空床停留时间(s)': 14, '入口表面负荷(g/m³·h)': 18, '进气流量': 10,
          '目标污染物入口浓度(mg/m³)': 20, '目标污染物入口浓度(ppm)': 20,
          '去除率': 12, 'ph_数值': 9, '温度_数值': 9, '去除率_数值': 10}
for i, h in enumerate(headers, start=1):
    ws.column_dimensions[get_column_letter(i)].width = widths.get(h, 12)
ws.freeze_panes = 'A2'

xlsx_path = os.path.join(OUT, "BTF_数据表_参数拆分.xlsx")
wb.save(xlsx_path)
print("已保存 Excel:", xlsx_path)
print("DONE")
