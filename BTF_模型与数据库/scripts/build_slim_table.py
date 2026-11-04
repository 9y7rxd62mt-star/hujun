# -*- coding: utf-8 -*-
"""
生成精简数据表: 仅保留 9 个参数列
文献名 / 污染物 / 温度 / ph / 空床停留时间 / 入口表面负荷 / 进气流量 / 目标污染物入口浓度 / 去除率
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE = r"C:\Users\15787\WorkBuddy\2026-11-03-11-42-33\BTF_ML"
OUT = os.path.join(BASE, "output")

df = pd.read_csv(os.path.join(OUT, "BTF_数据表_参数拆分.csv"))
print("原表:", df.shape)

# 列映射
out = pd.DataFrame({
    '文献名': df['论文标题'],
    '污染物': df['污染物名称'],
    '温度': df['温度_数值'],
    'ph': df['ph_数值'],
    '空床停留时间(s)': df['空床停留时间(s)'],
    '入口表面负荷(g/m³·h)': df['入口表面负荷(g/m³·h)'],
    '去除率(%)': df['去除率_数值'],
})
print("精简后列:", list(out.columns))
print("行数:", len(out))
print("各列非空统计:")
for c in out.columns:
    print(f"  {c}: {int(out[c].notna().sum())} 条 ({out[c].notna().mean()*100:.1f}%)")

# 导出 CSV
csv_path = os.path.join(OUT, "BTF_数据库_精简版.csv")
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
HDR_FONT = Font(name='微软雅黑', bold=True, color='FFFFFF', size=11)
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

# 数值列格式
num_cols = {3, 4, 5, 6, 7}
for r in range(2, len(out) + 2):
    for c in num_cols:
        ws.cell(row=r, column=c).number_format = '0.0'
# 列宽
widths = {'文献名': 40, '污染物': 14, '温度': 9, 'ph': 8, '空床停留时间(s)': 15,
          '入口表面负荷(g/m³·h)': 20, '去除率(%)': 10}
for i, h in enumerate(headers, start=1):
    ws.column_dimensions[get_column_letter(i)].width = widths.get(h, 12)
ws.freeze_panes = 'A2'

xlsx_path = os.path.join(OUT, "BTF_数据库_精简版.xlsx")
wb.save(xlsx_path)
print("已保存 Excel:", xlsx_path)
print("DONE")
