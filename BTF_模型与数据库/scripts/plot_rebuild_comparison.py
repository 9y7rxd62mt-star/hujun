# -*- coding: utf-8 -*-
"""生成「拆分数据重建模」对比图: 配置A / 配置B / 原模型"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

FIG = r"C:\Users\15787\WorkBuddy\2026-11-03-11-42-33\BTF_ML\output\figures"

labels = ['训练集 R²', '测试集 R²', '5折CV R²', '按文献\nGroupKFold R²']
orig = [0.8485, 0.4662, 0.4397, -0.0178]
cfgA = [0.8449, 0.4879, 0.4240, -0.0386]
cfgB = [0.8471, 0.4638, 0.4414, -0.0197]

x = np.arange(len(labels))
w = 0.26
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w, orig, w, label='原模型(ml_pipeline)', color='#4C72B0', edgecolor='white')
b2 = ax.bar(x, cfgB, w, label='配置B: 拆分数据+营养特征', color='#C44E52', edgecolor='white')
b3 = ax.bar(x + w, cfgA, w, label='配置A: 拆分数据(无营养特征)', color='#55A868', edgecolor='white')
for bars in (b1, b2, b3):
    for r in bars:
        v = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, v + (0.015 if v >= 0 else -0.06),
                f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
ax.axhline(0, color='k', lw=1)
ax.set_xticks(x, labels)
ax.set_ylabel('R² (决定系数)')
ax.set_title('以拆分后数据重新建模 — 与原模型对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(-0.35, 1.02)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
p = os.path.join(FIG, 'fig13_拆分数据重建模对比.png')
fig.savefig(p, bbox_inches='tight', dpi=150)
print("已保存:", p)
