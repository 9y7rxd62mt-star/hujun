BTF 生物滴滤塔去除率预测 —— 模型脚本与数据库
====================================================

目录结构
--------
scripts/      模型 Python 脚本
数据库/       数据表文件
可视化图/     模型与数据分析生成的全部图表

一、模型脚本（scripts/）
------------------------
1. ml_pipeline.py
   主建模流水线：数据清洗 → 特征工程 → 10 模型训练调参 →
   特征重要性/残差/学习曲线/部分依赖分析。
   产出 fig01~fig10 及全部 CSV。

2. export_excel_and_prediction.py
   复现最优模型（随机森林，seed=42），导出整理数据库 + 预测可视化。
   产出 BTF_建模数据库_整理.xlsx、fig11。

3. generate_fig12.py
   综合可视化：预测散点 + 特征重要性 + 各运行阶段预测率 + 误差分布。
   产出 fig12。

4. generate_fig14_15.py
   训练/测试分布与线性拟合图 + 特征重要性图。
   产出 fig14、fig15。

5. generate_prediction_lines.py
   训练/测试「实际 vs 预测」折线图 + 特征重要性图。

6. plot_rebuild_comparison.py
   拆分数据重建模与原模型对比图。产出 fig13。

7. rebuild_model_from_split.py
   以「参数拆分后」数据重新建模并对比原模型。

8. build_slim_table.py
   生成精简数据表（仅保留核心参数列）。

9. split_nutrition_params.py
   将营养条件相关参数拆分为空床停留时间/入口表面负荷/
   进气流量/目标污染物入口浓度等独立列。

二、数据库（数据库/）
---------------------
- BTF_数据库_精简版.xlsx      精简数据表（7 列：文献名/污染物/温度/
                              ph/空床停留时间/入口表面负荷/去除率，374 行）
- BTF_ML_features.csv          特征工程后的建模数据（336 行 × 34 列）
- BTF_data_parsed.csv          初步解析后的原始数据（374 行）
- model_comparison.csv         10 个候选模型性能汇总
- test_predictions.csv         测试集逐条预测结果

三、可视化图（可视化图/）
-------------------------
fig01~fig15 共 16 张图，涵盖数据概览、相关性、污染物分析、
模型对比、预测效果、特征重要性、残差、学习曲线、部分依赖等。

四、运行环境
------------
Python 解释器：
C:/Users/15787/.workbuddy/binaries/python/envs/default/Scripts/python.exe
已安装 pandas / scikit-learn / matplotlib / openpyxl。

运行示例：
"C:/Users/15787/.workbuddy/binaries/python/envs/default/Scripts/python.exe" ml_pipeline.py

五、核心结论
------------
- 最优模型：随机森林
  训练 R² = 0.8485，测试 R² = 0.4662，5 折 CV R² = 0.4397
- 最重要特征：常见污染物(0.165)、进气浓度缺失指示(0.113)、
  污染物类别(0.094)、进气浓度(0.093)
- 注意：按文献分组 GroupKFold R² ≈ -0.02，模型仅适合同文献体系
  内趋势内插，不具备跨文献绝对预测能力（详见分析报告）。
