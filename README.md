# Molecular ML Pipeline

预测分子 ΔEST 和 kRISC 的机器学习流水线，基于 RDKit 2D 描述符 + Morgan 指纹 + MACCS 指纹。

---

## 项目结构

```
ML/
├── main.py               # 一键运行入口
├── config.py             # 路径 & 超参数统一配置
├── preprocess_ML.py      # 分子特征提取（RDKit）
├── feature_selection.py  # 三步特征筛选 + 文件读写
├── models.py             # 模型训练与评估
├── visualization.py      # 所有绘图函数
├── dim reduction.ipynb   # 特征筛选（交互式，可选）
├── train ML-SHAP.ipynb   # 训练 & SHAP（交互式，可选）
├── results/              # 自动创建，存中间 .npy 文件
└── figures/              # 自动创建，存所有输出图片
```

数据集需放在 **项目目录的上一级**：

```
dataset/
└── regression/
    └── dataset_krisc_est/
        ├── data_train.txt
        └── data_test.txt
ML/          ← 本项目在这里
```

---

## 环境依赖

```bash
pip install numpy pandas scikit-learn xgboost scikit-optimize \
            boruta shap seaborn matplotlib rdkit-pypi
```

> 如果 `rdkit-pypi` 安装失败，改用 conda：
> ```bash
> conda install -c conda-forge rdkit
> ```

---

## 一键运行

```bash
cd /path/to/ML
python main.py
```

程序会依次执行以下五个阶段，全程无需人工干预：

| 阶段 | 内容 |
|------|------|
| Step 0 | 读取原始数据，计算 RDKit 描述符 + Morgan 指纹 + MACCS 指纹 |
| Step 1 | 删除含 NaN/Inf、零方差、近零方差、几乎全零的特征列 |
| Step 2 | Pearson 相关系数去冗余（RDKit 描述符间 \|r\| > 0.95） |
| Step 3 | Boruta 重要性筛选（分别对 ΔEST 和 kRISC 各跑一次） |
| Step 4 | 5 折 CV + BayesSearchCV 训练 5 个模型，打印汇总表 |
| Step 5 | 输出散点图、指标对比图、预测曲线、Pearson 热图、SHAP 图 |

---

## 输出文件

**`results/`**（中间数据，供重复使用）

| 文件 | 内容 |
|------|------|
| `X_train_step1.npy` 等 | Step 1 筛选后的特征矩阵 |
| `X_train_step2.npy` 等 | Step 2 筛选后的特征矩阵 |
| `X_train_boruta_est.npy` 等 | Boruta 筛选后（ΔEST） |
| `X_train_boruta_krisc.npy` 等 | Boruta 筛选后（kRISC） |
| `feature_names_*.txt` | 各阶段保留的特征名称列表 |
| `y_train_est.npy` 等 | 标签数组 |

**`figures/`**（论文/报告用图）

| 文件 | 内容 |
|------|------|
| `retained_features_score_est.png` | ΔEST Boruta 保留特征重要性 |
| `retained_features_score_krisc.png` | kRISC Boruta 保留特征重要性 |
| `ΔEST_scatter_comparison.png` | ΔEST 预测 vs 实验散点图 |
| `kRISC_scatter_comparison.png` | kRISC 预测 vs 实验散点图 |
| `ΔEST_metrics_comparison.png` | ΔEST R²/MAE/RMSE 对比 |
| `kRISC_metrics_comparison.png` | kRISC R²/MAE/RMSE 对比 |
| `pred_curve_ΔEST_test.png` | ΔEST 预测曲线（测试集） |
| `pred_curve_kRISC_test.png` | kRISC 预测曲线（测试集） |
| `pearson_correlation_heatmap_EST.png` | ΔEST Pearson 热图 |
| `pearson_correlation_heatmap_KRISC.png` | kRISC Pearson 热图 |
| `shap_ΔEST_beeswarm.png` | ΔEST SHAP beeswarm |
| `shap_kRISC_beeswarm.png` | kRISC SHAP beeswarm |

---

## 修改超参数

所有可调参数集中在 `config.py`，**不需要改其他文件**：

```python
# 数据集
DEFAULT_TASK    = 'regression'
DEFAULT_DATASET = 'dataset_krisc_est'

# 标准化常数（用真实训练集均值/标准差替换）
EST_MEAN   = 0.1734
EST_STD    = 0.1396
KRISC_MEAN = 4.0897
KRISC_STD  = 1.8414

# 特征筛选
VARIANCE_THRESHOLD   = 0.01
ZERO_RATIO_THRESHOLD = 0.95
PEARSON_THRESHOLD    = 0.95

# Boruta
BORUTA_N_ESTIMATORS = 500
BORUTA_MAX_ITER     = 200

# 模型训练
CV_FOLDS     = 5
BAYES_N_ITER = 50
```
