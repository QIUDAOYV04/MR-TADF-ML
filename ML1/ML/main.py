"""
Main entry point — runs the full pipeline end-to-end:

  1. Feature extraction   (preprocess_ML)
  2. Step-1 filtering     (feature_selection)
  3. Step-2 Pearson       (feature_selection)
  4. Boruta selection     (feature_selection)
  5. Model training       (models)
  6. Visualisation        (visualization)

Usage:
    python main.py
"""

import numpy as np

import preprocess_ML as pp
import feature_selection as fs
import models as m
import visualization as viz
from config import (
    DEFAULT_TASK, DEFAULT_DATASET,
    MORGAN_RADIUS,
    VARIANCE_THRESHOLD, ZERO_RATIO_THRESHOLD, PEARSON_THRESHOLD,
    BORUTA_N_ESTIMATORS, BORUTA_MAX_ITER, BORUTA_RANDOM_STATE,
    EST_MEAN, EST_STD, KRISC_MEAN, KRISC_STD,
    EST_OUTLIER_CAP, BAYES_N_ITER,
)


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# Step 0 — Feature extraction
# ============================================================
banner("Step 0 · Feature Extraction")

(X_train, y_train_est, y_train_krisc,
 X_dev,   y_dev_est,   y_dev_krisc,
 X_test,  y_test_est,  y_test_krisc,
 feature_names) = pp.create_datasets(
    task=DEFAULT_TASK,
    dataset=DEFAULT_DATASET,
    radius=MORGAN_RADIUS,
    device=None,
)

fs.save_labels(
    y_train_est, y_train_krisc,
    y_dev_est,   y_dev_krisc,
    y_test_est,  y_test_krisc,
)


# ============================================================
# Step 1 — Remove invalid features
# ============================================================
banner("Step 1 · Remove Invalid Features")

(X_train_s1, X_dev_s1, X_test_s1,
 names_s1, report_s1) = fs.step1_remove_invalid_features(
    X_train, X_dev, X_test, feature_names,
    variance_threshold=VARIANCE_THRESHOLD,
    zero_ratio_threshold=ZERO_RATIO_THRESHOLD,
)
fs.save_step1(X_train_s1, X_dev_s1, X_test_s1, names_s1)


# ============================================================
# Step 2 — Pearson redundancy removal
# ============================================================
banner("Step 2 · Pearson Redundancy Removal")

(X_train_s2, X_dev_s2, X_test_s2,
 names_s2, report_s2) = fs.step2_pearson_redundancy(
    X_train_s1, X_dev_s1, X_test_s1, names_s1,
    corr_threshold=PEARSON_THRESHOLD,
)
fs.save_step2(X_train_s2, X_dev_s2, X_test_s2, names_s2)

print(f"\n原始特征: {len(feature_names)} → "
      f"Step 1 后: {len(names_s1)} → "
      f"Step 2 后: {len(names_s2)}")


# ============================================================
# Step 3 — Boruta feature selection
# ============================================================
banner("Step 3 · Boruta Feature Selection")

names_s2_arr = np.array(names_s2)

est_confirmed, est_tentative, _, _ = fs.run_boruta(
    X_train_s2, y_train_est, names_s2_arr,
    target_name='ΔEST',
    n_estimators=BORUTA_N_ESTIMATORS,
    max_iter=BORUTA_MAX_ITER,
    random_state=BORUTA_RANDOM_STATE,
)

krisc_confirmed, krisc_tentative, _, _ = fs.run_boruta(
    X_train_s2, y_train_krisc, names_s2_arr,
    target_name='kRISC',
    n_estimators=BORUTA_N_ESTIMATORS,
    max_iter=BORUTA_MAX_ITER,
    random_state=BORUTA_RANDOM_STATE,
)

# Apply filter and save
(X_tr_est, X_dv_est, X_te_est,
 names_est) = fs.apply_boruta_filter(
    X_train_s2, X_dev_s2, X_test_s2, names_s2_arr,
    est_confirmed, est_tentative, target_name='ΔEST',
)
fs.save_boruta('est', X_tr_est, X_dv_est, X_te_est, names_est)

(X_tr_krisc, X_dv_krisc, X_te_krisc,
 names_krisc) = fs.apply_boruta_filter(
    X_train_s2, X_dev_s2, X_test_s2, names_s2_arr,
    krisc_confirmed, krisc_tentative, target_name='kRISC',
)
fs.save_boruta('krisc', X_tr_krisc, X_dv_krisc, X_te_krisc, names_krisc)

# Boruta importance charts
viz.plot_retained_features(
    X_train_s2, y_train_est, names_s2_arr,
    confirmed_list=est_confirmed, tentative_list=est_tentative,
    target_name='ΔEST', filename='retained_features_score_est.png',
)
viz.plot_retained_features(
    X_train_s2, y_train_krisc, names_s2_arr,
    confirmed_list=krisc_confirmed, tentative_list=krisc_tentative,
    target_name='kRISC', filename='retained_features_score_krisc.png',
)


# ============================================================
# Step 4 — Model training & evaluation
# ============================================================
banner("Step 4 · Model Training & Evaluation")

data = m.load_data()

# ── ΔEST ──────────────────────────────────────────────────────
X_tr_est_m, y_tr_est_m = m.filter_valid_samples(
    data['X_traindev'], data['y_traindev_est'])
X_te_est_m, y_te_est_m = m.filter_valid_samples(
    data['X_test_all'], data['y_test_est'])

tr_real = y_tr_est_m * EST_STD + EST_MEAN
X_tr_est_m = X_tr_est_m[tr_real <= EST_OUTLIER_CAP]
y_tr_est_m = y_tr_est_m[tr_real <= EST_OUTLIER_CAP]

te_real = y_te_est_m * EST_STD + EST_MEAN
X_te_est_m = X_te_est_m[te_real <= EST_OUTLIER_CAP]
y_te_est_m = y_te_est_m[te_real <= EST_OUTLIER_CAP]

print(f"ΔEST 有效样本 — Train: {len(y_tr_est_m)}, Test: {len(y_te_est_m)}")

est_results = m.train_and_evaluate(
    X_tr_est_m, y_tr_est_m,
    X_te_est_m, y_te_est_m,
    y_mean=EST_MEAN, y_std=EST_STD,
    target_name='ΔEST',
    feature_names=data['feat_names_est'],
    n_iter=BAYES_N_ITER,
)

# ── kRISC ─────────────────────────────────────────────────────
X_tr_krisc_m, y_tr_krisc_m = m.filter_valid_samples(
    data['X_traindev'], data['y_traindev_krisc'])
X_te_krisc_m, y_te_krisc_m = m.filter_valid_samples(
    data['X_test_all'], data['y_test_krisc'])

print(f"kRISC 有效样本 — Train: {len(y_tr_krisc_m)}, Test: {len(y_te_krisc_m)}")

krisc_results = m.train_and_evaluate(
    X_tr_krisc_m, y_tr_krisc_m,
    X_te_krisc_m, y_te_krisc_m,
    y_mean=KRISC_MEAN, y_std=KRISC_STD,
    target_name='kRISC',
    feature_names=data['feat_names_krisc'],
    n_iter=BAYES_N_ITER,
)

all_results = {'ΔEST': est_results, 'kRISC': krisc_results}
m.print_summary_table(all_results)


# ============================================================
# Step 5 — Visualisation
# ============================================================
banner("Step 5 · Visualisation")

# Scatter
viz.plot_prediction_scatter(est_results,   'ΔEST')
viz.plot_prediction_scatter(krisc_results, 'kRISC')

# Metrics bar-charts
viz.plot_metrics_comparison(est_results,   'ΔEST')
viz.plot_metrics_comparison(krisc_results, 'kRISC')

# Prediction curves
viz.plot_prediction_curve(
    est_results,   'ΔEST',  model_name='ExtraTrees',
    unit='eV', save_prefix='pred_curve',
)
viz.plot_prediction_curve(
    krisc_results, 'kRISC', model_name='XGBoost',
    unit='ns⁻¹', save_prefix='pred_curve',
)

# Pearson heat-maps
viz.plot_correlation_matrix(
    X_tr_est_m, y_tr_est_m,
    data['feat_names_est'],
    target_name='ΔEST', target_label='ΔEST',
    top_n=15, save_path='pearson_correlation_heatmap_EST.png',
)
viz.plot_correlation_matrix(
    X_tr_krisc_m, y_tr_krisc_m,
    data['feat_names_krisc'],
    target_name='kRISC', target_label='kRISC',
    top_n=15, save_path='pearson_correlation_heatmap_KRISC.png',
)

# SHAP
viz.plot_shap_analysis(
    est_results, X_tr_est_m,
    data['feat_names_est'],
    target_name='ΔEST', model_name='XGBoost', save_prefix='shap',
)
viz.plot_shap_analysis(
    krisc_results, X_tr_krisc_m,
    data['feat_names_krisc'],
    target_name='kRISC', model_name='XGBoost', save_prefix='shap',
)

banner("Pipeline complete — all outputs saved to results/ and figures/")
