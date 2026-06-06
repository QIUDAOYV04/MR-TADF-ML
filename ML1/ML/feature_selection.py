"""
Feature selection pipeline.

Steps
------
1. step1_remove_invalid_features  – remove NaN/Inf, zero-variance, near-zero
                                     variance, and mostly-zero columns.
2. step2_pearson_redundancy        – remove pairwise-correlated RDKit
                                     descriptors (|r| > threshold).
3. run_boruta                      – Boruta importance-based selection per
                                     target variable.
4. apply_boruta_filter             – apply confirmed + tentative mask.

I/O helpers
-----------
save_step1 / load_step1
save_step2 / load_step2
save_boruta / load_boruta
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

from config import (
    RESULTS_DIR,
    VARIANCE_THRESHOLD,
    ZERO_RATIO_THRESHOLD,
    PEARSON_THRESHOLD,
    BORUTA_N_ESTIMATORS,
    BORUTA_MAX_ITER,
    BORUTA_RANDOM_STATE,
)


# ============================================================
# Step 1 — Remove invalid features
# ============================================================

def step1_remove_invalid_features(
    X_train, X_dev, X_test, feature_names,
    variance_threshold=VARIANCE_THRESHOLD,
    zero_ratio_threshold=ZERO_RATIO_THRESHOLD,
):
    """
    Remove uninformative features from the full feature matrix.

    Rules applied (in order):
      [1] Columns with NaN or Inf values.
      [2] Zero-variance columns (constant value across all training samples).
      [3] RDKit descriptors only: near-zero variance (< variance_threshold).
      [4] RDKit descriptors only: mostly-zero columns (> zero_ratio_threshold).

    Parameters
    ----------
    X_train, X_dev, X_test : np.ndarray
    feature_names           : list[str]
    variance_threshold      : float
    zero_ratio_threshold    : float

    Returns
    -------
    X_train_f, X_dev_f, X_test_f : np.ndarray
    filtered_names               : list[str]
    report                       : dict
    """
    feature_names = np.array(feature_names)
    n_original = X_train.shape[1]

    is_rdkit = np.array([
        not n.startswith('MorganFP_') and not n.startswith('MACCS_')
        for n in feature_names
    ])
    is_fingerprint = ~is_rdkit

    print(f"  RDKit描述符: {is_rdkit.sum()} 个 | 指纹: {is_fingerprint.sum()} 个")
    report = {
        'initial': n_original,
        'rdkit_original': int(is_rdkit.sum()),
        'fp_original': int(is_fingerprint.sum()),
    }
    mask = np.ones(n_original, dtype=bool)

    # [1] NaN / Inf
    bad_cols = np.any(np.isnan(X_train), axis=0) | np.any(np.isinf(X_train), axis=0)
    n_bad = bad_cols.sum()
    mask &= ~bad_cols
    print(f"\n[1] 含NaN/Inf的列: {n_bad} 个")
    report['nan_inf_removed'] = int(n_bad)

    # [2] Zero variance
    variances = np.var(X_train.astype(np.float64), axis=0)
    zero_var = variances == 0
    new_zero_var = zero_var & mask
    n_zero_var = new_zero_var.sum()
    mask &= ~zero_var
    all_zero = np.all(X_train == 0, axis=0) & new_zero_var
    const_nonzero = new_zero_var & ~all_zero
    print(f"\n[2] 零方差列: {n_zero_var} 个  "
          f"(全零: {all_zero.sum()}, 恒定非零: {const_nonzero.sum()})")
    report.update({
        'zero_var_removed': int(n_zero_var),
        'zero_var_rdkit': int((zero_var & is_rdkit).sum()),
        'zero_var_fp': int((zero_var & is_fingerprint).sum()),
    })

    # [3] RDKit near-zero variance
    low_var = (variances > 0) & (variances < variance_threshold) & is_rdkit & mask
    n_low_var = low_var.sum()
    mask &= ~low_var
    print(f"\n[3] 仅描述符 - 近零方差列（方差 < {variance_threshold}）: {n_low_var} 个")
    report['rdkit_low_var_removed'] = int(n_low_var)

    # [4] RDKit mostly-zero
    zero_ratios = (X_train == 0).sum(axis=0) / X_train.shape[0]
    mostly_zero = (zero_ratios >= zero_ratio_threshold) & is_rdkit & mask
    n_mostly_zero = mostly_zero.sum()
    mask &= ~mostly_zero
    print(f"\n[4] 仅描述符 - 几乎全零列（≥{zero_ratio_threshold*100:.0f}%为0）: "
          f"{n_mostly_zero} 个")
    report['rdkit_mostly_zero_removed'] = int(n_mostly_zero)

    # Apply
    X_train_f = X_train[:, mask]
    X_dev_f   = X_dev[:, mask]
    X_test_f  = X_test[:, mask]
    filtered_names = feature_names[mask].tolist()

    n_remaining = mask.sum()
    report['remaining'] = int(n_remaining)
    report['total_removed'] = int(n_original - n_remaining)

    rdkit_r = sum(1 for n in filtered_names
                  if not n.startswith('MorganFP_') and not n.startswith('MACCS_'))
    morgan_r = sum(1 for n in filtered_names if n.startswith('MorganFP_'))
    maccs_r  = sum(1 for n in filtered_names if n.startswith('MACCS_'))
    print(f"\n总计删除: {report['total_removed']} 个 → 剩余: {n_remaining} 个")
    print(f"  RDKit描述符: {rdkit_r} | Morgan: {morgan_r} | MACCS: {maccs_r}")

    return X_train_f, X_dev_f, X_test_f, filtered_names, report


# ============================================================
# Step 2 — Pearson redundancy removal
# ============================================================

def step2_pearson_redundancy(
    X_train, X_dev, X_test, feature_names,
    corr_threshold=PEARSON_THRESHOLD,
):
    """
    Remove redundant RDKit descriptors whose pairwise Pearson |r| exceeds
    corr_threshold.  Fingerprint bits are left untouched.

    For each highly-correlated pair the feature with higher mean correlation
    to all other features (i.e. the more redundant one) is dropped.

    Returns
    -------
    X_train_f, X_dev_f, X_test_f : np.ndarray
    filtered_names               : list[str]
    report                       : dict
    """
    n_original = X_train.shape[1]
    print(f"输入特征数: {n_original}")

    rdkit_features = [f for f in feature_names
                      if not f.startswith('MorganFP_') and not f.startswith('MACCS_')]
    df = pd.DataFrame(X_train, columns=feature_names)
    corr_matrix = df[rdkit_features].corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_pairs = [
        (row, col, corr_matrix.loc[row, col])
        for col in upper.columns
        for row in upper.index[upper[col] > corr_threshold]
    ]
    print(f"\n发现 {len(high_corr_pairs)} 对高相关特征（|r| > {corr_threshold}）")

    to_drop = set()
    for f1, f2, _ in high_corr_pairs:
        if f1 in to_drop or f2 in to_drop:
            continue
        if corr_matrix[f1].mean() >= corr_matrix[f2].mean():
            to_drop.add(f1)
        else:
            to_drop.add(f2)

    keep_mask = np.array([name not in to_drop for name in feature_names])
    X_train_f = X_train[:, keep_mask]
    X_dev_f   = X_dev[:, keep_mask]
    X_test_f  = X_test[:, keep_mask]
    filtered_names = [n for n in feature_names if n not in to_drop]

    n_remaining = len(filtered_names)
    rdkit_r = sum(1 for n in filtered_names
                  if not n.startswith('MorganFP_') and not n.startswith('MACCS_'))
    morgan_r = sum(1 for n in filtered_names if n.startswith('MorganFP_'))
    maccs_r  = sum(1 for n in filtered_names if n.startswith('MACCS_'))

    report = {
        'input': n_original,
        'high_corr_pairs': len(high_corr_pairs),
        'removed': len(to_drop),
        'removed_rdkit': sum(1 for n in to_drop
                             if not n.startswith('MorganFP_') and
                             not n.startswith('MACCS_')),
        'remaining': n_remaining,
        'remaining_rdkit': rdkit_r,
        'remaining_morgan': morgan_r,
        'remaining_maccs': maccs_r,
    }
    print(f"总计删除: {len(to_drop)} 个 → 剩余: {n_remaining} 个")
    print(f"  RDKit描述符: {rdkit_r} | Morgan: {morgan_r} | MACCS: {maccs_r}")
    return X_train_f, X_dev_f, X_test_f, filtered_names, report


# ============================================================
# Step 3 — Boruta feature selection
# ============================================================

def run_boruta(
    X, y, feature_names, target_name,
    n_estimators=BORUTA_N_ESTIMATORS,
    max_iter=BORUTA_MAX_ITER,
    random_state=BORUTA_RANDOM_STATE,
):
    """
    Run Boruta feature selection for a single target variable.

    Samples with label == -999 (missing) are excluded automatically.

    Returns
    -------
    confirmed  : list[str]  – features confirmed as useful
    tentative  : list[str]  – features not conclusively decided
    rejected   : list[str]  – features confirmed as useless
    ranking    : np.ndarray – per-feature Boruta ranking
    """
    feature_names = np.array(feature_names)
    valid_mask = y != -999.0
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    print(f"\nBoruta — {target_name}  "
          f"(有效样本: {valid_mask.sum()}/{len(y)}, 特征: {X_valid.shape[1]})")

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=7,
        n_jobs=-1,
        random_state=random_state,
    )
    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=max_iter,
        random_state=random_state,
        verbose=2,
    )
    boruta.fit(X_valid, y_valid)

    confirmed = feature_names[boruta.support_].tolist()
    tentative = feature_names[boruta.support_weak_].tolist()
    rejected  = feature_names[~(boruta.support_ | boruta.support_weak_)].tolist()

    print(f"确认: {len(confirmed)} | 待定: {len(tentative)} | 拒绝: {len(rejected)}")
    return confirmed, tentative, rejected, boruta.ranking_


def apply_boruta_filter(X_train, X_dev, X_test, feature_names,
                        confirmed, tentative, target_name):
    """Keep only confirmed + tentative features."""
    feature_names = np.array(feature_names)
    keep = set(confirmed + tentative)
    mask = np.array([n in keep for n in feature_names])
    filtered_names = feature_names[mask].tolist()
    print(f"{target_name} Boruta筛选后: {mask.sum()} 个特征")
    return (X_train[:, mask], X_dev[:, mask], X_test[:, mask], filtered_names)


# ============================================================
# I/O helpers  (all files go to RESULTS_DIR)
# ============================================================

def _npy(name): return RESULTS_DIR / f'{name}.npy'
def _txt(name): return RESULTS_DIR / f'{name}.txt'


def _save_arrays(tag, X_train, X_dev, X_test, feature_names):
    np.save(_npy(f'X_train_{tag}'), X_train)
    np.save(_npy(f'X_dev_{tag}'),   X_dev)
    np.save(_npy(f'X_test_{tag}'),  X_test)
    _txt(f'feature_names_{tag}').write_text('\n'.join(feature_names))
    print(f"已保存 → results/{{X_train,X_dev,X_test}}_{tag}.npy  +  "
          f"feature_names_{tag}.txt")


def _load_arrays(tag):
    X_train = np.load(_npy(f'X_train_{tag}'))
    X_dev   = np.load(_npy(f'X_dev_{tag}'))
    X_test  = np.load(_npy(f'X_test_{tag}'))
    names   = _txt(f'feature_names_{tag}').read_text().splitlines()
    return X_train, X_dev, X_test, names


def save_labels(y_train_est, y_train_krisc,
                y_dev_est,   y_dev_krisc,
                y_test_est,  y_test_krisc):
    for arr, name in [
        (y_train_est,   'y_train_est'),
        (y_train_krisc, 'y_train_krisc'),
        (y_dev_est,     'y_dev_est'),
        (y_dev_krisc,   'y_dev_krisc'),
        (y_test_est,    'y_test_est'),
        (y_test_krisc,  'y_test_krisc'),
    ]:
        np.save(_npy(name), arr)
    print("标签文件已保存至 results/")


def load_labels():
    return (
        np.load(_npy('y_train_est')),
        np.load(_npy('y_train_krisc')),
        np.load(_npy('y_dev_est')),
        np.load(_npy('y_dev_krisc')),
        np.load(_npy('y_test_est')),
        np.load(_npy('y_test_krisc')),
    )


def save_step1(X_train, X_dev, X_test, feature_names):
    _save_arrays('step1', X_train, X_dev, X_test, feature_names)


def load_step1():
    return _load_arrays('step1')


def save_step2(X_train, X_dev, X_test, feature_names):
    _save_arrays('step2', X_train, X_dev, X_test, feature_names)


def load_step2():
    return _load_arrays('step2')


def save_boruta(target_tag, X_train, X_dev, X_test, feature_names):
    """target_tag: 'est' or 'krisc'"""
    _save_arrays(f'boruta_{target_tag}', X_train, X_dev, X_test, feature_names)


def load_boruta(target_tag):
    return _load_arrays(f'boruta_{target_tag}')
