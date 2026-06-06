"""
Model training and evaluation module.

Public API
----------
load_data            – load step-2 features and all label arrays.
filter_valid_samples – drop rows where label == -999 (missing).
filter_outlier       – drop rows where |y| > threshold.
get_models           – return dict of model name → {model, search space}.
train_and_evaluate   – BayesSearchCV + 5-fold CV for all models on one target.
print_summary_table  – print a formatted summary across all targets + models.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor

from config import (
    RESULTS_DIR,
    CV_FOLDS,
    BAYES_N_ITER,
    RANDOM_STATE,
)
from feature_selection import load_step2, load_labels


# ============================================================
# Data loading
# ============================================================

def load_data():
    """
    Load step-2 filtered features and label arrays.

    Returns
    -------
    dict with keys:
        X_train_all, X_dev_all, X_test_all
        feat_names_est, feat_names_krisc  (same list, kept symmetric)
        y_train_est, y_dev_est, y_test_est
        y_train_krisc, y_dev_krisc, y_test_krisc
        X_traindev            – train + dev stacked
        y_traindev_est        – concatenated train+dev labels for EST
        y_traindev_krisc      – concatenated train+dev labels for kRISC
    """
    X_train, X_dev, X_test, feat_names = load_step2()
    (y_train_est, y_train_krisc,
     y_dev_est,   y_dev_krisc,
     y_test_est,  y_test_krisc) = load_labels()

    return {
        'X_train_all':      X_train,
        'X_dev_all':        X_dev,
        'X_test_all':       X_test,
        'feat_names_est':   feat_names,
        'feat_names_krisc': feat_names,
        'y_train_est':      y_train_est,
        'y_dev_est':        y_dev_est,
        'y_test_est':       y_test_est,
        'y_train_krisc':    y_train_krisc,
        'y_dev_krisc':      y_dev_krisc,
        'y_test_krisc':     y_test_krisc,
        'X_traindev':       np.vstack([X_train, X_dev]),
        'y_traindev_est':   np.concatenate([y_train_est,   y_dev_est]),
        'y_traindev_krisc': np.concatenate([y_train_krisc, y_dev_krisc]),
    }


# ============================================================
# Sample filtering
# ============================================================

def filter_valid_samples(X, y):
    """Remove rows where the label is the missing-value sentinel -999."""
    valid = y != -999.0
    return X[valid], y[valid]


def filter_outlier(X, y, threshold=5.0):
    """Remove rows where |y| > threshold."""
    mask = np.abs(y) <= threshold
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  删除 {n_removed} 个异常点 (|y| > {threshold}): {y[~mask]}")
    return X[mask], y[mask]


# ============================================================
# Model / search-space definitions
# ============================================================

def get_models():
    """
    Return a dict of model configurations.

    Each entry: { 'model': estimator, 'space': BayesSearchCV param space }
    """
    return {
        'XGBoost': {
            'model': XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'space': {
                'n_estimators':      Integer(100, 600),
                'max_depth':         Integer(2, 6),
                'learning_rate':     Real(0.01, 0.2, prior='log-uniform'),
                'subsample':         Real(0.5, 0.9),
                'colsample_bytree':  Real(0.1, 0.6),
                'reg_alpha':         Real(0.01, 10,  prior='log-uniform'),
                'reg_lambda':        Real(0.1,  20,  prior='log-uniform'),
                'min_child_weight':  Integer(3, 15),
            },
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'space': {
                'n_estimators':      Integer(200, 800),
                'max_depth':         Integer(3, 15),
                'min_samples_split': Integer(2, 15),
                'min_samples_leaf':  Integer(1, 8),
                'max_features':      Real(0.1, 0.7),
            },
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'space': {
                'n_estimators':      Integer(200, 800),
                'max_depth':         Integer(3, 12),
                'min_samples_split': Integer(5, 20),
                'min_samples_leaf':  Integer(2, 10),
                'max_features':      Real(0.1, 0.6),
            },
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=RANDOM_STATE),
            'space': {
                'n_estimators':      Integer(100, 500),
                'max_depth':         Integer(2, 5),
                'learning_rate':     Real(0.01, 0.2, prior='log-uniform'),
                'subsample':         Real(0.5, 0.9),
                'min_samples_split': Integer(5, 20),
                'min_samples_leaf':  Integer(2, 10),
                'max_features':      Real(0.1, 0.6),
            },
        },
        'SVR': {
            'model': SVR(),
            'space': {
                'kernel':   Categorical(['rbf']),
                'C':        Real(0.1, 50,   prior='log-uniform'),
                'epsilon':  Real(0.01, 0.5, prior='log-uniform'),
                'gamma':    Categorical(['scale', 'auto']),
            },
        },
    }


# ============================================================
# Training and evaluation
# ============================================================

def _calc_metrics(y_true, y_pred):
    return {
        'R2':   r2_score(y_true, y_pred),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MSE':  mean_squared_error(y_true, y_pred),
    }


def train_and_evaluate(
    X_train, y_train,
    X_test,  y_test,
    y_mean,  y_std,
    target_name,
    feature_names,
    n_iter=BAYES_N_ITER,
):
    """
    Standardise features, run BayesSearchCV with CV_FOLDS-fold CV for every
    model returned by get_models(), and return performance metrics in both
    standardised and original scale.

    Parameters
    ----------
    X_train, y_train : training features / labels (already filtered)
    X_test,  y_test  : test  features / labels (already filtered)
    y_mean, y_std    : normalisation constants used to invert predictions
    target_name      : label used in print output
    feature_names    : list[str]  (not used in training, kept for callers)
    n_iter           : number of Bayesian search iterations

    Returns
    -------
    results : dict  { model_name: { model, best_params, scaler, cv_r2,
                                    train: {true, pred, metrics},
                                    test:  {true, pred, metrics} } }
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n目标: {target_name}  |  特征: {X_train.shape[1]}  "
          f"|  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"{CV_FOLDS}折CV + BayesSearchCV ({n_iter} 次迭代)")

    results = {}
    for name, cfg in get_models().items():
        print(f"\n  [{name}] 超参数搜索...")
        search = BayesSearchCV(
            cfg['model'], cfg['space'],
            n_iter=n_iter, cv=cv, scoring='r2',
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search.fit(X_train_s, y_train)
        best = search.best_estimator_

        # Predictions in normalised space → invert to original scale
        train_pred = best.predict(X_train_s) * y_std + y_mean
        test_pred  = best.predict(X_test_s)  * y_std + y_mean
        y_train_r  = y_train * y_std + y_mean
        y_test_r   = y_test  * y_std + y_mean

        results[name] = {
            'model':       best,
            'best_params': dict(search.best_params_),
            'scaler':      scaler,
            'cv_r2':       search.best_score_,
            'train':       {'true': y_train_r, 'pred': train_pred,
                            'metrics': _calc_metrics(y_train_r, train_pred)},
            'test':        {'true': y_test_r,  'pred': test_pred,
                            'metrics': _calc_metrics(y_test_r,  test_pred)},
        }
        tm, vm = results[name]['train']['metrics'], results[name]['test']['metrics']
        print(f"  CV R²={search.best_score_:.4f}")
        print(f"    {'Split':<6} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
        print(f"    {'Train':<6} {tm['R2']:>8.4f} {tm['MAE']:>8.4f} {tm['RMSE']:>8.4f}")
        print(f"    {'Test':<6} {vm['R2']:>8.4f} {vm['MAE']:>8.4f} {vm['RMSE']:>8.4f}")

    return results


# ============================================================
# Summary table
# ============================================================

def print_summary_table(all_results):
    """Pretty-print a cross-target × cross-model results table."""
    print("\n" + "=" * 85)
    print("最终结果汇总")
    print("=" * 85)
    print(f"{'目标':<8} {'模型':<18} {'数据集':<8} "
          f"{'CV R²':>8} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'MSE':>10}")
    print("-" * 85)
    for target_name, results in all_results.items():
        for model_name, res in results.items():
            cv_r2 = res['cv_r2']
            for split_label, split_key in [('Train', 'train'), ('Test', 'test')]:
                m = res[split_key]['metrics']
                cv_disp = f"{cv_r2:>8.4f}" if split_label == 'Train' else "        "
                print(f"{target_name:<8} {model_name:<18} {split_label:<8} "
                      f"{cv_disp} {m['R2']:>8.4f} {m['MAE']:>8.4f} "
                      f"{m['RMSE']:>8.4f} {m['MSE']:>10.6f}")
            print()
