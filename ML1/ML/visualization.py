"""
Visualization utilities for the ML project.

Functions
---------
plot_retained_features     – Boruta-selected feature importance bar chart.
plot_prediction_scatter    – Predicted vs. actual scatter for all models.
plot_metrics_comparison    – R² / MAE / RMSE bar-chart comparison across models.
plot_prediction_curve      – Sorted prediction vs. experimental curve.
plot_correlation_matrix    – Pearson correlation heat-map (descriptors + target).
plot_shap_analysis         – SHAP beeswarm + bar charts.

All figures are saved under config.FIGURES_DIR.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import FIGURES_DIR


def _savefig(filename: str):
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {path}")


def _descriptor_mask(feature_names):
    """Return (indices, names) for RDKit descriptors (non-fingerprint features)."""
    idx, names = [], []
    for i, n in enumerate(feature_names):
        if not n.startswith('MorganFP_') and not n.startswith('MACCS_'):
            idx.append(i)
            names.append(n)
    return np.array(idx), names


# ============================================================
# Boruta feature importance chart
# ============================================================

def _gradient_barh(ax, y_pos, width, height=0.6, cmap_name='Blues', alpha=0.9):
    gradient = np.linspace(0.2, 0.85, 256).reshape(1, -1)
    cmap = plt.colormaps[cmap_name]
    ax.imshow(cmap(gradient),
              extent=[0, width, y_pos - height / 2, y_pos + height / 2],
              aspect='auto', alpha=alpha, zorder=2)
    ax.add_patch(plt.Rectangle(
        (0, y_pos - height / 2), width, height,
        fill=False, edgecolor='black', linewidth=0.5, zorder=3,
    ))


def plot_retained_features(X_train, y_train, feature_names,
                           confirmed_list, tentative_list,
                           target_name, filename):
    """
    Horizontal bar chart of RF feature-importance for Boruta-retained features.
    Confirmed features use a blue gradient, tentative ones use orange.
    """
    valid = y_train != -999.0
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train[valid], y_train[valid])

    keep = set(confirmed_list + tentative_list)
    imp_df = pd.DataFrame({'Feature': feature_names,
                           'Importance': rf.feature_importances_})
    imp_df = imp_df[imp_df['Feature'].isin(keep)].copy()
    imp_df['Status'] = imp_df['Feature'].apply(
        lambda x: 'Confirmed' if x in confirmed_list else 'Tentative'
    )
    imp_df = imp_df.sort_values('Importance', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(imp_df) * 0.45)))
    for i, row in imp_df.iterrows():
        cmap_name = 'Blues' if row['Status'] == 'Confirmed' else 'Oranges'
        _gradient_barh(ax, i, row['Importance'], cmap_name=cmap_name)

    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df['Feature'], fontsize=16)
    ax.set_xlim(0, imp_df['Importance'].max() * 1.1)
    ax.set_ylim(-0.5, len(imp_df) - 0.5)
    ax.set_xlabel('Feature Importance Score (Random Forest)', fontsize=16)
    ax.set_ylabel('Features', fontsize=16)
    ax.set_title(f'Boruta Selected Features — {target_name}',
                 fontsize=14, fontweight='bold')
    conf_patch = mpatches.Patch(color=plt.colormaps['Blues'](0.6),
                                label=f'{target_name} Confirmed')
    tent_patch = mpatches.Patch(color=plt.colormaps['Oranges'](0.6),
                                label=f'{target_name} Tentative')
    ax.legend(handles=[conf_patch, tent_patch], loc='lower right', fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    _savefig(filename)
    plt.show()


# ============================================================
# Scatter: predicted vs. actual
# ============================================================

def plot_prediction_scatter(results, target_name):
    """One subplot per model: train (blue) and test (orange) scatter."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        ax.scatter(res['train']['true'], res['train']['pred'],
                   c='#1f77b4', marker='o', s=20, alpha=0.6, label='Train', zorder=2)
        ax.scatter(res['test']['true'], res['test']['pred'],
                   c='#ff7f0e', marker='o', s=28, alpha=0.8, label='Test', zorder=4)

        all_vals = np.concatenate([
            res['train']['true'], res['test']['true'],
            res['train']['pred'], res['test']['pred'],
        ])
        vmin, vmax = all_vals.min(), all_vals.max()
        pad = (vmax - vmin) * 0.05
        ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad],
                'k--', lw=1.2, alpha=0.5, zorder=1)

        train_m, test_m = res['train']['metrics'], res['test']['metrics']
        ax.text(0.95, 0.05,
                f"Train: R²={train_m['R2']:.3f}  MAE={train_m['MAE']:.3f}\n"
                f"Test:  R²={test_m['R2']:.3f}  MAE={test_m['MAE']:.3f}",
                transform=ax.transAxes, fontsize=12, va='bottom', ha='right')

        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(model_name, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=12, frameon=False)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(f'{target_name} — Predicted vs. Actual', fontsize=15, y=1.02)
    plt.tight_layout()
    _savefig(f'{target_name}_scatter_comparison.png')
    plt.show()


# ============================================================
# Metrics bar-chart comparison
# ============================================================

def plot_metrics_comparison(results, target_name):
    """Side-by-side R² / MAE / RMSE bar charts for all models."""
    model_names = list(results.keys())
    metrics_cfg = [('R2', 'R²'), ('MAE', 'MAE'), ('RMSE', 'RMSE')]
    split_cfg   = [('train', 'Train', '#1f77b4'), ('test', 'Test', '#2ca02c')]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(model_names))
    width = 0.35

    for ax, (metric, title) in zip(axes, metrics_cfg):
        for i, (split, label, color) in enumerate(split_cfg):
            vals = [results[m][split]['metrics'][metric] for m in model_names]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=label, color=color, alpha=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace('RandomForest', 'Random\nForest')
              .replace('GradientBoosting', 'Gradient\nBoosting')
             for n in model_names],
            fontsize=10,
        )
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                  ncol=2, fontsize=12, frameon=False)
        ax.set_ylim(bottom=0)

    fig.suptitle(f'{target_name} — Model Performance Comparison',
                 fontsize=14, y=0.84)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    _savefig(f'{target_name}_metrics_comparison.png')
    plt.show()


# ============================================================
# Prediction curve (sorted by experimental value)
# ============================================================

def plot_prediction_curve(results, target_name, model_name,
                          unit='', save_prefix='pred_curve'):
    """
    Two figures:
      1. Test-set only: sorted true vs. predicted with error fill.
      2. Train + Test side-by-side.
    """
    res = results[model_name]
    test_true, test_pred = res['test']['true'], res['test']['pred']
    sort_idx = np.argsort(test_true)
    tt_s, tp_s = test_true[sort_idx], test_pred[sort_idx]
    x = np.arange(len(tt_s))

    # ── Figure 1: test only ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, tt_s, '-o', color='#8da0cb', markersize=3.5,
            linewidth=1.5, label='True', zorder=3)
    ax.plot(x, tp_s, '--s', color='#d62728', markersize=3.2,
            linewidth=1.2, alpha=0.85, label='Predicted', zorder=3)
    ax.fill_between(x, tt_s, tp_s, alpha=0.15, color='gray',
                    label='Prediction Error')
    test_m = res['test']['metrics']
    ax.text(0.02, 0.97,
            f"R² = {test_m['R2']:.3f}\n"
            f"MAE = {test_m['MAE']:.3f} {unit}\n"
            f"RMSE = {test_m['RMSE']:.3f} {unit}",
            transform=ax.transAxes, fontsize=11, va='top', ha='left')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel(f'{target_name} ({unit})', fontsize=12)
    ax.set_title(f'{target_name} — Predicted vs. Experimental ({model_name})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=11, frameon=False)
    ax.grid(True, linestyle='-', alpha=0.18)
    plt.tight_layout()
    _savefig(f'{save_prefix}_{target_name}_test.png')
    plt.show()

    # ── Figure 2: train + test ───────────────────────────────
    train_true, train_pred = res['train']['true'], res['train']['pred']
    sort_tr = np.argsort(train_true)
    tr_t_s, tr_p_s = train_true[sort_tr], train_pred[sort_tr]
    x_tr = np.arange(len(tr_t_s))

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 5),
        gridspec_kw={'width_ratios': [len(train_true), len(test_true)]},
    )
    train_m = res['train']['metrics']
    axes[0].plot(x_tr, tr_t_s, 'o-', color='#1f77b4', markersize=3, linewidth=1,
                 label='Experimental')
    axes[0].plot(x_tr, tr_p_s, '^--', color='#2ca02c', markersize=3,
                 linewidth=0.8, alpha=0.8, label=f'{model_name} Predicted')
    axes[0].fill_between(x_tr, tr_t_s, tr_p_s, alpha=0.1, color='green')
    axes[0].set_title(f'Train (n={len(train_true)}, R²={train_m["R2"]:.3f})',
                      fontsize=12)
    axes[0].set_xlabel('Sample Index', fontsize=11)
    axes[0].set_ylabel(f'{target_name} {unit}', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    axes[1].plot(x, tt_s, 'o-', color='#1f77b4', markersize=5,
                 linewidth=1.5, label='Experimental')
    axes[1].plot(x, tp_s, '^--', color='#d62728', markersize=5,
                 linewidth=1.2, alpha=0.85, label=f'{model_name} Predicted')
    axes[1].fill_between(x, tt_s, tp_s, alpha=0.15, color='gray')
    axes[1].set_title(f'Test (n={len(test_true)}, R²={test_m["R2"]:.3f})',
                      fontsize=12)
    axes[1].set_xlabel('Sample Index', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle(f'{target_name} — Prediction Accuracy ({model_name})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    _savefig(f'{save_prefix}_{target_name}_train_test.png')
    plt.show()


# ============================================================
# Pearson correlation heat-map
# ============================================================

def plot_correlation_matrix(X, y, feature_names, target_name, target_label,
                            top_n=15, save_path='corr_matrix.png'):
    """
    Select the top_n descriptors by |Pearson r| with the target, then
    draw a full correlation heat-map including the target column.
    """
    desc_idx, desc_names = _descriptor_mask(feature_names)
    X_desc = X[:, desc_idx]

    corr_with_target = np.array([
        np.corrcoef(X_desc[:, i], y)[0, 1] for i in range(X_desc.shape[1])
    ])
    corr_with_target = np.nan_to_num(corr_with_target)

    top_idx = np.argsort(np.abs(corr_with_target))[::-1][:top_n]
    selected_names = [desc_names[i] for i in top_idx]

    print(f'{target_name}: Top-{top_n} descriptors by |r|')
    for rank, idx in enumerate(top_idx, 1):
        print(f'  {rank:2d}. {desc_names[idx]:<30s}  r={corr_with_target[idx]:+.4f}')

    all_names = selected_names + [target_label]
    data_matrix = np.column_stack([X_desc[:, top_idx], y])
    corr_matrix = np.nan_to_num(np.corrcoef(data_matrix.T))
    df_corr = pd.DataFrame(corr_matrix, index=all_names, columns=all_names)

    n = len(all_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(7, n * 0.55)))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(df_corr, annot=True, fmt='.2f', cmap=cmap,
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={'size': 9},
                cbar_kws={'label': 'Pearson r', 'shrink': 0.75},
                square=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Pearson r', fontsize=15)
    ax.set_title(f'Correlation Matrix — {target_name}\n', fontsize=13, pad=15)
    ax.set_xticklabels(all_names, rotation=30, ha='right',
                       rotation_mode='anchor', fontsize=12)
    ax.set_yticklabels(all_names, rotation=0, va='center', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    _savefig(save_path)
    plt.show()
    return df_corr


# ============================================================
# SHAP analysis
# ============================================================

def plot_shap_analysis(results, X_train, feature_names, target_name,
                       model_name='XGBoost', save_prefix='shap'):
    """
    Fit a dedicated XGBoost on RDKit descriptors only, compute SHAP values,
    and produce a beeswarm plot and a bar chart.

    Returns
    -------
    shap_values : np.ndarray
    desc_names  : list[str]
    """
    desc_idx, desc_names = _descriptor_mask(feature_names)
    X_desc = X_train[:, desc_idx]

    res = results[model_name]
    scaler = StandardScaler()
    X_desc_s = scaler.fit_transform(X_desc)
    y_train = res['train']['true']

    shap_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1, reg_lambda=5,
        random_state=42, n_jobs=-1,
    )
    shap_model.fit(X_desc_s, y_train)
    train_r2 = r2_score(y_train, shap_model.predict(X_desc_s))
    print(f'SHAP解释模型 ({target_name}) Train R²={train_r2:.4f}')

    explainer   = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_desc_s)
    X_desc_df   = pd.DataFrame(X_desc, columns=desc_names)

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_desc_df, feature_names=desc_names,
                      max_display=20, show=False, plot_type='dot',
                      color_bar=True, plot_size=(10, 8))
    fig = plt.gcf()
    cbar_ax = fig.axes[-1]
    plt.title(f'SHAP Analysis — {target_name}\n', fontsize=13, pad=10)
    plt.xlabel('SHAP value\n(impact on model output)', fontsize=14)
    ax = plt.gca()
    x_max = np.abs(shap_values).max() * 1.2
    ax.set_xlim(-x_max, x_max)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    cbar_ax.set_ylabel('Feature value', fontsize=18, fontweight='bold')
    cbar_ax.tick_params(labelsize=16)
    plt.tight_layout()
    _savefig(f'{save_prefix}_{target_name}_beeswarm.png')
    plt.show()

    # Bar
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_desc_df, feature_names=desc_names,
                      plot_type='bar', max_display=15, show=False)
    plt.title(f'Mean |SHAP| — {target_name}', fontsize=14, pad=10)
    plt.tight_layout()
    _savefig(f'{save_prefix}_{target_name}_bar.png')
    plt.show()

    # Top-10 summary
    mean_abs  = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:10]
    print(f'\n=== {target_name} SHAP Top-10 ===')
    for rank, idx in enumerate(top_idx, 1):
        direction = '正↑' if mean_shap[idx] > 0 else '负↓'
        print(f'  {rank:<4d} {desc_names[idx]:<35s} '
              f'|SHAP|={mean_abs[idx]:.4f}  {direction}')

    return shap_values, desc_names
