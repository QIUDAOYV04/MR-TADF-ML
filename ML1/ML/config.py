"""
Centralized path and constant configuration.
All paths are resolved relative to this file so the project
works regardless of where Python is invoked or where the repo is cloned.
"""
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent          # .../ML/
DATASET_DIR  = PROJECT_ROOT.parent / 'dataset'  # .../dataset/
RESULTS_DIR  = PROJECT_ROOT / 'results'       # .../ML/results/  (intermediate .npy)
FIGURES_DIR  = PROJECT_ROOT / 'figures'       # .../ML/figures/  (saved plots)

# Create output directories on import so callers never have to think about it
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── Dataset constants ─────────────────────────────────────────────────────────
DEFAULT_TASK    = 'regression'
DEFAULT_DATASET = 'dataset_krisc_est'

# Normalisation constants (computed from training set; update if data changes)
EST_MEAN   = 0.1734
EST_STD    = 0.1396
KRISC_MEAN = 4.0897
KRISC_STD  = 1.8414

# ── Feature-extraction hyper-params ──────────────────────────────────────────
MORGAN_RADIUS = 2
MORGAN_BITS   = 2048

# ── Feature-selection thresholds ─────────────────────────────────────────────
VARIANCE_THRESHOLD    = 0.01   # step 1: near-zero variance (RDKit only)
ZERO_RATIO_THRESHOLD  = 0.95   # step 1: mostly-zero (RDKit only)
PEARSON_THRESHOLD     = 0.95   # step 2: redundancy removal

# ── Boruta settings ───────────────────────────────────────────────────────────
BORUTA_N_ESTIMATORS = 500
BORUTA_MAX_ITER     = 200
BORUTA_RANDOM_STATE = 42

# ── Training settings ─────────────────────────────────────────────────────────
TRAIN_RATIO    = 0.75   # train / (train+dev) split
RANDOM_STATE   = 42
CV_FOLDS       = 5
BAYES_N_ITER   = 50
EST_OUTLIER_CAP = 0.5   # drop ΔEST samples with real value > this
