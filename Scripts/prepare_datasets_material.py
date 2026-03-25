"""
Offline Dataset Preprocessing Script  (CSV multi-material version)
====================================================================
Reads config.yaml for per-model feature definitions and normalization methods.
Loads pre-split train / test data from a folder tree that looks like:

    <data_root>/
        train/
            Material_A/
                B_Field.csv
                H_Field.csv
                Frequency.csv
                Temperature.csv
                Volumetric_Loss.csv
                Effective_Area.csv
                Effective_Length.csv
                Effective_Volume.csv
                Primary_Turns.csv
                Secondary_Turns.csv
            Material_B/  ...
        test/
            Material_A/  ...
            Material_B/  ...

Each CSV file contains one row per sample (waveforms: rows x time-steps,
scalars: rows x 1 or a 1-D column).

Physics features B and H are already provided as CSVs — no voltage/current
integration needed.  Volumetric_Loss is read directly.

Outlier removal is applied to the TRAINING split only, before normalization.
Val split is carved out of training data (configurable ratio, default 10 %).

Supported model modes and their __getitem__ return shapes
---------------------------------------------------------
  scaler / scalerv2  ->  x (n_inputs,),                    y (1,)
  sequence       --|
  cnn            --+--> B (T,1),  scalars (n,),             y (1,)
  cnnv2          --|    same features/norm as cnn
  transformer    --|    architecture difference only
  cnnv3          ->    B (T,1),  H (T,1),  scalars (n,),   y (1,)
  seq2seq        ->    B (T,1),  H (T,1)

Output per material per model:
  <output_dir>/<material>/<model_name>/train.npz
  <output_dir>/<material>/<model_name>/val.npz
  <output_dir>/<material>/<model_name>/test.npz
  <output_dir>/<material>/<model_name>/stats.json
  <output_dir>/<material>/outlier_report.json
  <output_dir>/preprocessing_summary.json

Recommended usage — all materials, cnnv3 only:
  python prepare_datasets.py \\
      --data    path/to/data_root \\
      --config  config.yaml \\
      --output  processed_data/ \\
      --models  cnnv3

Other useful options:
  # Specific materials:
  python prepare_datasets.py ... --materials Material_A Material_C

  # Outlier removal:
  python prepare_datasets.py ... \\
      --outlier-method    iqr \\
      --outlier-threshold 3.0 \\
      --outlier-features  Loss B_pk H_pk

  # Disable outlier removal:
  python prepare_datasets.py ... --no-outlier-removal

  # Validation split ratio carved from training data (default 0.1):
  python prepare_datasets.py ... --val-ratio 0.1

  # Multiple models at once:
  python prepare_datasets.py ... --models cnn cnnv2 cnnv3 transformer
"""

import os
import json
import argparse
import numpy as np
import yaml


# ══════════════════════════════════════════════════════════════════
# CSV file-name  ->  internal feature-pool key
# Add or rename entries here if your CSVs are named differently.
# ══════════════════════════════════════════════════════════════════
CSV_NAME_MAP = {
    'B_Field.csv':          'B',
    'H_Field.csv':          'H',
    'Frequency.csv':        'Frequency',
    'Temperature.csv':      'Temperature',
    'Volumetric_Loss.csv':  'Loss',
    'Effective_Area.csv':   'Effective_Area',
    'Effective_Length.csv': 'Effective_Length',
    'Effective_Volume.csv': 'Effective_Volume',
    'Primary_Turns.csv':    'Primary_Turns',
    'Secondary_Turns.csv':  'Secondary_Turns',
}

# Waveform keys for cnnv3 -- everything else in inputs is a scalar condition
_CNNV3_WAVEFORMS = frozenset({'B', 'H'})


# ══════════════════════════════════════════════════════════════════
# 1.  CONFIG LOADING
# ══════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load and return the YAML config as a plain dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"  Loaded config from: {config_path}")
    return cfg


def get_model_feature_config(cfg: dict, model_name: str) -> dict:
    """
    Extract the features block for one model from the global config.

    Expected config structure (config.yaml):
        models:
          <model_name>:
            features:
              inputs:
                <FeatureName>: <norm_method>   # standard | minmax | log10 | none
              targets:
                <FeatureName>: <norm_method>

    Returns
    -------
    {
      'inputs':  { 'FeatureName': 'norm_method', ... },
      'targets': { 'FeatureName': 'norm_method', ... }
    }
    """
    if 'models' not in cfg:
        raise KeyError("No 'models' section found in config.yaml.")
    if model_name not in cfg['models']:
        available = list(cfg['models'].keys())
        raise KeyError(
            f"Model '{model_name}' not found in config.yaml. "
            f"Available models: {available}"
        )
    model_cfg = cfg['models'][model_name]
    if 'features' not in model_cfg:
        raise KeyError(
            f"No 'features' section under models.{model_name} in config.yaml."
        )
    features = model_cfg['features']
    inputs   = dict(features.get('inputs',  {}) or {})
    targets  = dict(features.get('targets', {}) or {})
    if not inputs and not targets:
        raise ValueError(
            f"models.{model_name}.features defines no inputs or targets."
        )
    return {'inputs': inputs, 'targets': targets}


# ══════════════════════════════════════════════════════════════════
# 2.  CSV DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_csv_material(material_dir: str) -> dict:
    """
    Load all CSV files in a material directory into a dict.

    The CSV name is mapped to an internal key via CSV_NAME_MAP.
    Any CSV not in the map is loaded under its bare stem name.

    Returns
    -------
    dict  key -> np.ndarray float32
        shape (N, T) for waveforms  or  (N, 1) for scalars
    """
    if not os.path.isdir(material_dir):
        raise FileNotFoundError(f"Material directory not found: {material_dir}")

    pool = {}
    for fname in sorted(os.listdir(material_dir)):
        if not fname.endswith('.csv'):
            continue

        key  = CSV_NAME_MAP.get(fname, os.path.splitext(fname)[0])
        path = os.path.join(material_dir, fname)

        arr = np.loadtxt(path, delimiter=',', dtype=np.float32)

        # Scalar constant (single value, e.g. Effective_Area for whole material)
        if arr.ndim == 0:
            pool[key] = arr.reshape(1)
            print(f"    {fname:35s} -> '{key}'  shape=scalar (constant)")
            continue

        # Ensure at least 2-D
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        pool[key] = arr
        print(f"    {fname:35s} -> '{key}'  shape={arr.shape}")

    if not pool:
        raise ValueError(f"No CSV files found in {material_dir}")

    return pool


# ══════════════════════════════════════════════════════════════════
# 3.  FEATURE POOL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════

def build_feature_pool(raw: dict) -> dict:
    """
    Derive all available features from the raw CSV data.

    If B and H waveforms are present, peak scalars B_pk and H_pk are computed
    automatically. All other keys are passed through unchanged.

    Returns
    -------
    dict  name -> np.ndarray (N, ...) float32

    Available feature names (subject to which CSVs are present)
    -----------------------------------------------------------
    B, H              waveforms  (N, T)
    B_pk, H_pk        scalars    (N, 1)   derived
    Frequency,
    Temperature,
    Loss, ...         scalars    (N, 1)
    """
    pool = dict(raw)  # shallow copy

    def col(a: np.ndarray) -> np.ndarray:
        return a.reshape(-1, 1) if a.ndim == 1 else a

    # Ensure scalar-like arrays are (N, 1)
    for key in list(pool.keys()):
        arr = pool[key]
        if arr.ndim == 1:
            pool[key] = arr.reshape(-1, 1)

    # Derive peak scalars from waveforms if present
    if 'B' in pool:
        B = pool['B']
        if B.ndim == 2 and B.shape[1] > 1:
            pool['B_pk'] = col((B.max(axis=1) - B.min(axis=1)) / 2.0)
            print(f"    Derived B_pk  shape={pool['B_pk'].shape}")

    if 'H' in pool:
        H = pool['H']
        if H.ndim == 2 and H.shape[1] > 1:
            pool['H_pk'] = col((H.max(axis=1) - H.min(axis=1)) / 2.0)
            print(f"    Derived H_pk  shape={pool['H_pk'].shape}")

    return pool


# ══════════════════════════════════════════════════════════════════
# 4.  OUTLIER REMOVAL  (training split ONLY)
# ══════════════════════════════════════════════════════════════════

def remove_outliers(
    pool:       dict,
    train_idx:  np.ndarray,
    features:   list,
    method:     str   = 'iqr',
    threshold:  float = 3.0,
    output_dir: str   = None,
) -> np.ndarray:
    """
    Remove outlier samples from training indices only.

    Detection is in physics space (raw, un-normalized).
    A sample is dropped if it is an outlier in ANY of the listed features.

    Parameters
    ----------
    pool        : feature pool from build_feature_pool()
    train_idx   : 1-D int array of original training indices
    features    : list of pool feature names to inspect
    method      : 'iqr' | 'zscore'
    threshold   : IQR multiplier (1.5 aggressive - 3.0 conservative)
                  or z-score cutoff (2.5 - 3.5 typical)
    output_dir  : if given, writes outlier_report.json here

    Returns
    -------
    clean_train_idx : np.ndarray
    """
    print(f"\n  Outlier removal -- method='{method}', threshold={threshold}")
    print(f"  Features checked : {features}")
    print(f"  Training samples before: {len(train_idx):,}")

    if method not in ('iqr', 'zscore'):
        raise ValueError(f"Unknown method '{method}'. Choose 'iqr' or 'zscore'.")

    keep_mask = np.ones(len(train_idx), dtype=bool)
    report    = {'method': method, 'threshold': threshold, 'features': {}}

    for feat_name in features:
        if feat_name not in pool:
            print(f"  WARNING: '{feat_name}' not in pool -- skipping.")
            continue

        raw_arr = pool[feat_name][train_idx]

        # Collapse to a representative scalar per sample
        if raw_arr.ndim == 1:
            values = raw_arr
        elif raw_arr.ndim == 2 and raw_arr.shape[1] == 1:
            values = raw_arr.flatten()
        else:
            values = (raw_arr.max(axis=1) - raw_arr.min(axis=1)) / 2.0
            print(f"    '{feat_name}' is a waveform -- using peak amplitude")

        if method == 'iqr':
            q1, q3    = np.percentile(values, [25, 75])
            iqr       = q3 - q1
            lo, hi    = q1 - threshold * iqr, q3 + threshold * iqr
            feat_mask = (values >= lo) & (values <= hi)
        else:
            mean, std = values.mean(), values.std()
            if std == 0:
                print(f"    '{feat_name}' has zero std -- skipping.")
                continue
            lo, hi    = mean - threshold * std, mean + threshold * std
            feat_mask = np.abs((values - mean) / std) <= threshold

        n_removed = int((~feat_mask).sum())
        pct       = 100.0 * n_removed / len(train_idx)
        print(
            f"    {feat_name:12s}  "
            f"range=[{values.min():.4g}, {values.max():.4g}]  "
            f"bounds=[{lo:.4g}, {hi:.4g}]  "
            f"removed={n_removed:,} ({pct:.2f}%)"
        )
        report['features'][feat_name] = {
            'n_removed':   n_removed,
            'pct_removed': round(pct, 4),
            'bounds':      [float(lo), float(hi)],
            'data_range':  [float(values.min()), float(values.max())],
        }
        keep_mask &= feat_mask

    clean_train_idx = train_idx[keep_mask]
    total_removed   = int(len(train_idx) - len(clean_train_idx))
    total_pct       = 100.0 * total_removed / len(train_idx)

    report.update({
        'total_removed':     total_removed,
        'pct_total_removed': round(total_pct, 4),
        'train_size_before': int(len(train_idx)),
        'train_size_after':  int(len(clean_train_idx)),
    })

    print(f"\n  Total removed          : {total_removed:,} ({total_pct:.2f}%)")
    print(f"  Clean training samples : {len(clean_train_idx):,}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        rpath = os.path.join(output_dir, 'outlier_report.json')
        with open(rpath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  Outlier report -> {rpath}")

    return clean_train_idx


# ══════════════════════════════════════════════════════════════════
# 5.  NORMALIZATION
# ══════════════════════════════════════════════════════════════════

VALID_METHODS = {'standard', 'minmax', 'log10', 'none'}


def normalize(data: np.ndarray, method: str, stats: dict = None):
    """
    Normalize an array.

    Parameters
    ----------
    data   : np.ndarray
    method : 'standard' | 'minmax' | 'log10' | 'none'
    stats  : pre-computed stats dict; if None stats are computed from data.
             Always pass pre-computed stats for val/test to prevent leakage.

    Returns
    -------
    norm_data : np.ndarray float32
    stats     : dict  (empty for 'none' and 'log10')
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Valid: {VALID_METHODS}"
        )

    data = data.astype(np.float32)

    if method == 'none':
        return data, {}

    if method == 'log10':
        return np.log10(np.abs(data) + 1e-6).astype(np.float32), {}

    if method == 'standard':
        if stats is None:
            m = float(np.mean(data))
            s = float(np.std(data))
            if s == 0.0:
                s = 1.0
            stats = {'mean': m, 'std': s}
        normed = (data - stats['mean']) / stats['std']
        return normed.astype(np.float32), stats

    if method == 'minmax':
        if stats is None:
            lo = float(np.min(data))
            hi = float(np.max(data))
            if hi == lo:
                hi = lo + 1.0
            stats = {'min': lo, 'max': hi}
        denom = stats['max'] - stats['min']
        if denom == 0:
            denom = 1.0
        normed = (data - stats['min']) / denom
        return normed.astype(np.float32), stats


# ══════════════════════════════════════════════════════════════════
# 6.  CORE PREPROCESSING PIPELINE  (per model)
# ══════════════════════════════════════════════════════════════════

def preprocess_model(
    train_pool:  dict,
    test_pool:   dict,
    train_idx:   np.ndarray,
    val_idx:     np.ndarray,
    test_idx:    np.ndarray,
    feat_config: dict,
) -> tuple:
    """
    Split and normalize data for one model.

    Normalization stats are ALWAYS fitted on the clean training split only,
    then applied identically to val and test (no data leakage).

    Parameters
    ----------
    train_pool  : feature pool from training folder (build_feature_pool)
    test_pool   : feature pool from test folder (build_feature_pool)
    train_idx   : CLEAN training indices (outliers already removed)
    val_idx     : validation indices (carved from original train data)
    test_idx    : test indices (indices into test_pool -- always full range)
    feat_config : {'inputs': {name: method}, 'targets': {name: method}}

    Returns
    -------
    splits : {
        'train': {'inputs': {name: array}, 'targets': {name: array}},
        'val':   { ... },
        'test':  { ... }
    }
    stats : { feature_name: {'method': str, ...norm_params} }
    """
    splits = {s: {'inputs': {}, 'targets': {}} for s in ('train', 'val', 'test')}
    stats  = {}

    for role in ('inputs', 'targets'):
        role_cfg = feat_config[role]

        for feat_name, method in role_cfg.items():

            # Validate feature exists in both pools
            if feat_name not in train_pool:
                raise KeyError(
                    f"Feature '{feat_name}' not found in training pool.\n"
                    f"  Available: {list(train_pool.keys())}\n"
                    f"  Check config.yaml for typos or missing CSV files."
                )
            if feat_name not in test_pool:
                raise KeyError(
                    f"Feature '{feat_name}' not found in test pool.\n"
                    f"  Available: {list(test_pool.keys())}\n"
                    f"  Ensure the test folder has the same CSV files."
                )

            full_train = train_pool[feat_name]
            full_test  = test_pool[feat_name]

            # Fit normalization stats on CLEAN TRAINING data only
            train_data   = full_train[train_idx]
            _, fit_stats = normalize(train_data, method)
            stats[feat_name] = {'method': method, **fit_stats}

            # Apply to all three splits
            for split_name, idx, pool_arr in [
                ('train', train_idx, full_train),
                ('val',   val_idx,   full_train),
                ('test',  test_idx,  full_test),
            ]:
                norm_arr, _ = normalize(
                    pool_arr[idx], method,
                    stats=fit_stats if fit_stats else None
                )
                splits[split_name][role][feat_name] = norm_arr

            params_str = '  '.join(
                f"{k}={v:.5g}" for k, v in fit_stats.items()
            ) if fit_stats else '(stateless)'
            print(
                f"    [{role:7s}] {feat_name:14s}  "
                f"method={method:8s}  {params_str}"
            )

    return splits, stats


# ══════════════════════════════════════════════════════════════════
# 7.  SAVE / LOAD UTILITIES
# ══════════════════════════════════════════════════════════════════

def save_split(split_data: dict, path: str):
    """Save one split's inputs + targets into a single compressed .npz."""
    flat = {}
    for role in ('inputs', 'targets'):
        for feat_name, arr in split_data[role].items():
            flat[f"{role}__{feat_name}"] = arr
    np.savez_compressed(path, **flat)


def load_split(path: str) -> dict:
    """Reload a split saved by save_split()."""
    data   = np.load(path)
    result = {'inputs': {}, 'targets': {}}
    for key in data.files:
        role, feat_name = key.split('__', 1)
        result[role][feat_name] = data[key]
    return result


def stats_to_serializable(stats: dict) -> dict:
    """Convert numpy scalar types to plain Python types for JSON."""
    out = {}
    for feat, s in stats.items():
        out[feat] = {
            k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v
            for k, v in s.items()
        }
    return out


# ══════════════════════════════════════════════════════════════════
# 8.  MATERIAL DISCOVERY
# ══════════════════════════════════════════════════════════════════

def _find_split_root(data_root: str, candidates: list) -> str:
    """Return the first existing sub-directory from candidates."""
    for name in candidates:
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            return path
    tried = [os.path.join(data_root, c) for c in candidates]
    raise FileNotFoundError(
        f"Could not find a train or test split folder under: {data_root}\n"
        f"  Tried: {tried}\n"
        f"  Sub-directories present: "
        f"{[d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]}"
    )


def discover_materials(data_root: str) -> list:
    """
    Return matched material entries from train/ and test/.

    Automatically detects common split-folder naming conventions:
        train/, final_training/, training/, Train/
        test/,  final_testing/,  testing/,  Test/

    Folder names are normalised before matching -- spaces and underscores
    are treated as equivalent so 'Material_A' matches 'Material A'.

    Returns a list of dicts:
        [{'canonical':  'Material_A',
          'train_dir':  'Material_A',
          'test_dir':   'Material A',
          'train_root': '/path/to/train',
          'test_root':  '/path/to/test'}, ...]
    """
    TRAIN_NAMES = ['train', 'final_training', 'training', 'Train', 'TRAIN']
    TEST_NAMES  = ['test',  'final_testing',  'testing',  'Test',  'TEST']

    train_root = _find_split_root(data_root, TRAIN_NAMES)
    test_root  = _find_split_root(data_root, TEST_NAMES)
    print(f"  Train root : {train_root}")
    print(f"  Test root  : {test_root}")

    def _norm(name: str) -> str:
        return name.lower().replace(' ', '_')

    train_dirs = {
        d: _norm(d) for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    }
    test_dirs = {
        d: _norm(d) for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    }

    test_norm_to_actual = {v: k for k, v in test_dirs.items()}

    matched         = []
    unmatched_train = []
    unmatched_test  = set(test_dirs.keys())

    for train_actual, train_norm in sorted(train_dirs.items()):
        if train_norm in test_norm_to_actual:
            test_actual = test_norm_to_actual[train_norm]
            matched.append({
                'canonical':  train_actual,
                'train_dir':  train_actual,
                'test_dir':   test_actual,
                'train_root': train_root,
                'test_root':  test_root,
            })
            unmatched_test.discard(test_actual)
        else:
            unmatched_train.append(train_actual)

    if unmatched_train:
        print(f"  WARNING: no test match for train folders: {sorted(unmatched_train)}")
    if unmatched_test:
        print(f"  WARNING: no train match for test folders: {sorted(unmatched_test)}")

    if not matched:
        raise ValueError(
            "No materials could be matched between train/ and test/.\n"
            f"  train/ folders : {sorted(train_dirs)}\n"
            f"  test/  folders : {sorted(test_dirs)}\n"
            "Check that both folders contain matching material sub-directories."
        )

    for m in matched:
        if m['train_dir'] != m['test_dir']:
            print(f"  INFO: name mismatch resolved -- "
                  f"train/'{m['train_dir']}' <-> test/'{m['test_dir']}'")

    return matched


# ══════════════════════════════════════════════════════════════════
# 9.  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MagNet offline preprocessing -- CSV multi-material version.\n\n"
            "Data root must contain  train/<Material>/  and  test/<Material>/\n"
            "directories, each holding the CSV files listed in CSV_NAME_MAP.\n\n"
            "Feature selection and normalization methods are read entirely from\n"
            "config.yaml -- no hard-coded values in this script.\n\n"
            "Outlier removal is applied to the TRAINING split only,\n"
            "in physics space (before normalization)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data',   required=True,
                        help='Root data directory containing train/ and test/ sub-dirs')
    parser.add_argument('--config', required=True,
                        help='Path to config.yaml')
    parser.add_argument('--output', default='processed_data',
                        help='Root output directory (default: processed_data/)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to preprocess (default: all defined in config)')
    parser.add_argument('--materials', nargs='+', default=None,
                        help='Materials to process (default: all discovered)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Fraction of training data to use for validation (default: 0.1)')

    parser.add_argument(
        '--outlier-method', type=str, default='iqr', choices=['iqr', 'zscore'],
        help="Outlier detection method (default: iqr)."
    )
    parser.add_argument(
        '--outlier-threshold', type=float, default=3.0,
        help="Threshold multiplier for outlier removal (default: 3.0)."
    )
    parser.add_argument(
        '--outlier-features', nargs='+',
        default=['Loss', 'B_pk', 'H_pk', 'Frequency', 'Temperature'],
        help=(
            "Feature(s) to use for outlier detection "
            "(default: Loss B_pk H_pk Frequency Temperature). "
            "A sample is removed if it is an outlier in ANY listed feature."
        )
    )
    parser.add_argument(
        '--no-outlier-removal', action='store_true', default=False,
        help="Disable outlier removal entirely."
    )

    args = parser.parse_args()

    # ── STEP 1: Load config ─────────────────────────────────────────────────
    print("\n" + "="*64)
    print("  STEP 1 -- Load config.yaml")
    print("="*64)
    cfg = load_config(args.config)

    available_models = list(cfg.get('models', {}).keys())
    if not available_models:
        raise ValueError("No 'models' entries found in config.yaml.")

    models_to_run = args.models if args.models else available_models
    unknown_models = [m for m in models_to_run if m not in available_models]
    if unknown_models:
        raise ValueError(
            f"Requested model(s) {unknown_models} not in config.yaml.\n"
            f"Available: {available_models}"
        )
    print(f"  Models to process: {models_to_run}")

    # ── STEP 2: Discover materials ──────────────────────────────────────────
    print("\n" + "="*64)
    print("  STEP 2 -- Discover materials")
    print("="*64)
    all_materials = discover_materials(args.data)
    canonical_names = [m['canonical'] for m in all_materials]
    print(f"  Materials matched: {canonical_names}")

    if args.materials:
        unknown_mats = [n for n in args.materials if n not in canonical_names]
        if unknown_mats:
            raise ValueError(
                f"Requested material(s) {unknown_mats} not found.\n"
                f"Available: {canonical_names}"
            )
        materials_to_run = [m for m in all_materials if m['canonical'] in args.materials]
    else:
        materials_to_run = all_materials

    print(f"  Materials to process: {[m['canonical'] for m in materials_to_run]}")

    global_summary = {}

    # ══════════════════════════════════════════════════════════════════
    # Per-material loop
    # ══════════════════════════════════════════════════════════════════
    for mat in materials_to_run:
        material       = mat['canonical']
        train_dir_name = mat['train_dir']
        test_dir_name  = mat['test_dir']

        print("\n\n" + "#"*64)
        print(f"  MATERIAL : {material}")
        print("#"*64)
        global_summary[material] = {}

        mat_out_dir = os.path.join(args.output, material)
        os.makedirs(mat_out_dir, exist_ok=True)

        # ── STEP 3: Load CSV data ───────────────────────────────────────────
        train_csv_dir = os.path.join(mat['train_root'], train_dir_name)
        test_csv_dir  = os.path.join(mat['test_root'],  test_dir_name)

        print(f"\n  -- Loading training CSVs from: {train_csv_dir}")
        train_raw = load_csv_material(train_csv_dir)

        print(f"\n  -- Loading test CSVs from: {test_csv_dir}")
        test_raw  = load_csv_material(test_csv_dir)

        # Size checks (skip scalar constants with shape (1,))
        def _is_per_sample(arr):
            return arr.ndim >= 1 and not (arr.ndim == 1 and arr.shape[0] == 1)

        train_sizes = {k: v.shape[0] for k, v in train_raw.items() if _is_per_sample(v)}
        test_sizes  = {k: v.shape[0] for k, v in test_raw.items()  if _is_per_sample(v)}

        if not train_sizes:
            raise ValueError(f"[{material}] No per-sample arrays found in training CSVs.")
        if not test_sizes:
            raise ValueError(f"[{material}] No per-sample arrays found in test CSVs.")

        N_train_raw = next(iter(train_sizes.values()))
        N_test_raw  = next(iter(test_sizes.values()))

        if len(set(train_sizes.values())) != 1:
            raise ValueError(
                f"[{material}] Inconsistent row counts in training CSVs: {train_sizes}"
            )
        if len(set(test_sizes.values())) != 1:
            raise ValueError(
                f"[{material}] Inconsistent row counts in test CSVs: {test_sizes}"
            )

        print(f"\n  Raw counts -- train: {N_train_raw:,}  test: {N_test_raw:,}")

        # ── STEP 4: Build feature pools ─────────────────────────────────────
        print(f"\n  -- Building feature pools ...")
        print(f"  Training pool:")
        train_pool = build_feature_pool(train_raw)
        print(f"  Test pool:")
        test_pool  = build_feature_pool(test_raw)

        # ── STEP 5: Create train / val split indices ─────────────────────────
        # Test folder IS the test split.
        # Val is carved from the END of the training data.
        val_ratio  = max(0.0, min(0.5, args.val_ratio))
        val_size   = int(N_train_raw * val_ratio)
        train_size = N_train_raw - val_size

        all_train_idx = np.arange(N_train_raw, dtype=np.int64)
        raw_train_idx = all_train_idx[:train_size]
        val_idx       = all_train_idx[train_size:]
        test_idx      = np.arange(N_test_raw, dtype=np.int64)

        print(f"\n  Split sizes (before outlier removal):")
        print(f"    train : {len(raw_train_idx):>8,}")
        print(f"    val   : {len(val_idx):>8,}")
        print(f"    test  : {len(test_idx):>8,}  (from test folder)")

        # ── STEP 6: Outlier removal ─────────────────────────────────────────
        print(f"\n  -- Outlier removal ...")
        if args.no_outlier_removal:
            print("  Skipped (--no-outlier-removal).")
            clean_train_idx = raw_train_idx
        else:
            clean_train_idx = remove_outliers(
                pool        = train_pool,
                train_idx   = raw_train_idx,
                features    = args.outlier_features,
                method      = args.outlier_method,
                threshold   = args.outlier_threshold,
                output_dir  = mat_out_dir,
            )

        n_removed = int(len(raw_train_idx) - len(clean_train_idx))
        print(f"\n  Final split sizes:")
        print(f"    train (clean) : {len(clean_train_idx):>8,}  ({n_removed:,} removed)")
        print(f"    val           : {len(val_idx):>8,}")
        print(f"    test          : {len(test_idx):>8,}")

        # ── STEP 7: Per-model preprocessing ─────────────────────────────────
        print(f"\n  -- Per-model preprocessing ...")
        for model_name in models_to_run:
            print(f"\n  +{'─'*60}+")
            print(f"  | Model : {model_name.upper():<50}|")
            print(f"  +{'─'*60}+")

            feat_config = get_model_feature_config(cfg, model_name)
            print(f"  inputs  : {feat_config['inputs']}")
            print(f"  targets : {feat_config['targets']}")
            print()

            splits, model_stats = preprocess_model(
                train_pool  = train_pool,
                test_pool   = test_pool,
                train_idx   = clean_train_idx,
                val_idx     = val_idx,
                test_idx    = test_idx,
                feat_config = feat_config,
            )

            model_out_dir = os.path.join(mat_out_dir, model_name)
            os.makedirs(model_out_dir, exist_ok=True)

            print()
            for split_name in ('train', 'val', 'test'):
                path    = os.path.join(model_out_dir, f"{split_name}.npz")
                save_split(splits[split_name], path)
                size_mb = os.path.getsize(path) / 1e6
                print(f"  Saved {path}  ({size_mb:.1f} MB)")

            stats_path = os.path.join(model_out_dir, 'stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats_to_serializable(model_stats), f, indent=2)
            print(f"  Saved {stats_path}")

            global_summary[material][model_name] = {
                'inputs':      feat_config['inputs'],
                'targets':     feat_config['targets'],
                'split_sizes': {
                    'train': int(len(clean_train_idx)),
                    'val':   int(len(val_idx)),
                    'test':  int(len(test_idx)),
                },
                'outlier_removal': {
                    'enabled':       not args.no_outlier_removal,
                    'method':        args.outlier_method,
                    'threshold':     args.outlier_threshold,
                    'features':      args.outlier_features,
                    'removed_count': n_removed,
                },
                'stats':    stats_to_serializable(model_stats),
                'saved_to': model_out_dir,
            }

    # ── STEP 8: Write global summary ────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(args.output, 'preprocessing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n  Global summary -> {summary_path}")
    print("\n" + "="*64)
    print("  OK  Preprocessing complete!")
    print("="*64 + "\n")


# ══════════════════════════════════════════════════════════════════
# 10.  PYTORCH DATASET WRAPPER  (drop-in for training scripts)
# ══════════════════════════════════════════════════════════════════

class PreprocessedMagNetDataset:
    """
    Lightweight PyTorch Dataset backed by pre-saved .npz files.
    No physics recomputation at training time.

    Parameters
    ----------
    model_dir : str
        Directory produced by this script, e.g.
        'processed_data/Material_A/cnnv3'
    split : str
        'train' | 'val' | 'test'
    mode : str
        'scaler' | 'scalerv2' | 'sequence' | 'cnn' | 'cnnv2' |
        'cnnv3'  | 'transformer' | 'seq2seq'

    Item format per mode
    --------------------
    scaler / scalerv2  ->  x  (n_inputs,),                    y  (1,)
    sequence       --|
    cnn            --+--> B  (T,1),  scalars (n,),              y  (1,)
    cnnv2          --|    same features/norm as cnn
    transformer    --|    architecture difference only
    cnnv3          ->    B  (T,1),  H  (T,1),  scalars (n,),   y  (1,)
    seq2seq        ->    B  (T,1),  H  (T,1)

    Example -- cnnv3 single material
    ---------------------------------
        from prepare_datasets import PreprocessedMagNetDataset
        from torch.utils.data import DataLoader

        ds     = PreprocessedMagNetDataset(
                     'processed_data/Material_A/cnnv3',
                     split='train', mode='cnnv3')
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        B, H, scalars, y = next(iter(loader))
        # B       : (batch, T, 1)
        # H       : (batch, T, 1)
        # scalars : (batch, 2)   -- Frequency, Temperature
        # y       : (batch, 1)
    """

    def __init__(self, model_dir: str, split: str = 'train', mode: str = 'scaler'):
        import torch
        self._torch = torch
        self.mode   = mode

        npz_path = os.path.join(model_dir, f"{split}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Preprocessed file not found: {npz_path}\n"
                f"Run prepare_datasets.py first."
            )
        data = load_split(npz_path)
        self.inputs  = data['inputs']
        self.targets = data['targets']

        stats_path = os.path.join(model_dir, 'stats.json')
        with open(stats_path) as f:
            self.stats = json.load(f)

        first = (next(iter(self.inputs.values())) if self.inputs
                 else next(iter(self.targets.values())))
        self.length = len(first)

    def __len__(self) -> int:
        return self.length

    def _t(self, arr: np.ndarray) -> 'torch.Tensor':
        return self._torch.tensor(arr, dtype=self._torch.float32)

    def __getitem__(self, idx: int):
        inp = self.inputs
        tgt = self.targets

        if self.mode in ('scaler', 'scalerv2'):
            x = self._torch.cat([self._t(inp[f][idx]).flatten() for f in inp])
            y = self._t(tgt['Loss'][idx]).flatten()
            return x, y

        elif self.mode in ('sequence', 'cnn', 'cnnv2', 'transformer'):
            B       = self._t(inp['B'][idx]).unsqueeze(-1)          # (T, 1)
            scalars = self._torch.cat([
                self._t(inp[f][idx]).flatten() for f in inp if f != 'B'
            ])
            y = self._t(tgt['Loss'][idx]).flatten()
            return B, scalars, y

        elif self.mode == 'cnnv3':
            B       = self._t(inp['B'][idx]).unsqueeze(-1)          # (T, 1)
            H       = self._t(inp['H'][idx]).unsqueeze(-1)          # (T, 1)
            scalars = self._torch.cat([
                self._t(inp[f][idx]).flatten()
                for f in inp if f not in _CNNV3_WAVEFORMS
            ])
            y = self._t(tgt['Loss'][idx]).flatten()
            return B, H, scalars, y

        elif self.mode == 'seq2seq':
            B = self._t(inp['B'][idx]).unsqueeze(-1)                # (T, 1)
            H = self._t(tgt['H'][idx]).unsqueeze(-1)                # (T, 1)
            return B, H

        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. "
                f"Valid: scaler, scalerv2, sequence, cnn, cnnv2, cnnv3, "
                f"transformer, seq2seq"
            )


if __name__ == '__main__':
    main()
