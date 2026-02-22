"""
Offline Dataset Preprocessing Script
=====================================
Reads config.yaml for per-model feature definitions and normalization methods,
uses dataset_split.json for train/val/test indices, computes physics features
(B, H, Loss) once, then normalizes and saves each model's data to disk.

Output per model:
  <output_dir>/<model_name>/train.npz
  <output_dir>/<model_name>/val.npz
  <output_dir>/<model_name>/test.npz
  <output_dir>/<model_name>/stats.json
  <output_dir>/preprocessing_summary.json

Usage:
  python prepare_datasets.py \\
      --data   path/to/data.mat \\
      --split  dataset_split.json \\
      --config config.yaml \\
      --output processed_data/

  # Preprocess only specific models:
  python prepare_datasets.py ... --models cnn transformer
"""

import os
import json
import argparse
import numpy as np
import yaml
from scipy.integrate import cumulative_trapezoid, trapezoid


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
            f"No 'features' section under models.{model_name} in config.yaml.\n"
            f"Expected:\n"
            f"  models:\n"
            f"    {model_name}:\n"
            f"      features:\n"
            f"        inputs:\n"
            f"          B: standard\n"
            f"        targets:\n"
            f"          Loss: log10"
        )

    features = model_cfg['features']
    inputs   = dict(features.get('inputs',  {}) or {})
    targets  = dict(features.get('targets', {}) or {})

    if not inputs and not targets:
        raise ValueError(
            f"models.{model_name}.features defines no inputs or targets in config.yaml."
        )

    return {'inputs': inputs, 'targets': targets}


# ══════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_full_dataset(file_path: str) -> dict:
    """
    Load entire .mat file into memory.
    Supports scipy (MATLAB v7 and earlier) and h5py (MATLAB v7.3 / HDF5).

    Returns
    -------
    {
      'voltage': (N, T) float32,
      'current': (N, T) float32,
      'freq':    (N,)   float32,
      'temp':    (N,)   float32,
      'hdc':     (N,)   float32,
      'duty':    (N,)   float32,
      'meta':    { N_prim, N_sec, Ae, Le, dt }
    }
    """
    import scipy.io
    import h5py

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    def _scipy_load():
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        D = mat['Data']
        return {
            'voltage': D.Voltage.astype(np.float32),
            'current': D.Current.astype(np.float32),
            'freq':    D.Frequency_command.astype(np.float32),
            'temp':    D.Temperature_command.astype(np.float32),
            'hdc':     D.Hdc_command.astype(np.float32),
            'duty':    D.DutyP_command.astype(np.float32),
            'meta': {
                'N_prim': float(D.Primary_Turns),
                'N_sec':  float(D.Secondary_Turns),
                'Ae':     float(D.Effective_Area),
                'Le':     float(D.Effective_Length),
                'dt':     float(np.array(D.Sampling_Time).flat[0])
            }
        }

    def _h5py_load():
        with h5py.File(file_path, 'r') as f:
            D = f['Data']
            def arr(k):
                return np.array(D[k]).flatten().astype(np.float32)
            # h5py stores arrays transposed relative to MATLAB
            return {
                'voltage': np.array(D['Voltage']).T.astype(np.float32),
                'current': np.array(D['Current']).T.astype(np.float32),
                'freq':    arr('Frequency_command'),
                'temp':    arr('Temperature_command'),
                'hdc':     arr('Hdc_command'),
                'duty':    arr('DutyP_command'),
                'meta': {
                    'N_prim': float(np.array(D['Primary_Turns']).item()),
                    'N_sec':  float(np.array(D['Secondary_Turns']).item()),
                    'Ae':     float(np.array(D['Effective_Area']).item()),
                    'Le':     float(np.array(D['Effective_Length']).item()),
                    'dt':     float(np.array(D['Sampling_Time']).flat[0])
                }
            }

    try:
        return _scipy_load()
    except NotImplementedError:
        print("  [Loader] scipy failed (v7.3 HDF5 file) — falling back to h5py ...")
        return _h5py_load()


# ══════════════════════════════════════════════════════════════════
# 3.  PHYSICS FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════

def calc_B(voltage: np.ndarray, dt: float, N_sec: float, Ae: float) -> np.ndarray:
    """Flux density B — shape (N, T) float32."""
    v    = voltage - voltage.mean(axis=1, keepdims=True)
    flux = cumulative_trapezoid(v, axis=-1, initial=0) * dt
    B    = flux / (N_sec * Ae)
    B   -= B.mean(axis=1, keepdims=True)
    return B.astype(np.float32)


def calc_H(current: np.ndarray, N_prim: float, Le: float) -> np.ndarray:
    """Magnetizing force H — shape (N, T) float32."""
    return (N_prim * current / Le).astype(np.float32)


def calc_Loss(B: np.ndarray, H: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Volumetric power loss — shape (N,) float32."""
    energy = np.abs(trapezoid(y=H, x=B, axis=-1))
    return (energy * freq).astype(np.float32)


def build_feature_pool(raw: dict) -> dict:
    """
    Compute every possible feature from raw data **once**.
    All downstream model preprocessing shares this pool — no redundant
    physics computation per model.

    Returns
    -------
    dict  name -> np.ndarray (N, ...) float32

    Available feature names
    -----------------------
    B, H            waveforms    (N, T)
    B_pk, H_pk      peak values  (N, 1)   = (max - min) / 2
    Frequency,
    Temperature,
    Hdc, Duty       scalars      (N, 1)
    Loss            power loss   (N, 1)
    """
    meta = raw['meta']
    N    = raw['voltage'].shape[0]

    def col(a: np.ndarray) -> np.ndarray:
        """Ensure shape (N, 1) for 1-D scalar arrays."""
        return a.reshape(N, 1) if a.ndim == 1 else a

    print("  Computing B   ...", end=' ', flush=True)
    B = calc_B(raw['voltage'], meta['dt'], meta['N_sec'], meta['Ae'])
    print("done")

    print("  Computing H   ...", end=' ', flush=True)
    H = calc_H(raw['current'], meta['N_prim'], meta['Le'])
    print("done")

    print("  Computing Loss...", end=' ', flush=True)
    Loss = calc_Loss(B, H, raw['freq'])
    print("done")

    pool = {
        'B':           B,
        'H':           H,
        'B_pk':        col((B.max(axis=1) - B.min(axis=1)) / 2),
        'H_pk':        col((H.max(axis=1) - H.min(axis=1)) / 2),
        'Frequency':   col(raw['freq']),
        'Temperature': col(raw['temp']),
        'Hdc':         col(raw['hdc']),
        'Duty':        col(raw['duty']),
        'Loss':        col(Loss),
    }

    print(f"\n  Feature pool ready  ({N:,} samples):")
    for name, arr in pool.items():
        print(f"    {name:12s}  shape={str(arr.shape):18s}  dtype={arr.dtype}")

    return pool


# ══════════════════════════════════════════════════════════════════
# 4.  NORMALIZATION
# ══════════════════════════════════════════════════════════════════

VALID_METHODS = {'standard', 'minmax', 'log10', 'none'}


def normalize(data: np.ndarray, method: str, stats: dict = None):
    """
    Normalize an array.

    Parameters
    ----------
    data   : np.ndarray
    method : 'standard' | 'minmax' | 'log10' | 'none'
    stats  : pre-computed stats dict; if None, stats are computed from data
             (always pass pre-computed stats for val/test splits)

    Returns
    -------
    norm_data : np.ndarray  float32
    stats     : dict        (empty dict for 'none' and 'log10')
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Valid options: {VALID_METHODS}"
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
# 5.  CORE PREPROCESSING PIPELINE  (per model)
# ══════════════════════════════════════════════════════════════════

def preprocess_model(
    pool:        dict,
    train_idx:   np.ndarray,
    val_idx:     np.ndarray,
    test_idx:    np.ndarray,
    feat_config: dict,
) -> tuple:
    """
    Split and normalize data for **one model** using the feature config
    read directly from config.yaml.

    Normalization stats are ALWAYS fitted on the training split only —
    then applied identically to val and test (prevents data leakage).

    Parameters
    ----------
    pool        : full feature pool from build_feature_pool()
    train_idx   : np.ndarray  — indices into pool arrays
    val_idx     : np.ndarray
    test_idx    : np.ndarray
    feat_config : dict  {'inputs': {name: method}, 'targets': {name: method}}
                  — comes directly from get_model_feature_config()

    Returns
    -------
    splits : {
        'train': {'inputs': {name: array}, 'targets': {name: array}},
        'val':   { ... },
        'test':  { ... }
    }
    stats : {
        feature_name: {'method': str, ...norm_params}
    }
    """
    splits  = {s: {'inputs': {}, 'targets': {}} for s in ('train', 'val', 'test')}
    stats   = {}
    idx_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    for role in ('inputs', 'targets'):
        # role_cfg = {feature_name: norm_method} — read straight from config.yaml
        role_cfg = feat_config[role]

        for feat_name, method in role_cfg.items():

            # ── Validate feature exists in pool ───────────────────────────
            if feat_name not in pool:
                raise KeyError(
                    f"Feature '{feat_name}' (config.yaml  models → {role}) "
                    f"not found in the feature pool.\n"
                    f"  Available features: {list(pool.keys())}\n"
                    f"  Check your config.yaml for typos."
                )

            full_arr = pool[feat_name]       # (N_total, ...)

            # ── Fit normalization stats on TRAINING data only ─────────────
            train_data  = full_arr[train_idx]
            _, fit_stats = normalize(train_data, method)

            # Save stats with method name for full traceability
            stats[feat_name] = {'method': method, **fit_stats}

            # ── Apply to every split using the same train-fitted stats ────
            for split_name, idx in idx_map.items():
                norm_arr, _ = normalize(
                    full_arr[idx], method,
                    stats=fit_stats if fit_stats else None
                )
                splits[split_name][role][feat_name] = norm_arr

            # ── Log ───────────────────────────────────────────────────────
            params_str = '  '.join(
                f"{k}={v:.5g}" for k, v in fit_stats.items()
            ) if fit_stats else '(stateless)'
            print(
                f"    [{role:7s}] {feat_name:12s}  "
                f"method={method:8s}  {params_str}"
            )

    return splits, stats


# ══════════════════════════════════════════════════════════════════
# 6.  SAVE / LOAD UTILITIES
# ══════════════════════════════════════════════════════════════════

def save_split(split_data: dict, path: str):
    """
    Save one split's inputs + targets into a single compressed .npz.
    Archive keys follow '<role>__<feature_name>', e.g. 'inputs__B'.
    """
    flat = {}
    for role in ('inputs', 'targets'):
        for feat_name, arr in split_data[role].items():
            flat[f"{role}__{feat_name}"] = arr
    np.savez_compressed(path, **flat)


def load_split(path: str) -> dict:
    """
    Reload a split saved by save_split().

    Returns
    -------
    {'inputs': {name: array, ...}, 'targets': {name: array, ...}}
    """
    data   = np.load(path)
    result = {'inputs': {}, 'targets': {}}
    for key in data.files:
        role, feat_name = key.split('__', 1)
        result[role][feat_name] = data[key]
    return result


def stats_to_serializable(stats: dict) -> dict:
    """Convert numpy scalar types → plain Python types for JSON."""
    out = {}
    for feat, s in stats.items():
        out[feat] = {
            k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v
            for k, v in s.items()
        }
    return out


# ══════════════════════════════════════════════════════════════════
# 7.  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MagNet offline preprocessing.\n"
            "Feature selection and normalization methods are read entirely "
            "from config.yaml — no hard-coded values in this script."
        )
    )
    parser.add_argument('--data',   required=True,
                        help='Path to .mat data file')
    parser.add_argument('--split',  required=True,
                        help='Path to dataset_split.json')
    parser.add_argument('--config', required=True,
                        help='Path to config.yaml')
    parser.add_argument('--output', default='processed_data',
                        help='Root output directory (default: processed_data/)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to preprocess (default: all defined in config)')
    args = parser.parse_args()

    # ── STEP 1: Load config.yaml ────────────────────────────────────────────
    print("\n" + "═"*62)
    print("  STEP 1 — Load config.yaml")
    print("═"*62)
    cfg = load_config(args.config)

    available_models = list(cfg.get('models', {}).keys())
    if not available_models:
        raise ValueError("No 'models' entries found in config.yaml.")

    # Default: preprocess all models defined in config
    models_to_run = args.models if args.models else available_models

    # Validate
    unknown = [m for m in models_to_run if m not in available_models]
    if unknown:
        raise ValueError(
            f"Requested model(s) {unknown} not in config.yaml.\n"
            f"Available: {available_models}"
        )

    print(f"  Models to run: {models_to_run}")

    # ── STEP 2: Load split indices ──────────────────────────────────────────
    print("\n" + "═"*62)
    print("  STEP 2 — Load dataset_split.json")
    print("═"*62)
    if not os.path.exists(args.split):
        raise FileNotFoundError(f"Split file not found: {args.split}")

    with open(args.split) as f:
        split_json = json.load(f)

    train_idx = np.array(split_json['train_idx'], dtype=np.int64)
    val_idx   = np.array(split_json['val_idx'],   dtype=np.int64)
    test_idx  = np.array(split_json['test_idx'],  dtype=np.int64)

    print(f"  train = {len(train_idx):>8,}")
    print(f"  val   = {len(val_idx):>8,}")
    print(f"  test  = {len(test_idx):>8,}")

    # ── STEP 3: Load raw dataset ────────────────────────────────────────────
    print("\n" + "═"*62)
    print("  STEP 3 — Load raw .mat dataset")
    print("═"*62)
    raw = load_full_dataset(args.data)
    N   = raw['voltage'].shape[0]
    T   = raw['voltage'].shape[1]
    print(f"  Total samples   : {N:,}")
    print(f"  Waveform length : {T}")
    print(f"  Meta            : {raw['meta']}")

    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if all_idx.max() >= N:
        raise IndexError(
            f"Max index {all_idx.max()} in split file exceeds dataset size {N}."
        )
    if all_idx.min() < 0:
        raise IndexError("Negative indices found in split file.")

    # ── STEP 4: Build feature pool once ────────────────────────────────────
    print("\n" + "═"*62)
    print("  STEP 4 — Build feature pool  (physics computed once)")
    print("═"*62)
    pool = build_feature_pool(raw)

    # ── STEP 5: Per-model preprocessing driven by config.yaml ──────────────
    print("\n" + "═"*62)
    print("  STEP 5 — Per-model preprocessing")
    print("═"*62)

    global_summary = {}

    for model_name in models_to_run:
        print(f"\n  ┌{'─'*58}┐")
        print(f"  │  Model : {model_name.upper():<48}│")
        print(f"  └{'─'*58}┘")

        # Read features + norm methods straight from config.yaml
        feat_config = get_model_feature_config(cfg, model_name)

        print(f"  inputs  (config.yaml) : {feat_config['inputs']}")
        print(f"  targets (config.yaml) : {feat_config['targets']}")
        print()

        splits, stats = preprocess_model(
            pool, train_idx, val_idx, test_idx, feat_config
        )

        # ── Save splits ───────────────────────────────────────────────────
        out_dir = os.path.join(args.output, model_name)
        os.makedirs(out_dir, exist_ok=True)

        print()
        for split_name in ('train', 'val', 'test'):
            path    = os.path.join(out_dir, f"{split_name}.npz")
            save_split(splits[split_name], path)
            size_mb = os.path.getsize(path) / 1e6
            print(f"  Saved {path}  ({size_mb:.1f} MB)")

        # ── Save stats.json ───────────────────────────────────────────────
        stats_path = os.path.join(out_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_to_serializable(stats), f, indent=2)
        print(f"  Saved {stats_path}")

        # ── Summary entry ─────────────────────────────────────────────────
        global_summary[model_name] = {
            'inputs':      feat_config['inputs'],
            'targets':     feat_config['targets'],
            'split_sizes': {
                'train': int(len(train_idx)),
                'val':   int(len(val_idx)),
                'test':  int(len(test_idx)),
            },
            'stats':    stats_to_serializable(stats),
            'saved_to': out_dir,
        }

    # ── STEP 6: Write global summary ────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(args.output, 'preprocessing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n  Global summary → {summary_path}")
    print("\n" + "═"*62)
    print("  ✓  Preprocessing complete!")
    print("═"*62 + "\n")


# ══════════════════════════════════════════════════════════════════
# 8.  PYTORCH DATASET WRAPPER  (drop-in for training scripts)
# ══════════════════════════════════════════════════════════════════

class PreprocessedMagNetDataset:
    """
    Lightweight PyTorch Dataset backed by pre-saved .npz files.
    No physics recomputation at training time.

    Item format per mode
    --------------------
    scaler      ->  x  (n_inputs,),           y  (1,)
    sequence  ┐
    cnn       ├->  B  (T, 1),  scalars (n,),  y  (1,)
    transformer┘
    seq2seq     ->  B  (T, 1),  H  (T, 1)

    Parameters
    ----------
    model_dir : str   directory produced by this script, e.g. 'processed_data/cnn'
    split     : str   'train' | 'val' | 'test'
    mode      : str   'scaler' | 'sequence' | 'cnn' | 'transformer' | 'seq2seq'

    Example
    -------
        from prepare_datasets import PreprocessedMagNetDataset
        from torch.utils.data import DataLoader

        ds     = PreprocessedMagNetDataset('processed_data/cnn', 'train', 'cnn')
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        B, scalars, y = next(iter(loader))
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

        if self.mode == 'scaler':
            x = self._torch.cat([self._t(inp[f][idx]).flatten() for f in inp])
            y = self._t(tgt['Loss'][idx]).flatten()
            return x, y

        elif self.mode in ('sequence', 'cnn', 'transformer'):
            B       = self._t(inp['B'][idx]).unsqueeze(-1)          # (T, 1)
            scalars = self._torch.cat([
                self._t(inp[f][idx]).flatten() for f in inp if f != 'B'
            ])
            y = self._t(tgt['Loss'][idx]).flatten()
            return B, scalars, y

        elif self.mode == 'seq2seq':
            B = self._t(inp['B'][idx]).unsqueeze(-1)                # (T, 1)
            H = self._t(tgt['H'][idx]).unsqueeze(-1)                # (T, 1)
            return B, H

        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. "
                f"Valid: scaler, sequence, cnn, transformer, seq2seq"
            )


if __name__ == '__main__':
    main()