"""
sample_test_set.py
==================
Draws a **representative** sample from the test split for model evaluation.

"Representative" means the sample preserves the joint distribution of the
key experimental conditions (Frequency, Temperature, Hdc) AND the target
variable (Loss) — so model performance on the sample reflects performance
on the full test set.

Strategy
--------
1.  Load scalar features (Frequency, Temperature, Hdc) and compute Loss for
    every sample in the test split.
2.  Bin each feature into n_bins quantile bins (handles log-spaced freq, etc.)
3.  Assign each sample a joint stratum label (freq_bin, temp_bin, hdc_bin, loss_bin).
4.  Use the largest-remainder method to allocate slots proportionally so that
    the final count equals --n exactly.
5.  Save indices JSON, coverage stats JSON, and optionally per-model .npz files.

Fix vs previous version
------------------------
The old implementation used  max(1, round(prop * n))  per stratum.
With 5 bins × 4 features there can be up to 5⁴ = 625 active strata, so the
min-1 floor forced at least 625 samples regardless of --n.
This version uses pure proportional (largest-remainder) allocation which
strictly honours the requested --n.

Usage
-----
  # Draw sample and export per-model .npz files
  python sample_test_set.py \\
      --data      path/to/data.mat \\
      --split     dataset_split.json \\
      --processed processed_data/ \\
      --output    representative_sample/ \\
      --n         100 --bins 5

  # Inspect a saved .npz file (no .mat required)
  python sample_test_set.py \\
      --inspect /content/representative_sample/scaler/representative_test.npz
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NPZ INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

def inspect_npz(npz_path: str) -> dict:
    """
    Print a full human-readable summary of any .npz produced by this pipeline:
      • All keys with shapes and dtypes
      • Per-feature statistics: min, p25, mean, p50, p75, max, std

    Can be used as a CLI shortcut:
        python sample_test_set.py --inspect path/to/file.npz

    Or imported and called directly:
        from sample_test_set import inspect_npz
        inspect_npz('/content/representative_sample/scaler/representative_test.npz')

    Returns the parsed dict so callers can also use the arrays.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path)

    print("\n" + "═" * 68)
    print(f"  NPZ Inspector")
    print(f"  File : {npz_path}")
    print("═" * 68)
    print(f"  Total keys : {len(data.files)}")

    # Group keys by role (inputs / targets)
    roles = {}
    for key in data.files:
        if '__' in key:
            role, feat = key.split('__', 1)
        else:
            role, feat = 'data', key
        roles.setdefault(role, {})[feat] = data[key]

    result = {}
    for role in ('inputs', 'targets', 'data'):
        if role not in roles:
            continue
        print(f"\n  ── {role.upper()} {'─' * (60 - len(role))}")
        for feat, arr in roles[role].items():
            flat = arr.flatten().astype(np.float64)
            print(f"\n    Feature  : {feat}")
            print(f"    Key      : {role}__{feat}")
            print(f"    Shape    : {arr.shape}    Dtype : {arr.dtype}")
            print(f"    Samples  : {len(arr):,}")
            print(f"    ┌──────────────────────────────┐")
            print(f"    │ min  = {flat.min():>21.6g} │")
            print(f"    │ p25  = {np.percentile(flat,25):>21.6g} │")
            print(f"    │ mean = {flat.mean():>21.6g} │")
            print(f"    │ p50  = {np.percentile(flat,50):>21.6g} │")
            print(f"    │ p75  = {np.percentile(flat,75):>21.6g} │")
            print(f"    │ max  = {flat.max():>21.6g} │")
            print(f"    │ std  = {flat.std():>21.6g} │")
            print(f"    └──────────────────────────────┘")
            result.setdefault(role, {})[feat] = arr

    print("\n" + "═" * 68 + "\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RAW SCALAR LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_scalars(mat_path: str, indices: np.ndarray) -> dict:
    """
    Load scalar channels needed for stratification (no waveforms).
    Returns dict  {name: np.ndarray (N,) float32}
    """
    import scipy.io

    def _scipy():
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        D   = mat['Data']
        return {
            'Frequency':   np.array(D.Frequency_command,   dtype=np.float32),
            'Temperature': np.array(D.Temperature_command, dtype=np.float32),
            'Hdc':         np.array(D.Hdc_command,         dtype=np.float32),
            'Duty':        np.array(D.DutyP_command,        dtype=np.float32),
        }

    def _h5py():
        import h5py
        with h5py.File(mat_path, 'r') as f:
            D = f['Data']
            def a(k): return np.array(D[k]).flatten().astype(np.float32)
            return {
                'Frequency':   a('Frequency_command'),
                'Temperature': a('Temperature_command'),
                'Hdc':         a('Hdc_command'),
                'Duty':        a('DutyP_command'),
            }

    try:
        full = _scipy()
    except NotImplementedError:
        print("  [Loader] scipy failed — falling back to h5py ...")
        full = _h5py()

    return {k: v[indices] for k, v in full.items()}


def compute_loss_for_indices(mat_path: str, indices: np.ndarray) -> np.ndarray:
    """Compute volumetric power loss for a subset. Returns (N,) float32."""
    from scipy.integrate import cumulative_trapezoid, trapezoid
    import scipy.io

    def _scipy():
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        D   = mat['Data']
        meta = {
            'N_prim': float(D.Primary_Turns),  'N_sec': float(D.Secondary_Turns),
            'Ae':     float(D.Effective_Area), 'Le':    float(D.Effective_Length),
            'dt':     float(np.array(D.Sampling_Time).flat[0]),
        }
        return (meta,
                np.array(D.Frequency_command, dtype=np.float32)[indices],
                np.array(D.Voltage,           dtype=np.float32)[indices],
                np.array(D.Current,           dtype=np.float32)[indices])

    def _h5py():
        import h5py
        with h5py.File(mat_path, 'r') as f:
            D = f['Data']
            meta = {
                'N_prim': float(np.array(D['Primary_Turns']).item()),
                'N_sec':  float(np.array(D['Secondary_Turns']).item()),
                'Ae':     float(np.array(D['Effective_Area']).item()),
                'Le':     float(np.array(D['Effective_Length']).item()),
                'dt':     float(np.array(D['Sampling_Time']).flat[0]),
            }
            freq    = np.array(D['Frequency_command']).flatten().astype(np.float32)[indices]
            voltage = np.array(D['Voltage'])[:, indices].T.astype(np.float32)
            current = np.array(D['Current'])[:, indices].T.astype(np.float32)
        return meta, freq, voltage, current

    try:
        meta, freq, voltage, current = _scipy()
    except NotImplementedError:
        meta, freq, voltage, current = _h5py()

    v   = voltage - voltage.mean(axis=1, keepdims=True)
    B   = cumulative_trapezoid(v, axis=-1, initial=0) * meta['dt'] / (meta['N_sec'] * meta['Ae'])
    B  -= B.mean(axis=1, keepdims=True)
    H   = meta['N_prim'] * current / meta['Le']
    return (np.abs(trapezoid(y=H, x=B, axis=-1)) * freq).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  STRATIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based binning — 0-based labels, handles skewed distributions."""
    edges    = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-9
    return np.digitize(values, edges[1:-1])


def build_strata(scalars: dict, loss: np.ndarray, n_bins: int) -> np.ndarray:
    """Joint stratum label = (freq_bin, temp_bin, hdc_bin, loss_bin)."""
    stratify_on = {
        'Frequency':   scalars['Frequency'],
        'Temperature': scalars['Temperature'],
        'Hdc':         scalars['Hdc'],
        'Loss':        loss,
    }
    bin_arrays = {}
    print(f"\n  Binning strategy  (n_bins={n_bins}, quantile-based):")
    for name, vals in stratify_on.items():
        b = quantile_bin(vals, n_bins)
        bin_arrays[name] = b
        print(f"    {name:12s}  range=[{vals.min():.4g}, {vals.max():.4g}]"
              f"  → {len(np.unique(b))} active bins")

    N      = len(loss)
    labels = np.empty(N, dtype=object)
    for i in range(N):
        labels[i] = tuple(arr[i] for arr in bin_arrays.values())
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# 4.  STRATIFIED SAMPLER  (largest-remainder, exact n)
# ══════════════════════════════════════════════════════════════════════════════

def stratified_sample(
    global_indices: np.ndarray,
    strata_labels:  np.ndarray,
    n_samples:      int,
    seed:           int = 42,
) -> np.ndarray:
    """
    Draw exactly n_samples indices proportionally from each stratum.

    Uses the largest-remainder method so the total always equals n_samples.
    No min-1 floor — strata too small to earn a slot by proportion receive 0.
    """
    rng = np.random.default_rng(seed)

    stratum_to_positions = defaultdict(list)
    for pos, label in enumerate(strata_labels):
        stratum_to_positions[label].append(pos)

    total    = len(global_indices)
    n_strata = len(stratum_to_positions)

    print(f"\n  Stratified sampling:")
    print(f"    Total test samples : {total:,}")
    print(f"    Unique strata      : {n_strata:,}")
    print(f"    Target sample size : {n_samples:,}")

    if n_strata > n_samples:
        print(
            f"    NOTE: {n_strata} strata > {n_samples} target.\n"
            f"          Small strata will receive 0 slots.\n"
            f"          Increase --n or decrease --bins for full coverage."
        )

    # ── Largest-remainder proportional allocation ─────────────────────────────
    labels_sorted = sorted(stratum_to_positions.keys())
    sizes         = np.array([len(stratum_to_positions[l]) for l in labels_sorted],
                             dtype=np.float64)
    raw_alloc     = sizes / sizes.sum() * n_samples
    floor_alloc   = np.floor(raw_alloc).astype(int)
    deficit       = n_samples - floor_alloc.sum()
    top_up        = np.argsort(-(raw_alloc - floor_alloc))[:deficit]
    floor_alloc[top_up] += 1

    # ── Sample ────────────────────────────────────────────────────────────────
    sampled_pos = []
    for label, alloc in zip(labels_sorted, floor_alloc):
        if alloc == 0:
            continue
        positions = stratum_to_positions[label]
        chosen    = rng.choice(positions, size=min(alloc, len(positions)),
                               replace=False)
        sampled_pos.extend(chosen.tolist())

    # Deduplicate and enforce exact count
    sampled_pos = list(dict.fromkeys(sampled_pos))
    if len(sampled_pos) < n_samples:
        remaining   = list(set(range(total)) - set(sampled_pos))
        extra       = rng.choice(remaining, size=n_samples - len(sampled_pos),
                                 replace=False)
        sampled_pos.extend(extra.tolist())
    sampled_pos = sampled_pos[:n_samples]

    print(f"    ✓ Exact sample size: {len(sampled_pos):,}")
    return global_indices[np.array(sampled_pos, dtype=np.int64)]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  COVERAGE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def coverage_report(
    full_scalars:   dict, full_loss:   np.ndarray,
    sample_scalars: dict, sample_loss: np.ndarray,
) -> dict:
    """Compare distribution statistics — full test set vs sample."""
    features = {**full_scalars, 'Loss': full_loss}
    samples  = {**sample_scalars, 'Loss': sample_loss}
    report   = {}

    print("\n  Coverage report  (full test set  vs  sample):")
    hdr = f"  {'Feature':12s}  {'Metric':5s}  {'Full test':>13}  {'Sample':>13}  {'Δ%':>8}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for name in features:
        fv, sv = features[name].astype(np.float64), samples[name].astype(np.float64)
        stats  = {}
        for metric, f_val, s_val in [
            ('mean', np.mean(fv),          np.mean(sv)),
            ('std',  np.std(fv),           np.std(sv)),
            ('min',  np.min(fv),           np.min(sv)),
            ('p25',  np.percentile(fv,25), np.percentile(sv,25)),
            ('p50',  np.percentile(fv,50), np.percentile(sv,50)),
            ('p75',  np.percentile(fv,75), np.percentile(sv,75)),
            ('max',  np.max(fv),           np.max(sv)),
        ]:
            delta = ((s_val - f_val) / (abs(f_val) + 1e-12)) * 100
            stats[metric] = {'full_test': float(f_val), 'sample': float(s_val),
                             'delta_pct': float(delta)}
            print(f"  {name:12s}  {metric:5s}  {f_val:>13.5g}  {s_val:>13.5g}  {delta:>+7.2f}%")
        report[name] = stats

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EXPORT PER-MODEL .npz
# ══════════════════════════════════════════════════════════════════════════════

def export_model_npz(
    sample_global: np.ndarray,
    test_global:   np.ndarray,
    processed_root: str,
    output_dir:     str,
) -> list:
    """Slice each model's test.npz to the sampled rows and save."""
    g2l       = {int(g): i for i, g in enumerate(test_global)}
    local_rows = np.array([g2l[int(g)] for g in sample_global], dtype=np.int64)

    exported = []
    for model_name in sorted(os.listdir(processed_root)):
        test_npz = os.path.join(processed_root, model_name, 'test.npz')
        if not os.path.exists(test_npz):
            continue
        data     = np.load(test_npz)
        out_dir  = os.path.join(output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'representative_test.npz')
        np.savez_compressed(out_path, **{k: data[k][local_rows] for k in data.files})
        size_mb  = os.path.getsize(out_path) / 1e6
        print(f"    [{model_name}]  {out_path}  ({size_mb:.2f} MB)")
        print(f"      keys   : {list(data.files)}")
        print(f"      rows   : {len(local_rows):,}  (from {data[data.files[0]].shape[0]:,})")
        exported.append(model_name)

    return exported


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Representative test-set sampler with stratification.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--inspect', default=None, metavar='NPZ_PATH',
                        help='Inspect a saved .npz and print statistics, then exit.')
    parser.add_argument('--data',      default=None)
    parser.add_argument('--split',     default=None)
    parser.add_argument('--output',    default=None)
    parser.add_argument('--n',         type=int, default=500,
                        help='Exact number of samples to draw (default: 500)')
    parser.add_argument('--bins',      type=int, default=5,
                        help='Quantile bins per feature (default: 5)')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--processed', default=None,
                        help='processed_data/ root — exports per-model .npz files')
    args = parser.parse_args()

    # ── Inspect mode ──────────────────────────────────────────────────────────
    if args.inspect:
        inspect_npz(args.inspect)
        return

    for flag, val in [('--data', args.data), ('--split', args.split),
                      ('--output', args.output)]:
        if val is None:
            parser.error(f"{flag} is required unless --inspect is used.")

    os.makedirs(args.output, exist_ok=True)

    # STEP 1 — indices
    print("\n" + "═"*62)
    print("  STEP 1 — Load test split indices")
    print("═"*62)
    with open(args.split) as f:
        split_json = json.load(f)
    test_idx = np.array(split_json['test_idx'], dtype=np.int64)
    print(f"  Test set size : {len(test_idx):,}")

    if args.n >= len(test_idx):
        print(f"  n={args.n} ≥ test size — saving full test set.")
        with open(os.path.join(args.output, 'representative_test_indices.json'), 'w') as f:
            json.dump({'n_samples': int(len(test_idx)),
                       'sampled_indices': test_idx.tolist()}, f, indent=2)
        return

    # STEP 2 — scalars
    print("\n" + "═"*62)
    print("  STEP 2 — Load scalar features")
    print("═"*62)
    scalars = load_raw_scalars(args.data, test_idx)
    for name, arr in scalars.items():
        print(f"    {name:12s}  min={arr.min():.4g}  max={arr.max():.4g}  mean={arr.mean():.4g}")

    # STEP 3 — Loss
    print("\n" + "═"*62)
    print("  STEP 3 — Compute Loss")
    print("═"*62)
    loss = compute_loss_for_indices(args.data, test_idx)
    print(f"    Loss  min={loss.min():.4g}  max={loss.max():.4g}  mean={loss.mean():.4g}")

    # STEP 4 — strata
    print("\n" + "═"*62)
    print("  STEP 4 — Build joint strata")
    print("═"*62)
    strata = build_strata(scalars, loss, n_bins=args.bins)

    # STEP 5 — sample
    print("\n" + "═"*62)
    print("  STEP 5 — Draw stratified sample")
    print("═"*62)
    sampled = stratified_sample(test_idx, strata, args.n, seed=args.seed)

    # STEP 6 — coverage
    print("\n" + "═"*62)
    print("  STEP 6 — Coverage report")
    print("═"*62)
    g2l          = {int(g): i for i, g in enumerate(test_idx)}
    sample_local = np.array([g2l[int(g)] for g in sampled], dtype=np.int64)
    report       = coverage_report(
        scalars, loss,
        {k: v[sample_local] for k, v in scalars.items()},
        loss[sample_local],
    )

    # STEP 7 — save JSON outputs
    print("\n" + "═"*62)
    print("  STEP 7 — Save outputs")
    print("═"*62)
    idx_path = os.path.join(args.output, 'representative_test_indices.json')
    with open(idx_path, 'w') as f:
        json.dump({
            'description':     'Representative test sample — sample_test_set.py',
            'n_samples':       int(len(sampled)),
            'n_total_test':    int(len(test_idx)),
            'n_bins':          args.bins,
            'seed':            args.seed,
            'stratify_on':     ['Frequency', 'Temperature', 'Hdc', 'Loss'],
            'sampled_indices': sampled.tolist(),
        }, f, indent=2)
    print(f"  Saved {idx_path}")

    stats_path = os.path.join(args.output, 'representative_test_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'sample_config': {'n_samples': int(len(sampled)),
                              'n_total_test': int(len(test_idx)),
                              'n_bins': args.bins, 'seed': args.seed},
            'coverage': report,
        }, f, indent=2)
    print(f"  Saved {stats_path}")

    # STEP 8 — export per-model .npz
    if args.processed:
        print("\n" + "═"*62)
        print("  STEP 8 — Export per-model representative_test.npz")
        print("═"*62)
        exported = export_model_npz(sampled, test_idx, args.processed, args.output)
        print(f"\n  Exported models: {exported}")

        # Auto-inspect the scaler npz
        scaler_npz = os.path.join(args.output, 'scaler', 'representative_test.npz')
        if os.path.exists(scaler_npz):
            print()
            inspect_npz(scaler_npz)

    print("\n" + "═"*62)
    print(f"  ✓  Done!  {len(sampled):,} samples → {args.output}/")
    print("═"*62 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  IMPORTABLE INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def load_representative_sample(
    indices_json:   str,
    processed_root: str,
    model_name:     str,
    split_json:     str = None,
) -> dict:
    """
    Load the representative sample for one model without needing the .mat file.

    Returns {'inputs': {name: array}, 'targets': {name: array}}
    """
    with open(indices_json) as f:
        meta    = json.load(f)
    sampled = np.array(meta['sampled_indices'], dtype=np.int64)

    test_npz = os.path.join(processed_root, model_name, 'test.npz')
    if not os.path.exists(test_npz):
        raise FileNotFoundError(f"test.npz not found: {test_npz}")

    if split_json is None:
        split_json = os.path.join(processed_root, '..', 'dataset_split.json')
    with open(split_json) as f:
        test_global = np.array(json.load(f)['test_idx'], dtype=np.int64)

    g2l        = {int(g): i for i, g in enumerate(test_global)}
    local_rows = np.array([g2l[int(g)] for g in sampled], dtype=np.int64)

    raw    = np.load(test_npz)
    result = {'inputs': {}, 'targets': {}}
    for key in raw.files:
        role, feat = key.split('__', 1)
        result[role][feat] = raw[key][local_rows]
    return result


if __name__ == '__main__':
    main()