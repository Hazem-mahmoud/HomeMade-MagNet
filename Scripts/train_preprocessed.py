"""
train_preprocessed.py
=====================
Training entry point that runs on pre-processed data produced by
prepare_datasets.py — no raw .mat file or physics computation needed.

Additions vs original
---------------------
* cnnv3 model support  (B + H waveforms + scalar FiLM conditioning)
* --predictions-dir    new CLI flag: directory where prediction CSVs are saved
* After training, val AND test predictions + ground-truth are saved as:
      <predictions_dir>/cnnv3_v<version>_val_predictions.csv
      <predictions_dir>/cnnv3_v<version>_test_predictions.csv
  CSV columns: prediction, target

Usage
-----
  python train_preprocessed.py \\
      --processed     processed_data/ \\
      --config        Scripts/config/config.yaml \\
      --model         cnnv3 \\
      --predictions-dir  /content/drive/MyDrive/MagNet/predictions

  # Train all models (predictions saved for every model)
  python train_preprocessed.py \\
      --processed     processed_data/ \\
      --config        Scripts/config/config.yaml \\
      --predictions-dir  /content/drive/MyDrive/MagNet/predictions
"""

import os
import csv
import json
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

# ── Project imports ──────────────────────────────────────────────────────────
from src.models.scaler_model      import ScalerNetwork
from src.models.sequence_model    import SequenceToScalerNetwork
from src.models.seq2seq_model     import Seq2SeqNetwork
from src.models.cnn_model         import CNNNetwork
from src.models.transformer_model import TransformerNetwork
from src.training.train           import train_model
from src.utils.visualization      import plot_loss_curve, plot_prediction_scatter


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _load_split(npz_path: str) -> dict:
    data   = np.load(npz_path)
    result = {'inputs': {}, 'targets': {}}
    for key in data.files:
        role, feat = key.split('__', 1)
        result[role][feat] = data[key]
    return result


# Waveform keys for cnnv3 — all other input keys are treated as scalars
_CNNV3_WAVEFORMS = frozenset({'B', 'H'})


class PreprocessedDataset(Dataset):
    """
    PyTorch Dataset backed by a single pre-processed .npz split.

    Item format per mode
    ----------------------------------------------------------
    scaler / scalerv2  ->  x (n_feats,),                       y (1,)
    sequence / cnn
    cnnv2 / transformer->  B (T,1),  scalars (n,),             y (1,)
    cnnv3              ->  B (T,1),  H (T,1),  scalars (n,),   y (1,)
    seq2seq            ->  B (T,1),  H (T,1)
    """

    def __init__(self, npz_path: str, stats: dict, mode: str):
        super().__init__()
        self.mode  = mode
        self.stats = stats

        data = _load_split(npz_path)
        self.inputs  = data['inputs']
        self.targets = data['targets']

        first = (next(iter(self.inputs.values())) if self.inputs
                 else next(iter(self.targets.values())))
        self.length = len(first)

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx: int):
        inp = self.inputs
        tgt = self.targets

        if self.mode in ('scaler', 'scalerv2'):
            x = torch.cat([self._t(inp[f][idx]).flatten() for f in inp])
            y = self._t(tgt['Loss'][idx]).flatten()
            return x, y

        elif self.mode in ('sequence', 'cnn', 'cnnv2', 'transformer'):
            B       = self._t(inp['B'][idx]).unsqueeze(-1)
            scalars = torch.cat([
                self._t(inp[f][idx]).flatten() for f in inp if f != 'B'
            ])
            y = self._t(tgt['Loss'][idx]).flatten()
            return B, scalars, y

        elif self.mode == 'cnnv3':
            B       = self._t(inp['B'][idx]).unsqueeze(-1)          # (T, 1)
            H       = self._t(inp['H'][idx]).unsqueeze(-1)          # (T, 1)
            scalars = torch.cat([
                self._t(inp[f][idx]).flatten()
                for f in inp if f not in _CNNV3_WAVEFORMS
            ])
            y = self._t(tgt['Loss'][idx]).flatten()
            return B, H, scalars, y

        elif self.mode == 'seq2seq':
            B = self._t(inp['B'][idx]).unsqueeze(-1)
            H = self._t(tgt['H'][idx]).unsqueeze(-1)
            return B, H

        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")


def build_loaders(
    model_name:     str,
    processed_root: str,
    batch_size:     int,
) -> tuple:
    """Build train and val DataLoaders from pre-processed .npz files."""
    model_dir  = os.path.join(processed_root, model_name)
    stats_path = os.path.join(model_dir, 'stats.json')

    for split in ('train', 'val'):
        p = os.path.join(model_dir, f"{split}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Preprocessed file not found: {p}\n"
                f"Run prepare_datasets.py first."
            )
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stats file not found: {stats_path}\n"
            f"Run prepare_datasets.py first."
        )

    with open(stats_path) as f:
        stats = json.load(f)

    dataset_mode = _dataset_mode(model_name)

    train_ds = PreprocessedDataset(
        os.path.join(model_dir, 'train.npz'), stats, mode=dataset_mode
    )
    val_ds = PreprocessedDataset(
        os.path.join(model_dir, 'val.npz'), stats, mode=dataset_mode
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"  train samples : {len(train_ds):,}")
    print(f"  val   samples : {len(val_ds):,}")

    return train_loader, val_loader, stats


def _dataset_mode(model_name: str) -> str:
    """Map model name → dataset item format."""
    _mode_map = {
        'scalerv2': 'scaler',
        'cnnv2':    'cnn',
        # cnnv3 has its own mode
    }
    return _mode_map.get(model_name, model_name)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, config: dict, stats: dict, seq_len: int = 1024):
    """Instantiate the correct network from config.yaml."""
    model_conf  = config['models'][model_name]
    version     = model_conf['version']
    version_tag = f"{model_name}_v{version}"

    if model_name == 'scaler':
        input_dim = len(
            config['models']['scaler'].get('features', {}).get('inputs', {})
        ) or 3
        model = ScalerNetwork(
            input_dim  = input_dim,
            hidden_dim = model_conf['hidden_dim'],
            num_layers = model_conf['layers'],
            output_dim = 1,
        )

    elif model_name == 'scalerv2':
        from src.models.scaler_v2 import ScalerNetwork as ScalerNetworkV2
        input_dim = len(
            config['models']['scalerv2'].get('features', {}).get('inputs', {})
        ) or 4
        model = ScalerNetworkV2(
            input_dim  = input_dim,
            hidden_dim = model_conf['hidden_dim'],
            num_layers = model_conf['layers'],
            output_dim = 1,
            dropout    = model_conf.get('dropout', 0.2),
        )

    elif model_name == 'sequence':
        model = SequenceToScalerNetwork(
            input_dim  = 1,
            hidden_dim = model_conf['hidden_dim'],
            output_dim = 1,
            num_layers = model_conf['num_layers'],
        )

    elif model_name == 'seq2seq':
        model = Seq2SeqNetwork(
            input_dim  = 1,
            hidden_dim = model_conf['encoder_dim'],
            output_dim = 1,
        )

    elif model_name == 'cnn':
        freq_stats = {}
        if 'Frequency' in stats:
            f_s = stats['Frequency']
            if 'mean' in f_s:
                freq_stats = {'freq_mean': f_s['mean'], 'freq_std': f_s['std']}
        model = CNNNetwork(input_dim=1, stats=freq_stats)

    elif model_name == 'cnnv2':
        from src.models.cnn_v2 import CNNNetwork as CNNNetworkV2
        input_features = config['models']['cnnv2'].get('features', {}).get('inputs', {})
        scalar_dim = len([f for f in input_features if f != 'B']) or 3
        model = CNNNetworkV2(
            input_dim    = 1,
            num_channels = model_conf.get('num_channels', 96),
            num_layers   = model_conf.get('num_layers',   4),
            scalar_dim   = scalar_dim,
            dropout      = model_conf.get('dropout',      0.15),
            stats        = None,
        )

    elif model_name == 'cnnv3':
        # Import the new model (new_model.py / cnn_v3.py)
        # Try canonical project path first, fall back to new_model.py
        try:
            from src.models.cnn_v3 import CNNNetwork as CNNNetworkV3
        except ImportError:
            # Allow running with new_model.py placed alongside this script
            import importlib.util, sys
            _candidates = [
                os.path.join(os.path.dirname(__file__), 'new_model.py'),
                os.path.join(os.path.dirname(__file__), 'src', 'models', 'new_model.py'),
            ]
            _spec = None
            for _p in _candidates:
                if os.path.exists(_p):
                    _spec = importlib.util.spec_from_file_location('cnn_v3', _p)
                    break
            if _spec is None:
                raise ImportError(
                    "Cannot find cnnv3 model. Place new_model.py at "
                    "Scripts/src/models/cnn_v3.py or alongside train_preprocessed.py."
                )
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules['cnn_v3'] = _mod
            _spec.loader.exec_module(_mod)
            CNNNetworkV3 = _mod.CNNNetwork

        # Derive scalar_dim from config: inputs minus waveform keys (B, H)
        input_features = config['models']['cnnv3'].get('features', {}).get('inputs', {})
        scalar_dim = len([f for f in input_features if f not in _CNNV3_WAVEFORMS])
        if scalar_dim == 0:
            scalar_dim = model_conf.get('scalar_dim', 2)  # fallback from config key

        model = CNNNetworkV3(
            input_dim    = 3,                                   # B, H, B*H channels
            num_channels = model_conf.get('num_channels', 96),
            num_layers   = model_conf.get('num_layers',   4),
            scalar_dim   = scalar_dim,
            dropout      = model_conf.get('dropout',      0.15),
            stats        = None,
        )
        print(f"  CNNv3 — num_channels={model_conf.get('num_channels', 96)}, "
              f"num_layers={model_conf.get('num_layers', 4)}, "
              f"scalar_dim={scalar_dim}")

    elif model_name == 'transformer':
        model = TransformerNetwork(
            B_in_channel            = seq_len,
            dim_hidden              = model_conf['d_model'],
            n_encoder_layers        = model_conf['num_layers'],
            dim_feedforward_encoder = model_conf['dim_feedforward'],
            n_heads                 = model_conf['nhead'],
            dropout_encoder         = model_conf['dropout'],
        )

    else:
        raise ValueError(f"Unknown model name '{model_name}'.")

    return model, version_tag


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PREDICTION CSV SAVER
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions_csv(
    model:           torch.nn.Module,
    loader:          DataLoader,
    model_name:      str,
    version_tag:     str,
    split_name:      str,
    predictions_dir: str,
    device:          str,
):
    """
    Run inference on *loader* and write predictions + ground-truth to a CSV.

    File path
    ---------
      <predictions_dir>/<version_tag>_<split_name>_predictions.csv
      e.g.  predictions/cnnv3_v1_val_predictions.csv

    CSV columns
    -----------
      prediction, target
    """
    os.makedirs(predictions_dir, exist_ok=True)

    csv_filename = f"{version_tag}_{split_name}_predictions.csv"
    csv_path     = os.path.join(predictions_dir, csv_filename)

    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            preds_batch, targets_batch = _infer_batch(model, batch, model_name, device)
            all_preds.append(preds_batch)
            all_targets.append(targets_batch)

    all_preds   = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction', 'target'])
        for p, t in zip(all_preds, all_targets):
            writer.writerow([float(p), float(t)])

    print(f"  Predictions saved → {csv_path}  ({len(all_preds):,} rows)")
    return csv_path


def _infer_batch(
    model:      torch.nn.Module,
    batch:      tuple,
    model_name: str,
    device:     str,
) -> tuple:
    """
    Run one batch through *model* and return (preds_np, targets_np).
    Handles all batch signatures used across models.
    """
    if model_name == 'cnnv3':
        B, H, scalars, y = batch
        out = model(B.to(device), H.to(device), scalars.to(device))
        return out.cpu().numpy(), y.numpy()

    elif model_name in ('scaler', 'scalerv2'):
        x, y = batch
        out  = model(x.to(device))
        return out.cpu().numpy(), y.numpy()

    elif model_name == 'seq2seq':
        B, H = batch
        out  = model(B.to(device))
        return out.cpu().numpy(), H.numpy()

    else:
        # sequence / cnn / cnnv2 / transformer  →  (B, scalars, y)
        B, scalars, y = batch
        out = model(B.to(device), scalars.to(device))
        return out.cpu().numpy(), y.numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP WRAPPER  (cnnv3-aware)
# ══════════════════════════════════════════════════════════════════════════════

def _build_cnnv3_train_wrappers(train_loader, val_loader, device):
    """
    train_model() in train.py expects batches of length 2 or 3.
    cnnv3 batches have length 4: (B, H, scalars, y).

    We wrap the DataLoaders so their batches look like (combined_input, y)
    where combined_input is a tuple (B, H, scalars) — then monkey-patch the
    model forward to accept that signature inside train_model.

    A cleaner approach: subclass the model.  We do the minimal patch here so
    train.py needs zero modification.
    """
    # We expose a thin adapter Dataset that re-packs the 4-tuple into a 3-tuple
    # by treating (B, H) as a merged "input" object and scalars separately.
    # train.py checks len(batch) == 3 → calls model(inputs, scalars)
    # We make inputs = (B, H) and patch the model to unpack it.
    pass   # See _wrap_model_for_train below


class _CnnV3TrainAdapter(Dataset):
    """
    Wraps a cnnv3 PreprocessedDataset to emit 3-tuples:
      (B,  scalars,  y)   where B is actually (B_tensor, H_tensor) packed into one object

    This lets train_model() call model(inputs=B_packed, scalars=scalars)
    while our wrapper model unpacks B and H internally.
    """

    def __init__(self, base_ds: Dataset):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        B, H, scalars, y = self.base[idx]
        # Pack (B, H) as a simple tensor by stacking on last dim: (T, 2)
        BH = torch.cat([B, H], dim=-1)   # (T, 2)
        return BH, scalars, y


class _CnnV3ModelWrapper(torch.nn.Module):
    """
    Wraps CNNNetworkV3 so it accepts the packed (T, 2) BH tensor from the adapter.
    Called as: wrapper(BH, scalars)  →  unpacks to model(B, H, scalars)
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, BH: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        B = BH[..., 0:1]   # (batch, T, 1)
        H = BH[..., 1:2]   # (batch, T, 1)
        return self.model(B, H, scalars)

    # Expose underlying model parameters / state_dict transparently
    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)

    def state_dict(self, **kwargs):
        return self.model.state_dict(**kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        return self.model.load_state_dict(state_dict, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODELS = [
    'scaler', 'scalerv2', 'sequence', 'seq2seq',
    'cnn', 'cnnv2', 'cnnv3', 'transformer',
]


def main():
    parser = argparse.ArgumentParser(
        description="Train MagNet models on pre-processed data."
    )
    parser.add_argument('--processed', required=True,
                        help='Root directory of pre-processed data (e.g. processed_data/)')
    parser.add_argument('--config', type=str, default='Scripts/config/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--model', type=str,
                        choices=ALL_MODELS + ['all'], default='all',
                        help='Model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument(
        '--predictions-dir',
        type=str,
        default=None,
        help=(
            'Directory to save prediction CSVs after training.\n'
            'Files are named  <model_name>_v<version>_<split>_predictions.csv\n'
            'Both val and test splits are saved.\n'
            'If not set, predictions are not saved to disk.'
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['epochs'] = args.epochs

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = config['data'].get('batch_size', 32)
    save_root  = config['training']['save_dir']

    print(f"\nDevice          : {device}")
    print(f"Batch size      : {batch_size}")
    print(f"Epochs          : {config['training']['epochs']}")
    print(f"Save root       : {save_root}")
    if args.predictions_dir:
        print(f"Predictions dir : {args.predictions_dir}")

    models_to_run = ALL_MODELS if args.model == 'all' else [args.model]

    # ─────────────────────────────────────────────────────────────────────────
    for model_name in models_to_run:
        print(f"\n{'='*20} Training {model_name.upper()} {'='*20}")

        # ── 1. DataLoaders ───────────────────────────────────────────────────
        print("Loading pre-processed data ...")
        train_loader, val_loader, stats = build_loaders(
            model_name, args.processed, batch_size
        )

        # Infer waveform sequence length (used by Transformer)
        seq_len = 1024
        try:
            sample_path = os.path.join(args.processed, model_name, 'train.npz')
            sample_data = np.load(sample_path)
            b_key = next((k for k in sample_data.files if k.endswith('__B')), None)
            if b_key is not None:
                seq_len = sample_data[b_key].shape[1]
                print(f"  Waveform seq_len : {seq_len}")
        except Exception:
            pass

        # ── 2. Build model ────────────────────────────────────────────────────
        print("Building model ...")
        model, version_tag = build_model(model_name, config, stats, seq_len)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Version    : {version_tag}")
        print(f"  Parameters : {total_params:,}")

        # ── 3. cnnv3 — adapt loaders + wrap model for train_model() ──────────
        if model_name == 'cnnv3':
            # Wrap datasets so train.py sees 3-tuple batches
            train_ds_v3 = _CnnV3TrainAdapter(train_loader.dataset)
            val_ds_v3   = _CnnV3TrainAdapter(val_loader.dataset)

            train_loader_for_train = DataLoader(
                train_ds_v3, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=True,
            )
            val_loader_for_train = DataLoader(
                val_ds_v3, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True,
            )
            model_for_train = _CnnV3ModelWrapper(model)
        else:
            train_loader_for_train = train_loader
            val_loader_for_train   = val_loader
            model_for_train        = model

        # ── 4. Train ──────────────────────────────────────────────────────────
        print("Starting training ...")
        save_dir = os.path.join(save_root, model_name)
        os.makedirs(save_dir, exist_ok=True)

        train_config             = dict(config['training'])
        train_config['save_dir'] = save_dir
        run_config               = dict(config)
        run_config['training']   = train_config

        model_for_train = model_for_train.to(device)
        trained_wrapper, history, metrics, preds, targets_out = train_model(
            model_for_train, train_loader_for_train,
            val_loader_for_train, model_name, run_config, device,
        )

        # Unwrap so we have the pure model for inference
        trained_model = (
            trained_wrapper.model
            if isinstance(trained_wrapper, _CnnV3ModelWrapper)
            else trained_wrapper
        )

        print(f"  Validation metrics : {metrics}")

        # ── 5. Test-split relative-error statistics ───────────────────────────
        _SCALAR_MODELS = ('scaler', 'scalerv2', 'sequence', 'cnn', 'cnnv2', 'cnnv3', 'transformer')
        if model_name in _SCALAR_MODELS:
            try:
                dataset_mode = _dataset_mode(model_name)
                test_ds = PreprocessedDataset(
                    os.path.join(args.processed, model_name, 'test.npz'),
                    stats, mode=dataset_mode,
                )
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=True,
                )
                trained_model.eval()
                all_preds_t, all_targets_t = [], []

                with torch.no_grad():
                    for batch in test_loader:
                        p, t = _infer_batch(trained_model, batch, model_name, device)
                        all_preds_t.append(p)
                        all_targets_t.append(t)

                all_preds_t   = np.concatenate(all_preds_t).flatten()
                all_targets_t = np.concatenate(all_targets_t).flatten()

                rel_errors    = np.abs(
                    (all_preds_t - all_targets_t) / (np.abs(all_targets_t) + 1e-8)
                )
                p95_rel_error = float(np.percentile(rel_errors, 95))
                max_rel_error = float(np.max(rel_errors))

                print(f"  Test 95th-pct relative error : {p95_rel_error:.6f}")
                print(f"  Test max relative error       : {max_rel_error:.6f}")
                metrics['test_p95_relative_error'] = p95_rel_error
                metrics['test_max_relative_error'] = max_rel_error

                model_cfg           = config['models'][model_name]
                experiment_log_path = os.path.join(save_dir, 'experiments.txt')
                with open(experiment_log_path, 'a') as f:
                    f.write("Test Set Relative-Error Metrics:\n")
                    f.write(f"  P95 Relative Error: {p95_rel_error:.6f}\n")
                    f.write(f"  Max Relative Error: {max_rel_error:.6f}\n")
                    f.write("=" * 70 + "\n\n")

            except FileNotFoundError:
                print("  test.npz not found — skipping relative-error statistics.")
                test_loader = None
                all_preds_t   = None
                all_targets_t = None

        # ── 6. Save prediction CSVs ───────────────────────────────────────────
        if args.predictions_dir:
            print(f"\n  Saving prediction CSVs → {args.predictions_dir}")

            # --- Validation predictions ---
            # Rebuild val loader with original (non-adapted) dataset for cnnv3
            if model_name == 'cnnv3':
                val_pred_loader = DataLoader(
                    val_loader.dataset, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=True,
                )
            else:
                val_pred_loader = val_loader

            save_predictions_csv(
                model           = trained_model,
                loader          = val_pred_loader,
                model_name      = model_name,
                version_tag     = version_tag,
                split_name      = 'val',
                predictions_dir = args.predictions_dir,
                device          = device,
            )

            # --- Test predictions ---
            test_npz = os.path.join(args.processed, model_name, 'test.npz')
            if os.path.exists(test_npz):
                test_ds_for_csv = PreprocessedDataset(
                    test_npz, stats, mode=_dataset_mode(model_name)
                )
                test_loader_for_csv = DataLoader(
                    test_ds_for_csv, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=True,
                )
                save_predictions_csv(
                    model           = trained_model,
                    loader          = test_loader_for_csv,
                    model_name      = model_name,
                    version_tag     = version_tag,
                    split_name      = 'test',
                    predictions_dir = args.predictions_dir,
                    device          = device,
                )
            else:
                print("  test.npz not found — skipping test prediction CSV.")

        # ── 7. Plots ──────────────────────────────────────────────────────────
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        loss_plot_path = os.path.join(plots_dir, f'{version_tag}_loss_curve.png')
        plot_loss_curve(
            history,
            title     = f'{model_name} Training Loss',
            save_path = loss_plot_path,
        )
        print(f"  Loss plot saved → {loss_plot_path}")

        if model_name in _SCALAR_MODELS:
            pred_plot_path = os.path.join(
                plots_dir, f'{version_tag}_prediction_scatter.png'
            )
            plot_prediction_scatter(
                preds,
                targets_out,
                title     = f'{model_name}: Pred vs Actual Loss',
                save_path = pred_plot_path,
            )
            print(f"  Scatter plot saved → {pred_plot_path}")

        elif model_name == 'seq2seq':
            try:
                from src.utils.visualization import plot_bh_loop
                val_iter         = iter(val_loader)
                b_batch, h_batch = next(val_iter)
                b_batch          = b_batch.to(device)
                trained_model.eval()
                with torch.no_grad():
                    pred_h_batch = trained_model(b_batch)
                sample_idx = 0
                bh_plot_path = os.path.join(
                    plots_dir, f'{version_tag}_bh_loop_comparison.png'
                )
                plot_bh_loop(
                    b_batch[sample_idx].cpu().squeeze().numpy(),
                    pred_h_batch[sample_idx].cpu().squeeze().numpy(),
                    b_batch[sample_idx].cpu().squeeze().numpy(),
                    h_batch[sample_idx].cpu().squeeze().numpy(),
                    title     = f'{model_name} B-H Loop (Normalised)',
                    save_path = bh_plot_path,
                )
                print(f"  B-H loop plot saved → {bh_plot_path}")
            except Exception as e:
                print(f"  Could not plot B-H loop: {e}")

    print(f"\n{'='*20} All done {'='*20}\n")


if __name__ == '__main__':
    main()
