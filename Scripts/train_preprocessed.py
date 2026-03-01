"""
train_preprocessed.py
=====================
Training entry point that runs on **pre-processed** data produced by
prepare_datasets.py — no raw .mat file or physics computation needed at
training time.

Mirrors the full functionality of main.py:
  • Loads each model's train / val splits from .npz files
  • Instantiates the correct network from config.yaml
  • Trains with train_model()
  • Evaluates with evaluate_model()
  • Saves loss curves and prediction / B-H plots

Usage
-----
  # Train all models
  python train_preprocessed.py \\
      --processed  processed_data/ \\
      --config     Scripts/config/config.yaml

  # Train a single model
  python train_preprocessed.py \\
      --processed  processed_data/ \\
      --config     Scripts/config/config.yaml \\
      --model      cnn

  # Override epochs
  python train_preprocessed.py ... --epochs 50

Directory layout expected (output of prepare_datasets.py)
----------------------------------------------------------
  processed_data/
    <model_name>/
      train.npz
      val.npz
      test.npz
      stats.json
"""

import os
import json
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

# ── Project imports (same as main.py) ───────────────────────────────────────
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
# 2.  DATASET  (reads pre-processed .npz files)
# ══════════════════════════════════════════════════════════════════════════════

def _load_split(npz_path: str) -> dict:
    """
    Load one .npz split produced by prepare_datasets.py.

    Returns {'inputs': {name: array}, 'targets': {name: array}}
    """
    data   = np.load(npz_path)
    result = {'inputs': {}, 'targets': {}}
    for key in data.files:
        role, feat = key.split('__', 1)
        result[role][feat] = data[key]
    return result


class PreprocessedDataset(Dataset):
    """
    PyTorch Dataset backed by a single pre-processed .npz split.

    Item format per mode (matches MagNetDataset / dataset.py)
    ----------------------------------------------------------
    scaler      ->  x (n_feats,),          y (1,)
    sequence  ┐
    cnn       ├->  B (T,1),  scalars (n,), y (1,)
    transformer┘
    seq2seq     ->  B (T,1),  H (T,1)

    Parameters
    ----------
    npz_path : str   path to train.npz / val.npz / test.npz
    stats    : dict  normalization stats loaded from stats.json
    mode     : str   model mode string
    """

    def __init__(self, npz_path: str, stats: dict, mode: str):
        super().__init__()
        self.mode  = mode
        self.stats = stats

        data = _load_split(npz_path)
        self.inputs  = data['inputs']
        self.targets = data['targets']

        # Determine dataset length from first available array
        first = (next(iter(self.inputs.values()))  if self.inputs
                 else next(iter(self.targets.values())))
        self.length = len(first)

    # expose stats at dataset level so main loop can read them (same pattern
    # as MagNetDataset.stats used in main.py for CNN freq_stats)
    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx: int):
        inp = self.inputs
        tgt = self.targets

        # ── scaler: concatenate all input features → flat vector ──────────
        if self.mode == 'scaler':
            x = torch.cat([self._t(inp[f][idx]).flatten() for f in inp])
            y = self._t(tgt['Loss'][idx]).flatten()
            return x, y

        # ── sequence / cnn / transformer: waveform B + scalar context ─────
        elif self.mode in ('sequence', 'cnn', 'transformer'):
            B       = self._t(inp['B'][idx]).unsqueeze(-1)            # (T, 1)
            scalars = torch.cat([
                self._t(inp[f][idx]).flatten() for f in inp if f != 'B'
            ])
            y = self._t(tgt['Loss'][idx]).flatten()
            return B, scalars, y

        # ── seq2seq: B waveform → H waveform ─────────────────────────────
        elif self.mode == 'seq2seq':
            B = self._t(inp['B'][idx]).unsqueeze(-1)                  # (T, 1)
            H = self._t(tgt['H'][idx]).unsqueeze(-1)                  # (T, 1)
            return B, H

        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")


def build_loaders(
    model_name:     str,
    processed_root: str,
    batch_size:     int,
) -> tuple:
    """
    Build train and val DataLoaders from pre-processed .npz files.

    Returns
    -------
    train_loader, val_loader, stats  (dict loaded from stats.json)
    """
    model_dir  = os.path.join(processed_root, model_name)
    stats_path = os.path.join(model_dir, 'stats.json')

    # ── Validate paths ────────────────────────────────────────────────────
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

    train_ds = PreprocessedDataset(
        os.path.join(model_dir, 'train.npz'), stats, mode=model_name
    )
    val_ds = PreprocessedDataset(
        os.path.join(model_dir, 'val.npz'), stats, mode=model_name
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"  train samples : {len(train_ds):,}")
    print(f"  val   samples : {len(val_ds):,}")

    return train_loader, val_loader, stats


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL FACTORY  (identical logic to main.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, config: dict, stats: dict, seq_len: int = 1024):
    """
    Instantiate the correct network from config.yaml.

    Parameters
    ----------
    model_name : str   e.g. 'cnn'
    config     : dict  full config loaded from config.yaml
    stats      : dict  normalization stats from stats.json
                       (used by CNN to embed frequency info)
    seq_len    : int   waveform length — used for TransformerNetwork

    Returns
    -------
    (model, version_tag)
    """
    model_conf  = config['models'][model_name]
    version     = model_conf['version']
    version_tag = f"{model_name}_v{version}"

    # ── scaler ────────────────────────────────────────────────────────────
    if model_name == 'scaler':
        # input_dim = number of input features defined in config
        input_dim = len(
            config['models']['scaler'].get('features', {}).get('inputs', {})
        )
        if input_dim == 0:
            input_dim = 3   # legacy fallback

        model = ScalerNetwork(
            input_dim  = input_dim,
            hidden_dim = model_conf['hidden_dim'],
            num_layers = model_conf['layers'],
            output_dim = 1,
        )

    # ── sequence ─────────────────────────────────────────────────────────
    elif model_name == 'sequence':
        model = SequenceToScalerNetwork(
            input_dim  = 1,
            hidden_dim = model_conf['hidden_dim'],
            output_dim = 1,
            num_layers = model_conf['num_layers'],
        )

    # ── seq2seq ──────────────────────────────────────────────────────────
    elif model_name == 'seq2seq':
        model = Seq2SeqNetwork(
            input_dim  = 1,
            hidden_dim = model_conf['encoder_dim'],
            output_dim = 1,
        )

    # ── cnn ──────────────────────────────────────────────────────────────
    elif model_name == 'cnn':
        # Pass frequency normalization stats so CNN can embed freq correctly
        freq_stats = {}
        if 'Frequency' in stats:
            f_s = stats['Frequency']
            if 'mean' in f_s:
                freq_stats['freq_mean'] = f_s['mean']
                freq_stats['freq_std']  = f_s['std']
        print(f"  Passing stats to CNNNetwork: {freq_stats}")

        model = CNNNetwork(input_dim=1, stats=freq_stats)

    # ── transformer ──────────────────────────────────────────────────────
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
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODELS = ['scaler', 'sequence', 'seq2seq', 'cnn', 'transformer']


def main():
    parser = argparse.ArgumentParser(
        description="Train MagNet models on pre-processed data (prepare_datasets.py output)."
    )
    parser.add_argument(
        '--processed', required=True,
        help='Root directory of pre-processed data (e.g. processed_data/)'
    )
    parser.add_argument(
        '--config', type=str, default='Scripts/config/config.yaml',
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--model', type=str,
        choices=ALL_MODELS + ['all'],
        default='all',
        help='Model to train (default: all)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of epochs from config'
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    config = load_config(args.config)
    if args.epochs:
        config['training']['epochs'] = args.epochs

    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size  = config['data'].get('batch_size', 32)
    save_root   = config['training']['save_dir']

    print(f"\nDevice     : {device}")
    print(f"Batch size : {batch_size}")
    print(f"Epochs     : {config['training']['epochs']}")
    print(f"Save root  : {save_root}")

    models_to_run = ALL_MODELS if args.model == 'all' else [args.model]

    # ─────────────────────────────────────────────────────────────────────────
    for model_name in models_to_run:
        print(f"\n{'='*20} Training {model_name.upper()} Model {'='*20}")

        # ── 1. DataLoaders from pre-processed .npz ───────────────────────────
        print("Loading pre-processed data ...")
        train_loader, val_loader, stats = build_loaders(
            model_name, args.processed, batch_size
        )

        # Infer waveform sequence length from train split (used by Transformer)
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

        # ── 3. Train ──────────────────────────────────────────────────────────
        print("Starting training ...")
        save_dir = os.path.join(save_root, model_name)
        os.makedirs(save_dir, exist_ok=True)

        # Give train_model its own copy of save_dir so models don't collide
        train_config = dict(config['training'])
        train_config['save_dir'] = save_dir

        run_config = dict(config)
        run_config['training'] = train_config

        model = model.to(device)
        # train_model now evaluates the best checkpoint internally and returns
        # metrics + predictions so everything is written to the same log block.
        trained_model, history, metrics, preds, targets_out = train_model(
            model, train_loader, val_loader, model_name, run_config, device
        )
        print(f"  Validation metrics : {metrics}")

        # ── Additional relative-error statistics on the test split ────────────
        if model_name in ('scaler', 'sequence', 'cnn', 'transformer'):
            try:
                test_ds = PreprocessedDataset(
                    os.path.join(args.processed, model_name, 'test.npz'),
                    stats, mode=model_name,
                )
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=True,
                )
                trained_model.eval()
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        if model_name == 'scaler':
                            x, y = batch
                            out = trained_model(x.to(device))
                        else:
                            B, scalars, y = batch
                            out = trained_model(B.to(device), scalars.to(device))
                        all_preds.append(out.cpu().numpy())
                        all_targets.append(y.numpy())

                all_preds   = np.concatenate(all_preds).flatten()
                all_targets = np.concatenate(all_targets).flatten()

                rel_errors    = np.abs((all_preds - all_targets) / (np.abs(all_targets) + 1e-8))
                p95_rel_error = float(np.percentile(rel_errors, 95))
                max_rel_error = float(np.max(rel_errors))

                print(f"  Test 95th-pct relative error : {p95_rel_error:.6f}")
                print(f"  Test max relative error       : {max_rel_error:.6f}")
                metrics['test_p95_relative_error'] = p95_rel_error
                metrics['test_max_relative_error'] = max_rel_error

                # ── Append test metrics to the same experiments.txt log ───────
                model_cfg        = config['models'][model_name]
                model_name_clean = model_cfg['name']
                experiment_log_path = os.path.join(save_dir, model_name_clean, 'experiments.txt')
                # Fallback: log lives directly in save_dir when train.py uses save_dir as model_dir
                if not os.path.exists(experiment_log_path):
                    experiment_log_path = os.path.join(save_dir, 'experiments.txt')

                with open(experiment_log_path, 'a') as f:
                    f.write("Test Set Relative-Error Metrics:\n")
                    f.write(f"  P95 Relative Error: {p95_rel_error:.6f}\n")
                    f.write(f"  Max Relative Error: {max_rel_error:.6f}\n")
                    f.write("=" * 70 + "\n\n")

            except FileNotFoundError:
                print("  test.npz not found — skipping relative-error statistics.")

        # ── 5. Plots ──────────────────────────────────────────────────────────
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Loss curve (all models)
        loss_plot_path = os.path.join(plots_dir, f'{version_tag}_loss_curve.png')
        plot_loss_curve(
            history,
            title     = f'{model_name} Training Loss',
            save_path = loss_plot_path,
        )
        print(f"  Loss plot saved → {loss_plot_path}")

        # Prediction scatter (all models except seq2seq)
        if model_name in ('scaler', 'sequence', 'cnn', 'transformer'):
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

        # B-H loop plot (seq2seq only)
        elif model_name == 'seq2seq':
            try:
                from src.utils.visualization import plot_bh_loop

                val_iter = iter(val_loader)
                b_batch, h_batch = next(val_iter)
                b_batch = b_batch.to(device)

                trained_model.eval()
                with torch.no_grad():
                    pred_h_batch = trained_model(b_batch)

                sample_idx = 0
                actual_b = b_batch[sample_idx].cpu().squeeze().numpy()
                actual_h = h_batch[sample_idx].cpu().squeeze().numpy()
                pred_h   = pred_h_batch[sample_idx].cpu().squeeze().numpy()

                bh_plot_path = os.path.join(
                    plots_dir, f'{version_tag}_bh_loop_comparison.png'
                )
                plot_bh_loop(
                    actual_b, pred_h,
                    actual_b, actual_h,
                    title     = f'{model_name} B-H Loop (Normalised)',
                    save_path = bh_plot_path,
                )
                print(f"  B-H loop plot saved → {bh_plot_path}")

            except Exception as e:
                print(f"  Could not plot B-H loop: {e}")

    print(f"\n{'='*20} All done {'='*20}\n")


if __name__ == '__main__':
    main()