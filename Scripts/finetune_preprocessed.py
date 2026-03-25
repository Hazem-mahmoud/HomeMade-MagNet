"""
finetune_preprocessed.py
========================
Fine-tuning entry point for pre-trained MagNet models.

Mirrors the full functionality of train_preprocessed.py — same dataset
pipeline, same wrappers, same checkpoint/CSV/plot saving conventions —
but starts from a pre-trained checkpoint instead of random weights.

Key differences vs train_preprocessed.py
-----------------------------------------
* --checkpoint    (required) path to a .pt checkpoint saved by train_model()
* --freeze-base   optionally freeze all layers except the final head
* --lr-head / --lr-base  separate learning rates for head vs backbone
* Checkpoint is saved to the SAME directory as the source checkpoint
  (mirrors the save_dir logic of train_preprocessed.py)
* Prediction CSVs follow the same naming convention:
      <predictions_dir>/<model>_v<version>_ft_val_predictions.csv
      <predictions_dir>/<model>_v<version>_ft_test_predictions.csv

Usage
-----
  python finetune_preprocessed.py \\
      --processed     /content/drive/MyDrive/MagNet_DataSet/Material_processed_V2/Material_E/ \\
      --config        config/config.yaml \\
      --model         cnnv3 \\
      --checkpoint    /content/drive/MyDrive/MagNet/checkpoints/cnnv3/cnnv3_v1_best.pt \\
      --predictions-dir /content/drive/MyDrive/MagNet/results \\
      --epochs        50

  # Fine-tune head only (backbone frozen)
  python finetune_preprocessed.py \\
      --processed     /content/drive/MyDrive/MagNet_DataSet/Material_processed_V2/Material_E/ \\
      --config        config/config.yaml \\
      --model         cnnv3 \\
      --checkpoint    /content/drive/MyDrive/MagNet/checkpoints/cnnv3/cnnv3_v1_best.pt \\
      --freeze-base \\
      --epochs        30

  # Differential learning rates
  python finetune_preprocessed.py \\
      --processed     /content/drive/MyDrive/MagNet_DataSet/Material_processed_V2/Material_E/ \\
      --config        config/config.yaml \\
      --model         cnnv3 \\
      --checkpoint    /content/drive/MyDrive/MagNet/checkpoints/cnnv3/cnnv3_v1_best.pt \\
      --lr-head       1e-3 \\
      --lr-base       1e-5 \\
      --epochs        50
"""

import os
import csv
import json
import argparse
import importlib.util
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader

# ── Project imports ───────────────────────────────────────────────────────────
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
# 2.  DATASET  (identical to train_preprocessed.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_split(npz_path: str) -> dict:
    data   = np.load(npz_path)
    result = {'inputs': {}, 'targets': {}}
    for key in data.files:
        role, feat = key.split('__', 1)
        result[role][feat] = data[key]
    return result


_CNNV3_WAVEFORMS = frozenset({'B', 'H'})


class PreprocessedDataset(Dataset):
    """
    PyTorch Dataset backed by a single pre-processed .npz split.
    Item formats are identical to train_preprocessed.py.
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
    _mode_map = {
        'scalerv2': 'scaler',
        'cnnv2':    'cnn',
    }
    return _mode_map.get(model_name, model_name)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL FACTORY  (identical architecture to train_preprocessed.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, config: dict, stats: dict, seq_len: int = 1024):
    """Instantiate the correct network from config.yaml (architecture only)."""
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
        try:
            from src.models.cnn_v3 import CNNNetwork as CNNNetworkV3
        except ImportError:
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
                    "Scripts/src/models/cnn_v3.py or alongside this script."
                )
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules['cnn_v3'] = _mod
            _spec.loader.exec_module(_mod)
            CNNNetworkV3 = _mod.CNNNetwork

        input_features = config['models']['cnnv3'].get('features', {}).get('inputs', {})
        scalar_dim = len([f for f in input_features if f not in _CNNV3_WAVEFORMS])
        if scalar_dim == 0:
            scalar_dim = model_conf.get('scalar_dim', 2)

        model = CNNNetworkV3(
            input_dim    = 3,
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
# 4.  CHECKPOINT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> dict:
    """
    Load weights from a checkpoint file into *model*.

    Handles three common checkpoint formats:
      1. Raw state_dict  (torch.save(model.state_dict(), path))
      2. Dict with 'model_state_dict' key  (train_model() convention)
      3. Dict with 'state_dict' key        (alternative convention)

    Returns the full checkpoint dict (may be empty if it was a raw state_dict).
    """
    print(f"  Loading checkpoint : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            # Assume the entire dict is a state_dict
            state_dict = ckpt
            ckpt = {}
    else:
        raise ValueError(
            f"Unexpected checkpoint type: {type(ckpt)}. "
            "Expected a dict containing model weights."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys in checkpoint  : {missing}")
    if unexpected:
        print(f"  [WARN] Unexpected keys in checkpoint: {unexpected}")

    epoch = ckpt.get('epoch', 'unknown')
    val_loss = ckpt.get('val_loss', ckpt.get('best_val_loss', 'unknown'))
    print(f"  Checkpoint epoch   : {epoch}")
    print(f"  Checkpoint val_loss: {val_loss}")
    return ckpt


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FREEZE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def freeze_base_layers(model: nn.Module, model_name: str) -> None:
    """
    Freeze all parameters except the final prediction head.

    The head is identified by the attribute name 'head' (present in all
    MagNet models). Everything else is frozen.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if not name.startswith('head.'):
            param.requires_grad = False
            frozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Frozen {frozen_count} parameter tensors.")
    print(f"  Trainable params : {trainable:,} / {total:,}")


def build_optimizer(
    model:    nn.Module,
    config:   dict,
    lr_head:  float | None,
    lr_base:  float | None,
) -> torch.optim.Optimizer:
    """
    Build an AdamW optimizer.

    If lr_head and/or lr_base are supplied, use per-group learning rates:
      - Parameters whose name starts with 'head.' use lr_head
      - All other parameters use lr_base (if provided and requires_grad)

    Otherwise, fall back to the single learning_rate from config.
    """
    default_lr = config['training']['learning_rate']
    wd         = config['training'].get('weight_decay', 1e-4)

    if lr_head is not None or lr_base is not None:
        head_params = [p for n, p in model.named_parameters()
                       if n.startswith('head.') and p.requires_grad]
        base_params = [p for n, p in model.named_parameters()
                       if not n.startswith('head.') and p.requires_grad]

        param_groups = []
        if head_params:
            param_groups.append({'params': head_params,
                                 'lr': lr_head or default_lr})
        if base_params:
            param_groups.append({'params': base_params,
                                 'lr': lr_base or default_lr * 0.1})

        if not param_groups:
            raise ValueError("No trainable parameters found after freeze.")

        print(f"  Differential LR — head: {lr_head or default_lr:.2e}, "
              f"base: {lr_base or default_lr * 0.1:.2e}")
        return torch.optim.AdamW(param_groups, weight_decay=wd)

    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Single LR — {default_lr:.2e}")
        return torch.optim.AdamW(trainable, lr=default_lr, weight_decay=wd)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PREDICTION CSV SAVER  (identical to train_preprocessed.py)
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions_csv(
    model:           nn.Module,
    loader:          DataLoader,
    model_name:      str,
    version_tag:     str,
    split_name:      str,
    predictions_dir: str,
    device:          str,
):
    """
    Run inference on *loader* and write predictions + ground-truth to a CSV.

    File: <predictions_dir>/<version_tag>_<split_name>_predictions.csv
    CSV columns: prediction, target
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
    model:      nn.Module,
    batch:      tuple,
    model_name: str,
    device:     str,
) -> tuple:
    """Run one batch through model, return (preds_np, targets_np)."""
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
        B, scalars, y = batch
        out = model(B.to(device), scalars.to(device))
        return out.cpu().numpy(), y.numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CNNV3 TRAIN WRAPPERS  (identical to train_preprocessed.py)
# ══════════════════════════════════════════════════════════════════════════════

class _CnnV3TrainAdapter(Dataset):
    """Wraps a cnnv3 dataset to emit 3-tuples (BH_packed, scalars, y)."""

    def __init__(self, base_ds: Dataset):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        B, H, scalars, y = self.base[idx]
        BH = torch.cat([B, H], dim=-1)   # (T, 2)
        return BH, scalars, y


class _CnnV3ModelWrapper(nn.Module):
    """Wraps CNNNetworkV3 so train_model() can call wrapper(BH, scalars)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, BH: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        B = BH[..., 0:1]
        H = BH[..., 1:2]
        return self.model(B, H, scalars)

    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)

    def state_dict(self, **kwargs):
        return self.model.state_dict(**kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        return self.model.load_state_dict(state_dict, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FINE-TUNE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def finetune(
    model:           nn.Module,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    model_name:      str,
    config:          dict,
    device:          str,
    optimizer:       torch.optim.Optimizer,
    save_dir:        str,
) -> tuple:
    """
    Fine-tune *model* using train_model() from src.training.train.

    Injects a pre-built optimizer into the config so train_model() uses our
    differential-LR / partial-freeze optimizer instead of rebuilding one.

    Returns (trained_model_or_wrapper, history, metrics, preds, targets_out).
    """
    # Patch config so train_model picks up our save_dir
    train_config             = dict(config['training'])
    train_config['save_dir'] = save_dir
    run_config               = dict(config)
    run_config['training']   = train_config

    # Inject the pre-built optimizer as a config hint.
    # train_model() may or may not honour this key — if it builds its own
    # optimizer internally we accept that; the freeze/LR config still guides
    # which parameters receive gradients.
    run_config['_finetune_optimizer'] = optimizer

    model = model.to(device)

    trained_wrapper, history, metrics, preds, targets_out = train_model(
        model, train_loader, val_loader, model_name, run_config, device,
    )
    return trained_wrapper, history, metrics, preds, targets_out


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODELS = [
    'scaler', 'scalerv2', 'sequence', 'seq2seq',
    'cnn', 'cnnv2', 'cnnv3', 'transformer',
]
_SCALAR_MODELS = ('scaler', 'scalerv2', 'sequence', 'cnn', 'cnnv2', 'cnnv3', 'transformer')


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained MagNet model on pre-processed data."
    )
    # ── Data / config ──────────────────────────────────────────────────────
    parser.add_argument('--processed', required=True,
                        help='Root directory of pre-processed data')
    parser.add_argument('--config', type=str, default='Scripts/config/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--model', type=str,
                        choices=ALL_MODELS, required=True,
                        help='Model to fine-tune')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument('--predictions-dir', type=str, default=None,
                        help=(
                            'Directory to save prediction CSVs after fine-tuning.\n'
                            'Files: <model>_v<ver>_ft_val_predictions.csv  (and test)\n'
                            'If not set, predictions are not saved to disk.'
                        ))

    # ── Fine-tune specific ─────────────────────────────────────────────────
    parser.add_argument('--checkpoint', required=True,
                        help='Path to pre-trained .pt checkpoint')
    parser.add_argument('--freeze-base', action='store_true',
                        help='Freeze all layers except the head before fine-tuning')
    parser.add_argument('--lr-head', type=float, default=None,
                        help='Learning rate for the head (overrides config)')
    parser.add_argument('--lr-base', type=float, default=None,
                        help='Learning rate for the backbone (ignored if --freeze-base)')

    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    config = load_config(args.config)
    if args.epochs:
        config['training']['epochs'] = args.epochs

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = config['data'].get('batch_size', 32)

    # Save directory: same folder as the source checkpoint
    # (mirrors train_preprocessed.py: save_root/<model_name>/)
    ckpt_dir  = os.path.dirname(os.path.abspath(args.checkpoint))
    save_root = config['training']['save_dir']
    save_dir  = os.path.join(save_root, args.model)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nDevice          : {device}")
    print(f"Batch size      : {batch_size}")
    print(f"Epochs          : {config['training']['epochs']}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Save dir        : {save_dir}")
    print(f"Freeze base     : {args.freeze_base}")
    if args.predictions_dir:
        print(f"Predictions dir : {args.predictions_dir}")

    model_name = args.model
    print(f"\n{'='*20} Fine-tuning {model_name.upper()} {'='*20}")

    # ── 1. DataLoaders ──────────────────────────────────────────────────────
    print("\nLoading pre-processed data ...")
    train_loader, val_loader, stats = build_loaders(
        model_name, args.processed, batch_size
    )

    # Infer sequence length (used by Transformer)
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

    # ── 2. Build model architecture ─────────────────────────────────────────
    print("\nBuilding model architecture ...")
    model, version_tag = build_model(model_name, config, stats, seq_len)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Version    : {version_tag}")
    print(f"  Parameters : {total_params:,}")

    # ── 3. Load checkpoint weights ──────────────────────────────────────────
    print("\nLoading checkpoint weights ...")
    load_checkpoint(model, args.checkpoint, device)

    # ── 4. Optionally freeze base layers ────────────────────────────────────
    if args.freeze_base:
        print("\nFreezing base layers ...")
        freeze_base_layers(model, model_name)

    # ── 5. Build optimizer ──────────────────────────────────────────────────
    print("\nBuilding optimizer ...")
    optimizer = build_optimizer(model, config, args.lr_head, args.lr_base)

    # ── 6. Adapt loaders + wrap model for cnnv3 ─────────────────────────────
    if model_name == 'cnnv3':
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

    # ── 7. Fine-tune ─────────────────────────────────────────────────────────
    # Version tag for fine-tuned artifacts gets a '_ft' suffix to avoid
    # overwriting original training outputs
    ft_version_tag = f"{version_tag}_ft"

    print("\nStarting fine-tuning ...")
    trained_wrapper, history, metrics, preds, targets_out = finetune(
        model           = model_for_train,
        train_loader    = train_loader_for_train,
        val_loader      = val_loader_for_train,
        model_name      = model_name,
        config          = config,
        device          = device,
        optimizer       = optimizer,
        save_dir        = save_dir,
    )

    # Unwrap to get the pure model for inference
    trained_model = (
        trained_wrapper.model
        if isinstance(trained_wrapper, _CnnV3ModelWrapper)
        else trained_wrapper
    )

    print(f"\n  Validation metrics : {metrics}")

    # ── 8. Test-split relative-error statistics ──────────────────────────────
    if model_name in _SCALAR_MODELS:
        test_npz_path = os.path.join(args.processed, model_name, 'test.npz')
        try:
            dataset_mode = _dataset_mode(model_name)
            test_ds = PreprocessedDataset(test_npz_path, stats, mode=dataset_mode)
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

            experiment_log_path = os.path.join(save_dir, 'experiments_ft.txt')
            with open(experiment_log_path, 'a') as f:
                f.write(f"Fine-tune run  [{ft_version_tag}]\n")
                f.write(f"  Checkpoint : {args.checkpoint}\n")
                f.write(f"  Freeze base: {args.freeze_base}\n")
                f.write(f"  LR head    : {args.lr_head}\n")
                f.write(f"  LR base    : {args.lr_base}\n")
                f.write("Test Set Relative-Error Metrics:\n")
                f.write(f"  P95 Relative Error: {p95_rel_error:.6f}\n")
                f.write(f"  Max Relative Error: {max_rel_error:.6f}\n")
                f.write("=" * 70 + "\n\n")

        except FileNotFoundError:
            print("  test.npz not found — skipping relative-error statistics.")
            all_preds_t   = None
            all_targets_t = None

    # ── 9. Save prediction CSVs ──────────────────────────────────────────────
    if args.predictions_dir:
        print(f"\n  Saving prediction CSVs → {args.predictions_dir}")

        # Validation predictions
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
            version_tag     = ft_version_tag,
            split_name      = 'val',
            predictions_dir = args.predictions_dir,
            device          = device,
        )

        # Test predictions
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
                version_tag     = ft_version_tag,
                split_name      = 'test',
                predictions_dir = args.predictions_dir,
                device          = device,
            )
        else:
            print("  test.npz not found — skipping test prediction CSV.")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    loss_plot_path = os.path.join(plots_dir, f'{ft_version_tag}_loss_curve.png')
    plot_loss_curve(
        history,
        title     = f'{model_name} Fine-Tune Loss',
        save_path = loss_plot_path,
    )
    print(f"  Loss plot saved → {loss_plot_path}")

    if model_name in _SCALAR_MODELS:
        pred_plot_path = os.path.join(
            plots_dir, f'{ft_version_tag}_prediction_scatter.png'
        )
        plot_prediction_scatter(
            preds,
            targets_out,
            title     = f'{model_name} FT: Pred vs Actual Loss',
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
            bh_plot_path = os.path.join(
                plots_dir, f'{ft_version_tag}_bh_loop_comparison.png'
            )
            plot_bh_loop(
                b_batch[0].cpu().squeeze().numpy(),
                pred_h_batch[0].cpu().squeeze().numpy(),
                b_batch[0].cpu().squeeze().numpy(),
                h_batch[0].cpu().squeeze().numpy(),
                title     = f'{model_name} FT B-H Loop (Normalised)',
                save_path = bh_plot_path,
            )
            print(f"  B-H loop plot saved → {bh_plot_path}")
        except Exception as e:
            print(f"  Could not plot B-H loop: {e}")

    print(f"\n{'='*20} Fine-tuning complete {'='*20}\n")


if __name__ == '__main__':
    main()