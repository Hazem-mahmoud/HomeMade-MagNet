"""
inference.py
============
Two inference modes:

  1. predict_single()
     Takes a raw user-provided sample (physical units), normalizes it using
     the training stats, runs the model, then decodes the output back to
     real physical units (W/m³).

  2. predict_from_npz()
     Reads the representative_test.npz (already normalized), runs the model
     on every sample, decodes the predictions, and appends a column named
     after the model to a CSV file:

         sample_index  | <model_name>
         --------------|-------------
         88140         | 45231.7
         9783          | 12087.3
         ...

Usage — command line
--------------------
  # Single sample inference (scaler model example)
  python inference.py single \\
      --model-name  scaler \\
      --model-path  checkpoints/scaler/scaler_v2/scaler_v2_best.pth \\
      --stats-path  preprocessing_summary.json \\
      --config      Scripts/config/config.yaml \\
      --input       B_pk=0.18 Frequency=100000 Temperature=60 Hdc=10

  # Batch inference from representative_test.npz → CSV
  python inference.py batch \\
      --model-name  cnn \\
      --model-path  checkpoints/cnn/cnn_v1/cnn_v1_best.pth \\
      --stats-path  preprocessing_summary.json \\
      --config      Scripts/config/config.yaml \\
      --npz-path    representative_sample/cnn/representative_test.npz \\
      --indices-json representative_sample/representative_test_indices.json \\
      --output-csv  results/predictions.csv

Usage — Python API
------------------
  from inference import ModelInference

  infer = ModelInference(
      model_name  = 'scaler',
      model_path  = 'checkpoints/scaler/scaler_v2/scaler_v2_best.pth',
      stats_path  = 'preprocessing_summary.json',
      config_path = 'Scripts/config/config.yaml',
  )

  # Single sample
  result = infer.predict_single({'B_pk': 0.18, 'Frequency': 100000,
                                  'Temperature': 60, 'Hdc': 10})
  print(result)
  # {'Loss_W_per_m3': 43251.8, 'Loss_normalized': 0.3412, ...}

  # Batch from npz
  infer.predict_from_npz(
      npz_path     = 'representative_sample/scaler/representative_test.npz',
      indices_json = 'representative_sample/representative_test_indices.json',
      output_csv   = 'results/predictions.csv',
  )
"""

import os
import json
import argparse

import numpy as np
import torch
import yaml
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NORMALIZATION  (mirrors prepare_datasets.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_value(value: np.ndarray, method: str, stats: dict) -> np.ndarray:
    """
    Apply normalization to a raw value using pre-fitted stats.

    Parameters
    ----------
    value  : np.ndarray  raw physical value(s)
    method : str         'standard' | 'minmax' | 'log10' | 'none'
    stats  : dict        keys depend on method:
                           standard → {'mean': float, 'std': float}
                           minmax   → {'min': float, 'max': float}
                           log10    → {}  (no params needed)
                           none     → {}

    Returns
    -------
    np.ndarray  normalized value(s)
    """
    value = np.array(value, dtype=np.float32)

    if method == 'none':
        return value

    if method == 'standard':
        return (value - stats['mean']) / stats['std']

    if method == 'minmax':
        denom = stats['max'] - stats['min']
        if denom == 0:
            denom = 1.0
        return (value - stats['min']) / denom

    if method == 'log10':
        return np.log10(np.abs(value) + 1e-6)

    raise ValueError(f"Unknown normalization method: '{method}'")


def denormalize_value(value: np.ndarray, method: str, stats: dict) -> np.ndarray:
    """
    Invert normalization to recover real physical units.

    Parameters
    ----------
    value  : np.ndarray  normalized / transformed value(s)
    method : str         must match the method used during normalization
    stats  : dict        same stats dict used during normalization

    Returns
    -------
    np.ndarray  original-scale value(s)
    """
    value = np.array(value, dtype=np.float64)

    if method == 'none':
        return value

    if method == 'standard':
        return value * stats['std'] + stats['mean']

    if method == 'minmax':
        return value * (stats['max'] - stats['min']) + stats['min']

    if method == 'log10':
        return np.power(10.0, value)

    raise ValueError(f"Unknown normalization method: '{method}'")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _build_model_from_state_dict(model_name: str, state_dict: dict,
                                  feature_stats: dict, seq_len: int) -> 'nn.Module':
    """
    Reconstruct the exact model architecture purely from the checkpoint state dict.
    Works for any ScalerNetwork depth/width, with or without BatchNorm.
    No config.yaml required.
    """
    import torch.nn as nn

    # ── ScalerNetwork: build Sequential directly from state dict keys ─────
    if model_name == 'scaler':
        # Parse all indexed layers  {idx: {suffix: tensor}}
        layer_info = {}
        for full_key, tensor in state_dict.items():
            parts = full_key.split('.')
            if parts[0] != 'model':
                continue
            idx    = int(parts[1])
            suffix = '.'.join(parts[2:])
            layer_info.setdefault(idx, {})[suffix] = tensor

        seq = nn.Sequential()
        for idx in sorted(layer_info.keys()):
            info = layer_info[idx]
            if 'running_mean' in info:
                # BatchNorm1d
                num_f = info['running_mean'].shape[0]
                seq.add_module(str(idx), nn.BatchNorm1d(num_f))
                seq.add_module(f'relu_{idx}', nn.ReLU())
            elif 'weight' in info and info['weight'].ndim == 2:
                # Linear
                out_f, in_f = info['weight'].shape
                seq.add_module(str(idx), nn.Linear(in_f, out_f))

        # Wrap in a module with a .model attribute so load_state_dict works
        class ScalerWrapper(nn.Module):
            def __init__(self, sequential):
                super().__init__()
                self.model = sequential
            def forward(self, x):
                return self.model(x)

        return ScalerWrapper(seq)

    # ── All other models: use original constructors (unchanged) ──────────
    from src.models.sequence_model    import SequenceToScalerNetwork
    from src.models.seq2seq_model     import Seq2SeqNetwork
    from src.models.cnn_model         import CNNNetwork
    from src.models.transformer_model import TransformerNetwork

    if model_name == 'sequence':
        w = next(v for k, v in state_dict.items() if 'weight_ih_l0' in k)
        hidden = int(w.shape[0] // 4)
        nlayers = sum(1 for k in state_dict if 'weight_ih_l' in k)
        return SequenceToScalerNetwork(input_dim=1, hidden_dim=hidden,
                                       output_dim=1, num_layers=nlayers)

    elif model_name == 'seq2seq':
        w = next(v for k, v in state_dict.items() if 'weight_ih_l0' in k)
        hidden = int(w.shape[0] // 4)
        return Seq2SeqNetwork(input_dim=1, hidden_dim=hidden, output_dim=1)

    elif model_name == 'cnn':
        freq_stats = {}
        if 'Frequency' in feature_stats:
            fs = feature_stats['Frequency']
            if 'mean' in fs:
                freq_stats = {'freq_mean': fs['mean'], 'freq_std': fs['std']}
        return CNNNetwork(input_dim=1, stats=freq_stats)

    elif model_name == 'transformer':
        # d_model from first 2-D weight
        dim_hidden = next(
            (int(v.shape[-1]) for k, v in state_dict.items()
             if 'weight' in k and v.ndim == 2), 64
        )
        n_enc = sum(1 for k in state_dict if 'self_attn.in_proj_weight' in k)
        ff_dim = next(
            (int(v.shape[0]) for k, v in state_dict.items() if 'linear1.weight' in k),
            256
        )
        return TransformerNetwork(
            B_in_channel=seq_len, dim_hidden=dim_hidden,
            n_encoder_layers=max(n_enc, 1), dim_feedforward_encoder=ff_dim,
            n_heads=8, dropout_encoder=0.0,
        )

    else:
        raise ValueError(f"Unknown model name '{model_name}'.")


def load_model(model_name: str, model_path: str, feature_stats: dict,
               n_inputs: int = 4, seq_len: int = 1024, config: dict = None):
    """
    Load a trained model checkpoint.

    Architecture is reconstructed directly from the state dict weight shapes —
    no config.yaml needed. If config is provided it is used as a fallback for
    models whose architecture cannot be fully inferred (e.g. Transformer nhead).

    Parameters
    ----------
    model_name    : str   'scaler'|'sequence'|'seq2seq'|'cnn'|'transformer'
    model_path    : str   path to .pth checkpoint file
    feature_stats : dict  per-feature stats (for CNNNetwork freq embedding)
    n_inputs      : int   number of scalar input features (scaler model)
    seq_len       : int   waveform length (Transformer B_in_channel)
    config        : dict  optional config.yaml dict (used only for Transformer nhead)

    Returns
    -------
    torch.nn.Module  eval mode, on CPU
    """
    import torch

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # If config available and has model section, use it for Transformer nhead
    if (config and 'models' in config and model_name in config['models']
            and model_name == 'transformer'):
        cfg = config['models'][model_name]
        from src.models.transformer_model import TransformerNetwork
        model = TransformerNetwork(
            B_in_channel            = seq_len,
            dim_hidden              = cfg['d_model'],
            n_encoder_layers        = cfg['num_layers'],
            dim_feedforward_encoder = cfg['dim_feedforward'],
            n_heads                 = cfg['nhead'],
            dropout_encoder         = cfg['dropout'],
        )
        print(f"  Architecture source: config.yaml")
    else:
        print(f"  Architecture source: inferred from checkpoint weights")
        model = _build_model_from_state_dict(
            model_name, state_dict, feature_stats, seq_len
        )

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Weights loaded  ←  {model_path}")
    return model

def build_input_tensors(
    model_name:      str,
    norm_inputs:     dict,          # {feature_name: np.ndarray (N, ...)}
) -> tuple:
    """
    Pack normalized feature arrays into the tuple of tensors expected by
    each model's forward() call.

    Returns
    -------
    tuple  matching what train.py's training loop unpacks:
      scaler      → (x,)             x shape (N, n_feats)
      sequence/
      cnn/
      transformer → (B, scalars)     B (N, T, 1), scalars (N, n_scalar)
      seq2seq     → (B,)             B (N, T, 1)
    """
    def t(arr):
        return torch.tensor(np.array(arr, dtype=np.float32))

    if model_name == 'scaler':
        # Concatenate all input features into a flat vector per sample
        parts = [t(norm_inputs[f]).reshape(-1, 1) for f in norm_inputs]
        x = torch.cat(parts, dim=1)                          # (N, n_feats)
        return (x,)

    elif model_name in ('sequence', 'cnn', 'transformer'):
        B       = t(norm_inputs['B']).unsqueeze(-1)           # (N, T, 1)
        scalar_keys = [k for k in norm_inputs if k != 'B']
        scalars = torch.cat(
            [t(norm_inputs[k]).reshape(-1, 1) for k in scalar_keys], dim=1
        )                                                     # (N, n_scalar)
        return (B, scalars)

    elif model_name == 'seq2seq':
        B = t(norm_inputs['B']).unsqueeze(-1)                 # (N, T, 1)
        return (B,)

    else:
        raise ValueError(f"Unknown model_name '{model_name}'.")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN INFERENCE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ModelInference:
    """
    Wraps a trained model with its normalization stats for easy inference.

    Parameters
    ----------
    model_name  : str        'scaler' | 'sequence' | 'seq2seq' | 'cnn' | 'transformer'
    model_path  : str        path to .pth checkpoint
    stats_path  : str        path to preprocessing_summary.json
    config_path : str | None path to config.yaml  ← OPTIONAL
                             If omitted, architecture is inferred automatically
                             from the checkpoint weight shapes.
    device      : str        'cpu' or 'cuda'
    seq_len     : int        waveform length (needed for Transformer)
    """

    def __init__(
        self,
        model_name:  str,
        model_path:  str,
        stats_path:  str,
        config_path: str = None,    # ← now optional
        device:      str = 'cpu',
        seq_len:     int = 1024,
    ):
        self.model_name = model_name
        self.device     = device

        # ── Load config (optional) ────────────────────────────────────────
        self.config = None
        if config_path is not None:
            if not os.path.exists(config_path):
                print(f"  WARNING: config_path '{config_path}' not found — "
                      f"architecture will be inferred from checkpoint.")
            else:
                with open(config_path) as f:
                    self.config = yaml.safe_load(f)
                print(f"  Config loaded from: {config_path}")

        # ── Load stats — supports two file formats ────────────────────────
        #
        #  Format A  preprocessing_summary.json  (all models in one file)
        #    { "scaler": { "inputs": {...}, "targets": {...}, "stats": {...} }, ... }
        #
        #  Format B  per-model stats.json  (single model, passed directly)
        #    { "B_pk": {"method": "standard", "mean": ..., "std": ...}, ... }
        #
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        with open(stats_path) as f:
            summary = json.load(f)

        # ── Detect which format was passed ────────────────────────────────
        first_val = next(iter(summary.values()))

        if isinstance(first_val, dict) and 'inputs' in first_val:
            # Format A: preprocessing_summary.json — nested under model name
            if model_name not in summary:
                raise KeyError(
                    f"Model '{model_name}' not found in {stats_path}.\n"
                    f"Available models: {list(summary.keys())}"
                )
            model_summary        = summary[model_name]
            self.input_features  = model_summary['inputs']
            self.target_features = model_summary['targets']
            self.feature_stats   = model_summary['stats']
            print(f"  Stats source : preprocessing_summary.json  (model='{model_name}')")

        else:
            # Format B: per-model stats.json — feature names are top-level keys
            # e.g. {"B_pk": {"method": "standard", "mean": 0.21, "std": 0.17}, ...}
            self.feature_stats = summary
            _target_names = {'Loss', 'H'}
            self.input_features  = {
                feat: info['method']
                for feat, info in summary.items()
                if feat not in _target_names
            }
            self.target_features = {
                feat: info['method']
                for feat, info in summary.items()
                if feat in _target_names
            }
            print(f"  Stats source : per-model stats.json")


        print(f"\n  Model        : {model_name}")
        print(f"  Input feats  : {self.input_features}")
        print(f"  Target feats : {self.target_features}")

        # ── Load model (config optional — arch inferred from weights if absent) ──
        n_inputs = len(self.input_features)
        self.model = load_model(
            model_name   = model_name,
            model_path   = model_path,
            feature_stats = self.feature_stats,
            n_inputs     = n_inputs,
            seq_len      = seq_len,
            config       = self.config,
        ).to(device)

    # ─────────────────────────────────────────────────────────────────────────
    # 4a.  Normalize a raw user input dict
    # ─────────────────────────────────────────────────────────────────────────
    def _normalize_inputs(self, raw_inputs: dict) -> dict:
        """
        Normalize raw physical-unit values using the training stats.

        Parameters
        ----------
        raw_inputs : {feature_name: scalar or np.ndarray}
                     Physical-unit values, e.g. {'Frequency': 100000, 'B_pk': 0.18}

        Returns
        -------
        {feature_name: np.ndarray (1, ...) or (1, T)}  normalized
        """
        norm = {}
        for feat, method in self.input_features.items():
            if feat not in raw_inputs:
                raise KeyError(
                    f"Missing input feature '{feat}' for model '{self.model_name}'.\n"
                    f"Required inputs: {list(self.input_features.keys())}"
                )
            stats = {k: v for k, v in self.feature_stats.get(feat, {}).items()
                     if k != 'method'}
            val   = np.array(raw_inputs[feat], dtype=np.float32)
            # Ensure at least shape (1, ...) for batch dim
            if val.ndim == 0:
                val = val.reshape(1)
            elif val.ndim == 1 and feat in ('B', 'H'):
                val = val.reshape(1, -1)     # (1, T) — single waveform
            elif val.ndim == 1:
                val = val.reshape(1, 1)      # (1, 1) — single scalar
            norm[feat] = normalize_value(val, method, stats)
        return norm

    # ─────────────────────────────────────────────────────────────────────────
    # 4b.  Decode normalized model output → real units
    # ─────────────────────────────────────────────────────────────────────────
    def _decode_output(self, raw_output: np.ndarray) -> dict:
        """
        Invert the target normalization to get real physical values.

        For Loss:
          standard → multiply by std, add mean  → W/m³
          log10    → 10^value                   → W/m³
          none     → unchanged

        Returns
        -------
        dict  {target_name: {'normalized': float, 'real': float, 'unit': str}}
        """
        results = {}
        out_flat = raw_output.flatten()
        for i, (target, method) in enumerate(self.target_features.items()):
            norm_val = float(out_flat[i]) if i < len(out_flat) else float(out_flat[0])
            stats    = {k: v for k, v in self.feature_stats.get(target, {}).items()
                        if k != 'method'}
            real_val = float(denormalize_value(np.array([norm_val]), method, stats)[0])
            results[target] = {
                'normalized': norm_val,
                'real':       real_val,
                'unit':       'W/m³' if target == 'Loss' else '',
                'method':     method,
            }
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # 4c.  PUBLIC: single sample inference
    # ─────────────────────────────────────────────────────────────────────────
    def predict_single(self, raw_inputs: dict) -> dict:
        """
        Predict for one user-provided sample in physical units.

        Parameters
        ----------
        raw_inputs : dict   Physical-unit values keyed by feature name.
                            Scalars: float.  Waveforms (B, H): list or np.ndarray.
                            Examples per model:
                              scaler  → {'B_pk': 0.18, 'Frequency': 100000,
                                         'Temperature': 60, 'Hdc': 10}
                              cnn     → {'B': [0.1, 0.08, ...], 'Frequency': 100000,
                                         'Temperature': 60, 'Hdc': 10}
                              seq2seq → {'B': [0.1, 0.08, ...]}

        Returns
        -------
        dict  {
          'model'      : str,
          'inputs_raw' : dict  (original values),
          'inputs_norm': dict  (normalized values seen by model),
          'Loss'       : {
              'normalized': float,
              'real'       : float,   ← W/m³
              'unit'       : 'W/m³',
              'method'     : str,
          }
        }
        """
        print(f"\n{'─'*50}")
        print(f"  Single-sample inference — {self.model_name}")
        print(f"{'─'*50}")
        print(f"  Raw inputs: {raw_inputs}")

        # 1. Normalize
        norm_inputs = self._normalize_inputs(raw_inputs)
        print(f"  Normalized: { {k: float(v.flatten()[0]) for k, v in norm_inputs.items()} }")

        # 2. Build tensors
        tensors = build_input_tensors(self.model_name, norm_inputs)
        tensors = tuple(t.to(self.device) for t in tensors)

        # 3. Forward pass
        self.model.eval()
        with torch.no_grad():
            if len(tensors) == 2:
                output = self.model(tensors[0], tensors[1])
            else:
                output = self.model(tensors[0])

        raw_out = output.cpu().numpy()

        # 4. Decode
        decoded = self._decode_output(raw_out)

        result = {
            'model':       self.model_name,
            'inputs_raw':  raw_inputs,
            'inputs_norm': {k: float(v.flatten()[0]) for k, v in norm_inputs.items()},
            **decoded,
        }

        print(f"\n  ── Result ──")
        for target, info in decoded.items():
            print(f"  {target}:")
            print(f"    Normalized output : {info['normalized']:.6f}")
            print(f"    Real value        : {info['real']:.4f} {info['unit']}")

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 4d.  PUBLIC: batch inference from representative_test.npz → CSV
    # ─────────────────────────────────────────────────────────────────────────
    def predict_from_npz(
        self,
        npz_path:     str,
        indices_json: str,
        output_csv:   str,
        batch_size:   int = 512,
    ) -> pd.DataFrame:
        """
        Run inference on all samples in a pre-processed .npz file, decode
        the predictions, and save/update a CSV with columns:

            sample_index | <model_name>

        If the CSV already exists, a new column is added (or overwritten if
        the model column already exists). The sample_index column is always
        preserved across multiple model runs.

        Parameters
        ----------
        npz_path     : path to representative_test.npz  (normalized arrays)
        indices_json : path to representative_test_indices.json
                       (maps rows → original dataset indices)
        output_csv   : path where the CSV will be written / updated
        batch_size   : inference batch size (default 512)

        Returns
        -------
        pd.DataFrame  with columns [sample_index, <model_name>]
        """
        print(f"\n{'─'*50}")
        print(f"  Batch inference — {self.model_name}")
        print(f"{'─'*50}")

        # ── Load .npz ────────────────────────────────────────────────────
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        raw     = np.load(npz_path)
        inputs  = {}
        for key in raw.files:
            role, feat = key.split('__', 1)
            if role == 'inputs':
                inputs[feat] = raw[key]

        # Validate all required features are present
        missing = [f for f in self.input_features if f not in inputs]
        if missing:
            raise KeyError(
                f"Features {missing} required by '{self.model_name}' are missing "
                f"from {npz_path}.\nAvailable: {list(inputs.keys())}"
            )

        N = next(iter(inputs.values())).shape[0]
        print(f"  Samples in npz : {N:,}")

        # ── Load sample indices ───────────────────────────────────────────
        if not os.path.exists(indices_json):
            raise FileNotFoundError(f"Indices JSON not found: {indices_json}")

        with open(indices_json) as f:
            idx_meta = json.load(f)
        sample_indices = np.array(idx_meta['sampled_indices'], dtype=np.int64)

        if len(sample_indices) != N:
            raise ValueError(
                f"Mismatch: npz has {N} rows but indices_json has "
                f"{len(sample_indices)} entries."
            )

        # ── Batch inference ───────────────────────────────────────────────
        all_preds_norm = []
        self.model.eval()

        for start in range(0, N, batch_size):
            end     = min(start + batch_size, N)
            batch   = {feat: arr[start:end] for feat, arr in inputs.items()
                       if feat in self.input_features}
            tensors = build_input_tensors(self.model_name, batch)
            tensors = tuple(t.to(self.device) for t in tensors)

            with torch.no_grad():
                if len(tensors) == 2:
                    out = self.model(tensors[0], tensors[1])
                else:
                    out = self.model(tensors[0])

            all_preds_norm.append(out.cpu().numpy())

        preds_norm = np.concatenate(all_preds_norm, axis=0).flatten()
        print(f"  Forward pass complete — {len(preds_norm):,} predictions")

        # ── Decode predictions → real W/m³ ────────────────────────────────
        # Determine target feature and its method (always Loss for loss-predicting models)
        target_name   = next(iter(self.target_features))
        target_method = self.target_features[target_name]
        target_stats  = {k: v for k, v in self.feature_stats.get(target_name, {}).items()
                         if k != 'method'}

        preds_real = denormalize_value(preds_norm, target_method, target_stats)
        print(f"  Decoded  (real units W/m³):")
        print(f"    min  = {preds_real.min():.4g}")
        print(f"    mean = {preds_real.mean():.4g}")
        print(f"    max  = {preds_real.max():.4g}")

        # ── Build / update CSV ────────────────────────────────────────────
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

        if os.path.exists(output_csv):
            # Load existing CSV and add / overwrite the model column
            df = pd.read_csv(output_csv)
            if 'sample_index' not in df.columns:
                raise ValueError(
                    f"Existing CSV {output_csv} has no 'sample_index' column."
                )
            # Align on sample_index in case a different order was used
            new_col              = pd.Series(preds_real, index=sample_indices,
                                             name=self.model_name)
            df                   = df.set_index('sample_index')
            df[self.model_name]  = new_col
            df                   = df.reset_index()
        else:
            df = pd.DataFrame({
                'sample_index':  sample_indices,
                self.model_name: preds_real,
            })

        df.to_csv(output_csv, index=False)
        print(f"\n  Saved predictions → {output_csv}")
        print(f"  CSV columns: {list(df.columns)}")
        print(f"  Preview (first 5 rows):")
        print(df.head().to_string(index=False))

        return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_input_features(pairs: list) -> dict:
    """
    Parse CLI key=value pairs like  B_pk=0.18  Frequency=100000
    into a dict of floats.
    """
    result = {}
    for pair in pairs:
        if '=' not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid input format '{pair}'. Use FeatureName=value."
            )
        key, val = pair.split('=', 1)
        result[key.strip()] = float(val.strip())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="MagNet inference — single sample or batch from .npz"
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    # ── Shared args ───────────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--model-name',  required=True,
                        choices=['scaler', 'sequence', 'seq2seq', 'cnn', 'transformer'])
    shared.add_argument('--model-path',  required=True,
                        help='Path to .pth checkpoint file')
    shared.add_argument('--stats-path',  required=True,
                        help='Path to preprocessing_summary.json')
    shared.add_argument('--config',      default=None,
                        help='Path to config.yaml (optional — architecture inferred '
                             'from checkpoint weights if omitted)')
    shared.add_argument('--device',      default='cpu', choices=['cpu', 'cuda'])

    # ── single mode ───────────────────────────────────────────────────────────
    p_single = sub.add_parser('single', parents=[shared],
                               help='Predict for one user-provided sample')
    p_single.add_argument(
        '--input', nargs='+', required=True, metavar='FEAT=VALUE',
        help=(
            'Feature values in physical units, e.g.:\n'
            '  scaler:  B_pk=0.18 Frequency=100000 Temperature=60 Hdc=10\n'
            '  cnn:     Frequency=100000 Temperature=60 Hdc=10  '
            '           (B waveform must be passed via Python API for CNN/seq models)'
        )
    )

    # ── batch mode ────────────────────────────────────────────────────────────
    p_batch = sub.add_parser('batch', parents=[shared],
                              help='Predict all samples in a representative_test.npz')
    p_batch.add_argument('--npz-path',     required=True,
                         help='Path to representative_test.npz')
    p_batch.add_argument('--indices-json', required=True,
                         help='Path to representative_test_indices.json')
    p_batch.add_argument('--output-csv',   required=True,
                         help='CSV file to write/update with predictions')
    p_batch.add_argument('--batch-size',   type=int, default=512)

    args = parser.parse_args()

    infer = ModelInference(
        model_name  = args.model_name,
        model_path  = args.model_path,
        stats_path  = args.stats_path,
        config_path = args.config,
        device      = args.device,
    )

    if args.mode == 'single':
        raw = _parse_input_features(args.input)
        infer.predict_single(raw)

    elif args.mode == 'batch':
        infer.predict_from_npz(
            npz_path     = args.npz_path,
            indices_json = args.indices_json,
            output_csv   = args.output_csv,
            batch_size   = args.batch_size,
        )


if __name__ == '__main__':
    main()