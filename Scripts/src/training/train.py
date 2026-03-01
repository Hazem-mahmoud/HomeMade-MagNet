"""
Training Loop Module.

This module contains the logic for training the neural networks.
It includes the training loop, validation step, loss calculation, checkpointing,
early stopping, and experiment logging.

Validation metrics (MSE / MAE / RMSE) are computed on the best model weights
and written into the same experiments.txt run block — no second call needed
from main.py / train_preprocessed.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import copy
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_model_description(desc_path, model_name):
    """
    Load model descriptions from a text file.
    Expects format:
        <model_name>: description text
    Returns description string for given model_name.
    """
    if not os.path.exists(desc_path):
        return "No description available."

    description = "No description available."
    with open(desc_path, "r") as f:
        lines = f.readlines()
        current_model = None
        buffer = []
        for line in lines:
            line_strip = line.strip()
            if line_strip.endswith(":"):
                if current_model == model_name:
                    description = "\n".join(buffer).strip()
                    break
                current_model = line_strip[:-1]  # remove colon
                buffer = []
            else:
                buffer.append(line_strip)
        else:
            if current_model == model_name:
                description = "\n".join(buffer).strip()
    return description


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute MSE, MAE, RMSE, 95th-percentile relative error, and max relative
    error between flat numpy arrays.

    Parameters
    ----------
    preds   : np.ndarray  — model predictions (any shape, will be flattened)
    targets : np.ndarray  — ground-truth values (same shape)

    Returns
    -------
    {'mse': float, 'mae': float, 'rmse': float,
     'p95_relative_error': float, 'max_relative_error': float}
    """
    preds   = preds.flatten().astype(np.float64)
    targets = targets.flatten().astype(np.float64)
    mse     = float(np.mean((preds - targets) ** 2))
    mae     = float(np.mean(np.abs(preds - targets)))
    rmse    = float(np.sqrt(mse))
    rel_errors         = np.abs((preds - targets) / (np.abs(targets) + 1e-8))
    p95_relative_error = float(np.percentile(rel_errors, 95))
    max_relative_error = float(np.max(rel_errors))
    return {'mse': mse, 'mae': mae, 'rmse': rmse,
            'p95_relative_error': p95_relative_error,
            'max_relative_error': max_relative_error}


def _evaluate_best_model(model, val_loader, device: str) -> tuple:
    """
    Run the best-weights model over the full validation loader and collect
    all predictions + targets.

    Returns
    -------
    metrics : dict   {'mse': float, 'mae': float, 'rmse': float,
                      'p95_relative_error': float, 'max_relative_error': float}
    preds   : np.ndarray  (N,)
    targets : np.ndarray  (N,)
    """
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                inputs, scalars, batch_targets = batch
                inputs  = inputs.to(device)
                scalars = scalars.to(device)
                outputs = model(inputs, scalars)
            else:
                inputs, batch_targets = batch
                inputs  = inputs.to(device)
                outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = _compute_metrics(preds, targets)
    return metrics, preds, targets


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, model_key, config, device='cpu'):
    """
    Train a model with optional early stopping and LR scheduler.

    At the end of training the best-checkpoint model is evaluated on the
    full validation split. Metrics (MSE / MAE / RMSE / P95 & Max Relative Error)
    are written into the same experiments.txt run block together with all other
    hyperparameters.

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader
    val_loader   : DataLoader
    model_key    : str    key under config['models'], e.g. 'cnn'
    config       : dict   full config loaded from config.yaml
    device       : str    'cuda' or 'cpu'

    Returns
    -------
    model   : nn.Module   loaded with best weights
    history : dict        {'train_loss': [...], 'val_loss': [...]}
    metrics : dict        {'mse': float, 'mae': float, 'rmse': float,
                           'p95_relative_error': float, 'max_relative_error': float}
    preds   : np.ndarray  validation predictions  (N, ...)
    targets : np.ndarray  validation ground-truth (N, ...)
    """

    model        = model.to(device)
    criterion    = nn.MSELoss()
    training_cfg = config['training']

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 0.001)
    )

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler     = None
    scheduler_cfg = training_cfg.get("scheduler", {})

    if scheduler_cfg:
        scheduler_type = scheduler_cfg.get("type", "ReduceLROnPlateau")

        if scheduler_type == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode     = 'min',
                factor   = float(scheduler_cfg.get("factor",   0.5)),
                patience = int(scheduler_cfg.get("patience",   5)),
                min_lr   = float(scheduler_cfg.get("min_lr",   1e-6)),
            )
        elif scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size = int(scheduler_cfg.get("step_size", 10)),
                gamma     = float(scheduler_cfg.get("gamma",   0.1)),
            )

    # ── Early stopping ────────────────────────────────────────────────────────
    early_cfg           = training_cfg.get("early_stopping", {})
    early_stop_enabled  = early_cfg.get("enabled",   False)
    early_stop_patience = int(early_cfg.get("patience",   10))
    early_stop_delta    = float(early_cfg.get("min_delta", 0.0))
    early_stop_counter  = 0

    # ── Paths & identifiers ───────────────────────────────────────────────────
    num_epochs = int(training_cfg.get('epochs', 100))
    save_dir   = training_cfg.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    model_cfg        = config['models'][model_key]
    model_name_clean = model_cfg['name']
    model_version    = model_cfg['version']
    version_tag      = f"{model_name_clean}_v{model_version}"

    model_dir   = os.path.join(save_dir, model_name_clean)
    version_dir = os.path.join(model_dir, version_tag)
    os.makedirs(version_dir, exist_ok=True)

    checkpoint_path     = os.path.join(version_dir, f"{version_tag}_best.pth")
    experiment_log_path = os.path.join(model_dir,   "experiments.txt")
    desc_path           = "/content/HomeMade-MagNet/Scripts/description.txt"
    model_description   = load_model_description(desc_path, model_name_clean)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_cfg     = config.get("tensorboard", {})
    tb_root    = tb_cfg.get("log_dir", "runs")
    run_id     = tb_cfg.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    tb_log_dir = os.path.join(tb_root, model_name_clean, f"v{model_version}", run_id)
    writer     = SummaryWriter(log_dir=tb_log_dir)

    # ── Training state ────────────────────────────────────────────────────────
    best_loss      = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch     = 0
    history        = {'train_loss': [], 'val_loss': []}

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(num_epochs):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"{version_tag} | Epoch {epoch+1}/{num_epochs}"
        )

        for batch in pbar:
            if len(batch) == 3:
                inputs, scalars, targets = batch
                inputs  = inputs.to(device)
                scalars = scalars.to(device)
                targets = targets.to(device)
                outputs = model(inputs, scalars)
            else:
                inputs, targets = batch
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        writer.add_scalar("Loss/Train", epoch_loss, epoch)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    inputs, scalars, targets = batch
                    inputs  = inputs.to(device)
                    scalars = scalars.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs, scalars)
                else:
                    inputs, targets = batch
                    inputs  = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)

                val_loss += criterion(outputs, targets).item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)

        # ── LR scheduler step ─────────────────────────────────────────────────
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LR", current_lr, epoch)

        print(
            f"{version_tag} | "
            f"Train Loss: {epoch_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        # ── Best checkpoint & early stopping ──────────────────────────────────
        if epoch_val_loss < best_loss - early_stop_delta:
            best_loss      = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            early_stop_counter = 0
            best_epoch         = epoch + 1      # 1-based
        else:
            early_stop_counter += 1

        if early_stop_enabled and early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    # ── Restore best weights ──────────────────────────────────────────────────
    model.load_state_dict(best_model_wts)
    epochs_run = epoch + 1

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATE BEST MODEL ON FULL VALIDATION SET
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nEvaluating best checkpoint (epoch {best_epoch}) on validation set ...")
    metrics, preds, val_targets = _evaluate_best_model(model, val_loader, device)

    # Log metrics to TensorBoard as well
    writer.add_scalar("Metrics/MSE",               metrics['mse'],               best_epoch)
    writer.add_scalar("Metrics/MAE",               metrics['mae'],               best_epoch)
    writer.add_scalar("Metrics/RMSE",              metrics['rmse'],              best_epoch)
    writer.add_scalar("Metrics/P95_Relative_Error", metrics['p95_relative_error'], best_epoch)
    writer.add_scalar("Metrics/Max_Relative_Error", metrics['max_relative_error'], best_epoch)

    print(
        f"  MSE               : {metrics['mse']:.6f}\n"
        f"  MAE               : {metrics['mae']:.6f}\n"
        f"  RMSE              : {metrics['rmse']:.6f}\n"
        f"  P95 Relative Error: {metrics['p95_relative_error']:.6f}\n"
        f"  Max Relative Error: {metrics['max_relative_error']:.6f}"
    )

    writer.close()

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT LOG  (single coherent run block)
    # ══════════════════════════════════════════════════════════════════════════
    with open(experiment_log_path, "a") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Model         : {model_name_clean}\n")
        f.write(f"Version       : v{model_version}\n")
        f.write(f"Run ID        : {run_id}\n")
        f.write(f"Date          : {datetime.now()}\n")
        f.write(f"Epochs Run    : {epochs_run}\n")

        if early_stop_enabled:
            f.write(f"Early Stopping: enabled  (patience={early_stop_patience})\n")
            f.write(f"Best Epoch    : {best_epoch}  (lowest val loss)\n")
        else:
            f.write(f"Early Stopping: disabled\n")
            f.write(f"Best Epoch    : {best_epoch}  (lowest val loss)\n")

        f.write("\nDescription:\n")
        f.write(f"{model_description}\n")

        f.write("\nHyperparameters:\n")
        for k, v in model_cfg.items():
            if k not in ("features", "name", "version"):
                f.write(f"  {k}: {v}\n")

        f.write("\nTraining Results:\n")
        f.write(f"  Best Val Loss   : {best_loss:.6f}\n")
        f.write(f"  Final Train Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Final Val Loss  : {history['val_loss'][-1]:.6f}\n")

        f.write("\nValidation Metrics (best checkpoint):\n")
        f.write(f"  MSE               : {metrics['mse']:.6f}\n")
        f.write(f"  MAE               : {metrics['mae']:.6f}\n")
        f.write(f"  RMSE              : {metrics['rmse']:.6f}\n")
        f.write(f"  P95 Relative Error: {metrics['p95_relative_error']:.6f}\n")
        f.write(f"  Max Relative Error: {metrics['max_relative_error']:.6f}\n")

        f.write("=" * 70 + "\n\n")

    return model, history, metrics, preds, val_targets