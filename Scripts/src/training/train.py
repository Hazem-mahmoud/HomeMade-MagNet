"""
Training Loop Module.

This module contains the logic for training the neural networks.
It includes the training loop, validation step, loss calculation, checkpointing,
early stopping, and experiment logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import copy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


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


def train_model(model, train_loader, val_loader, model_key, config, device='cpu'):
    """
    Train a model with optional early stopping and LR scheduler.
    Logs experiments to experiments.txt and uses TensorBoard.
    """

    model = model.to(device)
    criterion = nn.MSELoss()
    training_cfg = config['training']

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 0.001)
    )

    # ===================== Scheduler =====================
    scheduler = None
    scheduler_cfg = training_cfg.get("scheduler", {})

    if scheduler_cfg:
        scheduler_type = scheduler_cfg.get("type", "ReduceLROnPlateau")

        if scheduler_type == "ReduceLROnPlateau":
            # Remove verbose to avoid error in PyTorch 2.x
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 5)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-6))
            )
        elif scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_cfg.get("step_size", 10)),
                gamma=float(scheduler_cfg.get("gamma", 0.1))
            )
    # =====================================================

    # ===================== Early Stopping =====================
    early_cfg = training_cfg.get("early_stopping", {})
    early_stop_enabled = early_cfg.get("enabled", False)
    early_stop_patience = int(early_cfg.get("patience", 10))
    early_stop_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_stop_counter = 0
    # ==========================================================

    num_epochs = int(training_cfg.get('epochs', 100))
    save_dir = training_cfg.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # Model info
    model_cfg = config['models'][model_key]
    model_name_clean = model_cfg['name']
    model_version = model_cfg['version']
    version_tag = f"{model_name_clean}_v{model_version}"

    # Directories
    model_dir = os.path.join(save_dir, model_name_clean)
    version_dir = os.path.join(model_dir, version_tag)
    os.makedirs(version_dir, exist_ok=True)
    checkpoint_path = os.path.join(version_dir, f"{version_tag}_best.pth")
    experiment_log_path = os.path.join(model_dir, "experiments.txt")
    desc_path = "/content/HomeMade-MagNet/Scripts/description.txt"
    model_description = load_model_description(desc_path, model_name_clean)

    # TensorBoard
    tb_cfg = config.get("tensorboard", {})
    tb_root = tb_cfg.get("log_dir", "runs")
    run_id = tb_cfg.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    tb_log_dir = os.path.join(tb_root, model_name_clean, f"v{model_version}", run_id)
    writer = SummaryWriter(log_dir=tb_log_dir)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # ===================== Training =====================
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"{version_tag} | Epoch {epoch+1}/{num_epochs}"
        )

        for batch in pbar:
            if len(batch) == 3:
                inputs, scalars, targets = batch
                inputs = inputs.to(device)
                scalars = scalars.to(device)
                targets = targets.to(device)
                outputs = model(inputs, scalars)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
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

        # ===================== Validation =====================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    inputs, scalars, targets = batch
                    inputs = inputs.to(device)
                    scalars = scalars.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs, scalars)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)

        # ===================== Scheduler Step =====================
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

        # ===================== Best Model & Early Stopping =====================
        if epoch_val_loss < best_loss - early_stop_min_delta:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            early_stop_counter = 0
            best_epoch = epoch + 1  # 1-based
        else:
            early_stop_counter += 1

        if early_stop_enabled and early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)

    # ===================== Experiment Logging =====================
    with open(experiment_log_path, "a") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Model: {model_name_clean}\n")
        f.write(f"Version: v{model_version}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Total Epochs Run: {epoch+1}\n")
        if early_stop_enabled:
            f.write(f"Early Stopping Enabled: True\n")
            f.write(f"Best Epoch (Val Loss): {best_epoch}\n")
        else:
            f.write(f"Early Stopping Enabled: False\n")
        f.write("\nDescription:\n")
        f.write(f"{model_description}\n\n")
        f.write("Hyperparameters:\n")
        for k, v in model_cfg.items():
            if k not in ["features", "name", "version"]:
                f.write(f"  {k}: {v}\n")
        f.write("\nResults:\n")
        f.write(f"  Best Val Loss: {best_loss:.6f}\n")
        f.write(f"  Final Train Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Final Val Loss: {history['val_loss'][-1]:.6f}\n")
        f.write("=" * 70 + "\n\n")

    writer.close()
    return model, history