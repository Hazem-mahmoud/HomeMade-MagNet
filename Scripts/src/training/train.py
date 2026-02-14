"""
Training Loop Module.

This module contains the logic for training the neural networks.
It includes the training loop, validation step, loss calculation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import copy
from datetime import datetime


from torch.utils.tensorboard import SummaryWriter



def train_model(model, train_loader, val_loader, model_name, config, device='cpu'):
    model = model.to(device)

    criterion = nn.MSELoss()
    training_cfg = config['training']

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 0.001)
    )

    num_epochs = training_cfg.get('epochs', 100)
    save_dir = training_cfg.get('save_dir', 'checkpoints')
    print("save dir is :",save_dir )
    os.makedirs(save_dir, exist_ok=True)

    # ===================== Model info =====================
    model_key = model_name
    model_cfg = config['models'][model_key]
    model_name = model_cfg['name']
    model_version = model_cfg['version']

    version_tag = f"{model_name}_v{model_version}"

    model_dir = os.path.join(save_dir, model_name)
    version_dir = os.path.join(model_dir, version_tag)
    os.makedirs(version_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        version_dir,
        f"{version_tag}_best.pth"
    )

    experiment_log_path = os.path.join(
        model_dir,
        "experiments.txt"
    )
    # ======================================================

    # ===================== TensorBoard =====================
    tb_cfg = config.get("tensorboard", {})
    tb_root = tb_cfg.get("log_dir", "runs")
    run_id = tb_cfg.get(
        "run_id",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    tb_log_dir = os.path.join(
        tb_root,
        model_name,
        f"v{model_version}",
        run_id
    )

    writer = SummaryWriter(log_dir=tb_log_dir)
    # ======================================================

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

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

        # TensorBoard train loss
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

        # TensorBoard val loss
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)

        print(
            f"{version_tag} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}"
        )

        # ===================== Best model =====================
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)

            # TensorBoard best loss
            writer.add_scalar(
                "Loss/Best_Validation",
                epoch_val_loss,
                epoch
            )

    # ===================== Experiment logging =====================
    model.load_state_dict(best_model_wts)

    with open(experiment_log_path, "a") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Version: v{model_version}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Date: {datetime.now()}\n\n")

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
