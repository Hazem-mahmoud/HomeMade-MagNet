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

def train_model(model, train_loader, val_loader, config, device='cpu'):
    """
    Generic training loop.
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        config (dict): Configuration dictionary (lr, epochs, save_dir).
        device (str): 'cpu' or 'cuda'.
        
    Returns:
        model (nn.Module): Trained model (best weights).
        history (dict): Training history (loss).
    """
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    num_epochs = config.get('epochs', 100)
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar if interactive
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        # Deep Copy Best Model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history
