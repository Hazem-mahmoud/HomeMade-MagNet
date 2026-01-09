"""
Transfer Learning Module.

This module aids in adapting pre-trained models to new datasets or materials.
It provides utilities for freezing layers, replacing heads, and fine-tuning.
"""

import torch
import torch.nn as nn
import os

def load_pretrained(model, checkpoint_path, device='cpu'):
    """
    Loads model weights from a checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded weights from {checkpoint_path}")
    return model

def freeze_layers(model, layers_to_freeze=None):
    """
    Freezes parameters in the model.
    
    Args:
        model (nn.Module): The model.
        layers_to_freeze (list of str, optional): Names of modules to freeze.
                                                  If None, freeze all except the last layer (Head).
    """
    if layers_to_freeze is None:
        # Default strategy: Freeze feature extractor, keep head trainable.
        # This assumes model has a .head attribute or similar structure.
        # If standard sequential, we might freeze all but last.
        
        for name, param in model.named_parameters():
             param.requires_grad = False
             
        # Unfreeze head
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
            print("Frozen all layers except 'head'.")
        elif hasattr(model, 'fc'): # ResNet style
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            # Try to unfreeze last layer of 'model' sequential if exists
            # Fallback: User must specify.
            print("Warning: Could not identify head. All layers frozen.")
    else:
        # Freeze specific layers
        for name, param in model.named_parameters():
            for layer_name in layers_to_freeze:
                if layer_name in name:
                    param.requires_grad = False
        print(f"Frozen layers containing: {layers_to_freeze}")

def fine_tune(model, train_loader, val_loader, config, device='cpu'):
    """
    Wrapper for training loop specifically for fine-tuning.
    Usually uses lower learning rate.
    """
    # Reduce LR for fine-tuning
    if 'learning_rate' in config:
        config['learning_rate'] /= 10.0
        
    print("Starting Fine-tuning (LR reduced)...")
    from src.training.train import train_model
    return train_model(model, train_loader, val_loader, config, device)
