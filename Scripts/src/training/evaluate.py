"""
Evaluation Module.

This module provides functions to evaluate model performance on test datasets.
"""

import torch
import torch.nn as nn
import numpy as np

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluates the model and returns predictions and metrics.
    
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data.
        device (str): Device.
        
    Returns:
        metrics (dict): MSE, MAE, RMSE, P95 Relative Error, Max Relative Error.
        predictions (ndarray)
        targets (ndarray)
    """
    model.eval()
    model.to(device)
    
    preds = []
    actuals = []
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    total_mse = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
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
            
            mse = criterion_mse(outputs, targets)
            mae = criterion_mae(outputs, targets)
            
            total_mse += mse.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
            
            preds.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
            
    num_samples = len(test_loader.dataset)
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    
    # ---- Relative Error Calculations ----
    predictions = np.concatenate(preds)
    targets = np.concatenate(actuals)
    
    epsilon = 1e-8  # numerical stability
    rel_errors = np.abs(predictions - targets) / (np.abs(targets) + epsilon)
    
    p95_rel_error = np.percentile(rel_errors, 95)
    max_rel_error = np.max(rel_errors)
    
    metrics = {
        'mse': avg_mse,
        'mae': avg_mae,
        'rmse': np.sqrt(avg_mse),
        'p95_relative_error': p95_rel_error,
        'max_relative_error': max_rel_error
    }
    
    return metrics, predictions, targets