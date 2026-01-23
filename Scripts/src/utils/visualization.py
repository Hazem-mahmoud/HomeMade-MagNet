"""
Visualization Module.

This module provides plotting functions for inspecting model performance.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(history, title='Training History', save_path=None):
    """
    Plots Train vs Val Loss.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_prediction_scatter(preds, targets, title='Predictions vs Actuals', save_path=None):
    """
    Scatter plot for scalar regression.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5)
    
    # Perfect line
    min_val = min(np.min(targets), np.min(preds))
    max_val = max(np.max(targets), np.max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
def plot_bh_loop(pred_b, pred_h, actual_b, actual_h, title='B-H Loop Comparison', save_path=None):
    """
    Plots predicted vs actual B-H loop.
    Args: (Seq_Len,) arrays.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(actual_h, actual_b, 'b-', label='Actual', linewidth=2)
    plt.plot(pred_h, pred_b, 'r--', label='Predicted', linewidth=2)
    plt.xlabel('H (A/m)')
    plt.ylabel('B (T)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
