"""
MagNet Project Entry Point.

This script serves as the main interface for training and evaluating the neural network models.
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from src.data.dataset import MagNetDataset
from src.models import ScalerNetwork, SequenceToScalerNetwork, Seq2SeqNetwork, CNNNetwork, TransformerNetwork
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.utils.visualization import plot_loss_curve, plot_prediction_scatter
import yaml
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="MagNet Deep Learning Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--model', type=str, choices=['scaler', 'sequence', 'seq2seq', 'cnn', 'transformer', 'all'], required=True, help='Model type to train')
    parser.add_argument('--epochs', type=int, help='Override epochs in config')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override config
    if args.epochs:
        config['training']['epochs'] = args.epochs
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models_to_run = [args.model] if args.model != 'all' else ['scaler', 'sequence', 'seq2seq', 'cnn', 'transformer']
    
    for model_name in models_to_run:
        print(f"\n{'='*20} Training {model_name.upper()} Model {'='*20}")
        
        # 1. Dataset
        print("Preparing Dataset...")
        dataset = MagNetDataset(args.data, mode=model_name)
        
        # Split (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        
        batch_size = config['data'].get('batch_size', 32)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        
        # 2. Model
        if model_name == 'scaler':
            # Input: Freq, Temp, Hdc (3)
            # Output: Log Loss (1)
            model_conf = config['models']['scaler']
            model = ScalerNetwork(input_dim=3, hidden_dim=model_conf['hidden_dim'], num_layers=model_conf['layers'], output_dim=1)
            
        elif model_name == 'sequence':
            # Input: B (1)
            # Output: Log Loss (1)
            model_conf = config['models']['sequence']
            model = SequenceToScalerNetwork(input_dim=1, hidden_dim=model_conf['hidden_dim'], output_dim=1)
            
        elif model_name == 'seq2seq':
            # Input: B (1)
            # Output: H (1)
            model_conf = config['models']['seq2seq']
            model = Seq2SeqNetwork(input_dim=1, encoder_dim=model_conf['encoder_dim'], decoder_dim=model_conf['decoder_dim'], output_dim=1)
            
        elif model_name == 'cnn':
            # Input: B (1)
            # Output: Log Loss (1) (Scalar)
            model_conf = config['models']['cnn']
            model = CNNNetwork(input_dim=1, kernel_size=model_conf['kernel_size'], num_channels=model_conf['num_channels'], num_layers=model_conf['num_layers'], output_dim=1)
        
        elif model_name == 'transformer':
            # Input: B (1)
            # Output: Log Loss (1) (Scalar)
            model_conf = config['models']['transformer']
            model = TransformerNetwork(input_dim=1, d_model=model_conf['d_model'], nhead=model_conf['nhead'], num_layers=model_conf['num_layers'], dim_feedforward=model_conf['dim_feedforward'], dropout=model_conf['dropout'], output_dim=1)

        # 3. Train
        print("Starting training...")
        # Subset config for training
        train_config = config['training']
        train_config['save_dir'] = os.path.join(train_config['save_dir'], model_name)
        
        if not os.path.exists(train_config['save_dir']):
            os.makedirs(train_config['save_dir'])
        
        model = model.to(device)
        trained_model, history = train_model(model, train_loader, val_loader, train_config, device)
        
        # 4. Evaluate & Visualize
        print("Evaluating...")
        metrics, preds, targets = evaluate_model(trained_model, val_loader, device)
        print(f"Validation Metrics: {metrics}")
        
        # Plot Loss
        plot_loss_curve(history, title=f'{model_name} Training Loss')
        
        # Plot Predictions
        if model_name in ['scaler', 'sequence', 'cnn', 'transformer']:
            plot_prediction_scatter(preds, targets, title=f'{model_name}: Pred vs Actual Loss')
        elif model_name == 'seq2seq':
            pass

if __name__ == "__main__":
    main()
