"""
MagNet Project Entry Point.

This script serves as the main interface for training and evaluating the neural network models.
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from src.data.dataset import MagNetDataset
from src.models.scaler_model import ScalerNetwork
from src.models.sequence_model import SequenceToScalerNetwork
from src.models.seq2seq_model import Seq2SeqNetwork
from src.models.cnn_model import CNNNetwork
from src.models.transformer_model import TransformerNetwork
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
    parser.add_argument('--config', type=str, default='Scripts/config/config.yaml', help='Path to config file')
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
            # Input: Dynamic based on config (B_pk, Freq, Temp, Hdc...)
            # Output: Log Loss (1)
            model_conf = config['models']['scaler']
            input_dim = len(model_conf.get('features', {}).get('inputs', {})) # Default to 3 if missing? Better to rely on config.
            # Fallback if config matches default structure but keys missing?
            # Safe bet:
            if input_dim == 0: input_dim = 3 # Legacy fallback
            
            model = ScalerNetwork(input_dim=input_dim, hidden_dim=model_conf['hidden_dim'], num_layers=model_conf['layers'], output_dim=1)
            
        elif model_name == 'sequence':
            # Input: B (1)
            # Output: Log Loss (1)
            model_conf = config['models']['sequence']
            model = SequenceToScalerNetwork(input_dim=1, hidden_dim=model_conf['hidden_dim'], output_dim=1, num_layers=model_conf['num_layers'])
            
        elif model_name == 'seq2seq':
            # Input: B (1)
            # Output: H (1)
            model_conf = config['models']['seq2seq']
            model = Seq2SeqNetwork(input_dim=1, hidden_dim=model_conf['encoder_dim'], output_dim=1)
        elif model_name == 'cnn':    
            # Input: B (1)
            # Output: Log Loss (1) (Scalar)
            model_conf = config['models']['cnn']
            # Retrieve Frequency stats from the dataset
            # The keys depend on what preprocessing returns. 
            # Usually: dataset.stats['Frequency']['mean']
            freq_stats = {}
            if hasattr(dataset, 'stats') and 'Frequency' in dataset.stats:
                f_s = dataset.stats['Frequency']
                # Check if it's standard (has mean/std) or minmax
                # The model expects mean/std for 'standard' normalization.
                if 'mean' in f_s:
                    freq_stats['freq_mean'] = f_s['mean']
                    freq_stats['freq_std'] = f_s['std']
            
            print(f"Passing Stats to CNN: {freq_stats}")
            
            model = CNNNetwork(input_dim=1, stats=freq_stats)
        elif model_name == 'transformer':
            # Input: B (1)
            # Output: Log Loss (1) (Scalar)
            model_conf = config['models']['transformer']
            # Map config keys to Fuzhou Transformer args
            # d_model -> dim_hidden
            # num_layers -> n_encoder_layers
            # dim_feedforward -> dim_feedforward_encoder
            model = TransformerNetwork(
                B_in_channel=1024, # Default seq len
                dim_hidden=model_conf['d_model'], 
                n_encoder_layers=model_conf['num_layers'], 
                dim_feedforward_encoder=model_conf['dim_feedforward'],
                n_heads=model_conf['nhead'],
                dropout_encoder=model_conf['dropout']
            )
        else:
            print("Unknown model name")


        # 3. Train
        print("Starting training...")
        # Subset config for training
        train_config = config['training']
        train_config['save_dir'] = os.path.join(train_config['save_dir'], model_name)
        
        if not os.path.exists(train_config['save_dir']):
            os.makedirs(train_config['save_dir'])
        
        model = model.to(device)
        trained_model, history = train_model(model, train_loader, val_loader,model_name ,config, device)

        
        # 4. Evaluate & Visualize
        print("Evaluating...")
        metrics, preds, targets = evaluate_model(trained_model, val_loader, device)
        print(f"Validation Metrics: {metrics}")
        
        # Prepare plots directory
        plots_dir = os.path.join(train_config['save_dir'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot Loss
        loss_plot_path = os.path.join(plots_dir, 'loss_curve.png')
        plot_loss_curve(history, title=f'{model_name} Training Loss', save_path=loss_plot_path)
        print(f"Loss plot saved to {loss_plot_path}")
        
        # Plot Predictions
        if model_name in ['scaler', 'sequence', 'cnn', 'transformer']:
            pred_plot_path = os.path.join(plots_dir, 'prediction_scatter.png')
            plot_prediction_scatter(preds, targets, title=f'{model_name}: Pred vs Actual Loss', save_path=pred_plot_path)
            print(f"Prediction plot saved to {pred_plot_path}")
            
        elif model_name == 'seq2seq':
            # Visualizing B-H loops
            # We have sequences. Let's plot the first sample in the validation set.
            # preds: (N, Seq, 1), targets: (N, Seq, 1)
            # Input to seq2seq was B (dataset mode='seq2seq': x=B, y=H)
            
            # Note: We need the B field (Input) to plot B-H loop, but evaluate_model returns metrics, preds, targets.
            # It doesn't return inputs.
            # For B-H check, we need B and H. 
            # In seq2seq mode: Target is H. Pred is H. Input is B.
            # We can't recover B easily from just preds/targets unless we modify evaluate or re-fetch.
            
            # Quick fix: Let's fetch one batch from val_loader to get B and H
            # And assume shuffle=False for val_loader so it matches? 
            # evaluate_model iterates over whole loader.
            
            # Let's just grab the first batch from val_loader again for visualization purposes.
            # This is safer.
            
            # Also, we likely want to inverse normalize for real units? 
            # For now, let's plot normalized.
            
            from src.utils.visualization import plot_bh_loop # Ensure imported
            
            try:
                # Get one batch
                val_iter = iter(val_loader)
                b_batch, h_batch = next(val_iter)
                
                # Move to device to inference
                b_batch = b_batch.to(device)
                
                model.eval()
                with torch.no_grad():
                    pred_h_batch = model(b_batch)
                    
                # Take first sample
                # b_batch: (Batch, Seq, 1)
                sample_idx = 0
                original_idx = val_set.indices[sample_idx]
                print(f"Plotting sample from Validation Set (Batch Index: {sample_idx})")
                print(f"Original Dataset Index (Experiment ID): {original_idx}")
                
                actual_b = b_batch[sample_idx].cpu().squeeze().numpy()
                actual_h = h_batch[sample_idx].cpu().squeeze().numpy() # Target H
                pred_h = pred_h_batch[sample_idx].cpu().squeeze().numpy()
                
                bh_plot_path = os.path.join(plots_dir, 'bh_loop_comparison.png')
                plot_bh_loop(actual_b, pred_h, actual_b, actual_h, title=f'{model_name} B-H Loop (Norm)', save_path=bh_plot_path)
                print(f"B-H Loop plot saved to {bh_plot_path}")
                
            except Exception as e:
                print(f"Could not plot B-H loop: {e}")

if __name__ == "__main__":
    main()
