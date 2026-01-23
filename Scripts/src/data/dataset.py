"""
MagNet Dataset Module.

This module provides a PyTorch Dataset wrapper for the MagNet data.
It handles:
1. Loading the full dataset into memory.
2. Computing derived quantities (B, H, Power Loss).
3. Normalizing features.
4. Serving data based on the requested mode ('scaler', 'sequence', 'seq2seq').

Classes:
- MagNetDataset(Dataset)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from src.data import loader, preprocessing

class MagNetDataset(Dataset):
    def __init__(self, file_path, mode='scaler', transform=None):
        """
        Args:
            file_path (str): Path to .mat file.
            mode (str): 'scaler', 'sequence', 'seq2seq'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform
        
        # Load raw data
        print(f"Loading dataset from {file_path}...")
        raw_data = loader.load_full_dataset(file_path)
        
        self.voltage = raw_data['voltage'].astype(np.float32)
        self.current = raw_data['current'].astype(np.float32)
        self.freq = raw_data['freq'].astype(np.float32)
        self.temp = raw_data['temp'].astype(np.float32)
        self.hdc = raw_data['hdc'].astype(np.float32)
        self.duty = raw_data['duty'].astype(np.float32)
        self.meta = raw_data['meta']
        
        # Compute derived features (B, H, Power Loss)
        print("Computing B field...")
        # Note: B calculation can be vectorized or looped.
        # Vectorized implementation of cumulative trapezoid is cleaner but requires care with shapes.
        # We'll use a loop or apply_along_axis if needed, but simple integration is fast.
        # B = Integral(V) / (N * Ae)
        
        # Remove DC offset from V per experiment
        v_mean = np.mean(self.voltage, axis=1, keepdims=True)
        v_clean = self.voltage - v_mean
        
        # Integrate
        # cumulative_trapezoid works on last axis by default, which is samples (dim 1)
        from scipy.integrate import cumulative_trapezoid
        # We need dt for each experiment.
        # If dt is scalar, easy. If array, we need to broadcast.
        dt = self.meta['dt']
        if np.ndim(dt) == 0:
            dt = float(dt)
        else:
            # Reshape to (N, 1) for broadcasting if possible, 
            # but cumtrapz doesn't accept array dx easily for different rows.
            # Assuming dt is constant for each experiment relative to time steps.
            pass

        # Since dt might vary per experiment, we might need a loop or valid avg dt
        # For simplicity in 'scaler' mode, we might trust specific dt.
        # Let's assume dt is constant within one cycle.
        
        # Efficient vector integration if dt is scalar or we average it?
        # Let's loop for safety in this version or use simple cumsum * dt if uniform.
        # cumtrapz is (y[i] + y[i-1])/2 * dx.
        
        # Vectorized cumtrapz:
        flux = cumulative_trapezoid(v_clean, axis=1, initial=0) # * dt later
        if np.ndim(dt) > 0:
            flux = flux * dt[:, None]
        else:
            flux = flux * dt
            
        self.b_field = flux / (self.meta['N_sec'] * self.meta['Ae'])
        
        # Remove DC from B
        b_mean = np.mean(self.b_field, axis=1, keepdims=True)
        self.b_field = self.b_field - b_mean
        
        print("Computing H field...")
        # H = (N * I) / Le
        self.h_field = (self.meta['N_prim'] * self.current) / self.meta['Le']
        
        print("Computing Power Loss...")
        # Refactored to use shared function (B-H Loop Area)
        # Pv = Frequency * Area(B-H)
        self.power_loss = preprocessing.calculate_volumetric_loss(
            self.b_field, 
            self.h_field, 
            frequency=self.freq
        )
        
        # Normalize inputs (MinMax)
        # Store scalers for inversion? For now, simple minmax.
        # For ML, we should calculate stats on TRAINING set only.
        # But here we do full dataset.
        # Ideally: split first, then normalize.
        # For now, we normalize everything together (simple approach).
        
        self.norm_b, _ = preprocessing.normalize_data(self.b_field)
        self.norm_h, _ = preprocessing.normalize_data(self.h_field)
        self.norm_freq, _ = preprocessing.normalize_data(self.freq)
        self.norm_temp, _ = preprocessing.normalize_data(self.temp)
        self.norm_hdc, _ = preprocessing.normalize_data(self.hdc)
        
        # Target normalization (log scale is often good for loss)
        self.log_loss = np.log10(np.abs(self.power_loss) + 1e-6)
        
    def __len__(self):
        return self.voltage.shape[0]
        
    def __getitem__(self, idx):
        if self.mode == 'scaler':
            # Input: Freq, Temp, Hdc
            # Target: Power Loss
            x = torch.tensor([
                self.norm_freq[idx],
                self.norm_temp[idx],
                self.norm_hdc[idx]
            ], dtype=torch.float32)
            y = torch.tensor([self.log_loss[idx]], dtype=torch.float32)
            return x, y
            
        elif self.mode in ['sequence', 'cnn', 'transformer']:
            # Input: B waveform (or H)
            # Target: Power Loss
            # Shape: (Seq_Len, 1)
            b_seq = torch.tensor(self.norm_b[idx], dtype=torch.float32).unsqueeze(-1)
            y = torch.tensor([self.log_loss[idx]], dtype=torch.float32)
            return b_seq, y
            
        elif self.mode == 'seq2seq':
            # Input: B waveform
            # Target: H waveform (or vice versa)
            # Many papers predict H from B.
            x = torch.tensor(self.norm_b[idx], dtype=torch.float32).unsqueeze(-1)
            y = torch.tensor(self.norm_h[idx], dtype=torch.float32).unsqueeze(-1)
            return x, y
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
