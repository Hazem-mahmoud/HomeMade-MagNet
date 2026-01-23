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
import loader
import preprocessing

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
        self.b_field = preprocessing.calculate_flux_density(
            self.voltage, 
            self.meta['dt'], 
            self.meta['N_sec'], 
            self.meta['Ae']
        )
        
        print("Computing H field...")
        self.h_field = preprocessing.calculate_magnetizing_force(
            self.current, 
            self.meta['N_prim'], 
            self.meta['Le']
        )
        
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
        
        self.norm_b, _ = preprocessing.normalize_data(self.b_field, axis=None)
        self.norm_h, _ = preprocessing.normalize_data(self.h_field, axis=None)
        
        # For scalars (1D arrays), axis=0 or None yields same result (scalar min/max)
        self.norm_freq, _ = preprocessing.normalize_data(self.freq, axis=None)
        self.norm_temp, _ = preprocessing.normalize_data(self.temp, axis=None)
        self.norm_hdc, _ = preprocessing.normalize_data(self.hdc, axis=None)
        
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

# ==========================================
# TEST SCRIPT
# ==========================================
def test_dataset():
    """
    Validation function to ensure the dataset loads and computes fields correctly.
    """
    import os
    print("Running Dataset Validation Test...")
    
    # Define a test file path (assuming running from Scripts/src/data or project root)
    # Adjust relative path to find a valid .mat file.
    # Looking for '3C90_TX-25-15-10_Data1_Cycle.mat' in project root
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming path structure: .../Scripts/src/data/dataset.py
    # Data is in .../HomeMade MagNet/ (grandparent of Scripts?)
    # or just use the same logic as preprocessing test
    
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 
    # Scripts/src/data -> Scripts/src -> Scripts -> HomeMade MagNet
    
    test_file = os.path.join(project_root, '3C90_TX-25-15-10_Data1_Cycle.mat')

    print(f"Loading {test_file}...")
    try:
        ds = MagNetDataset(test_file, mode='scaler')
        
        print("Dataset Loaded Successfully.")
        print(f" - Num Samples: {len(ds)}")
        print(f" - Voltage Shape: {ds.voltage.shape}")
        print(f" - B-Field Shape: {ds.b_field.shape}, Mean: {ds.b_field.mean():.4e}")
        print(f" - H-Field Shape: {ds.h_field.shape}, Mean: {ds.h_field.mean():.4e}")
        print(f" - Power Loss Shape: {ds.power_loss.shape}, Mean: {ds.power_loss.mean():.4e}")
        
        # Verify Norms exist
        print(f" - Norm B Shape: {ds.norm_b.shape}")
        
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
