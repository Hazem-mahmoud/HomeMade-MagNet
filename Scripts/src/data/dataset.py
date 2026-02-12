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
try:
    from src.data import loader
    from src.data import preprocessing
except ImportError:   
    from . import loader
    from . import preprocessing
    
class MagNetDataset(Dataset):
    def __init__(self, file_path, mode='scaler', partition='train', transform=None, config_path=None):
        """
        Args:
            file_path (str): Path to .mat file.
            mode (str): 'scaler', 'sequence', 'seq2seq'.
            partition (str): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied.
            config_path (str): Path to config.yaml (optional).
        """
        self.mode = mode
        self.partition = partition
        self.transform = transform

        # Load raw data
        print(f"Loading dataset from {file_path} for partition '{partition}'...")
        raw_data = loader.load_full_dataset(file_path)

        # Extract Standard Args
        voltage = raw_data['voltage'].astype(np.float32)
        current = raw_data['current'].astype(np.float32)
        freq = raw_data['freq'].astype(np.float32)

        props = {
            'N_prim': raw_data['meta']['N_prim'],
            'N_sec': raw_data['meta']['N_sec'],
            'Ae': raw_data['meta']['Ae'],
            'Le': raw_data['meta']['Le']
        }
        dt = raw_data['meta']['dt']
        if isinstance(dt, (list, np.ndarray)) and len(dt) > 1:
            dt = dt[0]

        # Extra Features
        extra = {
            'Temperature': raw_data['temp'].astype(np.float32),
            'Hdc': raw_data['hdc'].astype(np.float32),
            'Duty': raw_data['duty'].astype(np.float32)
        }

        # Use Central Preprocessing
        print("Running centralized preprocessing...")

        # Determine config path if not provided
        if not config_path:
             # Try to find it relative to this file
             import os
             current_dir = os.path.dirname(os.path.abspath(__file__))
             # Scripts/src/data -> ... -> Scripts/config/config.yaml
             config_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'config', 'config.yaml'))

        if not os.path.exists(config_path):
            print(f"WARNING: Config not found at {config_path}. Using defaults.")
            config_path = None

        data_split, self.stats = preprocessing.process_magnet_dataset(
            voltage, current, freq, props, dt,
            model_type=mode,
            config_path=config_path,
            extra_features=extra
        )

        # Select Partition
        if partition not in data_split:
            raise ValueError(f"Partition '{partition}' not found in split data.")

        self.inputs = data_split[partition]['inputs']
        self.targets = data_split[partition]['targets']

        # Validation checks
        if not self.targets:
             print("WARNING: No targets found in processed data.")

        if not self.inputs:
             print("WARNING: No inputs found in processed data.")

        # Store length
        # Assuming all input arrays are same length
        any_key = next(iter(self.inputs)) if self.inputs else next(iter(self.targets))
        self.length = len(self.inputs[any_key]) if self.inputs else len(self.targets[any_key])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve normalized features from dictionary
        # NOTE: Keys depend on config.yaml!
        # We must align code here with what we expect to be in the config for each mode.
        # Fallback logic is needed if config is dynamic.

        # Helper to get datum safely
        def get(dic, key):
            if key in dic:
                val = dic[key][idx]
                return torch.tensor(val, dtype=torch.float32)
            else:
                 # Be noisy if missing expected feature
                 raise KeyError(f"Feature '{key}' missing from dataset. Check config.yaml or preprocessing.")

        # Helper to get sequence or scalar
        def get_tens(dic, key, unsqueeze=False):
            t = get(dic, key)
            if unsqueeze:
                return t.unsqueeze(-1)
            return t

        if self.mode == 'scaler':
            # Config MUST include Frequency, Temperature, Hdc in inputs
            # Config MUST include Loss in targets

            # Input: B_pk, Freq, Temp, Hdc
            # We assume these are 1D arrays (scalers) in the input dict
            b = get(self.inputs, 'B_pk')
            f = get(self.inputs, 'Frequency')
            t = get(self.inputs, 'Temperature')
            h = get(self.inputs, 'Hdc')

            x = torch.stack([b.squeeze(), f.squeeze(), t.squeeze(), h.squeeze()])
            # Note: normalized scalars might come as (1,) or scalar. Squeeze ensures (3,)

            y = get_tens(self.targets, 'Loss', unsqueeze=True) # (1,)
            return x, y

        elif self.mode in ['sequence', 'cnn', 'transformer']:
            # Input: B (or H)
            # Target: Loss
            b = get_tens(self.inputs, 'B', unsqueeze=True) # (Seq, 1)

            # Scalars: Freq, Temp, Hdc
            f = get(self.inputs, 'Frequency')
            t = get(self.inputs, 'Temperature')
            h = get(self.inputs, 'Hdc')
            scalars = torch.stack([f.squeeze(), t.squeeze(), h.squeeze()])

            y = get_tens(self.targets, 'Loss', unsqueeze=True)
            return b, scalars, y

        elif self.mode == 'seq2seq':
            # Input: B
            # Target: H
            b = get_tens(self.inputs, 'B', unsqueeze=True)
            h = get_tens(self.inputs, 'H', unsqueeze=True) # Assuming H is in inputs or targets?
            # Usually H is 'target' for B->H prediction? Or Input?
            # If we want to predict H, it should be in targets.
            # But process_magnet_dataset puts things in 'inputs' or 'targets' based on config.
            # Check where H is.

            if 'H' in self.targets:
                target = get_tens(self.targets, 'H', unsqueeze=True)
            elif 'H' in self.inputs:
                 target = get_tens(self.inputs, 'H', unsqueeze=True)
            else:
                raise KeyError("H field not found in inputs or targets for seq2seq")

            return b, target

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

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    test_file = "/content/drive/MyDrive/MagNet_DataSet/3C90_TX-25-15-10_Data1_Cycle.mat"


    print(f"Loading {test_file}...")
    try:
        # Test Train Split
        print("\n--- Testing 'train' partition (mode='scaler') ---")
        ds_train = MagNetDataset(test_file, mode='scaler', partition='train')
        print("Dataset Loaded Successfully.")
        print(f" - Num Samples: {len(ds_train)}")

        # Test Item
        x, y = ds_train[0]
        print(f" - Sample[0] Input Shape: {x.shape}, Target Shape: {y.shape}")

        # Test Test Split
        print("\n--- Testing 'test' partition (mode='cnn') ---")
        ds_test = MagNetDataset(test_file, mode='cnn', partition='test')
        print(f" - Num Samples: {len(ds_test)}")

        # Test Item
        b, scalars, y = ds_test[0]
        print(b.shape, scalars.shape, y.shape)



    except Exception as e:
        print(f"Dataset validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
