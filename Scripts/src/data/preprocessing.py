"""
Data Preprocessing Module.

Generic processing for Time-Series FNN.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

def normalize_data(data, method='standard', stats=None):
    """
    Normalizes data using Global Scaling (Statistics computed over ALL data).
    
    Args:
        data (np.array): Input data.
        method (str): 'standard' (Z-score), 'minmax', or 'none'.
        stats (dict, optional): Pre-computed stats.
        
    Returns:
        norm_data (np.array): Normalized data.
        stats (dict): Statistics used.
    """
    # Force float32
    data = data.astype(np.float32)

    if method == 'none':
        return data, {}

    if method == 'standard':
        if stats:
            mean = stats['mean']
            std = stats['std']
        else:
            mean = np.mean(data, axis=None) 
            std = np.std(data, axis=None)
            if std == 0: std = 1.0
            stats = {'mean': mean, 'std': std}
            
        norm_data = (data - mean) / std
        return norm_data, stats
        
    elif method == 'minmax':
        if stats:
            min_val = stats['min']
            max_val = stats['max']
        else:
            min_val = np.min(data, axis=None)
            max_val = np.max(data, axis=None)
            stats = {'min': min_val, 'max': max_val}
            
        denom = max_val - min_val
        if denom == 0: denom = 1.0
        
        norm_data = (data - min_val) / denom
        return norm_data, stats
    
    else:
        raise ValueError(f"Unknown method: {method}")


def prepare_dataset(features, targets, test_ratio=0.2, norm_config=None):
    """
    Generic function to split and normalize ANY set of features and targets.
    
    Args:
        features (dict): {'FeatureName': data_array_of_shape_N_x_...}
        targets (dict): {'TargetName': data_array_of_shape_N_x_...}
        test_ratio (float): Fraction of data to use for testing.
        norm_config (dict): {'Name': 'method'}. 
                            e.g. {'B': 'standard', 'Loss': 'log10'}.
                            Defaults to 'standard' for features and 'none' for targets if not specified.
                            
    Returns:
        data_split (dict): Contains 'train' and 'test' dictionaries.
                           data_split['train']['inputs']['FeatureName']
                           data_split['train']['targets']['TargetName']
        stats (dict): The statistics used for normalization for each feature/target.
    """
    if norm_config is None:
        norm_config = {}
        
    # 1. Determine Split Index
    # Assume all arrays have same length N in dimension 0.
    any_key = next(iter(features))
    N = len(features[any_key])
    split_idx = int(N * (1 - test_ratio))
    
    split_data = {
        'train': {'inputs': {}, 'targets': {}},
        'test': {'inputs': {}, 'targets': {}}
    }
    stats_out = {}
    
    print(f"Splitting Dataset: {N} Samples -> {split_idx} Train, {N - split_idx} Test")
    
    # 2. Process Inputs (Features)
    print("Processing Features...")
    for name, data in features.items():
        # Get method (Default to 'standard' for inputs)
        method = norm_config.get(name, 'standard')
        
        # Split
        train_part = data[:split_idx]
        test_part = data[split_idx:]
        
        # Normalize Train
        print(f"  - Normalizing '{name}' with {method}...")
        train_norm, stat = normalize_data(train_part, method=method)
        stats_out[name] = stat
        
        # Normalize Test (using Train stats)
        test_norm, _ = normalize_data(test_part, method=method, stats=stat)
        
        split_data['train']['inputs'][name] = train_norm
        split_data['test']['inputs'][name] = test_norm
        
    # 3. Process Targets
    print("Processing Targets...")
    for name, data in targets.items():
        # Get method (Default to 'none' for targets unless specified)
        # Often we don't normalize targets for regression unless specifically asked (like Log Loss)
        method = norm_config.get(name, 'none')
        
        # Split
        train_part = data[:split_idx]
        test_part = data[split_idx:]
        
        # Normalize Train
        if method != 'none':
             print(f"  - Normalizing Target '{name}' with {method}...")
             
        train_norm, stat = normalize_data(train_part, method=method)
        if method != 'none':
            stats_out[name] = stat
        
        # Normalize Test
        test_norm, _ = normalize_data(test_part, method=method, stats=stat)
        
        split_data['train']['targets'][name] = train_norm
        split_data['test']['targets'][name] = test_norm
        
    return split_data, stats_out


# --- PHYSICS FUNCTIONS (Vectorized) ---
# Kept as helpers, but they are just ONE way to generate features.

def calculate_flux_density(voltage, sampling_time, secondary_turns, effective_area):
    """ Calculates B (Flux Density). """
    v_mean = np.mean(voltage, axis=1, keepdims=True)
    v_clean = voltage - v_mean
    flux = cumulative_trapezoid(v_clean, axis=-1, initial=0)
    
    if np.ndim(sampling_time) == 0:
        flux = flux * sampling_time
    else:
        st = np.array(sampling_time).reshape(-1, 1) 
        flux = flux * st

    b = flux / (secondary_turns * effective_area)
    b_mean = np.mean(b, axis=1, keepdims=True)
    b = b - b_mean
    
    return b.astype(np.float32)

def calculate_magnetizing_force(current, primary_turns, effective_length):
    """ Calculates H (Magnetizing Force). """
    h = (primary_turns * current) / effective_length
    return h.astype(np.float32)

def calculate_volumetric_loss(b_field, h_field, frequency):
    """ Calculates Volumetric Power Loss (Target). """
    energy_density = trapezoid(y=h_field, x=b_field, axis=-1)
    energy_density = np.abs(energy_density)
    
    if np.ndim(frequency) > 0:
        freq = np.array(frequency).squeeze()
    else:
        freq = frequency
        
    pv = energy_density * freq
    return pv.astype(np.float32)


# --- WRAPPER FOR FULL MagNet PIPELINE ---

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_magnet_dataset(voltage, current, frequency, mag_props, dt, model_type='scaler', config_path=None):
    """
    Orchestrates the physics calculation -> Feature Assembly -> Normalization.
    
    Args:
        voltage, current: Raw arrays
        frequency: Raw frequency array
        mag_props, dt: Constants
        model_type (str): Key in config['models'] (e.g. 'scaler', 'cnn')
        config_path (str, optional): Path to config.yaml.
    """
    norm_config = {}
    
    # 1. Load Config if path provided
    if config_path:
        full_config = load_config(config_path)
        if 'models' in full_config and model_type in full_config['models']:
            model_conf = full_config['models'][model_type]
            if 'features' in model_conf:
                # Merge inputs and targets into one simple config dict for prepare_dataset
                # prepare_dataset expects {'B': 'method', 'Loss': 'method'}
                feats = model_conf['features'].get('inputs', {})
                targs = model_conf['features'].get('targets', {})
                norm_config = {**feats, **targs}
                print(f"Loaded config for '{model_type}': {norm_config}")
            else:
                print(f"WARNING: No 'features' section found for model '{model_type}'. Using defaults.")
        else:
             print(f"WARNING: Model '{model_type}' not found in config. Using defaults.")
    
    # Fallback default if empty
    if not norm_config:
        norm_config = {'B': 'standard', 'H': 'standard', 'Loss': 'standard', 'Frequency': 'standard'}

    print("--- 1. Physics Calculations ---")
    B_raw = calculate_flux_density(voltage, dt, mag_props['N_sec'], mag_props['Ae'])
    H_raw = calculate_magnetizing_force(current, mag_props['N_prim'], mag_props['Le'])
    Loss_raw = calculate_volumetric_loss(B_raw, H_raw, frequency)
    
    # Pack Potential Features
    all_features = {
        'B': B_raw,
        'H': H_raw,
        'Frequency': frequency.reshape(-1, 1) if np.ndim(frequency) > 0 else np.full((len(B_raw), 1), frequency)
    }
    
    all_targets = {
        'Loss': Loss_raw
    }
    
    # 2. Filter Features/Targets based on Config
    # We only include keys that are in the norm_config
    selected_features = {k: v for k, v in all_features.items() if k in norm_config}
    selected_targets = {k: v for k, v in all_targets.items() if k in norm_config}
    
    if not selected_features:
        print("WARNING: No features selected! Checking config...")
        # Fallback to keeping all if config was malformed to avoid returning nothing
        selected_features = all_features
        
    print(f"Selected Features: {list(selected_features.keys())}")
    print(f"Selected Targets: {list(selected_targets.keys())}")
    
    print("--- 2. Generic Normalization & Split ---")
    data_split, stats = prepare_dataset(selected_features, selected_targets, test_ratio=0.2, norm_config=norm_config)
    
    return data_split, stats

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    try:
        from data.loader import load_full_dataset
    except ImportError:
        # Fallback if running directly from script location
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from src.data.loader import load_full_dataset

    # --- CONFIGURATION ---
    # Adjust path to where the .mat file is located relative to this script
    # Script is in Scripts/src/data
    # File is in Scripts/3C90_TX-25-15-10_Data1_Cycle.mat
    MAT_FILE_NAME = '3C90_TX-25-15-10_Data1_Cycle.mat'
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to Scripts/
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) 
    # Check if file exists there, or one level deeper in Scripts
    possible_path_1 = os.path.join(data_dir, MAT_FILE_NAME)
    possible_path_2 = os.path.join(data_dir, 'Scripts', MAT_FILE_NAME) # If cwd was deeper
    # Based on previous file exploration, it seems to be in 'Scripts' folder which is parent of src?
    # Actually user said "Scripts\src\data\preprocessing.py"
    # And file found at "Scripts\3C90_TX-25-15-10_Data1_Cycle.mat"
    # So from src/data, we go up to src, then up to Scripts. -> ../..
    
    mat_file_path = os.path.abspath(os.path.join(current_dir, '..', '..', MAT_FILE_NAME))
    
    print(f"Looking for data at: {mat_file_path}")
    
    if not os.path.exists(mat_file_path):
        print(f"ERROR: Data file not found at {mat_file_path}")
        print("Please ensure the .mat file is in the correct directory.")
    else:
        print("Loading Full Dataset...")
        full_data = load_full_dataset(mat_file_path)
        
        # Extract features
        volts = full_data['voltage'] # (N_exp, N_samples)
        amps = full_data['current']
        freqs = full_data['freq']
        
        # Metadata
        props = {
            'N_prim': full_data['meta']['N_prim'],
            'N_sec': full_data['meta']['N_sec'],
            'Ae': full_data['meta']['Ae'],
            'Le': full_data['meta']['Le']
        }
        dt = full_data['meta']['dt']
        if isinstance(dt, (list, np.ndarray)) and len(dt) > 1:
            dt = dt[0] # Assume constant dt for now or take first
        
        print("\n--- Running Normalization Check ---")
        # Define a detailed config to test specific normalizations
        config_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'config', 'config.yaml'))
        
        # We can force a config here for testing purposes if we want to ensure 'standard' or 'minmax'
        # Let's verify 'standard' which is the default
        
        split_data, stats = process_magnet_dataset(
            volts, amps, freqs, props, dt, 
            model_type='scaler', 
            config_path=config_path
        )
        
        print("\nNormalization Stats Computed:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
            
        # --- VISUALIZATION ---
        print("\nPlotting Verification...")
        
        inputs = split_data['train']['inputs']
        targets = split_data['train']['targets']
        
        # Pick a random experiment index from the TRAIN set range
        # volts contains ALL data, but inputs contains only current split.
        # We want to verify that normalized data matches raw data FOR THE SAME SAMPLE.
        # The split logic does: train_part = data[:split_idx]
        # So inputs['B'][i] corresponds to volts[i] (and B_raw[i]) for i < split_idx.
        
        N_train = len(inputs['B']) if 'B' in inputs else len(list(inputs.values())[0])
        idx = 73680
        
        print(f"Verifying Sample Inded: {idx} (Train Split)")
        
        # Get Raw Data (Corresponding to the same index in the original array, since train is the first chunk)
        raw_v = volts[idx]
        raw_i = amps[idx]
        
        # Get Normalized Data
        # Note: process_magnet_dataset creates features B and H. 
        # But 'voltage' and 'current' themselves might not be in the output unless requested.
        # The default config usually asks for B, H, Loss.
        # Let's inspect what's in split_data
        
        inputs = split_data['train']['inputs']
        targets = split_data['train']['targets']
        
        # If B and H are there, let's plot those vs raw Voltage/Current (shape-wise comparison)
        # Or better, let's reconstruct B/H manually to compare apples to apples.
        
        B_norm = inputs['B'][idx] if 'B' in inputs else None
        H_norm = inputs['H'][idx] if 'H' in inputs else None
        
        # Calculate Raw B/H for comparison
        B_raw = calculate_flux_density(raw_v.reshape(1, -1), dt, props['N_sec'], props['Ae']).flatten()
        H_raw = calculate_magnetizing_force(raw_i.reshape(1, -1), props['N_prim'], props['Le']).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Flux Density (B)
        ax = axes[0, 0]
        ax.plot(B_raw, label='Raw Calculated B', color='blue')
        ax.set_title(f'Raw Flux Density (Exp {idx})')
        ax.grid(True)
        ax.legend()
        
        ax = axes[0, 1]
        if B_norm is not None:
            ax.plot(B_norm, label='Normalized B', color='orange')
            ax.set_title(f'Normalized B (Method: {stats["B"].get("mean", "N/A") if "B" in stats else "Unknown"})')
            # Add text for stats
            if 'B' in stats:
                txt = f"Mean: {stats['B'].get('mean', 0):.2e}\nStd: {stats['B'].get('std', 1):.2e}"
                ax.text(0.05, 0.95, txt, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "B not in output features", ha='center')
        ax.grid(True)
        ax.legend()
        
        # Plot 2: Magnetizing Force (H)
        ax = axes[1, 0]
        ax.plot(H_raw, label='Raw Calculated H', color='green')
        ax.set_title(f'Raw Magnetizing Force (Exp {idx})')
        ax.grid(True)
        ax.legend()
        
        ax = axes[1, 1]
        if H_norm is not None:
            ax.plot(H_norm, label='Normalized H', color='red')
            ax.set_title('Normalized H')
             # Add text for stats
            if 'H' in stats:
                txt = f"Mean: {stats['H'].get('mean', 0):.2e}\nStd: {stats['H'].get('std', 1):.2e}"
                ax.text(0.05, 0.95, txt, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
             ax.text(0.5, 0.5, "H not in output features", ha='center')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(current_dir, 'normalization_check.png')
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        # plt.show() # Comment out for headless environments

