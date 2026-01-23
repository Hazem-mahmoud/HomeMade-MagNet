"""
Data Preprocessing Module.

This module prepares the raw data for neural network training.
It includes normalization, sequence generation, and dataset splitting.

Functions:
- normalize_data(data, method='minmax'): Applies MinMax or Standard scaling.
- create_sequences(data, sequence_length): Generates sliding windows for time-series models.
- split_data(data_len, ratio): Returns indices for splitting.
"""
import numpy as np

def normalize_data(data, method='minmax', stats=None):
    """
    Normalizes data using MinMax or Standard scaling.
    
    Args:
        data (np.array): Input data.
        method (str): 'minmax' or 'standard'.
        stats (dict, optional): Pre-computed stats (min/max or mean/std) for consistency (e.g. valid/test set).
        
    Returns:
        norm_data (np.array): Normalized data.
        stats (dict): Statistics used for normalization.
    """
    if method == 'minmax':
        if stats:
            min_val = stats['min']
            max_val = stats['max']
        else:
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            stats = {'min': min_val, 'max': max_val}
            
        # Avoid division by zero
        denom = max_val - min_val
        if np.isscalar(denom):
            if denom == 0:
                denom = 1.0
        else:
            denom[denom == 0] = 1.0
        
        norm_data = (data - min_val) / denom
        
    elif method == 'standard':
        if stats:
            mean = stats['mean']
            std = stats['std']
        else:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            stats = {'mean': mean, 'std': std}
            
        if np.ndim(std) == 0:
            if std == 0:
                std = 1.0
        else:
            std[std == 0] = 1.0
            
        norm_data = (data - mean) / std
        
    return norm_data.astype(np.float32), stats

def calculate_flux_density(voltage, sampling_time, secondary_turns, effective_area):
    """
    Calculates Flux Density (B) from Voltage.
    B = Integral(V) / (N * Ae)
    """
    from scipy.integrate import cumulative_trapezoid
    
    # Remove DC offset
    v_clean = voltage - np.mean(voltage)
    
    # Integrate
    flux = cumulative_trapezoid(v_clean, dx=sampling_time, initial=0)
    
    # Calculate B
    b = flux / (secondary_turns * effective_area)
    
    # Remove Drift
    b = b - np.mean(b)
    
    return b

def calculate_magnetizing_force(current, primary_turns, effective_length):
    """
    Calculates Magnetizing Force (H) from Current.
    H = (N * I) / Le
    """
    h = (primary_turns * current) / effective_length
    return h

def calculate_volumetric_loss(b_field, h_field, frequency):
    """
    Calculates Volumetric Power Loss (Pv) in W/m^3 using B-H loop area.
    Pv = Frequency * Area(B-H Loop)
       = Frequency * abs(Integral(H dB))
    
    Args:
        b_field: (N, Samples) Flux Density [T]
        h_field: (N, Samples) Magnetizing Force [A/m]
        frequency: scalar or (N,) array [Hz]
    """
    from scipy.integrate import trapezoid
    
    # Calculate Energy Density (Area of B-H loop) in J/m^3
    # Integration is H dB
    # axis=-1 usually corresponds to the time/sample dimension
    energy_density = trapezoid(y=h_field, x=b_field, axis=-1)
    
    # Take absolute value as area is positive energy loss
    energy_density = np.abs(energy_density)
    
    # Power Loss = Energy Density * Frequency
    # Ensure frequency is broadcastable if needed
    if np.ndim(frequency) > 0 and np.ndim(energy_density) > 0:
        freq = np.array(frequency).squeeze()
    else:
        freq = frequency
        
    pv = energy_density * freq
        
    return pv



def test_preprocessing_functions():
    """
    Runs tests using the ACTUAL dataset ('3C90_TX-25-15-10_Data1_Cycle.mat').
    """
    import sys
    import os
    
    # Add project root to path to allow imports
    # Assuming this script is in Scripts/src/data/preprocessing.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # HomeMade MagNet/Scripts/src/data -> HomeMade MagNet/Scripts
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    try:
        import loader
    except ImportError:
        # Fallback if running from proper root
        import loader
        
    print("Running Preprocessing Tests with REAL Data...")
    
    # Path to data file
    data_file_name = '3C90_TX-25-15-10_Data1_Cycle.mat'
    data_path = os.path.join(project_root, data_file_name)
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return

    # Load Experiment 1
    exp_id = 1
    print(f"Loading Experiment {exp_id} from {data_file_name}...")
    try:
        data = loader.load_experiment(data_path, exp_id)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
        
    # Extract Raw Data
    voltage = data['voltage'].astype(np.float32)
    current = data['current'].astype(np.float32)
    time_scalar = data['sampling_time'] # This might be scalar or array
    
    # Construct Time Array
    # If sampling_time is a scalar dt, we construct t array.
    if np.ndim(time_scalar) == 0:
        dt = float(time_scalar)
        time = np.arange(len(voltage)) * dt
    else:
        time = time_scalar
        dt = time[1] - time[0] # Approx
        
    # Metadata
    N_prim = data['primary_turns']
    N_sec = data['secondary_turns']
    Ae = data['effective_area']
    Le = data['effective_length']
    Ve = Ae * Le
    
    print(f" - Loaded {len(voltage)} samples.")
    print(f" - dt: {dt:.2e} s")
    print(f" - Ae: {Ae}, Le: {Le}, Ve: {Ve}")
    
    # 1. Calculate B
    print("Testing B Calculation...")
    b_field = calculate_flux_density(voltage, dt, N_sec, Ae)
    print(f" - B: Min {b_field.min():.4f}, Max {b_field.max():.4f}, Mean {b_field.mean():.4f}")
    
    # 2. Calculate H
    print("Testing H Calculation...")
    h_field = calculate_magnetizing_force(current, N_prim, Le)
    print(f" - H: Min {h_field.min():.4f}, Max {h_field.max():.4f}, Mean {h_field.mean():.4f}")
    
    # 3. Calculate Power Loss
    print("Testing Power Loss Calculation (B-H Loop Area)...")
    # Extract frequency 
    freq_val = data['frequency']
    
    # We now pass B and H fields
    pv = calculate_volumetric_loss(b_field, h_field, frequency=freq_val)
    print(f" - Volumetric Power Loss: {pv:.4f} W/m^3")
    
    # 4. Normalize
    print("Testing Normalization...")
    # 4a. MinMax
    b_norm, stats_minmax = normalize_data(b_field, method='minmax')
    print(f" - MinMax Normalized B: Min {b_norm.min():.4f}, Max {b_norm.max():.4f}")
    
    # 4b. Standard (Z-Score)
    b_std, stats_std = normalize_data(b_field, method='standard')
    print(f" - Standard Normalized B: Mean {b_std.mean():.4f}, Std {b_std.std():.4f}")
    print(f"   (Stats: Mean {stats_std['mean']:.4f}, Std {stats_std['std']:.4f})")
    
    # 4c. Re-using Stats (Simulating Train/Val split)
    # Applying the SAME stats to the SAME data should result in identical output
    b_std_re, _ = normalize_data(b_field, method='standard', stats=stats_std)
    assert np.allclose(b_std, b_std_re), "Stats reuse failed!"
    print(" - Stats Reuse (Consistency Check): PASS")
    
    print("Real Data Test Complete.")

if __name__ == "__main__":
    test_preprocessing_functions()
