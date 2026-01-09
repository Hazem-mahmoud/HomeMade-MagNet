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
            
        std[std == 0] = 1.0
        norm_data = (data - mean) / std
        
    return norm_data, stats

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

def calculate_volumetric_loss(voltage, current, time, effective_volume):
    """
    Calculates Volumetric Power Loss (Pv) in W/m^3.
    Pv = (1 / (T * Ve)) * Integral(V * I) dt
    """
    from scipy.integrate import trapz
    
    # Instantaneous Power
    p_inst = voltage * current
    
    # Energy per cycle
    energy = trapz(p_inst, time)
    
    # Period
    period = time[-1] - time[0]
    
    # Power Loss
    pv = energy / (period * effective_volume)
    
    return pv

