"""
Data Loading Module.

This module is responsible for loading experimental data from MATLAB (.mat) files.
It handles both legacy (SciPy) and v7.3 (HDF5/h5py) formats.

Functions:
- load_experiment(file_path, experiment_id): Loads data for a single experiment.
"""

import numpy as np
import scipy.io
import h5py
import os

def load_experiment(file_path, experiment_id):
    """
    Loads data for a specific experiment ID from a .mat file.
    
    Args:
        file_path (str): Path to the .mat file.
        experiment_id (int): 1-based experiment index (as in MATLAB).
        
    Returns:
        dict: Dictionary containing experimental data (Voltage, Current, Metadata).
    """
    idx = experiment_id - 1  # Convert to 0-based index
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Try loading with scipy (for v7 and earlier)
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        Data = mat['Data']
        
        def get_val(obj, idx):
            if np.ndim(obj) > 0:
                return obj[idx]
            return obj

        data_dict = {
            'voltage': Data.Voltage[idx, :],
            'current': Data.Current[idx, :],
            'primary_turns': Data.Primary_Turns,
            'secondary_turns': Data.Secondary_Turns,
            'effective_area': Data.Effective_Area,
            'effective_length': Data.Effective_Length,
            'sampling_time': get_val(Data.Sampling_Time, idx),
            'hdc': get_val(Data.Hdc_command, idx),
            'temperature': get_val(Data.Temperature_command, idx),
            'duty_p': get_val(Data.DutyP_command, idx),
            'frequency': get_val(Data.Frequency_command, idx),
            'flux_cmd': get_val(Data.Flux_command, idx)
        }
        return data_dict

    except NotImplementedError:
        # Fallback to h5py for v7.3 files
        with h5py.File(file_path, 'r') as f:
            Data = f['Data']
            
            # Helper to access h5py data
            # Arrays are transposed in h5py relative to MATLAB
            # MATLAB: Voltage(ExperimentID, :) -> h5py: Voltage[Samples, ExperimentID]
            # We want all samples for the specific experiment column
            
            v_sec = Data['Voltage'][:, idx]
            i_prim = Data['Current'][:, idx]
            
            def get_scalar(key):
                val = np.array(Data[key])
                if val.size == 1:
                    return val.item()
                return val

            def get_indexed(key, idx):
                val = np.array(Data[key]).flatten()
                if len(val) > idx:
                    return val[idx]
                return val.item()

            data_dict = {
                'voltage': v_sec,
                'current': i_prim,
                'primary_turns': get_scalar('Primary_Turns'),
                'secondary_turns': get_scalar('Secondary_Turns'),
                'effective_area': get_scalar('Effective_Area'),
                'effective_length': get_scalar('Effective_Length'),
                'sampling_time': get_indexed('Sampling_Time', idx),
                'hdc': get_indexed('Hdc_command', idx),
                'temperature': get_indexed('Temperature_command', idx),
                'duty_p': get_indexed('DutyP_command', idx),
                'frequency': get_indexed('Frequency_command', idx),
                'flux_cmd': get_indexed('Flux_command', idx)
            }
            return data_dict

def get_dataset_info(file_path):
    """
    Retrieves metadata about the dataset without loading full data.
    
    Returns:
        num_experiments (int)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        # Try scipy
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        # shape might be (Experiments, samples) or transposed.
        # Based on prev code: Data.Voltage was accessed as [idx, :] -> (N_exp, N_samples)
        return mat['Data'].Voltage.shape[0]
    except NotImplementedError:
        with h5py.File(file_path, 'r') as f:
            # h5py: Voltage shape (Samples, Experiments)
            return f['Data']['Voltage'].shape[1]

def load_full_dataset(file_path):
    """
    Loads the entire dataset into memory.
    Optimized for bulk loading (reads full arrays).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        # SciPy
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        Data = mat['Data']
        
        return {
            'voltage': Data.Voltage,   # (N_exp, N_samples)
            'current': Data.Current,
            'freq': Data.Frequency_command,
            'temp': Data.Temperature_command,
            'hdc': Data.Hdc_command,
            'duty': Data.DutyP_command,
            # Add other fields as needed
            'meta': {
               'N_prim': Data.Primary_Turns,
               'N_sec': Data.Secondary_Turns,
               'Ae': Data.Effective_Area,
               'Le': Data.Effective_Length,
               'dt': Data.Sampling_Time
            }
        }
    except NotImplementedError:
        # h5py
        with h5py.File(file_path, 'r') as f:
            Data = f['Data']
            
            # Read all at once. Transpose to (N_exp, N_samples)
            # h5py stores as (samples, experiments).
            # T operation in numpy is cheap (view), but reading might be slow if chunked.
            # reading as [:] reads into memory.
            
            print("Reading full arrays from HDF5... this may take a moment.")
            v_all = np.array(Data['Voltage']).T
            i_all = np.array(Data['Current']).T
            
            # 1D arrays
            def get_arr(key):
                return np.array(Data[key]).flatten()
                
            data_dict = {
                'voltage': v_all,
                'current': i_all,
                'freq': get_arr('Frequency_command'),
                'temp': get_arr('Temperature_command'),
                'hdc': get_arr('Hdc_command'),
                'duty': get_arr('DutyP_command'),
                'meta': {
                   'N_prim': np.array(Data['Primary_Turns']).item(),
                   'N_sec': np.array(Data['Secondary_Turns']).item(),
                   'Ae': np.array(Data['Effective_Area']).item(),
                   'Le': np.array(Data['Effective_Length']).item(),
                   # Sampling time might be an array or scalar? 
                   # Assuming scalar for uniform sampling usually, or array.
                   'dt': np.array(Data['Sampling_Time']).flatten()
                }
            }
            return data_dict

# ==========================================
# VALIDATION SCRIPT (Append to end of file)
# ==========================================

if __name__ == "__main__":
    import os
    
    # --- CONFIGURATION ---
    # Use the filename you mentioned earlier
    TEST_FILE = '../../3C90_TX-25-15-10_Data1_Cycle.mat'
    TEST_ID = 3859  # The ID you used in your example
    
    # Get absolute path to ensure we find the file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, TEST_FILE)
    
    print(f"--- STARTING VALIDATION FOR: {TEST_FILE} ---")

    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
    else:
        # -------------------------------------------------
        # TEST 1: Single Experiment Loading
        # -------------------------------------------------
        print(f"\n[1] Testing load_experiment(id={TEST_ID})...")
        try:
            single_data = load_experiment(file_path, TEST_ID)
            
            v_shape = single_data['voltage'].shape
            i_shape = single_data['current'].shape
            freq = single_data['frequency']
            
            print(f"    Success!")
            print(f"    Voltage Shape: {v_shape} (Expected: 1D array, e.g., (10000,))")
            print(f"    Frequency:     {freq} Hz")
            
            if len(v_shape) != 1:
                print("    WARNING: Voltage should be 1D!")
        except Exception as e:
            print(f"    FAILED: {e}")

        # -------------------------------------------------
        # TEST 2: Full Dataset Loading
        # -------------------------------------------------
        print(f"\n[2] Testing load_full_dataset()...")
        try:
            full_data = load_full_dataset(file_path)
            
            full_v_shape = full_data['voltage'].shape
            
            print(f"    Success!")
            print(f"    Full Voltage Shape: {full_v_shape} (Expected: [N_experiments, N_samples])")
            print(f"    Total Experiments:  {full_v_shape[0]}")
            print(f"    Samples per Exp:    {full_v_shape[1]}")
            
        except Exception as e:
            print(f"    FAILED: {e}")
            full_data = None

        # -------------------------------------------------
        # TEST 3: Consistency Check (The Transpose Proof)
        # -------------------------------------------------
        if 'single_data' in locals() and full_data is not None:
            print(f"\n[3] Checking Consistency (Single vs Full)...")
            
            # The row in full dataset (0-based) must match the single load
            idx = TEST_ID - 1 
            
            # Extract row from full matrix
            full_slice_v = full_data['voltage'][idx, :]
            
            # Extract array from single load
            single_slice_v = single_data['voltage']
            
            # Check difference
            diff = np.max(np.abs(full_slice_v - single_slice_v))
            
            if diff < 1e-9:
                print(f"    PASSED: Data matches exactly (Diff: {diff})")
                print("    --> This confirms your Transpose/Index logic is CORRECT.")
            else:
                print(f"    FAILED: Data mismatch! (Diff: {diff})")
                print("    --> Check if Transpose (.T) is missing in load_full_dataset")
        
    print("\n--- VALIDATION COMPLETE ---")
