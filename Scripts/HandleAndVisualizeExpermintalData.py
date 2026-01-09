import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import h5py
import mplcursors

import os

# --- 1. Load Data ---
# Using the existing .mat file in the directory
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '3C90_TX-25-15-10_Data1_Cycle.mat')
experiment_id = 1  # 1-based index from MATLAB
idx = experiment_id - 1 # Convert to 0-based index for Python

print(f"Loading {file_path}...")

# Initialize variables to allow scope access later
v_sec = None
i_prim = None
n_prim = None
n_sec = None
ae = None
le = None
dt = None
hdc = None
temperature = None
dutyp = None
frequency = None
flux_cmd = None


with h5py.File(file_path, 'r') as f:
    # Access the Data group
    Data = f['Data']
    
    # Helper to read dataset and handle dimensions
    # MATLAB [rows, cols] -> HDF5 [cols, rows]
    # We need to be careful with column-major vs row-major.
    # usually extracting a column 'idx' in MATLAB (if accessing by row)
    # Wait:
    # MATLAB: V_sec = Data.Voltage(ExperimentID,:); -> Row 'ExperimentID'
    # HDF5 for 'Voltage': Shape will be (N_samples, N_experiments)
    # So we need the 'ExperimentID' column in HDF5.
    
    v_ds = Data['Voltage'] # Shape likely (1000, 60000)
    v_sec = v_ds[:, idx] # Get all samples for this experiment
    
    i_ds = Data['Current']
    i_prim = i_ds[:, idx]
    
    # Scalar/Small arrays
    def get_scalar(key):
        val = np.array(Data[key])
        if val.size == 1:
            return val.item()
        return val

    n_prim = get_scalar('Primary_Turns')
    n_sec = get_scalar('Secondary_Turns')
    ae = get_scalar('Effective_Area')
    le = get_scalar('Effective_Length')
    
    # 1D arrays
    def get_indexed(key, idx):
        val = np.array(Data[key]).flatten()
        if len(val) > idx:
            return val[idx]
        return val.item()

    dt = get_indexed('Sampling_Time', idx)
    hdc = get_indexed('Hdc_command', idx)
    temperature = get_indexed('Temperature_command', idx)
    dutyp = get_indexed('DutyP_command', idx)
    frequency = get_indexed('Frequency_command', idx)
    flux_cmd = get_indexed('Flux_command', idx)

print("Data loaded successfully with h5py.")

# Create time vector
t = np.arange(len(v_sec)) * dt

# --- 2. Calculate Magnetizing Force (H) ---
# H = (N * I) / Le
h = (n_prim * i_prim) / le

# --- 3. Calculate Flux Density (B) ---
# Remove DC offset from voltage
v_clean = v_sec - np.mean(v_sec)

# Integrate Voltage: Integral(V) dt
flux = cumulative_trapezoid(v_sec, t, initial=0)

# B = Flux / (N * Ae)
b = flux / (n_sec * ae)
# Remove DC offset from B
b = b - np.mean(b)

# --- 4. Plotting ---
fig = plt.figure('B-H Analysis', figsize=(12, 8), facecolor='white')

# Subplot 1: Voltage and Current (Top-Left)
ax1 = plt.subplot2grid((2, 2), (0, 0))
color_v = 'tab:blue'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)', color=color_v)
ax1.plot(t, v_sec, color=color_v, linewidth=1.5, label='V (Volts)')
ax1.tick_params(axis='y', labelcolor=color_v)
ax1.grid(True)

ax2 = ax1.twinx()
color_i = 'tab:orange'
ax2.set_ylabel('Current (A)', color=color_i)
ax2.plot(t, i_prim, color=color_i, linewidth=1.5, label='I (Amps)')
ax2.tick_params(axis='y', labelcolor=color_i)
ax1.set_title('Voltage & Current')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Subplot 2: B and H (Bottom-Left)
ax3 = plt.subplot2grid((2, 2), (1, 0))
color_b = 'tab:blue'
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Flux Density B (Tesla)', color=color_b)
ax3.plot(t, b, color=color_b, linewidth=1.5, label='B (Tesla)')
ax3.tick_params(axis='y', labelcolor=color_b)
ax3.grid(True)

ax4 = ax3.twinx()
color_h = 'tab:orange'
ax4.set_ylabel('Field Strength H (A/m)', color=color_h)
ax4.plot(t, h, color=color_h, linewidth=1.5, label='H (A/m)')
ax4.tick_params(axis='y', labelcolor=color_h)
ax3.set_title('B & H Waveforms')

lines_3, labels_3 = ax3.get_legend_handles_labels()
lines_4, labels_4 = ax4.get_legend_handles_labels()
ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right')

# Subplot 3: B-H Loop (Right side)
ax_bh = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ax_bh.plot(h, b, linewidth=2)
ax_bh.set_xlabel('Magnetizing Force H (A/m)')
ax_bh.set_ylabel('Flux Density B (Tesla)')
ax_bh.set_title('B-H Hysteresis Loop')
ax_bh.grid(True)
ax_bh.autoscale(enable=True, axis='both', tight=True)

# Add Info Box
info_str = (
    f"Hdc: {hdc} A/m\n"
    f"Temp: {temperature} °C\n"
    f"Freq: {round(frequency/1000)} kHz\n"
    f"Duty: {dutyp}\n"
    f"Flux Cmd: {flux_cmd:.3f} T"
)

props = dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=1.0)
ax_bh.text(0.05, 0.95, info_str, transform=ax_bh.transAxes, verticalalignment='top', bbox=props)

plt.tight_layout()
mplcursors.cursor(hover=True) 
plt.show()
