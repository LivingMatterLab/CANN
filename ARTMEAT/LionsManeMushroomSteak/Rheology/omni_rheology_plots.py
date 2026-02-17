import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os

# Custom colors
col1 = '#054700'   # Color for out-of-plane data
col2 = '#30ff1e'   # Color for in-plane data

#%% AMPLITUDE
def compute_stats(filename, sheet_name):
    data = pd.read_excel(filename, sheet_name=sheet_name)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    G_storage = data_num[:, 0] / 1000  # Pa to kPa
    G_loss = data_num[:, 1] / 1000
    return G_storage, G_loss

folder= 'amplitude'
# Sample names
inplane_samples = [os.path.join(folder, f'amplitude-inplane-{i}.xls') for i in range(1, 6)]
outplane_samples = [os.path.join(folder, f'amplitude-outplane-{i}.xls')for i in range(1, 6)]
# Read frequency values from one of the files
freqs = pd.read_excel(inplane_samples[0], sheet_name=2).iloc[2:, 9].to_numpy(dtype=np.float64)

# Accumulators
inplane_Gs = []
inplane_Gl = []
outplane_Gs = []
outplane_Gl = []

# Process in-plane samples
for sample in inplane_samples:
    Gs, Gl = compute_stats(sample, sheet_name=2)
    inplane_Gs.append(Gs)
    inplane_Gl.append(Gl)

# Process out-of-plane samples
for sample in outplane_samples:
    Gs, Gl = compute_stats(sample, sheet_name=2)
    outplane_Gs.append(Gs)
    outplane_Gl.append(Gl)

# Convert to numpy arrays
inplane_Gs = np.array(inplane_Gs)
inplane_Gl = np.array(inplane_Gl)
outplane_Gs = np.array(outplane_Gs)
outplane_Gl = np.array(outplane_Gl)

# Compute mean and std
mean_in_Gs = np.mean(inplane_Gs, axis=0)
std_in_Gs = np.std(inplane_Gs, axis=0)
mean_in_Gl = np.mean(inplane_Gl, axis=0)
std_in_Gl = np.std(inplane_Gl, axis=0)

mean_out_Gs = np.mean(outplane_Gs, axis=0)
std_out_Gs = np.std(outplane_Gs, axis=0)
mean_out_Gl = np.mean(outplane_Gl, axis=0)
std_out_Gl = np.std(outplane_Gl, axis=0)

# For G_storage
lower_Gs_in = np.maximum.accumulate(mean_in_Gs - std_in_Gs)
lower_Gs_out = np.maximum.accumulate(mean_out_Gs - std_out_Gs)

# For G_loss
lower_Gl_in = np.maximum.accumulate(mean_in_Gl - std_in_Gl)
lower_Gl_out = np.maximum.accumulate(mean_out_Gl - std_out_Gl)

# --- Plot G_storage ---
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
# Plot mean and fill ±std
ax.plot(freqs, mean_out_Gs, color=col1, linewidth=2, marker='o', markersize=5, label='cross-plane')
ax.fill_between(freqs, mean_out_Gs - std_out_Gs, mean_out_Gs + std_out_Gs, color=col1, alpha=0.3)

ax.plot(freqs, mean_in_Gs, color=col2, linewidth=2, marker='o', markersize=5, label='in-plane')
ax.fill_between(freqs, mean_in_Gs - std_in_Gs, mean_in_Gs + std_in_Gs, color=col2, alpha=0.3)

# Axis scaling and limits
# min_Gs = min(mean_in_Gs.min(), mean_out_Gs.min())
# corresponding_std_Gs = std_in_Gs[mean_in_Gs.argmin()] if mean_in_Gs.min() < mean_out_Gs.min() else std_out_Gs[mean_out_Gs.argmin()]
# ax.set_ylim(bottom=min_Gs - 0.5 * corresponding_std_Gs)
ax.set_xscale('log')
ax.set_xticks([0.01, 0.1, 2, 60])
ax.set_xticklabels(['0','0.001', '0.02', '0.6'])

# Remove spines and ticks
# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.set_xticks([])
# ax.tick_params(axis='x', which='both', length=0)
# ax.set_yticks([])

#Optional labels and legend
ax.set_ylabel("G' [kPa]")
ax.set_xlabel(r'$\log(\mathrm{\gamma})$ [-]')
ax.set_title("amplitude sweeps - storage modulus G' (n=5)")
ax.legend()

plt.tight_layout()
plt.show()

# --- Plot G_loss ---
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

ax.plot(freqs, mean_out_Gl, color=col1, linewidth=2.5, marker='o', label='cross-plane')
ax.fill_between(freqs, mean_out_Gl - std_out_Gl, mean_out_Gl + std_out_Gl, color=col1, alpha=0.3)

ax.plot(freqs, mean_in_Gl, color=col2, linewidth=2.5, marker='o', label='in-plane')
ax.fill_between(freqs, mean_in_Gl - std_in_Gl, mean_in_Gl + std_in_Gl, color=col2, alpha=0.3)

min_Gl = min(mean_in_Gl.min(), mean_out_Gl.min())
corresponding_std_Gl = std_in_Gl[mean_in_Gl.argmin()] if mean_in_Gl.min() < mean_out_Gl.min() else std_out_Gl[mean_out_Gl.argmin()]
#ax.set_ylim(bottom=min_Gl - 0.5 * corresponding_std_Gl)
ax.set_xscale('log')
ax.set_xticks([0.01, 0.1, 2, 60])
ax.set_xticklabels(['0','0.001', '0.02', '0.6'])
# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.set_xticks([])
# ax.tick_params(axis='x', which='both', length=0)
# ax.set_yticks([])

#Optional labels and legend
ax.set_ylabel("G'' [kPa]")
ax.set_xlabel(r'$\log(\mathrm{\gamma})$ [-]')
ax.set_title("amplitude sweeps - loss modulus G'' (n=5)")
ax.legend()

plt.tight_layout()
plt.show()


#%% FREQUENCY
def compute_stats(filename, sheet_name):
    data = pd.read_excel(filename, sheet_name=sheet_name)
    data_num = data.iloc[2:].to_numpy(dtype=np.float64)
    G_storage = data_num[:, 0] / 1000  # Pa to kPa
    G_loss = data_num[:, 1] / 1000
    return G_storage, G_loss

folder= 'frequency'
# Sample names
inplane_samples = [os.path.join(folder, f'frequency-inplane-{i}.xls') for i in range(1, 11)]
outplane_samples = [os.path.join(folder, f'frequency-outplane-{i}.xls')for i in range(1, 11)]
# Read frequency values from one of the files
freqs = pd.read_excel(inplane_samples[0], sheet_name=2).iloc[2:, 3].to_numpy(dtype=np.float64)

# Accumulators
inplane_Gs = []
inplane_Gl = []
outplane_Gs = []
outplane_Gl = []

# Process in-plane samples
for sample in inplane_samples:
    Gs, Gl = compute_stats(sample, sheet_name=2)
    inplane_Gs.append(Gs)
    inplane_Gl.append(Gl)

# Process out-of-plane samples
for sample in outplane_samples:
    Gs, Gl = compute_stats(sample, sheet_name=2)
    outplane_Gs.append(Gs)
    outplane_Gl.append(Gl)

# Convert to numpy arrays
inplane_Gs = np.array(inplane_Gs)
inplane_Gl = np.array(inplane_Gl)
outplane_Gs = np.array(outplane_Gs)
outplane_Gl = np.array(outplane_Gl)

# Compute mean and std
mean_in_Gs = np.mean(inplane_Gs, axis=0)
std_in_Gs = np.std(inplane_Gs, axis=0)
mean_in_Gl = np.mean(inplane_Gl, axis=0)
std_in_Gl = np.std(inplane_Gl, axis=0)

mean_out_Gs = np.mean(outplane_Gs, axis=0)
std_out_Gs = np.std(outplane_Gs, axis=0)
mean_out_Gl = np.mean(outplane_Gl, axis=0)
std_out_Gl = np.std(outplane_Gl, axis=0)

# For G_storage
lower_Gs_in = np.maximum.accumulate(mean_in_Gs - std_in_Gs)
lower_Gs_out = np.maximum.accumulate(mean_out_Gs - std_out_Gs)

# For G_loss
lower_Gl_in = np.maximum.accumulate(mean_in_Gl - std_in_Gl)
lower_Gl_out = np.maximum.accumulate(mean_out_Gl - std_out_Gl)

# --- Plot G_storage ---
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

# Plot mean and fill ±std
ax.plot(freqs, mean_out_Gs, color=col1, linewidth=2.5, marker='o', label='cross-plane')
ax.fill_between(freqs, mean_out_Gs - std_out_Gs, mean_out_Gs + std_out_Gs, color=col1, alpha=0.3)

ax.plot(freqs, mean_in_Gs, color=col2, linewidth=2.5, marker='o', label='in-plane')
ax.fill_between(freqs, mean_in_Gs - std_in_Gs, mean_in_Gs + std_in_Gs, color=col2, alpha=0.3)

# Axis scaling and limits
# min_Gs = min(mean_in_Gs.min(), mean_out_Gs.min())
# corresponding_std_Gs = std_in_Gs[mean_in_Gs.argmin()] if mean_in_Gs.min() < mean_out_Gs.min() else std_out_Gs[mean_out_Gs.argmin()]
# ax.set_ylim(bottom=min_Gs - 0.5 * corresponding_std_Gs)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
# Remove spines and ticks
# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.set_xticks([])
# ax.tick_params(axis='x', which='both', length=0)
# ax.set_yticks([])

#Optional labels and legend
ax.set_ylabel("G' [kPa]")
ax.set_xlabel(r'$\log(\mathrm{\omega})$ [rad/s]')
ax.set_title("frequency sweeps - storage modulus G' (n=10)")
ax.legend()

plt.tight_layout()
plt.show()

# --- Plot G_loss ---
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

ax.plot(freqs, mean_out_Gl, color=col1, linewidth=2, marker='o', markersize=5, label='cross-plane')
ax.fill_between(freqs, mean_out_Gl - std_out_Gl, mean_out_Gl + std_out_Gl, color=col1, alpha=0.3)

ax.plot(freqs, mean_in_Gl, color=col2, linewidth=2, marker='o', markersize=5, label='in-plane')
ax.fill_between(freqs, mean_in_Gl - std_in_Gl, mean_in_Gl + std_in_Gl, color=col2, alpha=0.3)

min_Gl = min(mean_in_Gl.min(), mean_out_Gl.min())
corresponding_std_Gl = std_in_Gl[mean_in_Gl.argmin()] if mean_in_Gl.min() < mean_out_Gl.min() else std_out_Gl[mean_out_Gl.argmin()]
#ax.set_ylim(bottom=min_Gl - 0.5 * corresponding_std_Gl)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())

# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.set_xticks([])
# ax.tick_params(axis='x', which='both', length=0)
# ax.set_yticks([])

#Optional labels and legend
ax.set_ylabel("G'' [kPa]")
ax.set_xlabel(r'$\log(\mathrm{\omega})$ [rad/s]')
ax.set_title("frequency sweeps - loss modulus G'' (n=10)")
ax.legend()

plt.tight_layout()
plt.show()


