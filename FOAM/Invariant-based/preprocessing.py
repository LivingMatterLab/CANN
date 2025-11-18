"""
process_foam_data.py

Port of the provided MATLAB script to Python using numpy, pandas, matplotlib.
Ensure your data folder structure matches the MATLAB script expectations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ---------- Settings (match MATLAB) ----------
root_folder = "./input/raw_data/asics/final_tcs/"
colors = ["r", "b"]
foam_types = ["leap", "turbo"]

n_pts_table = 13
n_pts_plt = 101

# helper to ensure consistent float printing in latex composer
def fmt_fixed2(x):
    return f"{x:.2f}"

# ---------- Utility functions ----------

def deriv(y, x):
    """
    central-difference derivative similar to MATLAB function in script.
    y and x are 1D numpy arrays
    returns dydx of same length
    """
    y = np.asarray(y)
    x = np.asarray(x)
    dydx = np.empty_like(y, dtype=float)
    if len(y) < 2:
        return np.array([0.0])
    # forward difference for first
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    # central for middle
    if len(y) > 2:
        dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    # backward for last
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dydx

def average_curves(x, y, n_cycles, n_pts, min_peak_dist, loading_mode, max_strain=-1):
    """
    Port of MATLAB average_curves function.

    x, y : 1d arrays (displacement, measured quantity)
    n_cycles : how many cycles to average
    n_pts : number of interpolation points
    min_peak_dist : min distance between peaks (in samples)
    loading_mode : "shear", "ten", or "com"
    max_strain : optional max strain; if negative, derive from peaks
    Returns: x_out, y_out (both 1D numpy arrays)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    is_shear = (loading_mode == "shear")
    is_ten = (loading_mode == "ten")
    if not (is_shear or is_ten or loading_mode == "com"):
        raise ValueError("Invalid loading_mode")

    # find peaks (maxima) in x
    peaks_vals, peaks_idx = None, None
    # Using scipy find_peaks gives indices of peaks; we need peak values too
    peaks_idx, _ = find_peaks(x, distance=min_peak_dist, height=0)
    peaks_vals = x[peaks_idx]

    if peaks_idx.size == 0:
        # fallback: treat entire time series as one segment
        minmax_val = x.max()
    else:
        # take first n_cycles peaks (if available)
        if peaks_idx.size >= n_cycles:
            minmax_val = np.min(peaks_vals[:n_cycles])
        else:
            minmax_val = np.min(peaks_vals)  # fallback

    # offset used to find minima (in COM case)
    offset_val = (1 - (1 if is_shear else 0)) * minmax_val
    # find minima by finding peaks in offset - x
    inverted = offset_val - x
    minima_idx, _ = find_peaks(inverted, distance=min_peak_dist, height=0)
   
    # For constructing segment boundaries we try to follow MATLAB logic.
    # Identify maxima and minima sequences, and choose start.
    # If too few peaks are found, fallback to simple segmentation (start=0, end=end)
    if len(peaks_idx) == 0 or len(minima_idx) == 0:
        start = 0
        # we'll treat the whole array as a single segment
        maxima = np.array([len(x)-1])
        minima = np.array([len(x)-1])
    else:
        maxima = peaks_idx.copy()
        minima = minima_idx.copy()

        if is_ten:
            # For tension, start at index 0 and select first n_cycles maxima and minima
            start = 0
            # keep maxima that are greater than start
            maxima = maxima[maxima > start]
            if maxima.size < n_cycles:
                maxima = maxima[:max(1, maxima.size)]
            else:
                maxima = maxima[:n_cycles]
            minima = minima[:n_cycles] if minima.size >= n_cycles else minima
        else:
            # compression or shear flow in script: minima after first maxima
            # MATLAB logic: minima = minima(minima > maxima(1)); start = minima(1);
            maxima_after = maxima[maxima > maxima[0]] if maxima.size > 1 else maxima
            minima_after = minima[minima > maxima[0]]
            if minima_after.size == 0:
                start = 0
            else:
                start = minima_after[0]
            # choose first n_cycles maxima after start
            maxima = maxima[maxima > start]
            if maxima.size >= n_cycles:
                maxima = maxima[:n_cycles]
            # minima for cycles: in MATLAB they used minima(2:(n_cycles+1)) for com
            # so shift minima selection
            # choose minima that occur after maxima[0]
            # minima = minima[minima > maxima[0]] if maxima.size>0 else minima
            if minima.size >= (n_cycles + 1):
                minima = minima[1:(n_cycles + 1)]
            else:
                minima = minima[:n_cycles]

    # Build sequence of endpoints: start, maxima[0], minima[0], maxima[1], minima[1], ...
    # In MATLAB they do: end_pts = [maxima, minima]'; end_pts = [start; end_pts(:)];
    # We'll interleave maxima and minima
    interleaved = []
    # ensure maxima and minima arrays are paired properly:
    mlen = min(len(maxima), len(minima))
    # if equal lengths, interleave maxima[0], minima[0], maxima[1], minima[1] ...
    for i in range(mlen):
        interleaved.append(maxima[i])
        interleaved.append(minima[i])
    # if maxima has extra, append them
    if len(maxima) > mlen:
        interleaved.extend(list(maxima[mlen:]))
    if len(minima) > mlen:
        interleaved.extend(list(minima[mlen:]))

    # Prepend start
    end_pts = [int(start)] + [int(i) for i in interleaved]
    # ensure end_pts are within bounds and sorted
    end_pts = [max(0, min(len(x)-1, int(i))) for i in end_pts]
    # ensure increasing order; if not, fix by uniq sorting
    end_pts = sorted(list(dict.fromkeys(end_pts)))  # preserve order-ish by dict trick then sort

    # fallback: if this gives only single endpoint, force segmentation to [0, len-1]
    if len(end_pts) < 2:
        end_pts = [0, len(x)-1]

    # decide max_strain
    if max_strain < 0:
        max_strain_final = minmax_val
    else:
        max_strain_final = max_strain

    if is_shear:
        x_interp = np.linspace(-max_strain_final, max_strain_final, n_pts)
    else:
        x_interp = np.linspace(0, max_strain_final, n_pts)

    y_interp_all = []

    # For each segment between end_pts[i] and end_pts[i+1], map x_segment values
    for i in range(len(end_pts)-1):
        i0 = end_pts[i]
        i1 = end_pts[i+1]
        if i1 <= i0:
            continue
        x_segment = x[i0:i1+1]
        y_segment = y[i0:i1+1]
        # require x_unique and average y for repeated x
        if x_segment.size == 0:
            continue
        xu, inv = np.unique(x_segment, return_inverse=True)
        # average y for each unique x
        y_u = np.zeros_like(xu, dtype=float)
        counts = np.zeros_like(xu, dtype=float)
        for k, idx in enumerate(inv):
            y_u[idx] += y_segment[k]
            counts[idx] += 1
        y_u = y_u / np.maximum(counts, 1)
        # Interpolate y_u on xu to x_interp
        # For values outside xu range, numpy.interp returns endpoints (good)
        y_interp = np.interp(x_interp, xu, y_u)
        # fix NaN at first position if occurs (mimic MATLAB)
        if np.isnan(y_interp[0]):
            y_interp[0] = y_u[0]
        y_interp_all.append(y_interp)

    if len(y_interp_all) == 0:
        # no segments found; return zeros
        y_mean = np.zeros_like(x_interp)
    else:
        y_interp_all = np.array(y_interp_all)  # shape (n_segments, n_pts)
        # MATLAB special handling:
        if is_ten:
            # they took just first and last (?) then averaged; original code: y_interp_all = [y_interp_all(1, :); y_interp_all(end, :)];
            # We'll pick first and last rows to follow that.
            if y_interp_all.shape[0] >= 2:
                y_interp_all = np.vstack([y_interp_all[0, :], y_interp_all[-1, :]])
        if is_shear:
            midpoint = (len(x_interp) // 2)
            # in MATLAB they do midpoint = floor(length/2)+1 and then take midpoint:end,
            # then y_interp_all = (y_interp_all - fliplr(y_interp_all)) / 2
            # replicate: compute symmetric diff across midpoint
            # Flip horizontally and compute (y - flipped) / 2
            y_flipped = np.fliplr(y_interp_all)
            # Make sure same shape
            if y_flipped.shape == y_interp_all.shape:
                y_interp_all = (y_interp_all - y_flipped) / 2.0
            # keep right half from midpoint to end
            y_interp_all = y_interp_all[:, midpoint:]
            x_interp = x_interp[midpoint:]
        # compute mean across segments (omit nan)
        y_mean = np.nanmean(y_interp_all, axis=0)

    # y_out: subtract initial value as in MATLAB
    y_out = y_mean - y_mean[0]
    x_out = x_interp
    return x_out, y_out

# ---------- Main processing ----------
def main():
    # Storage arrays (numpy)
    n_materials = len(foam_types)
    stretch_ten = np.zeros((n_pts_plt, n_materials))
    stress_ten = np.zeros((n_pts_plt, n_materials))
    stress_ten_std = np.zeros((n_pts_plt, n_materials))


    # --- Tension ----------
    n_cycles = 5
    max_strain_ten = 0.3

    for foam_idx, foam in enumerate(foam_types):
        # read measurements CSV
        meas_path = os.path.join(root_folder, f"{foam}_tension_measurements.csv")
        meas = pd.read_csv(meas_path, header=None).values  # table2array equivalent
        # widths: columns 4:6 in MATLAB are indices 3,4,5 (1-based). In python zero-based: 3:6
        widths_mm = np.mean(meas[:, 3:6], axis=1)
        heights_mm = np.mean(meas[:, 6:9], axis=1)
        areas_mm2 = widths_mm * heights_mm
        gauge_lens_mm = meas[:, 9]  # column 10 in MATLAB

        stress_all_plt = []

        for sample_idx in range(1,6):  # MATLAB samples 1..5
            data_path = os.path.join(root_folder, f"{foam}_tension_{sample_idx}_1.csv")
            # Skip metadata lines ("Results Table 1", "Results Table 2", etc.) and header rows
            # Read the file and find where numeric data starts
            try:
                # Try reading with skiprows to skip metadata (typically first 8 lines)
                data = pd.read_csv(data_path, header=None, skiprows=8, on_bad_lines='skip', engine='python').values
            except (TypeError, ValueError):
                # Fallback: read all and filter to find numeric data
                try:
                    raw_data = pd.read_csv(data_path, header=None, on_bad_lines='skip', engine='python')
                except TypeError:
                    raw_data = pd.read_csv(data_path, header=None, error_bad_lines=False, warn_bad_lines=False, engine='python')
                
                # Find first row where column 2 can be converted to float (actual data starts)
                data_start_row = None
                for idx in range(len(raw_data)):
                    try:
                        float(raw_data.iloc[idx, 2])
                        data_start_row = idx
                        break
                    except (ValueError, TypeError, IndexError):
                        continue
                
                if data_start_row is None:
                    raise ValueError(f"Could not find numeric data in {data_path}")
                
                data = raw_data.iloc[data_start_row:].values
            
            # displacement_mm = data(3:end, 3) in MATLAB -> python rows 2: , col index 2
            # After skipping metadata, data[0] is first data row, so we still skip first 2 rows if needed
            # But if we already skipped to data, we might not need to skip more
            # Check if first row is numeric or header
            try:
                float(data[0, 2])
                start_idx = 0  # Data starts immediately
            except (ValueError, TypeError, IndexError):
                start_idx = 2  # Skip header rows
            displacement_mm = data[start_idx:, 2].astype(float)
            force_n = data[start_idx:, 3].astype(float)
            # gauge_lens_mm is per sample; MATLAB uses gauge_lens_mm(sample_idx)
            strain = displacement_mm / gauge_lens_mm[sample_idx-1]
            stress_kpa = force_n / areas_mm2[sample_idx-1] * 1000.0

            min_peak_dist = 100
            x_interp_plt, stress_mean_kpa_plt = average_curves(strain, stress_kpa, n_cycles, n_pts_plt, min_peak_dist, "ten", max_strain_ten)
            stress_all_plt.append(stress_mean_kpa_plt)

        stress_all_plt = np.array(stress_all_plt)  # shape (n_samples, n_pts_plt)
        stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
        stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)

        # Resample for table
        # strain_interp_table = np.linspace(0.0, max_strain_ten, n_pts_table)
        # stress_mean_table = np.interp(strain_interp_table, x_interp_plt, stress_mean_plt)
        # stress_var_table = np.interp(strain_interp_table, x_interp_plt, stress_var_plt)

        stretch_ten[:, foam_idx] = 1.0 + x_interp_plt
        stress_ten[:, foam_idx] = stress_mean_plt
        stress_ten_std[:, foam_idx] = stress_var_plt


    # --- Compression ----------
    max_strain_com = 0.6
    offset = [1, 0]  # MATLAB offset array
    n_cycles = 4
    min_peak_dist = 1000

    stretch_com = np.zeros((n_pts_plt, n_materials))
    stress_com = np.zeros((n_pts_plt, n_materials))
    stress_com_std = np.zeros((n_pts_plt, n_materials))
    for foam_idx, foam in enumerate(foam_types):
        meas_path = os.path.join(root_folder, f"{foam}_comp_measurements.csv")
        meas = pd.read_csv(meas_path, header=None).values
        diameters_mm = np.mean(meas[:, 0:3], axis=1)  # columns 1:3 in MATLAB
        areas_mm2 = (diameters_mm ** 2) * np.pi / 4.0

        stress_all_plt = []
        for sample_idx in range(1,6):
            # note: MATLAB used file index sample_idx + offset(foam_idx)
            file_idx = sample_idx + offset[foam_idx]
            comp_path = os.path.join(root_folder, f"{foam}_comp_{file_idx}.txt")
            # many .txt used whitespace delim
            # Read as DataFrame first to handle string conversion and NaN columns
            df = pd.read_csv(comp_path, delim_whitespace=True, header=None)
            # Convert all columns to numeric, coercing errors to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            # Drop columns that are all NaN
            df = df.dropna(axis=1, how='all')
            # Convert to numpy array
            data = df.values
            
            gap_mm = data[1:, 5] / 1e3  # column 6 in MATLAB divided by 1000
            force_n = data[1:, 4]
            strain = (gap_mm[0] - gap_mm) / gap_mm[0]
            stress_kpa = force_n / areas_mm2[sample_idx-1] * 1000.0

            x_interp_plt, stress_mean_kpa = average_curves(strain, stress_kpa, n_cycles, n_pts_plt, min_peak_dist, "com", max_strain_com)
            stress_all_plt.append(stress_mean_kpa)

        stress_all_plt = np.array(stress_all_plt)
        stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
        stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)

        # # Resample for table
        # strain_interp_table = np.linspace(0.0, max_strain_com, n_pts_table)
        # stress_mean_table = np.interp(strain_interp_table, x_interp_plt, stress_mean_plt)
        # stress_var_table = np.interp(strain_interp_table, x_interp_plt, stress_var_plt)

        stretch_com[:, foam_idx] = 1.0 - x_interp_plt
        stress_com[:, foam_idx] = -stress_mean_plt
        stress_com_std[:, foam_idx] = stress_var_plt

       

    # --- Shear ----------
    max_shr = 0.15
    offset = [2, 0]
    n_cycles = 3
    min_peak_dist = 1000
    strain_interp_plt = np.linspace(0, max_shr, n_pts_plt)

    strain_shr = np.zeros((n_pts_plt, n_materials))
    stress_shr = np.zeros((n_pts_plt, n_materials))
    stress_shr_std = np.zeros((n_pts_plt, n_materials))
    for foam_idx, foam in enumerate(foam_types):
        meas_path = os.path.join(root_folder, f"{foam}_shear_measurements.csv")
        meas = pd.read_csv(meas_path, header=None).values
        radii_mm = np.mean(meas[:, 0:3], axis=1) / 2.0

        stress_all_plt = []
        for sample_idx in range(1,6):
            file_idx = sample_idx + offset[foam_idx]
            shear_path = os.path.join(root_folder, f"{foam}_shear_{file_idx}.xls")
            # sheet "Sine Strain - 3"
            # Read Excel file with explicit engine and handle data conversion
            try:
                df = pd.read_excel(shear_path, sheet_name="Sine Strain - 3", header=None, engine='xlrd')
            except Exception:
                # fallback try default sheet
                try:
                    df = pd.read_excel(shear_path, header=None, engine='xlrd')
                except Exception as e:
                    # If xlrd fails, try openpyxl (for .xlsx files) or other engines
                    try:
                        df = pd.read_excel(shear_path, sheet_name="Sine Strain - 3", header=None, engine='openpyxl')
                    except Exception:
                        df = pd.read_excel(shear_path, header=None, engine='openpyxl')
            
            # Convert all columns to numeric, coercing errors to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            # Drop columns that are all NaN
            df = df.dropna(axis=1, how='all')
            # Convert to numpy array
            data = df.values

            torque_nmm = data[3:, 0] / 1e3  # column 1 divided by 1000
            disp_rad = data[3:, 1]
            force_n = data[3:, 4]
            gap_mm = data[3:, 5] / 1e3
            height_mm = gap_mm[0]
            safety_factor = 1.01
            # disp_rad_max = height_mm / radius * max_shr * safety_factor
            r = radii_mm[sample_idx-1]
            disp_rad_max = (height_mm / r) * max_shr * safety_factor if r != 0 else max_shr

            disp_rad_interp, torque_nmm_interp_mean = average_curves(disp_rad, torque_nmm, n_cycles, 101, min_peak_dist, "shear", disp_rad_max)
            
            # compute shear strain: radii_mm * disp_rad_interp / height_mm
            strain_vals = r * disp_rad_interp / height_mm
            torque_times_disp_nmm = torque_nmm_interp_mean * disp_rad_interp
            # shear stress kPa formula from MATLAB:
            # shear_stress_kpa = 1000 / (2*pi*r^3) * (2*T + deriv(T*disp, disp))
            torque_times_disp_deriv = deriv(torque_times_disp_nmm, disp_rad_interp)
            shear_stress_kpa = 1000.0 / (2.0 * np.pi * r**3) * (2.0 * torque_nmm_interp_mean + torque_times_disp_deriv)

            # Interpolate shear stress onto common strain_interp_plt
            stress_interp_kpa = np.interp(strain_interp_plt, strain_vals, shear_stress_kpa, left=np.nan, right=np.nan)
            stress_all_plt.append(stress_interp_kpa)

        stress_all_plt = np.array(stress_all_plt)
        stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
        stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)

        # Resample for table
        # strain_interp_table = np.linspace(0.0, max_shr, n_pts_table)
        # stress_mean_table = np.interp(strain_interp_table, strain_interp_plt, stress_mean_plt)
        # stress_var_table = np.interp(strain_interp_table, strain_interp_plt, stress_var_plt)

        strain_shr[:, foam_idx] = strain_interp_plt
        stress_shr[:, foam_idx] = stress_mean_plt
        stress_shr_std[:, foam_idx] = stress_var_plt



    ## Interpolate to table
    stretch_ten_table = np.linspace(np.min(stretch_ten), np.max(stretch_ten), n_pts_table)
    stress_ten_table = np.stack([np.interp(stretch_ten_table, stretch_ten[:, foam_idx], stress_ten[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stress_ten_std_table = np.stack([np.interp(stretch_ten_table, stretch_ten[:, foam_idx], stress_ten_std[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stretch_com_table = np.linspace(np.min(stretch_com), np.max(stretch_com), n_pts_table)
    stress_com_table = np.stack([np.interp(stretch_com_table, stretch_com[::-1, foam_idx], stress_com[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stress_com_std_table = np.stack([np.interp(stretch_com_table, stretch_com[::-1, foam_idx], stress_com_std[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    strain_shr_table = np.linspace(np.min(strain_shr), np.max(strain_shr), n_pts_table)
    stress_shr_table = np.stack([np.interp(strain_shr_table, strain_shr[:, foam_idx], stress_shr[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stress_shr_std_table = np.stack([np.interp(strain_shr_table, strain_shr[:, foam_idx], stress_shr_std[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    print(stretch_ten_table.shape)
    # ---------- Create all plots at the end ----------
    # Tension plot
    plt.figure(figsize=(8,6))
    for foam_idx in range(n_materials):
        
        plt.plot(stretch_ten[:, foam_idx], stress_ten[:, foam_idx], colors[foam_idx], label=f"{foam} mean")
        plt.plot(stretch_ten_table, stress_ten_table[:, foam_idx], colors[foam_idx] + "o", markersize=4)
        plt.fill_between(stretch_ten[:, foam_idx],
                         stress_ten[:, foam_idx] - stress_ten_std[:, foam_idx],
                         stress_ten[:, foam_idx] + stress_ten_std[:, foam_idx],
                         color=colors[foam_idx], alpha=0.25)
    plt.xlim([1.0, 1.3])
    plt.xlabel("Stretch [-]")
    plt.ylabel("Stress [kPa]")
    plt.title("Tension")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compression plot
    plt.figure(figsize=(8,6))
    for foam_idx in range(n_materials):
        plt.plot(stretch_com[:, foam_idx], stress_com[:, foam_idx], colors[foam_idx], label=f"{foam} mean")
        plt.plot(stretch_com_table, stress_com_table[:, foam_idx], colors[foam_idx] + "o", markersize=4)
        plt.fill_between(stretch_com[:, foam_idx],
                         stress_com[:, foam_idx] - stress_com_std[:, foam_idx],
                         stress_com[:, foam_idx] + stress_com_std[:, foam_idx],
                         color=colors[foam_idx], alpha=0.25)
    plt.xlabel("Stretch")
    plt.ylabel("Stress [kPa]")
    plt.title("Compression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Shear plot
    plt.figure(figsize=(8,6))
    for foam_idx in range(n_materials):
        plt.plot(strain_shr[:, foam_idx], stress_shr[:, foam_idx], colors[foam_idx], label=f"{foam} mean")
        plt.plot(strain_shr_table, stress_shr_table[:, foam_idx], colors[foam_idx] + "o", markersize=4)
        plt.fill_between(strain_shr[:, foam_idx],
                         stress_shr[:, foam_idx] - stress_shr_std[:, foam_idx],
                         stress_shr[:, foam_idx] + stress_shr_std[:, foam_idx],
                         color=colors[foam_idx], alpha=0.25)
    plt.xlabel("Shear Strain [-]")
    plt.ylabel("Shear Stress [kPa]")
    plt.title("Shear")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    exit()
    # ---------- Write to LaTeX (strings) ----------
    stress_ten_kPa = stress_ten.copy()
    stress_com_kPa = np.abs(stress_com)
    stress_shr_kPa = stress_shr.copy()
    std_ten_kPa = stress_ten_std.copy()
    std_com_kPa = stress_com_std.copy()
    std_shr_kPa = stress_shr_std.copy()

    # Combine all data (order: tension, compression, shear)
    stretch_all = np.hstack([stretch_ten, stretch_com, strain_shr])
    stress_all = np.hstack([stress_ten_kPa, stress_com_kPa, stress_shr_kPa])
    std_all = np.hstack([std_ten_kPa, std_com_kPa, std_shr_kPa])

    nRows = stretch_all.shape[0]
    nModes = 3
    nMaterials = stress_ten.shape[1]
    stretchSymbols = [r'$\lambda$', r'$\lambda$', r'$\gamma$']

    materialTables = []

    for mat in range(nMaterials):
        # Determine max integer digits for mean and std per mode
        maxIntDigitsMean = np.zeros(nModes, dtype=int)
        maxIntDigitsStd = np.zeros(nModes, dtype=int)
        maxFracDigitsMean = 2
        maxFracDigitsStd = 2

        for mode in range(nModes):
            colIdx = 2*(mode) + mat  # MATLAB used 2*(mode-1)+mat with 1-based indexing
            # in MATLAB colIdx indexing tricky; here we approximate by selecting columns appropriately
            # but to follow original, assume stress_all layout: [ten_mat1, ten_mat2, com_mat1, com_mat2, shr_mat1, shr_mat2]
            # So indexing above should work if data arranged similarly.
            colIdx_global = colIdx
            # protect index
            if colIdx_global >= stress_all.shape[1]:
                valmax = 1.0
            else:
                valmax = np.max(np.maximum(1.0, stress_all[:, colIdx_global]))
            maxIntDigitsMean[mode] = int(np.floor(np.log10(valmax))) + 1 if valmax > 0 else 1
            stdmax = np.max(np.maximum(1.0, std_all[:, colIdx_global])) if colIdx_global < std_all.shape[1] else 1.0
            maxIntDigitsStd[mode] = int(np.floor(np.log10(stdmax))) + 1 if stdmax > 0 else 1

        lines = []
        lines.append(r'\begin{tabular}{ccc ccc ccc}')
        lines.append(r'\hline')
        # Header
        headerParts = []
        for mode in range(nModes):
            headerParts.append(f"{stretchSymbols[mode]} & $P$ [kPa]")
        lines.append(' & '.join(headerParts) + r' \\')
        lines.append(r'\hline')

        # Data rows
        for i in range(nRows):
            rowParts = []
            for mode in range(nModes):
                colIdx_global = 2*(mode) + mat
                if colIdx_global >= stress_all.shape[1]:
                    meanVal = 0.0
                    stdVal = 0.0
                    # Use a safe default stretch value (e.g., 1.0 for lambda, 0.0 for gamma)
                    if mode == 2:  # shear mode uses gamma
                        stretch_val = 0.0
                    else:  # tension/compression use lambda
                        stretch_val = 1.0
                else:
                    meanVal = stress_all[i, colIdx_global]
                    stdVal = std_all[i, colIdx_global]
                    stretch_val = stretch_all[i, colIdx_global]

                # round to 2 decimals as MATLAB: multiply by 100 then round
                meanRounded = int(np.round(meanVal * 100))
                meanInt = meanRounded // 100
                meanFrac = meanRounded - meanInt * 100
                stdRounded = int(np.round(stdVal * 100))
                stdInt = stdRounded // 100
                stdFrac = stdRounded - stdInt * 100

                nPhantomMean = maxIntDigitsMean[mode] - (int(np.floor(np.log10(max(1, meanInt)))) + 1 if meanInt>0 else 1)
                nPhantomStd = maxIntDigitsStd[mode] - (int(np.floor(np.log10(max(1, stdInt)))) + 1 if stdInt>0 else 1)
                meanPhantom = r'\phantom{0}' * max(0, nPhantomMean)
                stdPhantom = r'\phantom{0}' * max(0, nPhantomStd)

                rowParts.append(f"{stretch_val:.3f} & {meanPhantom}{meanInt}.{meanFrac:02d}\\hspace{{0.5em}}$\\pm$ {stdPhantom}{stdInt}.{stdFrac:02d}")
            lines.append(' & '.join(rowParts) + r' \\')
        lines.append(r'\hline')
        lines.append(r'\end{tabular}')
        materialTables.append("\n".join(lines))

    for mat_idx, tbl in enumerate(materialTables, start=1):
        print(f"LaTeX table for Material {mat_idx}:\n{tbl}\n\n")

    # ---------- Write to file (Excel) ----------
    # Recreate MATLAB final blocks: stretch_ut, stress_ut, stress_ut_std etc.
    stretch_ut = np.vstack([np.flipud(stretch_com), stretch_ten[1:, :]])
    stress_ut = np.vstack([np.flipud(stress_com), stress_ten[1:, :]])
    stress_ut_std = np.vstack([np.flipud(stress_com_std), stress_ten_std[1:, :]])

    strain_ss = np.vstack([-np.flipud(strain_shr), strain_shr[1:, :]])
    stress_ss = np.vstack([-np.flipud(stress_shr), stress_shr[1:, :]])
    stress_ss_std = np.vstack([np.flipud(stress_shr_std), stress_shr_std[1:, :]])

    data_cols = []
    headings = []
    for i, foam in enumerate(foam_types):
        # comten-stretch, comten-stress, comten-stddev, shr-strain, shr-stress, shr-stddev
        data_cols.append(stretch_ut[:, i])
        headings.append(f"{foam}-comten-stretch")
        data_cols.append(stress_ut[:, i])
        headings.append(f"{foam}-comten-stress")
        data_cols.append(stress_ut_std[:, i])
        headings.append(f"{foam}-comten-stddev")
        data_cols.append(strain_ss[:, i])
        headings.append(f"{foam}-shr-strain")
        data_cols.append(stress_ss[:, i])
        headings.append(f"{foam}-shr-stress")
        data_cols.append(stress_ss_std[:, i])
        headings.append(f"{foam}-shr-stddev")

    # Stack columns into DataFrame
    df_out = pd.DataFrame(np.column_stack(data_cols), columns=headings)
    out_dir = "./input"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FoamData2.xlsx")
    df_out.to_excel(out_path, index=False)
    print(f"Wrote output excel to: {out_path}")

if __name__ == "__main__":
    main()
