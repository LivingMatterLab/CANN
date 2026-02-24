"""
process_foam_data.py

Port of the provided MATLAB script to Python using numpy, pandas, matplotlib.
Ensure your data folder structure matches the MATLAB script expectations.
"""

# TODO: Recollect leap confcomp data / pull last confcomp data from lab computer
# TODO: Figure out what to do with original tension data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks

# ---------- Settings (match MATLAB) ----------
root_folder = "./input/raw_data/asics/final_tcs/"
out_dir = "./input"

colors = ["r", "b"]
foam_types = ["leap", "turbo"]

n_pts_table = 13
n_pts_plt = 101

# Global font size for all plot text elements
FONT_SIZE = 20

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
    # is_ten = (loading_mode == "ten")
    # if not (is_shear or is_ten or loading_mode == "com"):
    #     raise ValueError("Invalid loading_mode")

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

        ## No longer use this code since we will treat tension the same as comp/shr
        # if is_ten:
        #     # For tension, start at index 0 and select first n_cycles maxima and minima
        #     start = 0
        #     # keep maxima that are greater than start
        #     maxima = maxima[maxima > start]
        #     if maxima.size < n_cycles:
        #         maxima = maxima[:max(1, maxima.size)]
        #     else:
        #         maxima = maxima[:n_cycles]
        #     minima = minima[:n_cycles] if minima.size >= n_cycles else minima
        # else:
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
    segment_means = []
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
        segment_means.append(np.mean(y_u[xu > 0]))

    y_interp_all = np.array(y_interp_all)  # shape (n_segments, n_pts)
    # MATLAB special handling:
    # if is_ten:
    #     # they took just first and last (?) then averaged; original code: y_interp_all = [y_interp_all(1, :); y_interp_all(end, :)];
    #     # We'll pick first and last rows to follow that.
    #     if y_interp_all.shape[0] >= 2:
    #         y_interp_all = np.vstack([y_interp_all[0, :], y_interp_all[-1, :]])
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
    # compute mean across segments (omit nan) for full range (used for y_out)
    y_mean = np.nanmean(y_interp_all, axis=0)

    #### Compute Hysteresis
    loading_mean = np.mean(segment_means[0::2])
    unloading_mean = np.mean(segment_means[1::2])
    energy_loss = loading_mean - unloading_mean
    energy_storage = (loading_mean + unloading_mean) / 2
    if not is_shear:
        energy_storage = energy_storage - y_mean[0]
    hysteresis = energy_loss / energy_storage

        

    # y_out: subtract initial value as in MATLAB
    y_out = y_mean - y_mean[0]
    x_out = x_interp
    return x_out, y_out, hysteresis

# ---------- Main processing ----------
def main():
    # Storage arrays (numpy)
    n_materials = len(foam_types)
    stretch_ten = np.zeros((n_pts_plt, n_materials))
    stress_ten = np.zeros((n_pts_plt, n_materials))
    stress_ten_std = np.zeros((n_pts_plt, n_materials))

    # Storage for individual sample data (for subplot figure)
    # Structure: [foam_idx][sample_idx-1] = {'stretch'/'strain': array, 'stress': array}
    individual_samples_tension = [[] for _ in range(n_materials)]
    individual_samples_compression = [[] for _ in range(n_materials)]
    individual_samples_shear = [[] for _ in range(n_materials)]
    individual_samples_conf_compression = [[] for _ in range(n_materials)]
    hysteresis_ten = np.zeros((n_materials))
    hysteresis_ten_samples = [[] for _ in range(n_materials)]
    hysteresis_com = np.zeros((n_materials))
    hysteresis_com_samples = [[] for _ in range(n_materials)]
    hysteresis_shear = np.zeros((n_materials))
    hysteresis_shear_samples = [[] for _ in range(n_materials)]
    stiffness_ten = np.zeros((n_materials))
    stiffness_ten_std = np.zeros((n_materials))
    stiffness_ten_samples = [[] for _ in range(n_materials)]
    stiffness_com = np.zeros((n_materials))
    stiffness_com_std = np.zeros((n_materials))
    stiffness_com_samples = [[] for _ in range(n_materials)]
    stiffness_shear = np.zeros((n_materials))
    stiffness_shear_std = np.zeros((n_materials))
    stiffness_shear_samples = [[] for _ in range(n_materials)]
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
            data_path = os.path.join(root_folder, f"{foam}-tension-{sample_idx}_1.csv")
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
            time = data[start_idx:, 1].astype(float)
            # gauge_lens_mm is per sample; MATLAB uses gauge_lens_mm(sample_idx)
            strain = displacement_mm / gauge_lens_mm[sample_idx-1]
            stress_kpa = force_n / areas_mm2[sample_idx-1] * 1000.0
            
            ## Write time, stretch, and stress_kpa to a sheet in an excel file named "raw_data.xlsx"
            df = pd.DataFrame({'time_s': time, 'stretch': 1.0 + strain, 'stress_kpa': stress_kpa})
            excel_path = os.path.join(out_dir, f"{foam}_raw_data.xlsx")
            # Use 'a' mode if file exists, 'w' mode if it doesn't
            mode = 'a' if os.path.exists(excel_path) else 'w'
            if mode == 'a':
                with pd.ExcelWriter(excel_path, mode=mode, if_sheet_exists='replace', engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=f"sample_{sample_idx}_tension", index=False)
            else:
                with pd.ExcelWriter(excel_path, mode=mode, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=f"sample_{sample_idx}_tension", index=False)

            min_peak_dist = 100
            x_interp_plt, stress_mean_kpa_plt, hysteresis_sample = average_curves(strain, stress_kpa, n_cycles, n_pts_plt, min_peak_dist, "ten", max_strain_ten)
            stress_all_plt.append(stress_mean_kpa_plt)
            hysteresis_ten_samples[foam_idx].append(hysteresis_sample)
            # Store individual sample data for subplot
            individual_samples_tension[foam_idx].append({
                'stretch': 1.0 + strain,
                'stress': stress_kpa
            })

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
        hysteresis_ten[foam_idx] = np.mean(np.array(hysteresis_ten_samples[foam_idx]))

        ### Stiffness per sample
        for sample_idx in range(stress_all_plt.shape[0]):
            max_strain = 0.1
            x_data = np.linspace(0, max_strain, 101)
            y_data = np.interp(x_data, x_interp_plt, stress_all_plt[sample_idx, :])
            # Fit y = m x with zero intercept: m = (x^T y) / (x^T x)
            denom = np.dot(x_data, x_data)
            if denom > 0:
                stiffness = np.dot(x_data, y_data) / denom
            else:
                stiffness = 0.0
            stiffness_ten_samples[foam_idx].append(stiffness)
        stiffness_ten[foam_idx] = np.mean(np.array(stiffness_ten_samples[foam_idx]))
        stiffness_ten_std[foam_idx] = np.std(np.array(stiffness_ten_samples[foam_idx]), ddof=0)

    # --- Compression ----------
    max_strain_com = 0.6
    offset = [1, 0]  # MATLAB offset array
    n_cycles = 4
    min_peak_dist = 1000

    stretch_com = np.zeros((n_pts_plt, n_materials))
    stress_com = np.zeros((n_pts_plt, n_materials))
    stress_com_std = np.zeros((n_pts_plt, n_materials))
    hysteresis_com = np.zeros((n_materials))
    hysteresis_com_samples = [[] for _ in range(n_materials)]
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
            time = data[1:, 0].astype(float)
            stretch = 1.0 - strain

            ## Crop compression data
            # Find first time stretch drops below 0.99, then go back 0.1 seconds
            below_099_idx = np.where(stretch < 0.99)[0]
            if len(below_099_idx) > 0:
                first_below_099_idx = below_099_idx[0]
                # Find index 0.1 seconds before this point
                target_time = time[first_below_099_idx] - 0.1
                start_idx = np.where(time >= target_time)[0]
                if len(start_idx) > 0:
                    start_idx = start_idx[0]
                else:
                    start_idx = 0
            else:
                start_idx = 0
            
            # Find final stretch value, then find when stretch comes/stays within 0.01 of it, then add 0.1 seconds
            final_stretch = stretch[-1]
            # Find last index where stretch is within 0.01 of final stretch
            not_within_001 = np.where(np.abs(stretch - final_stretch) > 0.01)[0]
            if len(not_within_001) > 0:
                last_not_within_001_idx = not_within_001[-1]
                # Find index 0.1 seconds after this point
                target_time = time[last_not_within_001_idx] + 0.1
                end_idx = np.where(time <= target_time)[0]
                if len(end_idx) > 0:
                    end_idx = end_idx[-1] + 1  # +1 to include the last point
                else:
                    end_idx = len(time)
            else:
                end_idx = len(time)
            
            # Crop the data
            time_cropped = time[start_idx:end_idx]
            time_cropped = time_cropped - time_cropped[0]
            stretch_cropped = stretch[start_idx:end_idx]
            stress_kpa_cropped = -stress_kpa[start_idx:end_idx]

            ## Write time, stretch, and stress_kpa to a sheet in an excel file named "raw_data.xlsx"
            df = pd.DataFrame({'time_s': time_cropped, 'stretch': stretch_cropped, 'stress_kpa': stress_kpa_cropped})
            excel_path = os.path.join(out_dir, f"{foam}_raw_data.xlsx")
            # Use 'a' mode if file exists, 'w' mode if it doesn't
            mode = 'a' if os.path.exists(excel_path) else 'w'
            if mode == 'a':
                with pd.ExcelWriter(excel_path, mode=mode, if_sheet_exists='replace', engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=f"sample_{sample_idx}_compression", index=False)
            else:
                with pd.ExcelWriter(excel_path, mode=mode, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=f"sample_{sample_idx}_compression", index=False)

            x_interp_plt, stress_mean_kpa, hysteresis_sample = average_curves(
                strain, stress_kpa, n_cycles, n_pts_plt, min_peak_dist, "com", max_strain_com
            )
            stress_all_plt.append(stress_mean_kpa)
            hysteresis_com_samples[foam_idx].append(hysteresis_sample)
            
            # Store individual sample data for subplot
            individual_samples_compression[foam_idx].append({
                'stretch': 1.0 - strain,
                'stress': -stress_kpa
            })

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
        hysteresis_com[foam_idx] = np.mean(np.array(hysteresis_com_samples[foam_idx]))

        ### Stiffness per sample
        for sample_idx in range(stress_all_plt.shape[0]):
            max_strain = 0.1
            x_data = np.linspace(0, max_strain, 101)
            y_data = np.interp(x_data, x_interp_plt, stress_all_plt[sample_idx, :])
            denom = np.dot(x_data, x_data)
            if denom > 0:
                stiffness = np.dot(x_data, y_data) / denom
            else:
                stiffness = 0.0
            stiffness_com_samples[foam_idx].append(stiffness)
        stiffness_com[foam_idx] = np.mean(np.array(stiffness_com_samples[foam_idx]))
        stiffness_com_std[foam_idx] = np.std(np.array(stiffness_com_samples[foam_idx]), ddof=0)

    # --- Confined Compression ----------
    max_strain_com = 0.6
    offset = [0, 0]  # MATLAB offset array
    n_cycles = 4
    min_peak_dist = 1000

    stretch_conf_com = np.zeros((n_pts_plt, n_materials))
    stress_conf_com = np.zeros((n_pts_plt, n_materials))
    stress_conf_com_std = np.zeros((n_pts_plt, n_materials))
    for foam_idx, foam in enumerate(foam_types):
        meas_path = os.path.join(root_folder, f"{foam}_confcomp_measurements.csv")
        meas = pd.read_csv(meas_path, header=None).values
        diameters_mm = np.mean(meas[:, 0:3], axis=1)  # columns 1:3 in MATLAB
        areas_mm2 = (diameters_mm ** 2) * np.pi / 4.0

        stress_all_plt = []
        for sample_idx in range(1,6):
            # note: MATLAB used file index sample_idx + offset(foam_idx)
            file_idx = sample_idx + offset[foam_idx]
            comp_path = os.path.join(root_folder, f"{foam}-confcomp-{file_idx}.txt")
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

            
            if (np.max(strain) > 0.65):
                print(f"Strain too low for {foam} - {sample_idx}")
                end_idx = np.where(strain > 0.65)[0][0]
                print(f"End index: {end_idx}")
                strain = strain[:end_idx]
                stress_kpa = stress_kpa[:end_idx]
            x_interp_plt, stress_mean_kpa, _ = average_curves(
                strain, stress_kpa, n_cycles, n_pts_plt, min_peak_dist, "com", max_strain_com
            )
            stress_all_plt.append(stress_mean_kpa)
            
            # Store individual sample data for subplot
            individual_samples_conf_compression[foam_idx].append({
                'stretch': 1.0 - strain,
                'stress': -stress_kpa
            })

        stress_all_plt = np.array(stress_all_plt)
        stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
        stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)

        # # Resample for table
        # strain_interp_table = np.linspace(0.0, max_strain_com, n_pts_table)
        # stress_mean_table = np.interp(strain_interp_table, x_interp_plt, stress_mean_plt)
        # stress_var_table = np.interp(strain_interp_table, x_interp_plt, stress_var_plt)

        stretch_conf_com[:, foam_idx] = 1.0 - x_interp_plt
        stress_conf_com[:, foam_idx] = -stress_mean_plt
        stress_conf_com_std[:, foam_idx] = stress_var_plt

       

    # --- Shear ----------
    max_shr = 0.15
    offset = [2, 0]
    n_cycles = 3
    min_peak_dist = 1000
    strain_interp_plt = np.linspace(0, max_shr, n_pts_plt)

    strain_shr = np.zeros((n_pts_plt, n_materials))
    stress_shr = np.zeros((n_pts_plt, n_materials))
    stress_shr_std = np.zeros((n_pts_plt, n_materials))
    hysteresis_shear = np.zeros((n_materials))
    hysteresis_shear_samples = [[] for _ in range(n_materials)]
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
            time = data[3:, 2].astype(float)
            time = time - time[0]
            shear_strain = disp_rad * radii_mm[sample_idx-1] / gap_mm[0]
            stretch = shear_strain * 0.0 + 0.8
            stress_kpa = force_n / (np.pi * radii_mm[sample_idx-1]**3 / 2.0) * 1000.0

            ## Write time, stretch, shear_strain and stress to a sheet in an excel file named "raw_data.xlsx"
            df = pd.DataFrame({'time_s': time, 'stretch': stretch, 'shear_strain': shear_strain, 'stress_kpa': stress_kpa})
            excel_path = os.path.join(out_dir, f"{foam}_raw_data.xlsx")
            # Use 'a' mode if file exists, 'w' mode if it doesn't
            mode = 'a' if os.path.exists(excel_path) else 'w'
            if mode == 'a':
                with pd.ExcelWriter(excel_path, mode=mode, if_sheet_exists='replace', engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=f"sample_{sample_idx}_shear", index=False)
            else:
                with pd.ExcelWriter(excel_path, mode=mode, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=f"sample_{sample_idx}_shear", index=False)
            height_mm = gap_mm[0]
            safety_factor = 1.01
            # disp_rad_max = height_mm / radius * max_shr * safety_factor
            r = radii_mm[sample_idx-1]
            disp_rad_max = (height_mm / r) * max_shr * safety_factor if r != 0 else max_shr

            disp_rad_interp, torque_nmm_interp_mean, hysteresis_sample = average_curves(
                disp_rad, torque_nmm, n_cycles, 101, min_peak_dist, "shear", disp_rad_max
            )

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
            ## Compute linearized shear stress
            strain_raw = r * disp_rad / height_mm
            shear_stress_raw = 1000.0 * torque_nmm / (np.pi * r**3 / 2.0)

            # Store individual sample data for subplot
            individual_samples_shear[foam_idx].append({
                'strain': strain_raw,
                'stress': shear_stress_raw
            })
            hysteresis_shear_samples[foam_idx].append(hysteresis_sample)

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
        hysteresis_shear[foam_idx] = np.mean(np.array(hysteresis_shear_samples[foam_idx]))

        ### Stiffness per sample
        for sample_idx in range(stress_all_plt.shape[0]):
            max_strain = 0.1
            x_data = np.linspace(0, max_strain, 101)
            y_data = np.interp(x_data, strain_interp_plt, stress_all_plt[sample_idx, :])
            denom = np.dot(x_data, x_data)
            if denom > 0:
                stiffness = np.dot(x_data, y_data) / denom
            else:
                stiffness = 0.0
            stiffness_shear_samples[foam_idx].append(stiffness)
        stiffness_shear[foam_idx] = np.mean(np.array(stiffness_shear_samples[foam_idx]))
        stiffness_shear_std[foam_idx] = np.std(np.array(stiffness_shear_samples[foam_idx]), ddof=0)

    # --- Output Hysteresis Table ---
    print("\n--- Hysteresis Values ---")
    # Prepare means and stds (in percent)
    ten_means = hysteresis_ten * 100.0
    com_means = hysteresis_com * 100.0
    shr_means = hysteresis_shear * 100.0
    ten_stds = np.array([np.std(np.array(hysteresis_ten_samples[i]), ddof=0) for i in range(n_materials)]) * 100.0
    com_stds = np.array([np.std(np.array(hysteresis_com_samples[i]), ddof=0) for i in range(n_materials)]) * 100.0
    shr_stds = np.array([np.std(np.array(hysteresis_shear_samples[i]), ddof=0) for i in range(n_materials)]) * 100.0

    # Build LaTeX table: rows = materials, columns = modes
    lines = []
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\hline')
    lines.append(r'Material & Tension & Compression & Shear \\')
    lines.append(r'\hline')
    for foam_idx, foam in enumerate(foam_types):
        ten_mean = ten_means[foam_idx]
        com_mean = com_means[foam_idx]
        shr_mean = shr_means[foam_idx]
        ten_std = ten_stds[foam_idx]
        com_std = com_stds[foam_idx]
        shr_std = shr_stds[foam_idx]
        lines.append(
            f"{foam} & "
            f"{ten_mean:.1f} $\\pm$ {ten_std:.1f} & "
            f"{com_mean:.1f} $\\pm$ {com_std:.1f} & "
            f"{shr_mean:.1f} $\\pm$ {shr_std:.1f} \\\\"
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    hysteresis_table = "\n".join(lines)
    print(hysteresis_table)
    
    # Save hysteresis table to file
    output_dir = "./Results/RawData"
    os.makedirs(output_dir, exist_ok=True)
    hysteresis_table_path = os.path.join(output_dir, "hysteresis_table.tex")
    with open(hysteresis_table_path, 'w') as f:
        f.write(hysteresis_table)
    print(f"\nHysteresis table saved to: {hysteresis_table_path}\n")

    # --- Output Stiffness Table ---
    print("\n--- Stiffness Values ---")
    # Build LaTeX table: rows = materials, columns = modes
    lines_stiff = []
    lines_stiff.append(r'\begin{tabular}{lccc}')
    lines_stiff.append(r'\hline')
    lines_stiff.append(r'Material & Tension & Compression & Shear \\')
    lines_stiff.append(r'\hline')
    for foam_idx, foam in enumerate(foam_types):
        ten_mean = stiffness_ten[foam_idx]
        com_mean = stiffness_com[foam_idx]
        shr_mean = stiffness_shear[foam_idx]
        ten_std = stiffness_ten_std[foam_idx]
        com_std = stiffness_com_std[foam_idx]
        shr_std = stiffness_shear_std[foam_idx]
        lines_stiff.append(
            f"{foam} & "
            f"{ten_mean:.1f} $\\pm$ {ten_std:.1f} & "
            f"{com_mean:.1f} $\\pm$ {com_std:.1f} & "
            f"{shr_mean:.1f} $\\pm$ {shr_std:.1f} \\\\"
        )
    lines_stiff.append(r'\hline')
    lines_stiff.append(r'\end{tabular}')
    stiffness_table = "\n".join(lines_stiff)
    print(stiffness_table)
    
    # Save stiffness table to file
    stiffness_table_path = os.path.join(output_dir, "stiffness_table.tex")
    with open(stiffness_table_path, 'w') as f:
        f.write(stiffness_table)
    print(f"\nStiffness table saved to: {stiffness_table_path}\n")

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
    stretch_conf_com_table = np.linspace(np.min(stretch_conf_com), np.max(stretch_conf_com), n_pts_table)
    stress_conf_com_table = np.stack([np.interp(stretch_conf_com_table, stretch_conf_com[::-1, foam_idx], stress_conf_com[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stress_conf_com_std_table = np.stack([np.interp(stretch_conf_com_table, stretch_conf_com[::-1, foam_idx], stress_conf_com_std[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)

    print(stretch_ten_table.shape)
    # ---------- Create all plots at the end ----------
    # Create output directory
    output_dir = "./Results/RawData"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tension plot
    plt.figure(figsize=(7, 5))
    for foam_idx in range(n_materials):
        plt.plot(stretch_ten[:, foam_idx], stress_ten[:, foam_idx], colors[foam_idx], label=f"{foam_types[foam_idx]}")
        plt.plot(stretch_ten_table, stress_ten_table[:, foam_idx], colors[foam_idx] + "o", markersize=4)
        plt.fill_between(stretch_ten[:, foam_idx],
                         stress_ten[:, foam_idx] - stress_ten_std[:, foam_idx],
                         stress_ten[:, foam_idx] + stress_ten_std[:, foam_idx],
                         color=colors[foam_idx], alpha=0.25)
    plt.xlim([1.0, 1.3])
    plt.xlabel("Stretch [-]", fontsize=FONT_SIZE)
    plt.ylabel("Stress [kPa]", fontsize=FONT_SIZE)
    plt.title("Tension", fontsize=FONT_SIZE)
    plt.tick_params(labelsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Tension.pdf"), format='pdf', bbox_inches='tight')
    # plt.show()

    # Compression plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for foam_idx in range(n_materials):
        ax.plot(stretch_com[:, foam_idx], stress_com[:, foam_idx], colors[foam_idx], label=f"{foam_types[foam_idx]}")
        ax.plot(stretch_com_table, stress_com_table[:, foam_idx], colors[foam_idx] + "o", markersize=4)
        ax.fill_between(stretch_com[:, foam_idx],
                         stress_com[:, foam_idx] - stress_com_std[:, foam_idx],
                         stress_com[:, foam_idx] + stress_com_std[:, foam_idx],
                         color=colors[foam_idx], alpha=0.25)
    # Flip both axes: x-axis (stretch decreases left to right), y-axis (negative stress up)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Stretch [-]", fontsize=FONT_SIZE)
    ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
    ax.set_title("Compression", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Compression.pdf"), format='pdf', bbox_inches='tight')
    # plt.show()

    ## Confined compression plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for foam_idx in range(n_materials):
        ax.plot(stretch_conf_com[:, foam_idx], stress_conf_com[:, foam_idx], colors[foam_idx], label=f"{foam_types[foam_idx]}")
        ax.plot(stretch_conf_com_table, stress_conf_com_table[:, foam_idx], colors[foam_idx] + "o", markersize=4)
        ax.fill_between(stretch_conf_com[:, foam_idx],
                         stress_conf_com[:, foam_idx] - stress_conf_com_std[:, foam_idx],
                         stress_conf_com[:, foam_idx] + stress_conf_com_std[:, foam_idx],
                         color=colors[foam_idx], alpha=0.25)
    # Flip both axes: x-axis (stretch decreases left to right), y-axis (negative stress up)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Stretch [-]", fontsize=FONT_SIZE)
    ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
    ax.set_title("Confined Compression", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ConfinedCompression.pdf"), format='pdf', bbox_inches='tight')
    # plt.show()

    # Shear plot
    plt.figure(figsize=(7, 5))
    for foam_idx in range(n_materials):
        plt.plot(strain_shr[:, foam_idx], stress_shr[:, foam_idx], colors[foam_idx], label=f"{foam_types[foam_idx]}")
        plt.plot(strain_shr_table, stress_shr_table[:, foam_idx], colors[foam_idx] + "o", markersize=4)
        plt.fill_between(strain_shr[:, foam_idx],
                         stress_shr[:, foam_idx] - stress_shr_std[:, foam_idx],
                         stress_shr[:, foam_idx] + stress_shr_std[:, foam_idx],
                         color=colors[foam_idx], alpha=0.25)
    plt.xlabel("Shear Strain [-]", fontsize=FONT_SIZE)
    plt.ylabel("Shear Stress [kPa]", fontsize=FONT_SIZE)
    plt.title("Shear", fontsize=FONT_SIZE)
    # Set x-axis ticks every 0.05 and format to 2 decimal places
    ax_shear = plt.gca()
    ax_shear.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax_shear.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.tick_params(labelsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Shear.pdf"), format='pdf', bbox_inches='tight')
    # plt.show()


    # --- 5x4 Subplot Figures: Individual Samples (one per material) ---
    # 5 rows: Sample 1, Sample 2, Sample 3, Sample 4, Sample 5
    # 4 columns: Tension, Compression, Shear, Confined Compression
    # Create a separate figure for each material
    for foam_idx in range(n_materials):
        fig, axes = plt.subplots(5, 4, figsize=(14, 16))
        fig.suptitle(f'{foam_types[foam_idx]} - Individual Samples', fontsize=FONT_SIZE, fontweight='bold')
        
        # Compute axis limits for each test type (to make all samples in a column have same scale)
        # Tension limits
        ten_x_min, ten_x_max, ten_y_min, ten_y_max = np.inf, -np.inf, np.inf, -np.inf
        for sample_idx in range(len(individual_samples_tension[foam_idx])):
            sample_data = individual_samples_tension[foam_idx][sample_idx]
            ten_x_min = min(ten_x_min, np.nanmin(sample_data['stretch']))
            ten_x_max = max(ten_x_max, np.nanmax(sample_data['stretch']))
            ten_y_min = min(ten_y_min, np.nanmin(sample_data['stress']))
            ten_y_max = max(ten_y_max, np.nanmax(sample_data['stress']))
        
        # Compression limits
        com_x_min, com_x_max, com_y_min, com_y_max = np.inf, -np.inf, np.inf, -np.inf
        for sample_idx in range(len(individual_samples_compression[foam_idx])):
            sample_data = individual_samples_compression[foam_idx][sample_idx]
            com_x_min = min(com_x_min, np.nanmin(sample_data['stretch']))
            com_x_max = max(com_x_max, np.nanmax(sample_data['stretch']))
            com_y_min = min(com_y_min, np.nanmin(sample_data['stress']))
            com_y_max = max(com_y_max, np.nanmax(sample_data['stress']))

        # Confined compression limits
        conf_com_x_min, conf_com_x_max, conf_com_y_min, conf_com_y_max = np.inf, -np.inf, np.inf, -np.inf
        for sample_idx in range(len(individual_samples_conf_compression[foam_idx])):
            sample_data = individual_samples_conf_compression[foam_idx][sample_idx]
            conf_com_x_min = min(conf_com_x_min, np.nanmin(sample_data['stretch']))
            conf_com_x_max = max(conf_com_x_max, np.nanmax(sample_data['stretch']))
            conf_com_y_min = min(conf_com_y_min, np.nanmin(sample_data['stress']))
            conf_com_y_max = max(conf_com_y_max, np.nanmax(sample_data['stress']))
        
        # Shear limits
        shr_x_min, shr_x_max, shr_y_min, shr_y_max = np.inf, -np.inf, np.inf, -np.inf
        for sample_idx in range(len(individual_samples_shear[foam_idx])):
            sample_data = individual_samples_shear[foam_idx][sample_idx]
            shr_x_min = min(shr_x_min, np.nanmin(sample_data['strain']))
            shr_x_max = max(shr_x_max, np.nanmax(sample_data['strain']))
            shr_y_min = min(shr_y_min, np.nanmin(sample_data['stress']))
            shr_y_max = max(shr_y_max, np.nanmax(sample_data['stress']))
        
        # Column 0: Tension
        for sample_idx in range(5):  # Rows 0-4 for samples 1-5
            ax = axes[sample_idx, 0]
            if sample_idx < len(individual_samples_tension[foam_idx]):
                sample_data = individual_samples_tension[foam_idx][sample_idx]
                ax.plot(sample_data['stretch'], sample_data['stress'], 
                       colors[foam_idx], linewidth=1.5)
            ax.set_xlabel("Stretch [-]", fontsize=FONT_SIZE)
            ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
            ax.set_title(f"Tension \n Sample {sample_idx + 1}", fontsize=FONT_SIZE)
            ax.tick_params(labelsize=FONT_SIZE)
            ax.grid(True, alpha=0.3)
            # Set identical axis limits for all tension subplots
            if ten_x_max > ten_x_min and ten_y_max > ten_y_min:
                ax.set_xlim(1.0, 1.3)
                ax.set_ylim(ten_y_min, ten_y_max)
        
        # Column 1: Compression
        for sample_idx in range(5):  # Rows 0-4 for samples 1-5
            ax = axes[sample_idx, 1]
            if sample_idx < len(individual_samples_compression[foam_idx]):
                sample_data = individual_samples_compression[foam_idx][sample_idx]
                ax.plot(sample_data['stretch'], sample_data['stress'], 
                       colors[foam_idx], linewidth=1.5)
            # Flip both axes for compression

            ax.set_xlabel("Stretch [-]", fontsize=FONT_SIZE)
            ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
            ax.set_title(f"Compression \n Sample {sample_idx + 1}", fontsize=FONT_SIZE)
            ax.tick_params(labelsize=FONT_SIZE)
            ax.grid(True, alpha=0.3)
            # Set identical axis limits for all compression subplots
            if com_x_max > com_x_min and com_y_max > com_y_min:
                ax.set_xlim(com_x_min, com_x_max)
                ax.set_ylim(com_y_min, com_y_max)
            ax.invert_xaxis()
            ax.invert_yaxis()
        
        # Column 2: Shear
        for sample_idx in range(5):  # Rows 0-4 for samples 1-5
            ax = axes[sample_idx, 2]
            if sample_idx < len(individual_samples_shear[foam_idx]):
                sample_data = individual_samples_shear[foam_idx][sample_idx]
                # Plot both positive and negative values
                ax.plot(sample_data['strain'], 
                       sample_data['stress'], 
                       colors[foam_idx], linewidth=1.5)
            ax.set_xlabel("Shear Strain [-]", fontsize=FONT_SIZE)
            ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
            ax.set_title(f"Shear \n Sample {sample_idx + 1}", fontsize=FONT_SIZE)
            ax.tick_params(labelsize=FONT_SIZE)
            ax.grid(True, alpha=0.3)
            # Set identical axis limits for all shear subplots
            if shr_x_max > shr_x_min and shr_y_max > shr_y_min:
                ax.set_xlim(shr_x_min, shr_x_max)
                ax.set_ylim(shr_y_min, shr_y_max)
        
        # Column 3: Confined Compression
        for sample_idx in range(5):  # Rows 0-4 for samples 1-5
            ax = axes[sample_idx, 3]
            if sample_idx < len(individual_samples_conf_compression[foam_idx]):
                sample_data = individual_samples_conf_compression[foam_idx][sample_idx]
                ax.plot(sample_data['stretch'], sample_data['stress'], 
                       colors[foam_idx], linewidth=1.5)
            ax.set_xlabel("Stretch [-]", fontsize=FONT_SIZE)
            ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
            ax.set_title(f"Confined Compression\nSample {sample_idx + 1}", fontsize=FONT_SIZE)
            ax.tick_params(labelsize=FONT_SIZE)
            ax.grid(True, alpha=0.3)
            # Set identical axis limits for all confined compression subplots
            if conf_com_x_max > conf_com_x_min and conf_com_y_max > conf_com_y_min:
                ax.set_xlim(conf_com_x_min, conf_com_x_max)
                ax.set_ylim(conf_com_y_min, conf_com_y_max)
            ax.invert_xaxis()
            ax.invert_yaxis()
        
        plt.tight_layout()
        # Save figure with material name
        filename = f"{foam_types[foam_idx]}_individual_samples.pdf"
        plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
        # plt.show()

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
        foam_name = foam_types[mat]
        # Get table data (13 points)
        ten_stretch = stretch_ten_table
        ten_stress = stress_ten_table[:, mat]
        ten_std = stress_ten_std_table[:, mat]
        # Reverse compression data so largest stretch (1.0) is at top
        # Negate compression stress values to make them positive
        com_stretch = stretch_com_table[::-1]
        com_stress = -stress_com_table[::-1, mat]  # Negate to make positive
        com_std = stress_com_std_table[::-1, mat]  # Keep std positive
        shr_strain = strain_shr_table
        shr_stress = stress_shr_table[:, mat]
        shr_std = stress_shr_std_table[:, mat]
        
        # Get stiffness and energy return values
        E_ten = stiffness_ten[mat]
        E_ten_std = stiffness_ten_std[mat]
        E_com = stiffness_com[mat]
        E_com_std = stiffness_com_std[mat]
        G_shr = stiffness_shear[mat]
        G_shr_std = stiffness_shear_std[mat]
        # Energy return = 1 - hysteresis (convert hysteresis from fraction to %, then compute return)
        energy_return_ten = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_ten_samples[mat]])) * 100.0
        # For std: if hysteresis has std, energy return has same std (but opposite sign doesn't matter for std)
        energy_return_ten_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_ten_samples[mat]]), ddof=0) * 100.0
        energy_return_com = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_com_samples[mat]])) * 100.0
        energy_return_com_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_com_samples[mat]]), ddof=0) * 100.0
        energy_return_shr = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_shear_samples[mat]])) * 100.0
        energy_return_shr_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_shear_samples[mat]]), ddof=0) * 100.0

        lines = []
        lines.append(r'\begin{table*}[h]')
        lines.append(rf'\caption{{\sffamily{{\bfseries{{{foam_name} data from tension, compression, shear experiments.}}}}}} Recorded Piola stress $P$ at equally spaced axial stretch $\lambda$ or shear strain $\gamma$ intervals for the {foam_name} foam.')
        lines.append(rf'The first two columns represent uniaxial tension,')
        lines.append(rf'the middle two columns uniaxial compression, and')
        lines.append(rf'the last two columns simple shear.')
        lines.append(rf'Means and standard deviations are reported across $n=5$ samples.')
        lines.append(r'\vspace*{0.1cm}')
        lines.append(r'\small')
        lines.append(r'\centering')
        lines.append(rf'\label{{table:{foam_name}}}')
        lines.append(r'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        lines.append(r'\begin{tabular}{|cc||cc||cc|}')
        lines.append(r'\hline')
        lines.append(r'  \multicolumn{2}{|c||}{\sffamily{\bfseries{uniaxial tension}}}')
        lines.append(r'& \multicolumn{2}{c||} {\sffamily{\bfseries{uniaxial compression}}}')
        lines.append(r'& \multicolumn{2}{c|}  {\sffamily{\bfseries{simple shear}}} \\')
        lines.append(r'  \multicolumn{2}{|c||}{$n=5$}')
        lines.append(r'& \multicolumn{2}{c||}{$n=5$}')
        lines.append(r'& \multicolumn{2}{c|}{$n=5$} \\ \hline')
        lines.append(r'$\lambda$ & $P_{11}$ & $\lambda$ & $P_{11}$ & $\gamma$ & $P_{12}$  \\')
        lines.append(r'\,[-] & [kPa]  & [-] & [kPa]  & [-] & [kPa]  \\')
        lines.append(r'\hline \hline')

        # Helper function to format numbers with phantoms
        def format_with_phantoms(val, std, max_digits=3):
            # Handle negative values properly
            val_sign = -1 if val < 0 else 1
            val_abs = abs(val)
            # Round to 2 decimal places first to avoid floating point issues
            val_rounded = np.round(val_abs, 2)
            val_int = int(val_rounded)
            val_frac_raw = val_rounded - val_int
            val_frac = int(np.round(val_frac_raw * 100))
            # Ensure fractional part is between 0 and 99
            if val_frac < 0:
                val_frac = 0
            elif val_frac >= 100:
                val_int += 1
                val_frac = 0
            
            std_sign = -1 if std < 0 else 1
            std_abs = abs(std)
            # Round to 2 decimal places first
            std_rounded = np.round(std_abs, 2)
            std_int = int(std_rounded)
            std_frac_raw = std_rounded - std_int
            std_frac = int(np.round(std_frac_raw * 100))
            # Ensure fractional part is between 0 and 99
            if std_frac < 0:
                std_frac = 0
            elif std_frac >= 100:
                std_int += 1
                std_frac = 0
            
            val_digits = len(str(val_int)) if val_int != 0 else 1
            std_digits = len(str(std_int)) if std_int != 0 else 1
            
            val_phantom = r'\phantom{0}' * max(0, max_digits - val_digits)
            std_phantom = r'\phantom{0}' * max(0, max_digits - std_digits)
            
            # Format value with sign
            if val_int == 0 and val_frac == 0:
                val_str = r"\phantom{0}\phantom{0}0.00"
            elif val_int == 0:
                val_str = rf"\phantom{{0}}\phantom{{0}}0.{val_frac:02d}"
            else:
                sign_str = "-" if val_sign < 0 else ""
                val_str = rf"{sign_str}{val_phantom}{val_int}.{val_frac:02d}"
            
            # Format std with sign
            if std_int == 0 and std_frac == 0:
                std_str = r"\phantom{0}\phantom{0}0.00"
            elif std_int == 0:
                std_str = rf"\phantom{{0}}\phantom{{0}}0.{std_frac:02d}"
            else:
                sign_str = "-" if std_sign < 0 else ""
                std_str = rf"{sign_str}{std_phantom}{std_int}.{std_frac:02d}"
            
            return rf"{val_str}\hspace{{0.5em}}$\pm$ {std_str}"

        # Data rows (13 rows)
        for i in range(n_pts_table):
            ten_str = format_with_phantoms(ten_stress[i], ten_std[i])
            com_str = format_with_phantoms(com_stress[i], com_std[i])
            shr_str = format_with_phantoms(shr_stress[i], shr_std[i])
            
            # Add \hline after certain rows (matching the example)
            hline_after = (i == 0) or (i == 3) or (i == 4) or (i == 7) or (i == 8) or (i == 11)
            
            lines.append(
                f"{ten_stretch[i]:.3f} & {ten_str} & "
                f"{com_stretch[i]:.3f} & {com_str} & "
                f"{shr_strain[i]:.3f} & {shr_str}"
            )
            if hline_after:
                lines.append(r' \\ \hline')
            else:
                lines.append(r' \\')
        
        lines.append(r'\hline \hline')
        # Stiffness section
        lines.append(r'  \multicolumn{2}{|c||}{\sffamily{\bfseries{tensile stiffness}}}')
        lines.append(r'& \multicolumn{2}{c||} {\sffamily{\bfseries{compressive stiffness}}}')
        lines.append(r'& \multicolumn{2}{c|}  {\sffamily{\bfseries{shear stiffness}}} \\')
        lines.append(
            rf'  \multicolumn{{2}}{{|c||}}{{$\textsf{{E}}_{{\rm{{ten}}}} = {E_ten:.2f} \pm {E_ten_std:.2f}$\,kPa}}'
        )
        lines.append(
            rf'& \multicolumn{{2}}{{c||}} {{$\textsf{{E}}_{{\rm{{com}}}} = {E_com:.2f} \pm {E_com_std:.2f}$\,kPa}}'
        )
        lines.append(
            rf'& \multicolumn{{2}}{{c|}}  {{$\textsf{{G}}_{{\rm{{shr}}}} = {G_shr:.2f} \pm {G_shr_std:.2f}$\,kPa}} \\'
        )
        lines.append(r'\hline \hline')
        # Energy return section
        lines.append(r'  \multicolumn{2}{|c||}{\sffamily{\bfseries{energy return}}}')
        lines.append(r'& \multicolumn{2}{c||} {\sffamily{\bfseries{energy return}}}')
        lines.append(r'& \multicolumn{2}{c|}  {\sffamily{\bfseries{energy return}}} \\')
        lines.append(
            rf'  \multicolumn{{2}}{{|c||}}{{$\eta_{{\rm{{ten}}}}  = {energy_return_ten:.1f} \pm {energy_return_ten_std:.1f} \%$}}'
        )
        lines.append(
            rf'& \multicolumn{{2}}{{c||}} {{$\eta_{{\rm{{com}}}}  = {energy_return_com:.1f} \pm {energy_return_com_std:.1f}\%$}}'
        )
        lines.append(
            rf'& \multicolumn{{2}}{{c|}}  {{$\eta_{{\rm{{shr}}}}  = {energy_return_shr:.1f} \pm {energy_return_shr_std:.1f} \%$}} \\'
        )
        lines.append(r'\hline')
        lines.append(r'\end{tabular}')
        lines.append(rf'%% End {foam_name} table')
        lines.append(r'\end{table*}')
        materialTables.append("\n".join(lines))

    # Save material tables to files
    output_dir = "./Results/RawData"
    os.makedirs(output_dir, exist_ok=True)
    for mat_idx, (foam_name, tbl) in enumerate(zip(foam_types, materialTables), start=1):
        print(f"LaTeX table for Material {mat_idx} ({foam_name}):\n{tbl}\n\n")
        table_path = os.path.join(output_dir, f"{foam_name}_material_table.tex")
        with open(table_path, 'w') as f:
            f.write(tbl)
        print(f"Material table saved to: {table_path}\n")

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
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FoamData.xlsx")
    df_out.to_excel(out_path, index=False)
    print(f"Wrote output excel to: {out_path}")

    #### Print strain energies for paper
    mean_tensile_stress = np.mean(stress_ten[:, 0])
    comp_stretch_min = 0.65
    ten_stretch_max = np.max(stretch_ten[:, 0])
    mean_compressive_stress = np.mean(stress_com[stretch_com[:, 0] > comp_stretch_min, 0])
    strain_energy_tensile = mean_tensile_stress * (ten_stretch_max - 1.0)
    strain_energy_compressive = mean_compressive_stress * (1 - comp_stretch_min)
    print(f"Strain energy (tensile): {strain_energy_tensile:.2f} kJ/m^3")
    print(f"Strain energy (compressive): {strain_energy_compressive:.2f} kJ/m^3")

if __name__ == "__main__":
    main()
