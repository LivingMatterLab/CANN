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

from util import *
from stats_functions import *
from preproc_plotting import *

# All the settings which affect the preprocessing of experimental data and creation of the raw data plots
worn_shoe = True 
should_process = False # should be set to True the first time the script is run to process the data, subsequently can be set to False to skip the processing and just plot the data

root_folder = "./input/raw_data/asics/worn-shoe/" if worn_shoe else "./input/raw_data/asics/final-tcs/" # where the input experimental data is stored
transverse_folder = "./input/data_mm/" # where the input transverse data is stored
out_dir = "./input/" # where to output preprocessed stress data
out_dir_plots = "./Results/RawData" # where to output raw data plots

# Plotting parameters
colors = ["r", "g", "b"] * 2 if worn_shoe else ["r", "b"]
linestyles = (["-"] * 3 + ["--"] * 3) if worn_shoe else ["-", "-"]
# Foam types and how they should be shown in the plots
foam_types = ["new-toe", "new-mid", "new-heel",  "worn-toe", "worn-mid", "worn-heel"] if worn_shoe else ["leap", "turbo"]
foam_types_title = [x.replace("-", " ").title() for x in foam_types] if worn_shoe else ["FF LEAP\u2122", "FF TURBO\u2122 PLUS"] 

header = 0 if worn_shoe else None

n_pts_table = 13 # number of points to include in the raw data table
n_pts_plt = 101 # number of points to plot for the raw data plots

# Strain window for initial stiffness / Poisson's ratio fits
max_strain_linear = 0.1

# Global font size for all plot text elements
base_font_size = 20

def deriv(y, x):
    """
    Function to take the central difference derivative of a function y(x)
    """
    y = np.asarray(y)
    x = np.asarray(x)
    dydx = np.empty_like(y, dtype=float)
    if len(y) < 2:
        return np.array([0.0])
    # forward difference for first
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    # central difference for middle
    if len(y) > 2:
        dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    # backward difference for last
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dydx

def average_curves(x, y, y2, n_cycles, n_pts, min_peak_dist, loading_mode, max_strain=-1):
    """
    Function to average a time series of input (x) and outputs (y, y2) data in order to obtain 
    a single mean curve y(x).

    x, y, y2 : 1d arrays (displacement, stress, and transverse stretch)
    n_cycles : will average the first n_cycles loading cycles of the time series (excluding the first cycle)
    n_pts : number of interpolation points
    min_peak_dist : min distance between peaks (when identifying start and end of a loading cycle)
    loading_mode : "shear", "ten", or "com"
    max_strain : optional max strain; if negative, derive this from the data
    Returns: x_out, y_out (both 1D numpy arrays)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    y2 = np.asarray(y2).flatten()

    is_shear = (loading_mode == "shear")

    # Find peak indices and values of the input data x
    peaks_idx, _ = find_peaks(x, distance=min_peak_dist, height=0, prominence=0.1)
    peaks_vals = x[peaks_idx]

    # Compute the minimum peak stretch value
    if peaks_idx.size == 0:
        # fallback: treat entire time series as one segment
        minmax_val = x.max()
    else:
        # take first n_cycles peaks (if available)
        if peaks_idx.size >= n_cycles:
            minmax_val = np.min(peaks_vals[:n_cycles])
        else:
            minmax_val = np.min(peaks_vals)  # fallback

    # Find the relative minima by finding peaks of 1-x for uniaxial and -x for shear
    offset_val = (1 - (1 if is_shear else 0)) * minmax_val
    inverted = offset_val - x
    minima_idx, _ = find_peaks(inverted, distance=min_peak_dist, height=0, prominence=0.1)
    
    # Construct 
    if len(peaks_idx) == 0 or len(minima_idx) == 0:
        start = 0
        # we'll treat the whole array as a single segment
        maxima = np.array([len(x)-1])
        minima = np.array([len(x)-1])
    else:
        maxima = peaks_idx.copy()
        minima = minima_idx.copy()
        # Find the first minima after the first maximum, we will use this as the start of our data (since we do not want to use the first cycle)
        minima_after = minima[minima > maxima[0]]
        if minima_after.size == 0:
            start = 0
        else:
            start = minima_after[0]
        # choose the first n_cycles maxima after start
        maxima = maxima[maxima > start]
        if maxima.size >= n_cycles:
            maxima = maxima[:n_cycles]
        # minima for cycles: in MATLAB they used minima(2:(n_cycles+1)) for com
        # 
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
    y2_interp_all = []

    # For each segment between end_pts[i] and end_pts[i+1], map x_segment values
    segment_means = []
    for i in range(len(end_pts)-1):
        i0 = end_pts[i]
        i1 = end_pts[i+1]
        if i1 <= i0:
            continue
        x_segment = x[i0:i1+1]
        y_segment = y[i0:i1+1]
        y2_segment = y2[i0:i1+1]
        # require x_unique and average y for repeated x
        if x_segment.size == 0:
            continue
        xu, inv = np.unique(x_segment, return_inverse=True)
        # average y for each unique x
        y_u = np.zeros_like(xu, dtype=float)
        y2_u = np.zeros_like(xu, dtype=float)
        counts = np.zeros_like(xu, dtype=float)
        for k, idx in enumerate(inv):
            y_u[idx] += y_segment[k]
            y2_u[idx] += y2_segment[k]
            counts[idx] += 1
        y_u = y_u / np.maximum(counts, 1)
        y2_u = y2_u / np.maximum(counts, 1)
        # Interpolate y_u on xu to x_interp
        # For values outside xu range, numpy.interp returns endpoints (good)
        y_interp = np.interp(x_interp, xu, y_u)
        y2_interp = np.interp(x_interp, xu, y2_u)
        # fix NaN at first position if occurs (mimic MATLAB)
        if np.isnan(y_interp[0]):
            y_interp[0] = y_u[0]
        if np.isnan(y2_interp[0]):
            y2_interp[0] = y2_u[0]
        y_interp_all.append(y_interp)
        y2_interp_all.append(y2_interp)
        segment_means.append(np.mean(y_u[xu > 0]))
        if (np.isnan(np.mean(y_u[xu > 0]))):
            print(minima)
            plt.plot(x)
            plt.show()
            plt.plot(y)
            plt.show()
            print("xy_u")
            print(y_u)
            print(xu)
            print(y_u[xu > 0])
            test = y_u[xu > 0]
            for i in test.shape[0]:
                print(test[i])
            assert False
        

    y_interp_all = np.array(y_interp_all)  # shape (n_segments, n_pts)
    y2_interp_all = np.array(y2_interp_all)  # shape (n_segments, n_pts)
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
    y2_mean = np.nanmean(y2_interp_all, axis=0)

    #### Compute Energy return
    loading_mean = np.mean(segment_means[0::2])
    unloading_mean = np.mean(segment_means[1::2])
    if not is_shear:
        loading_mean = loading_mean - y_mean[0]
        unloading_mean = unloading_mean - y_mean[0]
    energy_return = unloading_mean / loading_mean

    # Shift y_out so it starts at 0
    y_out = y_mean - y_mean[0]
    x_out = x_interp

    return x_out, y_out, y2_mean, energy_return

def read_transverse_data(transverse_path):
    if os.path.exists(transverse_path):
        transverse_data = pd.read_csv(transverse_path, header=0).values
        transverse_time = transverse_data[:, 0].astype(float)
        transverse_width = transverse_data[:, 1].astype(float)
        transverse_height = transverse_data[:, 2].astype(float)
        axial_stretch = transverse_height / transverse_height[0]
        return transverse_time, transverse_width, axial_stretch
    else: 
        return np.array([0.0]), np.array([0.0]), np.array([0.0])


def get_transverse_width(transverse_path, time, strain, is_compression=False):

    # Read transverse data
    if os.path.exists(transverse_path):
        transverse_time, transverse_width, axial_stretch = read_transverse_data(transverse_path)

        # Find peaks of transverse stretch
        transverse_peaks_idx, _ = find_peaks((1 - axial_stretch) if is_compression else axial_stretch, distance=30, height=0, prominence=0.1)
        # Find peaks of strain
        strain_peaks_idx, _ = find_peaks(strain, distance=100, height=0, prominence=0.1)

        if transverse_peaks_idx.shape[0] > strain_peaks_idx.shape[0]:
            print(f"Error: Transverse stretch has more peaks than strain does: {transverse_peaks_idx.shape[0]} > {strain_peaks_idx.shape[0]}")
            assert False
        else:
            strain_peaks_idx = strain_peaks_idx[:transverse_peaks_idx.shape[0]]

        ## Find time offset between transverse stretch and strain peaks
        time_offset = np.median(time[strain_peaks_idx] - transverse_time[transverse_peaks_idx])
        # print(f"Time offset: {time_offset}")
        transverse_time = transverse_time + time_offset
        transverse_width_interp = np.interp(time, transverse_time, transverse_width)
    else:
        transverse_width_interp = np.zeros_like(time)
    return transverse_width_interp


def process_data(): 
    # Create storage arrays for processed data
    all_data_stress = np.zeros((3, 3, 2, 5, n_pts_plt))
    all_data_transverse = np.full((2, 3, 2, 5, n_pts_plt), np.nan)

    n_materials = len(foam_types)
    stretch_ten = np.zeros((n_pts_plt, n_materials))

    # Store individual sample stresses
    stress_ten_samples = [[] for _ in range(n_materials)]
    stress_com_samples = [[] for _ in range(n_materials)]
    stress_shear_samples = [[] for _ in range(n_materials)]
    stress_confcom_samples = [[] for _ in range(n_materials)]

    # Store individual sample transverse stretches
    transverse_stretch_ten_samples = [[] for _ in range(n_materials)]
    transverse_stretch_com_samples = [[] for _ in range(n_materials)]

    # Store individual sample energy returns
    energy_return_ten_samples = [[] for _ in range(n_materials)]
    energy_return_com_samples = [[] for _ in range(n_materials)]
    energy_return_shear_samples = [[] for _ in range(n_materials)]

    # --- Tension ----------
    n_cycles = 5
    max_strain_ten = 0.3
    for foam_idx, foam in enumerate(foam_types):
        # read measurements CSV
        meas_path = os.path.join(root_folder, f"{foam}-tension-measurements.csv")
        # meas_path = os.path.join(root_folder, f"tension/{foam}/tension-measurements-{foam}.csv")

        meas = pd.read_csv(meas_path, header=header).values  # table2array equivalent
        # widths: columns 4:6 in MATLAB are indices 3,4,5 (1-based). In python zero-based: 3:6
        offset = 3 if meas.shape[1] >= 10 else 0

        widths_mm = np.mean(meas[:, (offset):(offset+3)], axis=1)
        heights_mm = np.mean(meas[:, (offset+3):(offset+6)], axis=1)
        areas_mm2 = widths_mm * heights_mm
        gauge_lens_mm = meas[:, (offset+6)]  # column 10 in MATLAB

        stress_all_plt = []
        transverse_stretch_all_plt = []

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
            # Some files have an extra column and some don't
            offset = data.shape[1] - 3
            displacement_mm = data[start_idx:, offset+1].astype(float)
            force_n = data[start_idx:, offset+2].astype(float)
            time = data[start_idx:, offset].astype(float)
            # gauge_lens_mm is per sample; MATLAB uses gauge_lens_mm(sample_idx)
            strain = displacement_mm / gauge_lens_mm[sample_idx-1]
            stress_kpa = force_n / areas_mm2[sample_idx-1] * 1000.0
            transverse_path = os.path.join(transverse_folder, f"{foam}-tension-{sample_idx}.csv")
            transverse_width_interp = get_transverse_width(transverse_path, time, strain)
            transverse_stretch = transverse_width_interp / transverse_width_interp[0]
            

            ## Write time, stretch, and stress_kpa to a sheet in an excel file named "raw_data.xlsx"
            df = pd.DataFrame({'time_s': time, 'stretch': 1.0 + strain, 'stress_kpa': stress_kpa, 'transverse_stretch': transverse_stretch})
            excel_path = os.path.join(out_dir, f"{foam}-raw-data.xlsx")
            write_sheet_with_corrupt_recovery(df, excel_path, f"sample_{sample_idx}_tension")

            min_peak_dist = 100
            x_interp_plt, stress_mean_kpa_plt, transverse_stretch_plt, energy_return_sample = average_curves(strain, stress_kpa, transverse_stretch, n_cycles, n_pts_plt, min_peak_dist, "ten", max_strain_ten)
            stress_all_plt.append(stress_mean_kpa_plt)
            transverse_stretch_all_plt.append(transverse_stretch_plt / transverse_stretch_plt[0])
            energy_return_ten_samples[foam_idx].append(energy_return_sample)
            # Store individual sample data for subplot
            stress_ten_samples[foam_idx].append({
                'stretch': 1.0 + strain,
                'stress': stress_kpa
            })

        stress_all_plt = np.array(stress_all_plt)  # shape (n_samples, n_pts_plt)
        all_data_stress[0, foam_idx % 3, foam_idx // 3, :, :] = stress_all_plt
        transverse_stretch_all_plt = np.array(transverse_stretch_all_plt)  # shape (n_samples, n_pts_plt)
        all_data_transverse[0, foam_idx % 3, foam_idx // 3, :transverse_stretch_all_plt.shape[0], :] = (
            transverse_stretch_all_plt
        )
        stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
        stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)
        transverse_stretch_mean_plt = np.nanmean(transverse_stretch_all_plt, axis=0)
        transverse_stretch_var_plt = np.nanstd(transverse_stretch_all_plt, axis=0, ddof=0)

        stretch_ten[:, foam_idx] = 1.0 + x_interp_plt

    # --- Compression ----------
    max_strain_com = 0.6
    offset = [0] * 6 if worn_shoe else [1, 0]  # MATLAB offset array

    n_cycles = 4
    min_peak_dist = 1000

    stretch_com = np.zeros((n_pts_plt, n_materials))
    transverse_stretch_com = np.zeros((n_pts_plt, n_materials))
    transverse_stretch_com_std = np.zeros((n_pts_plt, n_materials))
    for foam_idx, foam in enumerate(foam_types):
        meas_path = os.path.join(root_folder, f"{foam}-comp-measurements.csv")
        meas = pd.read_csv(meas_path, header=header).values
        diameters_mm = np.mean(meas[:, 0:3], axis=1)  # columns 1:3 in MATLAB
        areas_mm2 = (diameters_mm ** 2) * np.pi / 4.0

        stress_all_plt = []
        transverse_stretch_all_plt = []
        for sample_idx in range(1,6):
            # note: MATLAB used file index sample_idx + offset(foam_idx)
            file_idx = sample_idx + offset[foam_idx]
            comp_path = os.path.join(root_folder, f"{foam}-comp-{file_idx}.txt")
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

            transverse_path = os.path.join(transverse_folder, f"{foam}-comp-{sample_idx}.csv")
            if not os.path.exists(transverse_path):
                foam_comp = foam.replace("_", "-")
                transverse_path = os.path.join(transverse_folder, f"{foam_comp}-comp-{sample_idx}.csv")
            transverse_time, transverse_width, axial_stretch = read_transverse_data(transverse_path)
            transverse_stretch = transverse_width / transverse_width[0]
            
            # transverse_width_interp = get_transverse_width(transverse_path, time_cropped, 1.0-stretch_cropped, True)
            # transverse_stretch = transverse_width_interp / transverse_width_interp[0]
            
            ## Write time, stretch, and stress_kpa to a sheet in an excel file named "raw_data.xlsx"
            df = pd.DataFrame({'time_s': time_cropped, 'stretch': stretch_cropped, 'stress_kpa': stress_kpa_cropped})
            excel_path = os.path.join(out_dir, f"{foam}-raw-data.xlsx")
            write_sheet_with_corrupt_recovery(df, excel_path, f"sample_{sample_idx}_compression")

            x_interp_plt, stress_mean_kpa, _, energy_return_sample = average_curves(
                strain, stress_kpa, np.zeros_like(stress_kpa), n_cycles, n_pts_plt, min_peak_dist, "com", max_strain_com
            )
            if np.any(axial_stretch != 0.0):
                _, _, transverse_stretch_plt, _ = average_curves(1 - axial_stretch, transverse_stretch, transverse_stretch, n_cycles, n_pts_plt, 50, "com", max_strain_com)
            else:
                transverse_stretch_plt = np.zeros_like(x_interp_plt) / 0.0
            
            # if np.all(np.isnan(transverse_width_interp)):
            
            stress_all_plt.append(stress_mean_kpa)
            transverse_stretch_all_plt.append(transverse_stretch_plt / transverse_stretch_plt[0])
            energy_return_com_samples[foam_idx].append(energy_return_sample)
            
            # Store individual sample data for subplot
            stress_com_samples[foam_idx].append({
                'stretch': 1.0 - strain,
                'stress': -stress_kpa
            })


        stress_all_plt = np.array(stress_all_plt)
        all_data_stress[1, foam_idx % 3, foam_idx // 3, :, :] = stress_all_plt
        transverse_stretch_all_plt = np.array(transverse_stretch_all_plt)  # shape (n_samples, n_pts_plt)
        all_data_transverse[1, foam_idx % 3, foam_idx // 3, :transverse_stretch_all_plt.shape[0], :] = (
            transverse_stretch_all_plt
        )
        stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
        stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)
        stretch_com[:, foam_idx] = 1.0 - x_interp_plt


    # --- Confined Compression ----------
    if not worn_shoe: 
        max_strain_com = 0.6
        offset = [0, 0]  # MATLAB offset array
        n_cycles = 4
        min_peak_dist = 1000

        stretch_conf_com = np.zeros((n_pts_plt, n_materials))
        stress_conf_com = np.zeros((n_pts_plt, n_materials))
        stress_conf_com_std = np.zeros((n_pts_plt, n_materials))
        for foam_idx, foam in enumerate(foam_types):
            meas_path = os.path.join(root_folder, f"{foam}-confcomp-measurements.csv")
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
                x_interp_plt, stress_mean_kpa, _, _ = average_curves(
                    strain, stress_kpa, np.zeros_like(stress_kpa), n_cycles, n_pts_plt, min_peak_dist, "com", max_strain_com
                )
                stress_all_plt.append(stress_mean_kpa)
                
                # Store individual sample data for subplot
                stress_confcom_samples[foam_idx].append({
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
    offset = [0] * 6 if worn_shoe else [2, 0]
    n_cycles = 3
    min_peak_dist = 1000
    strain_interp_plt = np.linspace(0, max_shr, n_pts_plt)

    strain_shr = np.zeros((n_pts_plt, n_materials))
    for foam_idx, foam in enumerate(foam_types):
        meas_path = os.path.join(root_folder, f"{foam}-shear-measurements.csv")
        meas = pd.read_csv(meas_path, header=header).values
        radii_mm = np.mean(meas[:, 0:3], axis=1) / 2.0

        stress_all_plt = []
        for sample_idx in range(1,6):
            if sample_idx == 4 and foam == "new":
                ## TODO: Fix this, currently duplicating data from new_shear_3 since new_shear_4 exported incorrectly
                shear_path = os.path.join(root_folder, f"{foam}-shear-{file_idx-1}.xls")
            else:
                file_idx = sample_idx + offset[foam_idx]
                shear_path = os.path.join(root_folder, f"{foam}-shear-{file_idx}.xls")
            # sheet "Sine Strain - 3"
            # Read Excel file with explicit engine and handle data conversion
            #### Handle new way of exporting data, each cycle in a new file
            import xlrd
            if "Sine Strain - 8" in xlrd.open_workbook(shear_path).sheet_names():
                start_time = -1
                for i in range(5, 9):
                    try:
                        df = pd.read_excel(shear_path, sheet_name=f"Sine Strain - {i}", header=None, engine='xlrd')
                    except Exception:
                        df = pd.read_excel(shear_path, sheet_name=f"Sine Strain - {i}", header=None, engine='openpyxl')
                    # Convert all columns to numeric, coercing errors to NaN
                    df = df.apply(pd.to_numeric, errors='coerce')
                    # Drop columns that are all NaN
                    df = df.dropna(axis=1, how='all')
                    # Convert to numpy array
                    subdata = df.values
                    if start_time < 0:
                        data = subdata
                        start_time = data[-1, 2].astype(float)
                    else:
                        subdata = subdata[3:, :]
                        subdata[:, 2] += start_time
                        start_time = subdata[-1, 2]
                        data = np.concatenate([data, subdata], axis=0)
            else:
                try:
                    df = pd.read_excel(shear_path, sheet_name="Sine Strain - 3", header=None, engine='xlrd')
                except Exception:
                    # fallback try default sheet
                    try:
                        df = pd.read_excel(shear_path, header=None, engine='xlrd')
                    except Exception as e:
                        # If xlrd fails, try openpyxl (for .xlsx files) or other engines
                        try:
                            df = pd.read_excel(shear_path, sheet_name="Sine Strain - 8", header=None, engine='openpyxl')
                        except Exception:
                            df = pd.read_excel(shear_path, header=None, engine='openpyxl')
                
                # Convert all columns to numeric, coercing errors to NaN
                df = df.apply(pd.to_numeric, errors='coerce')
                # Drop columns that are all NaN
                df = df.dropna(axis=1, how='all')
                # Convert to numpy array
                data = df.values


            ## For some reason the new data files have torque in g * cm instead of uN * m
            ## 1 g * cm = 0.001 kgf * 10 mm = 0.01 kgf * mm
            ## 1 kgf = 9.80665 N, so 0.01 kgf * mm = 0.0980665 N * mm
            ## g * cm is equivalent to 0.0000980665 N * m or 0.0980665 N * mm

            torque_nmm = data[3:, 0] * (0.0980665 if worn_shoe else 1e-3)
            disp_rad = data[3:, 1]
            force_n = data[3:, 4]
            gap_mm = data[3:, 5] / 1e3
            time = data[3:, 2].astype(float)
            time = time - time[0]
            shear_strain = disp_rad * radii_mm[sample_idx-1] / gap_mm[0]
            stretch = shear_strain * 0.0 + 0.8
            stress_kpa = torque_nmm / (np.pi * radii_mm[sample_idx-1]**3 / 2.0) * 1000.0
      

            ## Write time, stretch, shear_strain and stress to a sheet in an excel file named "raw_data.xlsx"
            df = pd.DataFrame({'time_s': time, 'stretch': stretch, 'shear_strain': shear_strain, 'stress_kpa': stress_kpa})
            excel_path = os.path.join(out_dir, f"{foam}-raw-data.xlsx")
            write_sheet_with_corrupt_recovery(df, excel_path, f"sample_{sample_idx}_shear")
            height_mm = gap_mm[0]
            safety_factor = 1.01
            # disp_rad_max = height_mm / radius * max_shr * safety_factor
            r = radii_mm[sample_idx-1]
            disp_rad_max = (height_mm / r) * max_shr * safety_factor if r != 0 else max_shr

            disp_rad_interp, torque_nmm_interp_mean, _, energy_return_sample = average_curves(
                disp_rad, torque_nmm, np.zeros_like(torque_nmm), n_cycles, 101, min_peak_dist, "shear", disp_rad_max
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
            stress_all_plt.append(stress_interp_kpa - stress_interp_kpa[0])
            ## Compute linearized shear stress
            strain_raw = r * disp_rad / height_mm
            shear_stress_raw = 1000.0 * torque_nmm / (np.pi * r**3 / 2.0)

            # Store individual sample data for subplot
            stress_shear_samples[foam_idx].append({
                'strain': strain_raw,
                'stress': shear_stress_raw
            })
            energy_return_shear_samples[foam_idx].append(energy_return_sample)

        stress_all_plt = np.array(stress_all_plt)
        all_data_stress[2, foam_idx % 3, foam_idx // 3, :, :] = stress_all_plt
        stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
        stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)

        # Resample for table
        # strain_interp_table = np.linspace(0.0, max_shr, n_pts_table)
        # stress_mean_table = np.interp(strain_interp_table, strain_interp_plt, stress_mean_plt)
        # stress_var_table = np.interp(strain_interp_table, strain_interp_plt, stress_var_plt)

        strain_shr[:, foam_idx] = strain_interp_plt

        # ### Stiffness per sample
        # for sample_idx in range(stress_all_plt.shape[0]):
        #     stiffness_shear_samples[foam_idx].append(
        #         fit_initial_slope(strain_interp_plt, stress_all_plt[sample_idx, :])
        #     )
        # stiffness_shear[foam_idx] = np.mean(np.array(stiffness_shear_samples[foam_idx]))
        # stiffness_shear_std[foam_idx] = np.std(np.array(stiffness_shear_samples[foam_idx]), ddof=0)

    if worn_shoe:
        stretch_conf_com = np.zeros((n_pts_plt, n_materials))
        stress_conf_com = np.zeros((n_pts_plt, n_materials))
        stress_conf_com_std = np.zeros((n_pts_plt, n_materials))

    ## Save relevant files so we can load them in main()
    ## Save all into one file
    ## Do we need to save all this or just all_data_...?
    ## Lets try to build the downstream stuff from just all_data and see 
    np.savez(
        os.path.join(out_dir, "all_data.npz"),
        all_data_stress=all_data_stress,
        all_data_transverse=all_data_transverse,
        stretch_ten=stretch_ten,
        stretch_com=stretch_com,
        strain_shr=strain_shr,
        stretch_conf_com=stretch_conf_com,
        energy_return_ten_samples=energy_return_ten_samples,
        energy_return_com_samples=energy_return_com_samples,
        energy_return_shear_samples=energy_return_shear_samples,
        stress_ten_samples=stress_ten_samples,
        stress_com_samples=stress_com_samples, 
        stress_shear_samples=stress_shear_samples,
        stress_confcom_samples=stress_confcom_samples
    )
    

    


# ---------- Main processing ----------
def main():

    if should_process:
        process_data()

    results = PreprocessingResults(
        foam_types=foam_types,
        foam_types_title=foam_types_title,
        colors=colors,
        linestyles=linestyles,
        worn_shoe=worn_shoe,
        n_pts_table=n_pts_table,
        n_pts_plt=n_pts_plt,
        FONT_SIZE=base_font_size,
        out_dir=out_dir
    )

    results.save_plots_and_tables(out_dir_plots)


    # #### Print strain energies for paper
    # mean_tensile_stress = np.mean(stress_ten[:, 0])
    # comp_stretch_min = 0.65
    # ten_stretch_max = np.max(stretch_ten[:, 0])
    # mean_compressive_stress = np.mean(stress_com[stretch_com[:, 0] > comp_stretch_min, 0])
    # strain_energy_tensile = mean_tensile_stress * (ten_stretch_max - 1.0)
    # strain_energy_compressive = mean_compressive_stress * (1 - comp_stretch_min)
    # print(f"Strain energy (tensile): {strain_energy_tensile:.2f} kJ/m^3")
    # print(f"Strain energy (compressive): {strain_energy_compressive:.2f} kJ/m^3")
    
    # ## Compute linearity of shear data
    # for foam_idx in range(n_materials):
    #     strain = strain_shr[:, foam_idx]
    #     stress = stress_shr[:, foam_idx]
    #     # Compute r2 using scipy
    #     r2_shr = pearsonr(strain, stress)
    #     print(f"R2 of shear data for {foam_types_title[foam_idx]}: {r2_shr[0]}")

if __name__ == "__main__":
    main()
