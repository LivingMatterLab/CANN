"""
process_foam_data.py

Port of the provided MATLAB script to Python using numpy, pandas, matplotlib.
Ensure your data folder structure matches the MATLAB script expectations.
"""

import os
from enum import StrEnum
from zipfile import BadZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from skfda.inference.anova import oneway_anova
from skfda.representation.grid import FDataGrid
from tabulate import tabulate

# ---------- Settings (match MATLAB) ----------
worn_shoe = True
root_folder = "./input/raw_data/asics/worn-shoe/" if worn_shoe else "./input/raw_data/asics/final-tcs/"
transverse_folder = "./input/data_mm/"
out_dir = "./input/"

colors = ["r", "g", "b"] * 2 if worn_shoe else ["r", "b"]
linestyles = (["-"] * 3 + ["--"] * 3) if worn_shoe else ["-", "-"]
foam_types = ["new-toe", "new-heel", "new-mid", "worn-toe", "worn-heel", "worn-mid"] if worn_shoe else ["leap", "turbo"]
foam_types_title = [x.replace("-", " ").title() for x in foam_types] if worn_shoe else ["FF LEAP\u2122", "FF TURBO\u2122 PLUS"] 

header = 0 if worn_shoe else None

n_pts_table = 13
n_pts_plt = 101

# Global font size for all plot text elements
FONT_SIZE = 20

# helper to ensure consistent float printing in latex composer
def fmt_fixed2(x):
    return f"{x:.2f}"

def fmt_p_value(p):
    """Format ANOVA p-values for LaTeX tables (decimal notation)."""
    if not np.isfinite(p):
        return ""

    def highlight(formatted):
        if p < 0.001:
            return rf"\cellcolor{{green!25}}{formatted}"
        if p < 0.05:
            return rf"\cellcolor{{yellow!25}}{formatted}"
        if p < 0.10:
            return rf"\cellcolor{{orange!25}}{formatted}"
        return rf"\cellcolor{{red!25}}{formatted}"

    if p == 0:
        return highlight(r"$<10^{-5}$")
    sig_figs = 1 if p < 0.0001 else 2
    exp = int(np.floor(np.log10(abs(p))))
    rounded = round(p / (10 ** exp), sig_figs - 1) * (10 ** exp)
    if rounded >= 10:
        rounded = round(rounded / 10, sig_figs - 1) * 10
    if rounded == 0:
        return highlight(r"$<10^{-5}$")
    exp_rounded = int(np.floor(np.log10(abs(rounded))))
    decimal_places = max(0, -exp_rounded + (sig_figs - 1))
    return highlight(rf"${rounded:.{decimal_places}f}$")

def fmt_ci_value(x):
    """Format confidence interval bounds for LaTeX tables."""
    if not np.isfinite(x):
        return ""
    if abs(x) >= 10:
        return rf"${int(np.round(x))}$"
    if x == 0:
        return r"$0$"
    return rf"${x:.2g}$"


def format_with_phantoms(val, std, max_digits=3, decimal_places=2):
    """Format mean ± std for stress tables with LaTeX phantoms for alignment."""
    scale = 10 ** decimal_places
    zero_frac = "0" * decimal_places

    val_sign = -1 if val < 0 else 1
    val_abs = abs(val)
    val_rounded = np.round(val_abs, decimal_places)
    val_int = int(val_rounded)
    val_frac_raw = val_rounded - val_int
    val_frac = int(np.round(val_frac_raw * scale))
    if val_frac < 0:
        val_frac = 0
    elif val_frac >= scale:
        val_int += 1
        val_frac = 0

    std_sign = -1 if std < 0 else 1
    std_abs = abs(std)
    std_rounded = np.round(std_abs, decimal_places)
    std_int = int(std_rounded)
    std_frac_raw = std_rounded - std_int
    std_frac = int(np.round(std_frac_raw * scale))
    if std_frac < 0:
        std_frac = 0
    elif std_frac >= scale:
        std_int += 1
        std_frac = 0

    val_digits = len(str(val_int)) if val_int != 0 else 1
    std_digits = len(str(std_int)) if std_int != 0 else 1

    val_phantom = r"\phantom{0}" * max(0, max_digits - val_digits)
    std_phantom = r"\phantom{0}" * max(0, max_digits - std_digits)

    if val_int == 0 and val_frac == 0:
        val_str = rf"\phantom{{0}}\phantom{{0}}0.{zero_frac}"
    elif val_int == 0:
        val_str = rf"\phantom{{0}}\phantom{{0}}0.{val_frac:0{decimal_places}d}"
    else:
        sign_str = "-" if val_sign < 0 else ""
        val_str = rf"{sign_str}{val_phantom}{val_int}.{val_frac:0{decimal_places}d}"

    if std_int == 0 and std_frac == 0:
        std_str = rf"\phantom{{0}}\phantom{{0}}0.{zero_frac}"
    elif std_int == 0:
        std_str = rf"\phantom{{0}}\phantom{{0}}0.{std_frac:0{decimal_places}d}"
    else:
        sign_str = "-" if std_sign < 0 else ""
        std_str = rf"{sign_str}{std_phantom}{std_int}.{std_frac:0{decimal_places}d}"

    return rf"{val_str}\hspace{{0.5em}}$\pm$ {std_str}"


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

def write_sheet_with_corrupt_recovery(df, excel_path, sheet_name):
    """
    Write a DataFrame to an xlsx sheet.
    If the existing workbook is corrupt, delete it and recreate it.
    """
    try:
        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    except BadZipFile:
        if os.path.exists(excel_path):
            os.remove(excel_path)
        print(f"Corrupt workbook detected and removed: {excel_path}")
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def oneway_anova_np(first, *rest, grid_points=None, n_reps=100000, return_dist=False,
                    random_state=None, p=2, equal_var=True):
    """
    One-way functional ANOVA for numpy curve samples.

    Each input array should have shape (n_samples, n_points), where each row is
    one functional observation on a common grid. Wraps skfda's oneway_anova by
    converting inputs to FDataGrid objects.
    """
    groups = [np.asarray(first, dtype=float), *[np.asarray(group, dtype=float) for group in rest]]
    if len(groups) < 2:
        raise ValueError("At least two groups must be provided.")

    n_points = groups[0].shape[-1]
    for group_idx, group in enumerate(groups):
        if group.ndim != 2:
            raise ValueError(
                f"Group {group_idx} must be 2D with shape (n_samples, n_points)."
            )
        if group.shape[-1] != n_points:
            raise ValueError("All groups must share the same number of grid points.")

    if grid_points is None:
        grid_points = np.arange(n_points, dtype=float)
    else:
        grid_points = np.asarray(grid_points, dtype=float)
        if grid_points.shape[0] != n_points:
            raise ValueError("grid_points length must match the number of columns.")

    fd_groups = [FDataGrid(group, grid_points=grid_points) for group in groups]
    return oneway_anova(
        *fd_groups,
        n_reps=n_reps,
        return_dist=return_dist,
        random_state=random_state,
        p=p,
        equal_var=equal_var,
    )

def average_curves(x, y, y2, n_cycles, n_pts, min_peak_dist, loading_mode, max_strain=-1):
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
    y2 = np.asarray(y2).flatten()
    is_shear = (loading_mode == "shear")
    # is_ten = (loading_mode == "ten")
    # if not (is_shear or is_ten or loading_mode == "com"):
    #     raise ValueError("Invalid loading_mode")

    # find peaks (maxima) in x
    peaks_vals, peaks_idx = None, None
    # Using scipy find_peaks gives indices of peaks; we need peak values too
    peaks_idx, _ = find_peaks(x, distance=min_peak_dist, height=0, prominence=0.1)
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
    minima_idx, _ = find_peaks(inverted, distance=min_peak_dist, height=0, prominence=0.1)
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

    return x_out, y_out, y2_mean, hysteresis

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
    all_data_stress = np.zeros((3, 3, 2, 5, n_pts_plt))
    ## Modify to work with worn shoe data
    # Treat each folder (new, toe, heel, mid) as a separate material 
    # Storage arrays (numpy)
    n_materials = len(foam_types)
    stretch_ten = np.zeros((n_pts_plt, n_materials))
    stress_ten = np.zeros((n_pts_plt, n_materials))
    stress_ten_std = np.zeros((n_pts_plt, n_materials))
    transverse_stretch_ten = np.zeros((n_pts_plt, n_materials))
    transverse_stretch_ten_std = np.zeros((n_pts_plt, n_materials))

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
            x_interp_plt, stress_mean_kpa_plt, transverse_stretch_plt, hysteresis_sample = average_curves(strain, stress_kpa, transverse_stretch, n_cycles, n_pts_plt, min_peak_dist, "ten", max_strain_ten)
            stress_all_plt.append(stress_mean_kpa_plt)
            transverse_stretch_all_plt.append(transverse_stretch_plt / transverse_stretch_plt[0])
            hysteresis_ten_samples[foam_idx].append(hysteresis_sample)
            # Store individual sample data for subplot
            individual_samples_tension[foam_idx].append({
                'stretch': 1.0 + strain,
                'stress': stress_kpa
            })

        stress_all_plt = np.array(stress_all_plt)  # shape (n_samples, n_pts_plt)
        all_data_stress[0, foam_idx % 3, foam_idx // 3, :, :] = stress_all_plt
        transverse_stretch_all_plt = np.array(transverse_stretch_all_plt)  # shape (n_samples, n_pts_plt)
        stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
        stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)
        transverse_stretch_mean_plt = np.nanmean(transverse_stretch_all_plt, axis=0)
        transverse_stretch_var_plt = np.nanstd(transverse_stretch_all_plt, axis=0, ddof=0)

        

        # Resample for table
        # strain_interp_table = np.linspace(0.0, max_strain_ten, n_pts_table)
        # stress_mean_table = np.interp(strain_interp_table, x_interp_plt, stress_mean_plt)
        # stress_var_table = np.interp(strain_interp_table, x_interp_plt, stress_var_plt)

        stretch_ten[:, foam_idx] = 1.0 + x_interp_plt
        stress_ten[:, foam_idx] = stress_mean_plt
        stress_ten_std[:, foam_idx] = stress_var_plt
        transverse_stretch_ten[:, foam_idx] = transverse_stretch_mean_plt
        transverse_stretch_ten_std[:, foam_idx] = transverse_stretch_var_plt
        hysteresis_ten[foam_idx] = np.mean(np.array(hysteresis_ten_samples[foam_idx]))

        # plt.plot(x_interp_plt, transverse_stretch_mean_plt)
        # plt.fill_between(x_interp_plt, transverse_stretch_mean_plt - transverse_stretch_var_plt, transverse_stretch_mean_plt + transverse_stretch_var_plt, alpha=0.25)
        # plt.show()

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
        offset = [0] * 6 if worn_shoe else [1, 0]  # MATLAB offset array

        n_cycles = 4
        min_peak_dist = 1000

        stretch_com = np.zeros((n_pts_plt, n_materials))
        stress_com = np.zeros((n_pts_plt, n_materials))
        stress_com_std = np.zeros((n_pts_plt, n_materials))
        transverse_stretch_com = np.zeros((n_pts_plt, n_materials))
        transverse_stretch_com_std = np.zeros((n_pts_plt, n_materials))
        hysteresis_com = np.zeros((n_materials))
        hysteresis_com_samples = [[] for _ in range(n_materials)]
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

                x_interp_plt, stress_mean_kpa, _, hysteresis_sample = average_curves(
                    strain, stress_kpa, np.zeros_like(stress_kpa), n_cycles, n_pts_plt, min_peak_dist, "com", max_strain_com
                )
                if np.any(axial_stretch != 0.0):
                    _, _, transverse_stretch_plt, _ = average_curves(1 - axial_stretch, transverse_stretch, transverse_stretch, n_cycles, n_pts_plt, 50, "com", max_strain_com)
                else:
                    transverse_stretch_plt = np.zeros_like(x_interp_plt) / 0.0
                
                # if np.all(np.isnan(transverse_width_interp)):
                
                stress_all_plt.append(stress_mean_kpa)
                transverse_stretch_all_plt.append(transverse_stretch_plt / transverse_stretch_plt[0])
                hysteresis_com_samples[foam_idx].append(hysteresis_sample)
                
                # Store individual sample data for subplot
                individual_samples_compression[foam_idx].append({
                    'stretch': 1.0 - strain,
                    'stress': -stress_kpa
                })
                # plt.plot(x_interp_plt, transverse_mean_stretch_plt)
                # plt.show()



            stress_all_plt = np.array(stress_all_plt)
            all_data_stress[1, foam_idx % 3, foam_idx // 3, :, :] = stress_all_plt
            transverse_stretch_all_plt = np.array(transverse_stretch_all_plt)  # shape (n_samples, n_pts_plt)
            stress_mean_plt = np.nanmean(stress_all_plt, axis=0)
            stress_var_plt = np.nanstd(stress_all_plt, axis=0, ddof=0)
            transverse_stretch_mean_plt = np.nanmean(transverse_stretch_all_plt, axis=0)
            transverse_stretch_var_plt = np.nanstd(transverse_stretch_all_plt, axis=0, ddof=0)

            # plt.plot(x_interp_plt, transverse_stretch_mean_plt)
            # plt.fill_between(x_interp_plt, transverse_stretch_mean_plt - transverse_stretch_var_plt, transverse_stretch_mean_plt + transverse_stretch_var_plt, alpha=0.25)
            # plt.show()

            # # Resample for table
            # strain_interp_table = np.linspace(0.0, max_strain_com, n_pts_table)
            # stress_mean_table = np.interp(strain_interp_table, x_interp_plt, stress_mean_plt)
            # stress_var_table = np.interp(strain_interp_table, x_interp_plt, stress_var_plt)

            stretch_com[:, foam_idx] = 1.0 - x_interp_plt
            stress_com[:, foam_idx] = -stress_mean_plt
            stress_com_std[:, foam_idx] = stress_var_plt
            transverse_stretch_com[:, foam_idx] = transverse_stretch_mean_plt
            transverse_stretch_com_std[:, foam_idx] = transverse_stretch_var_plt
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
    offset = [0] * 6 if worn_shoe else [2, 0]
    n_cycles = 3
    min_peak_dist = 1000
    strain_interp_plt = np.linspace(0, max_shr, n_pts_plt)

    strain_shr = np.zeros((n_pts_plt, n_materials))
    stress_shr = np.zeros((n_pts_plt, n_materials))
    stress_shr_std = np.zeros((n_pts_plt, n_materials))
    hysteresis_shear = np.zeros((n_materials))
    hysteresis_shear_samples = [[] for _ in range(n_materials)]
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

            disp_rad_interp, torque_nmm_interp_mean, _, hysteresis_sample = average_curves(
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
            individual_samples_shear[foam_idx].append({
                'strain': strain_raw,
                'stress': shear_stress_raw
            })
            hysteresis_shear_samples[foam_idx].append(hysteresis_sample)

        stress_all_plt = np.array(stress_all_plt)
        all_data_stress[2, foam_idx % 3, foam_idx // 3, :, :] = stress_all_plt
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

    if worn_shoe:
        stretch_conf_com = np.zeros((n_pts_plt, n_materials))
        stress_conf_com = np.zeros((n_pts_plt, n_materials))
        stress_conf_com_std = np.zeros((n_pts_plt, n_materials))

    ## Save relevant files so we can load them in main()
    ## Save all into one file
    np.savez(
        os.path.join(out_dir, "all_data.npz"),
        all_data_stress=all_data_stress,
        stretch_ten=stretch_ten,
        stress_ten=stress_ten,
        stress_ten_std=stress_ten_std,
        transverse_stretch_ten=transverse_stretch_ten,
        transverse_stretch_ten_std=transverse_stretch_ten_std,
        stretch_com=stretch_com,
        stress_com=stress_com,
        stress_com_std=stress_com_std,
        transverse_stretch_com=transverse_stretch_com,
        transverse_stretch_com_std=transverse_stretch_com_std,
        strain_shr=strain_shr,
        stress_shr=stress_shr,
        stress_shr_std=stress_shr_std,
        stretch_conf_com=stretch_conf_com,
        stress_conf_com=stress_conf_com,
        stress_conf_com_std=stress_conf_com_std,
        individual_samples_tension=individual_samples_tension,
        individual_samples_compression=individual_samples_compression,
        individual_samples_shear=individual_samples_shear,
        individual_samples_conf_compression=individual_samples_conf_compression,
        hysteresis_ten=hysteresis_ten,
        hysteresis_ten_samples=hysteresis_ten_samples,
        hysteresis_com=hysteresis_com,
        hysteresis_com_samples=hysteresis_com_samples,
        hysteresis_shear=hysteresis_shear,
        hysteresis_shear_samples=hysteresis_shear_samples,
        stiffness_ten=stiffness_ten,
        stiffness_ten_std=stiffness_ten_std,
        stiffness_com=stiffness_com,
        stiffness_com_std=stiffness_com_std,
        stiffness_shear=stiffness_shear,
        stiffness_shear_std=stiffness_shear_std,
    )

    ## Load all the data from the file
    

    

# ---------- Plotting ----------
def save_figure(fig, output_dir, filename, bbox_inches="tight"):
    """Save figure as PDF and PNG with the same basename."""
    os.makedirs(output_dir, exist_ok=True)
    stem, _ = os.path.splitext(filename)
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{stem}.{fmt}"), format=fmt, bbox_inches=bbox_inches)


def plot_difference_ci(
    all_data_stress,
    ci_min_all,
    ci_max_all,
    stretch_ten,
    stretch_com,
    strain_shr,
    regions,
    modes,
    output_dir="./Results/RawData",
):
    """New vs worn mean stress difference with 95% bootstrap CI (3x3 grid)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    x_data_by_mode = [stretch_ten, stretch_com, strain_shr]
    y_labels = [
        r"$P_{11,\mathrm{new}} - P_{11,\mathrm{worn}}$ [kPa]",
        r"$|P_{11,\mathrm{new}}| - |P_{11,\mathrm{worn}}|$ [kPa]",
        r"$P_{12,\mathrm{new}} - P_{12,\mathrm{worn}}$ [kPa]",
    ]
    x_labels = ["Stretch [-]", "Stretch [-]", "Shear strain [-]"]
    for mode in range(3):
        x_all = x_data_by_mode[mode]
        for region in range(3):
            ax = axes[mode, region]
            stress_data_new = all_data_stress[mode, region, 0, :, :]
            stress_data_worn = all_data_stress[mode, region, 1, :, :]
            mean_new = np.mean(stress_data_new, axis=0)
            mean_worn = np.mean(stress_data_worn, axis=0)
            mean_diff = mean_new - mean_worn
            ci_min = ci_min_all[mode, region]
            ci_max = ci_max_all[mode, region]
            x = x_all[:, region]
            ax.plot(x, mean_diff, label="Mean Difference")
            ax.fill_between(x, mean_diff + ci_min, mean_diff + ci_max, alpha=0.2, label="95% CI")
            ax.set_title(f"{modes[mode].capitalize()} - {regions[region].replace("worn-", "").capitalize()}", fontsize=FONT_SIZE)
            ax.grid(True)
            if region == 0:
                ax.set_ylabel(y_labels[mode], fontsize=FONT_SIZE)
            if mode == 2:
                ax.set_xlabel(x_labels[mode], fontsize=FONT_SIZE)
            if mode == 0 and region == 2:
                ax.legend(fontsize=FONT_SIZE)
            if mode == 1: 
                ax.invert_xaxis()
    fig.suptitle("New - Worn Mean Stress Difference with 95% CI", fontsize=FONT_SIZE, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "DifferenceCI.pdf")
    plt.close(fig)


class LoadingMode(StrEnum):
    TENSION = "tension"
    COMPRESSION = "compression"
    SHEAR = "shear"
    CONFINED_COMPRESSION = "confined_compression"

    @property
    def plot_title(self) -> str:
        if self is LoadingMode.CONFINED_COMPRESSION:
            return "Confined Compression"
        return self.capitalize()

    @property
    def plot_filename(self) -> str:
        if self is LoadingMode.CONFINED_COMPRESSION:
            return "ConfinedCompression.pdf"
        return f"{self.capitalize()}.pdf"

    @property
    def transverse_plot_filename(self) -> str:
        return f"{self.capitalize()}Transverse.pdf"


_TRANSVERSE_LOADING_MODES = frozenset({LoadingMode.TENSION, LoadingMode.COMPRESSION})


def plot_stress(
    mode,
    show_error_bars,
    x,
    stress,
    stress_std,
    x_table,
    stress_table,
    n_materials,
    output_dir,
):
    """Plot mean stress vs stretch/strain with std band and table resample markers."""
    if not isinstance(mode, LoadingMode):
        raise ValueError(f"mode must be one of {tuple(LoadingMode)}, got {mode!r}")

    title = mode.plot_title
    filename = mode.plot_filename

    fig, ax = plt.subplots(figsize=(7, 5))
    for foam_idx in range(n_materials):
        ax.plot(x[:, foam_idx], stress[:, foam_idx], colors[foam_idx] + linestyles[foam_idx], label=f"{foam_types_title[foam_idx]}")
        ax.plot(x_table, stress_table[:, foam_idx], colors[foam_idx] + "o", markersize=4)
        if show_error_bars:
            ax.fill_between(
                x[:, foam_idx],
                stress[:, foam_idx] - stress_std[:, foam_idx],
                stress[:, foam_idx] + stress_std[:, foam_idx],
                color=colors[foam_idx],
                alpha=0.25,
            )
    if mode in (LoadingMode.COMPRESSION, LoadingMode.CONFINED_COMPRESSION):
        ax.invert_xaxis()
        ax.invert_yaxis()
    if mode == LoadingMode.TENSION:
        ax.set_xlim(1.0, 1.3)
    xlabel = "Shear Strain [-]" if mode == LoadingMode.SHEAR else "Stretch [-]"
    ylabel = "Shear Stress [kPa]" if mode == LoadingMode.SHEAR else "Stress [kPa]"
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    if mode == LoadingMode.SHEAR:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(True)
    plt.tight_layout()
    save_figure(fig, output_dir, filename)
    plt.close(fig)


def plot_transverse_stretch(
    mode,
    show_error_bars,
    axial_stretch,
    transverse_stretch,
    transverse_stretch_std,
    n_materials,
    output_dir,
):
    """Plot transverse stretch vs axial stretch with optional std band."""
    if not isinstance(mode, LoadingMode):
        raise ValueError(f"mode must be one of {tuple(LoadingMode)}, got {mode!r}")
    if mode not in _TRANSVERSE_LOADING_MODES:
        raise ValueError(f"transverse plot mode must be tension or compression, got {mode!r}")

    title = mode.capitalize()
    filename = mode.transverse_plot_filename

    fig, ax = plt.subplots(figsize=(7, 5))
    for foam_idx in range(n_materials):
        ax.plot(
            axial_stretch[:, foam_idx],
            transverse_stretch[:, foam_idx],
            colors[foam_idx] + linestyles[foam_idx],
            label=f"{foam_types_title[foam_idx]}",
        )
        if show_error_bars:
            ax.fill_between(
                axial_stretch[:, foam_idx],
                transverse_stretch[:, foam_idx] - transverse_stretch_std[:, foam_idx],
                transverse_stretch[:, foam_idx] + transverse_stretch_std[:, foam_idx],
                color=colors[foam_idx],
                alpha=0.25,
            )
    if mode == LoadingMode.COMPRESSION:
        ax.invert_xaxis()
    if mode == LoadingMode.TENSION:
        ax.invert_yaxis()
    ax.set_xlabel("Axial Stretch [-]", fontsize=FONT_SIZE)
    ax.set_ylabel("Transverse Stretch [-]", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(True)
    plt.tight_layout()
    save_figure(fig, output_dir, filename)
    plt.close(fig)


def plot_individual_samples(
    foam_idx,
    individual_samples_tension,
    individual_samples_compression,
    individual_samples_shear,
    individual_samples_conf_compression,
    output_dir,
):
    """5x4 subplot figure: individual tension/compression/shear/confined samples for one material."""
    fig, axes = plt.subplots(5, 4, figsize=(14, 16))
    fig.suptitle(f"{foam_types_title[foam_idx]} - Individual Samples", fontsize=FONT_SIZE, fontweight="bold")

    ten_x_min, ten_x_max, ten_y_min, ten_y_max = np.inf, -np.inf, np.inf, -np.inf
    for sample_idx in range(len(individual_samples_tension[foam_idx])):
        sample_data = individual_samples_tension[foam_idx][sample_idx]
        ten_x_min = min(ten_x_min, np.nanmin(sample_data["stretch"]))
        ten_x_max = max(ten_x_max, np.nanmax(sample_data["stretch"]))
        ten_y_min = min(ten_y_min, np.nanmin(sample_data["stress"]))
        ten_y_max = max(ten_y_max, np.nanmax(sample_data["stress"]))

    com_x_min, com_x_max, com_y_min, com_y_max = np.inf, -np.inf, np.inf, -np.inf
    for sample_idx in range(len(individual_samples_compression[foam_idx])):
        sample_data = individual_samples_compression[foam_idx][sample_idx]
        com_x_min = min(com_x_min, np.nanmin(sample_data["stretch"]))
        com_x_max = max(com_x_max, np.nanmax(sample_data["stretch"]))
        com_y_min = min(com_y_min, np.nanmin(sample_data["stress"]))
        com_y_max = max(com_y_max, np.nanmax(sample_data["stress"]))

    conf_com_x_min = conf_com_x_max = conf_com_y_min = conf_com_y_max = None
    if not worn_shoe:
        conf_com_x_min, conf_com_x_max, conf_com_y_min, conf_com_y_max = np.inf, -np.inf, np.inf, -np.inf
        for sample_idx in range(len(individual_samples_conf_compression[foam_idx])):
            sample_data = individual_samples_conf_compression[foam_idx][sample_idx]
            conf_com_x_min = min(conf_com_x_min, np.nanmin(sample_data["stretch"]))
            conf_com_x_max = max(conf_com_x_max, np.nanmax(sample_data["stretch"]))
            conf_com_y_min = min(conf_com_y_min, np.nanmin(sample_data["stress"]))
            conf_com_y_max = max(conf_com_y_max, np.nanmax(sample_data["stress"]))

    shr_x_min, shr_x_max, shr_y_min, shr_y_max = np.inf, -np.inf, np.inf, -np.inf
    for sample_idx in range(len(individual_samples_shear[foam_idx])):
        sample_data = individual_samples_shear[foam_idx][sample_idx]
        shr_x_min = min(shr_x_min, np.nanmin(sample_data["strain"]))
        shr_x_max = max(shr_x_max, np.nanmax(sample_data["strain"]))
        shr_y_min = min(shr_y_min, np.nanmin(sample_data["stress"]))
        shr_y_max = max(shr_y_max, np.nanmax(sample_data["stress"]))

    for sample_idx in range(5):
        ax = axes[sample_idx, 0]
        if sample_idx < len(individual_samples_tension[foam_idx]):
            sample_data = individual_samples_tension[foam_idx][sample_idx]
            ax.plot(sample_data["stretch"], sample_data["stress"], colors[foam_idx], linewidth=1.5)
        ax.set_xlabel("Stretch [-]", fontsize=FONT_SIZE)
        ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
        ax.set_title(f"Tension \n Sample {sample_idx + 1}", fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        ax.grid(True, alpha=0.3)
        if ten_x_max > ten_x_min and ten_y_max > ten_y_min:
            ax.set_xlim(1.0, 1.3)
            ax.set_ylim(ten_y_min, ten_y_max)

    for sample_idx in range(5):
        ax = axes[sample_idx, 1]
        if sample_idx < len(individual_samples_compression[foam_idx]):
            sample_data = individual_samples_compression[foam_idx][sample_idx]
            ax.plot(sample_data["stretch"], sample_data["stress"], colors[foam_idx], linewidth=1.5)
        ax.set_xlabel("Stretch [-]", fontsize=FONT_SIZE)
        ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
        ax.set_title(f"Compression \n Sample {sample_idx + 1}", fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        ax.grid(True, alpha=0.3)
        if com_x_max > com_x_min and com_y_max > com_y_min:
            ax.set_xlim(com_x_min, com_x_max)
            ax.set_ylim(com_y_min, com_y_max)
        ax.invert_xaxis()
        ax.invert_yaxis()

    for sample_idx in range(5):
        ax = axes[sample_idx, 2]
        if sample_idx < len(individual_samples_shear[foam_idx]):
            sample_data = individual_samples_shear[foam_idx][sample_idx]
            ax.plot(sample_data["strain"], sample_data["stress"], colors[foam_idx], linewidth=1.5)
        ax.set_xlabel("Shear Strain [-]", fontsize=FONT_SIZE)
        ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
        ax.set_title(f"Shear \n Sample {sample_idx + 1}", fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        ax.grid(True, alpha=0.3)
        if shr_x_max > shr_x_min and shr_y_max > shr_y_min:
            ax.set_xlim(shr_x_min, shr_x_max)
            ax.set_ylim(shr_y_min, shr_y_max)

    if not worn_shoe:
        for sample_idx in range(5):
            ax = axes[sample_idx, 3]
            if sample_idx < len(individual_samples_conf_compression[foam_idx]):
                sample_data = individual_samples_conf_compression[foam_idx][sample_idx]
                ax.plot(sample_data["stretch"], sample_data["stress"], colors[foam_idx], linewidth=1.5)
            ax.set_xlabel("Stretch [-]", fontsize=FONT_SIZE)
            ax.set_ylabel("Stress [kPa]", fontsize=FONT_SIZE)
            ax.set_title(f"Confined Compression\nSample {sample_idx + 1}", fontsize=FONT_SIZE)
            ax.tick_params(labelsize=FONT_SIZE)
            ax.grid(True, alpha=0.3)
            if conf_com_x_max > conf_com_x_min and conf_com_y_max > conf_com_y_min:
                ax.set_xlim(conf_com_x_min, conf_com_x_max)
                ax.set_ylim(conf_com_y_min, conf_com_y_max)
            ax.invert_xaxis()
            ax.invert_yaxis()

    plt.tight_layout()
    filename = f"{foam_types[foam_idx]}_individual_samples.pdf"
    save_figure(fig, output_dir, filename)
    plt.close(fig)


# ---------- Table generation ----------
def _latex_tabular_from_tabulate(data, headers, colspec):
    table = tabulate(data, headers=headers, tablefmt="latex_raw")
    return table.replace(table.split("\n", 1)[0], rf"\begin{{tabular}}{{{colspec}}}", 1)


def save_anova_table(anova_p_values, output_dir="./Results/RawData"):
    test_names = np.array(
        [
            "6 way",
            "New vs Worn (toe)",
            "New vs Worn (heel)",
            "New vs Worn (mid)",
            "Toe vs Mid vs Heel",
            "Toe vs Heel",
            "Toe vs Mid",
            "Mid vs Heel",
        ]
    )[:, np.newaxis]
    headers = [
        r"\makecell{Test Name}",
        r"\makecell{Toe \\ Worn}",
        r"\makecell{Toe \\ New}",
        r"\makecell{Heel \\ Worn}",
        r"\makecell{Heel \\ New}",
        r"\makecell{Mid \\ Worn}",
        r"\makecell{Mid \\ New}",
        r"\makecell{$p$ Value \\ Ten}",
        r"\makecell{$p$ Value \\ Com}",
        r"\makecell{$p$ Value \\ Shear}",
    ]
    group_data = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0],
            [0, 0, 0, 0, 1, 2],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 0, 0],
            [1, 1, 0, 0, 2, 2],
            [0, 0, 1, 1, 2, 2],
        ]
    )
    group_data_formatted = np.where(group_data == 0, "", group_data.astype(object))
    anova_p_values_formatted = np.vectorize(fmt_p_value)(anova_p_values)
    data = np.concatenate([test_names, group_data_formatted, anova_p_values_formatted], axis=1)
    table = _latex_tabular_from_tabulate(data, headers, r"|l||c|c|c|c|c|c||l|l|l|")
    print(table)
    os.makedirs(output_dir, exist_ok=True)
    anova_table_path = os.path.join(output_dir, "anova_table.tex")
    with open(anova_table_path, "w") as f:
        f.write(table)
    print(f"ANOVA table saved to: {anova_table_path}")


def save_confidence_interval_table(ci_min_all, ci_max_all, regions, output_dir="./Results/RawData"):
    ci_headers = [
        r"\makecell{Region}",
        r"\makecell{Tension \\ Lower}",
        r"\makecell{Tension \\ Upper}",
        r"\makecell{Compression \\ Lower}",
        r"\makecell{Compression \\ Upper}",
        r"\makecell{Shear \\ Lower}",
        r"\makecell{Shear \\ Upper}",
    ]
    ci_rows = []
    for region in range(3):
        row = [regions[region].capitalize()]
        for mode in range(3):
            row.append(fmt_ci_value(ci_min_all[mode, region]))
            row.append(fmt_ci_value(ci_max_all[mode, region]))
        ci_rows.append(row)
    ci_table = _latex_tabular_from_tabulate(ci_rows, ci_headers, r"|l||c|c||c|c||c|c|")
    print("\nConfidence interval table:")
    print(ci_table)
    os.makedirs(output_dir, exist_ok=True)
    ci_table_path = os.path.join(output_dir, "confidence_interval_table.tex")
    with open(ci_table_path, "w") as f:
        f.write(ci_table)
    print(f"Confidence interval table saved to: {ci_table_path}")


def build_hysteresis_table_latex(
    hysteresis_ten,
    hysteresis_com,
    hysteresis_shear,
    hysteresis_ten_samples,
    hysteresis_com_samples,
    hysteresis_shear_samples,
    n_materials,
):
    ten_means = hysteresis_ten * 100.0
    com_means = hysteresis_com * 100.0
    shr_means = hysteresis_shear * 100.0
    ten_stds = np.array([np.std(np.array(hysteresis_ten_samples[i]), ddof=0) for i in range(n_materials)]) * 100.0
    com_stds = np.array([np.std(np.array(hysteresis_com_samples[i]), ddof=0) for i in range(n_materials)]) * 100.0
    shr_stds = np.array([np.std(np.array(hysteresis_shear_samples[i]), ddof=0) for i in range(n_materials)]) * 100.0

    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Material & Tension & Compression & Shear \\")
    lines.append(r"\hline")
    for foam_idx, foam in enumerate(foam_types):
        lines.append(
            f"{foam} & "
            f"{ten_means[foam_idx]:.1f} $\\pm$ {ten_stds[foam_idx]:.1f} & "
            f"{com_means[foam_idx]:.1f} $\\pm$ {com_stds[foam_idx]:.1f} & "
            f"{shr_means[foam_idx]:.1f} $\\pm$ {shr_stds[foam_idx]:.1f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def save_hysteresis_table(
    hysteresis_ten,
    hysteresis_com,
    hysteresis_shear,
    hysteresis_ten_samples,
    hysteresis_com_samples,
    hysteresis_shear_samples,
    n_materials,
    output_dir="./Results/RawData",
):
    print("\n--- Hysteresis Values ---")
    hysteresis_table = build_hysteresis_table_latex(
        hysteresis_ten,
        hysteresis_com,
        hysteresis_shear,
        hysteresis_ten_samples,
        hysteresis_com_samples,
        hysteresis_shear_samples,
        n_materials,
    )
    print(hysteresis_table)
    os.makedirs(output_dir, exist_ok=True)
    hysteresis_table_path = os.path.join(output_dir, "hysteresis_table.tex")
    with open(hysteresis_table_path, "w") as f:
        f.write(hysteresis_table)
    print(f"\nHysteresis table saved to: {hysteresis_table_path}\n")


def build_stiffness_table_latex(stiffness_ten, stiffness_com, stiffness_shear, stiffness_ten_std, stiffness_com_std, stiffness_shear_std):
    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Material & Tension & Compression & Shear \\")
    lines.append(r"\hline")
    for foam_idx, foam in enumerate(foam_types):
        lines.append(
            f"{foam} & "
            f"{stiffness_ten[foam_idx]:.1f} $\\pm$ {stiffness_ten_std[foam_idx]:.1f} & "
            f"{stiffness_com[foam_idx]:.1f} $\\pm$ {stiffness_com_std[foam_idx]:.1f} & "
            f"{stiffness_shear[foam_idx]:.1f} $\\pm$ {stiffness_shear_std[foam_idx]:.1f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def save_stiffness_table(
    stiffness_ten,
    stiffness_com,
    stiffness_shear,
    stiffness_ten_std,
    stiffness_com_std,
    stiffness_shear_std,
    output_dir="./Results/RawData",
):
    print("\n--- Stiffness Values ---")
    stiffness_table = build_stiffness_table_latex(
        stiffness_ten, stiffness_com, stiffness_shear, stiffness_ten_std, stiffness_com_std, stiffness_shear_std
    )
    print(stiffness_table)
    os.makedirs(output_dir, exist_ok=True)
    stiffness_table_path = os.path.join(output_dir, "stiffness_table.tex")
    with open(stiffness_table_path, "w") as f:
        f.write(stiffness_table)
    print(f"\nStiffness table saved to: {stiffness_table_path}\n")


def build_stress_table_latex(
    mat,
    stretch_ten_table,
    stress_ten_table,
    stress_ten_std_table,
    transverse_stretch_ten_table,
    transverse_stretch_ten_std_table,
    stretch_com_table,
    stress_com_table,
    stress_com_std_table,
    transverse_stretch_com_table,
    transverse_stretch_com_std_table,
    strain_shr_table,
    stress_shr_table,
    stress_shr_std_table,
    stiffness_ten,
    stiffness_ten_std,
    stiffness_com,
    stiffness_com_std,
    stiffness_shear,
    stiffness_shear_std,
    hysteresis_ten_samples,
    hysteresis_com_samples,
    hysteresis_shear_samples,
):
    foam_name = foam_types[mat]
    ten_stretch = stretch_ten_table
    ten_stress = stress_ten_table[:, mat]
    ten_std = stress_ten_std_table[:, mat]
    ten_trans = transverse_stretch_ten_table[:, mat]
    ten_trans_std = transverse_stretch_ten_std_table[:, mat]
    com_stretch = stretch_com_table[::-1]
    com_stress = -stress_com_table[::-1, mat]
    com_std = stress_com_std_table[::-1, mat]
    com_trans = transverse_stretch_com_table[::-1, mat]
    com_trans_std = transverse_stretch_com_std_table[::-1, mat]
    shr_strain = strain_shr_table
    shr_stress = stress_shr_table[:, mat]
    shr_std = stress_shr_std_table[:, mat]

    E_ten = stiffness_ten[mat]
    E_ten_std = stiffness_ten_std[mat]
    E_com = stiffness_com[mat]
    E_com_std = stiffness_com_std[mat]
    G_shr = stiffness_shear[mat]
    G_shr_std = stiffness_shear_std[mat]
    energy_return_ten = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_ten_samples[mat]])) * 100.0
    energy_return_ten_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_ten_samples[mat]]), ddof=0) * 100.0
    energy_return_com = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_com_samples[mat]])) * 100.0
    energy_return_com_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_com_samples[mat]]), ddof=0) * 100.0
    energy_return_shr = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_shear_samples[mat]])) * 100.0
    energy_return_shr_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_shear_samples[mat]]), ddof=0) * 100.0

    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(
        rf"\caption{{\sffamily{{\bfseries{{{foam_name} data from tension, compression, shear experiments.}}}}}} "
        rf"Recorded Piola stress $P$ at equally spaced axial stretch $\lambda$ or shear strain $\gamma$ "
        rf"intervals for the {foam_name} foam."
    )
    lines.append(r"The first two columns represent uniaxial tension,")
    lines.append(r"the middle two columns uniaxial compression, and")
    lines.append(r"the last two columns simple shear.")
    lines.append(r"Means and standard deviations are reported across $n=5$ samples.")
    lines.append(r"\vspace*{0.1cm}")
    lines.append(r"\small")
    lines.append(r"\centering")
    lines.append(rf"\label{{table:{foam_name}}}")
    lines.append(r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    lines.append(r"\begin{tabular}{|ccc||ccc||cc|}")
    lines.append(r"\hline")
    lines.append(r"  \multicolumn{3}{|c||}{\sffamily{\bfseries{uniaxial tension}}}")
    lines.append(r"& \multicolumn{3}{c||} {\sffamily{\bfseries{uniaxial compression}}}")
    lines.append(r"& \multicolumn{2}{c|}  {\sffamily{\bfseries{simple shear}}} \\")
    lines.append(r"  \multicolumn{3}{|c||}{$n=5$}")
    lines.append(r"& \multicolumn{3}{c||}{$n=5$}")
    lines.append(r"& \multicolumn{2}{c|}{$n=5$} \\ \hline")
    lines.append(r"$\lambda$ & $P_{11}$ & $\lambda_2$ & $\lambda$ & $P_{11}$ & $\lambda_2$ & $\gamma$ & $P_{12}$  \\")
    lines.append(r"\,[-] & [kPa] & [-] & [-] & [kPa] & [-] & [-] & [kPa]  \\")
    lines.append(r"\hline \hline")

    for i in range(n_pts_table):
        ten_str = format_with_phantoms(ten_stress[i], ten_std[i])
        ten_trans_str = format_with_phantoms(ten_trans[i], ten_trans_std[i], decimal_places=3)
        com_str = format_with_phantoms(com_stress[i], com_std[i])
        com_trans_str = format_with_phantoms(com_trans[i], com_trans_std[i], decimal_places=3)
        shr_str = format_with_phantoms(shr_stress[i], shr_std[i])
        hline_after = (i == 0) or (i == 3) or (i == 4) or (i == 7) or (i == 8) or (i == 11)
        lines.append(
            f"{ten_stretch[i]:.3f} & {ten_str} & {ten_trans_str} & "
            f"{com_stretch[i]:.3f} & {com_str} & {com_trans_str} & "
            f"{shr_strain[i]:.3f} & {shr_str}"
        )
        if hline_after:
            lines.append(r" \\ \hline")
        else:
            lines.append(r" \\")

    lines.append(r"\hline \hline")
    lines.append(r"  \multicolumn{3}{|c||}{\sffamily{\bfseries{tensile stiffness}}}")
    lines.append(r"& \multicolumn{3}{c||} {\sffamily{\bfseries{compressive stiffness}}}")
    lines.append(r"& \multicolumn{2}{c|}  {\sffamily{\bfseries{shear stiffness}}} \\")
    lines.append(rf"  \multicolumn{{3}}{{|c||}}{{$\textsf{{E}}_{{\rm{{ten}}}} = {E_ten:.2f} \pm {E_ten_std:.2f}$\,kPa}}")
    lines.append(rf"& \multicolumn{{3}}{{c||}} {{$\textsf{{E}}_{{\rm{{com}}}} = {E_com:.2f} \pm {E_com_std:.2f}$\,kPa}}")
    lines.append(rf"& \multicolumn{{2}}{{c|}}  {{$\textsf{{G}}_{{\rm{{shr}}}} = {G_shr:.2f} \pm {G_shr_std:.2f}$\,kPa}} \\")
    lines.append(r"\hline \hline")
    lines.append(r"  \multicolumn{3}{|c||}{\sffamily{\bfseries{energy return}}}")
    lines.append(r"& \multicolumn{3}{c||} {\sffamily{\bfseries{energy return}}}")
    lines.append(r"& \multicolumn{2}{c|}  {\sffamily{\bfseries{energy return}}} \\")
    lines.append(rf"  \multicolumn{{3}}{{|c||}}{{$\eta_{{\rm{{ten}}}}  = {energy_return_ten:.1f} \pm {energy_return_ten_std:.1f} \%$}}")
    lines.append(rf"& \multicolumn{{3}}{{c||}} {{$\eta_{{\rm{{com}}}}  = {energy_return_com:.1f} \pm {energy_return_com_std:.1f}\%$}}")
    lines.append(rf"& \multicolumn{{2}}{{c|}}  {{$\eta_{{\rm{{shr}}}}  = {energy_return_shr:.1f} \pm {energy_return_shr_std:.1f} \%$}} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"%% End {foam_name} table")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def save_stress_tables(
    n_materials,
    stretch_ten_table,
    stress_ten_table,
    stress_ten_std_table,
    transverse_stretch_ten_table,
    transverse_stretch_ten_std_table,
    stretch_com_table,
    stress_com_table,
    stress_com_std_table,
    transverse_stretch_com_table,
    transverse_stretch_com_std_table,
    strain_shr_table,
    stress_shr_table,
    stress_shr_std_table,
    stiffness_ten,
    stiffness_ten_std,
    stiffness_com,
    stiffness_com_std,
    stiffness_shear,
    stiffness_shear_std,
    hysteresis_ten_samples,
    hysteresis_com_samples,
    hysteresis_shear_samples,
    output_dir="./Results/RawData",
):
    os.makedirs(output_dir, exist_ok=True)
    for mat in range(n_materials):
        foam_name = foam_types[mat]
        tbl = build_stress_table_latex(
            mat,
            stretch_ten_table,
            stress_ten_table,
            stress_ten_std_table,
            transverse_stretch_ten_table,
            transverse_stretch_ten_std_table,
            stretch_com_table,
            stress_com_table,
            stress_com_std_table,
            transverse_stretch_com_table,
            transverse_stretch_com_std_table,
            strain_shr_table,
            stress_shr_table,
            stress_shr_std_table,
            stiffness_ten,
            stiffness_ten_std,
            stiffness_com,
            stiffness_com_std,
            stiffness_shear,
            stiffness_shear_std,
            hysteresis_ten_samples,
            hysteresis_com_samples,
            hysteresis_shear_samples,
        )
        print(f"LaTeX table for Material {mat + 1} ({foam_name}):\n{tbl}\n\n")
        table_path = os.path.join(output_dir, f"{foam_name}_stress_table.tex")
        with open(table_path, "w") as f:
            f.write(tbl)
        print(f"Stress table saved to: {table_path}\n")


def save_stress_excel(
    stretch_ten,
    stress_ten,
    stress_ten_std,
    stretch_com,
    stress_com,
    stress_com_std,
    transverse_stretch_ten,
    transverse_stretch_ten_std,
    transverse_stretch_com,
    transverse_stretch_com_std,
    strain_shr,
    stress_shr,
    stress_shr_std,
    excel_dir=None,
    excel_filename="WornFoamData.xlsx",
):
    """Write combined tension/compression/shear columns to Excel (MATLAB-style layout)."""
    if excel_dir is None:
        excel_dir = out_dir

    stretch_ut = np.vstack([np.flipud(stretch_com), stretch_ten[1:, :]])
    stress_ut = np.vstack([np.flipud(stress_com), stress_ten[1:, :]])
    stress_ut_std = np.vstack([np.flipud(stress_com_std), stress_ten_std[1:, :]])
    transverse_stretch_ut = np.vstack([np.flipud(transverse_stretch_com), transverse_stretch_ten[1:, :]])
    transverse_stretch_ut_std = np.vstack(
        [np.flipud(transverse_stretch_com_std), transverse_stretch_ten_std[1:, :]]
    )

    strain_ss = np.vstack([-np.flipud(strain_shr), strain_shr[1:, :]])
    stress_ss = np.vstack([-np.flipud(stress_shr), stress_shr[1:, :]])
    stress_ss_std = np.vstack([np.flipud(stress_shr_std), stress_shr_std[1:, :]])

    data_cols = []
    headings = []
    for i, foam in enumerate(foam_types):
        data_cols.append(stretch_ut[:, i])
        headings.append(f"{foam}-comten-ax-stretch")
        data_cols.append(transverse_stretch_ut[:, i])
        headings.append(f"{foam}-comten-trans-stretch")
        data_cols.append(transverse_stretch_ut_std[:, i])
        headings.append(f"{foam}-comten-trans-stretch-stddev")
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

    df_out = pd.DataFrame(np.column_stack(data_cols), columns=headings)
    os.makedirs(excel_dir, exist_ok=True)
    out_path = os.path.join(excel_dir, excel_filename)
    df_out.to_excel(out_path, index=False)
    print(f"Wrote output excel to: {out_path}")


# ---------- Main processing ----------
def main():

    should_process = False
    if should_process:
        process_data()
    
    ## Load all the data from the file
    n_materials = len(foam_types)
    all_data = np.load(os.path.join(out_dir, "all_data.npz"), allow_pickle=True)
    required_keys = [
        "stretch_com", "stress_com", "stress_com_std",
        "transverse_stretch_com", "transverse_stretch_com_std",
        "strain_shr", "stress_shr", "stress_shr_std",
        "stretch_conf_com", "stress_conf_com", "stress_conf_com_std",
    ]
    missing_keys = [key for key in required_keys if key not in all_data.files]
    if missing_keys:
        raise KeyError(
            f"Missing keys in all_data.npz: {missing_keys}. "
            "Run process_data() once (set should_process=True) to regenerate."
        )
    all_data_stress = all_data["all_data_stress"]
    stretch_ten = all_data["stretch_ten"]
    stress_ten = all_data["stress_ten"]
    stress_ten_std = all_data["stress_ten_std"]
    transverse_stretch_ten = all_data["transverse_stretch_ten"]
    transverse_stretch_ten_std = all_data["transverse_stretch_ten_std"]
    stretch_com = all_data["stretch_com"]
    stress_com = all_data["stress_com"]
    stress_com_std = all_data["stress_com_std"]
    transverse_stretch_com = all_data["transverse_stretch_com"]
    transverse_stretch_com_std = all_data["transverse_stretch_com_std"]
    strain_shr = all_data["strain_shr"]
    stress_shr = all_data["stress_shr"]
    stress_shr_std = all_data["stress_shr_std"]
    stretch_conf_com = all_data["stretch_conf_com"]
    stress_conf_com = all_data["stress_conf_com"]
    stress_conf_com_std = all_data["stress_conf_com_std"]
    individual_samples_tension = all_data["individual_samples_tension"]
    individual_samples_compression = all_data["individual_samples_compression"]
    individual_samples_shear = all_data["individual_samples_shear"]
    individual_samples_conf_compression = all_data["individual_samples_conf_compression"]
    hysteresis_ten = all_data["hysteresis_ten"]
    hysteresis_ten_samples = all_data["hysteresis_ten_samples"]
    hysteresis_com = all_data["hysteresis_com"]
    hysteresis_com_samples = all_data["hysteresis_com_samples"]
    hysteresis_shear = all_data["hysteresis_shear"]
    hysteresis_shear_samples = all_data["hysteresis_shear_samples"]
    stiffness_ten = all_data["stiffness_ten"]
    stiffness_ten_std = all_data["stiffness_ten_std"]
    stiffness_com = all_data["stiffness_com"]
    stiffness_com_std = all_data["stiffness_com_std"]
    stiffness_shear = all_data["stiffness_shear"]
    stiffness_shear_std = all_data["stiffness_shear_std"]

    

    # --- Perform statistical tests ---
    ## Perform FDA ANOVA on all the data for each mode
    regions = foam_types[3:]
    modes = ["tension", "compression", "shear"]
    # Rows: 0=six-way, 1-3=new vs worn (toe/heel/mid), 4=three-way,
    #       5-7=region pairs (toe-heel, toe-mid, heel-mid); cols=modes
    anova_p_values = np.full((8, 3), np.nan)
    region_pairs = [(0, 1), (0, 2), (1, 2)]
    
    print("6 way FDA ANOVA across all data:")
    for mode in range(3):# tension, compression, shear
        data_reshape = all_data_stress[mode, :, :, :, :].reshape(-1, 5, n_pts_plt)
        data_list = [data_reshape[i, :, :] for i in range(data_reshape.shape[0])]
        # Perform FDA ANOVA
        _, p_val = oneway_anova_np(*data_list)
        anova_p_values[0, mode] = p_val
        print(f"\tp value for {modes[mode]}: {p_val}")
    ## Perform pairwise FDA ANOVA for each mode and region
    print("Pairwise FDA ANOVA comparing new vs worn for each region and mode:")
    for mode in range(3):# tension, compression, shear
        for region in range(3):
            data_reshape = all_data_stress[mode, region, :, :, :].reshape(-1, 5, n_pts_plt)
            # Perform FDA ANOVA
            _, p_val = oneway_anova_np(data_reshape[0, :, :], data_reshape[1, :, :])
            anova_p_values[1 + region, mode] = p_val
            print(f"\tp value for {modes[mode]} and {regions[region]}: {p_val}")

    ## Perform 3 way FDA ANOVA ignoring worn vs new
    print("3 way FDA ANOVA ignoring worn vs new:")
    for mode in range(3):# tension, compression, shear
        data_reshape = all_data_stress[mode, :, :, :, :].reshape(3, -1, n_pts_plt)
        # Perform 3 way FDA ANOVA
        _, p_val = oneway_anova_np(data_reshape[0, :, :], data_reshape[1, :, :], data_reshape[2, :, :])
        anova_p_values[4, mode] = p_val
        print(f"\tp value for {modes[mode]}: {p_val}")

    ## Perform 2 way FDA ANOVA ignoring worn vs new
    print("2 way FDA ANOVA ignoring worn vs new:")
    for mode in range(3):# tension, compression, shear
        data_reshape = all_data_stress[mode, :, :, :, :].reshape(3, -1, n_pts_plt)
        for pair_idx, (region1, region2) in enumerate(region_pairs):
            _, p_val = oneway_anova_np(data_reshape[region1, :, :], data_reshape[region2, :, :])
            anova_p_values[5 + pair_idx, mode] = p_val
            print(f"\tp value for {modes[mode]} comparing {regions[region1]} and {regions[region2]}: {p_val}")

    ## Compute confidence intervals for the mean difference between new and worn shoes for each mode and region
    print("Confidence intervals:")
    ci_min_all = np.zeros((3, 3))
    ci_max_all = np.zeros((3, 3))
    for mode in range(3):# tension, compression, shear
        for region in range(3):
            stress_data_new = all_data_stress[mode, region, 0, :, :]
            stress_data_worn = all_data_stress[mode, region, 1, :, :]
            # Compute mean and sample variance of stress_data_new and stress_data_worn
            mean_new = np.mean(stress_data_new, axis=0)
            var_new = np.mean(np.var(stress_data_new, axis=0, ddof=1))
            mean_worn = np.mean(stress_data_worn, axis=0)
            var_worn = np.mean(np.var(stress_data_worn, axis=0, ddof=1))
            mean_diff = mean_new - mean_worn
            var_diff = var_new + var_worn
            # Perform bootstrapping to get confidence intervals
            n_samples = stress_data_new.shape[-2]
            n_pts = stress_data_new.shape[-1]
            n_bootstraps = 1000
            bootstrap_means_new = np.zeros((n_bootstraps, n_pts))
            bootstrap_vars_new = np.zeros((n_bootstraps, 1))
            for i in range(n_bootstraps):
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_means_new[i, :] = np.mean(stress_data_new[bootstrap_indices, :], axis=0)
                bootstrap_vars_new[i] = np.mean(np.var(stress_data_new[bootstrap_indices, :], axis=0, ddof=1)) if n_samples > 1 else 0
            bootstrap_means_worn = np.zeros((n_bootstraps, n_pts))
            bootstrap_vars_worn = np.zeros((n_bootstraps, 1))
            for i in range(n_bootstraps):
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_means_worn[i, :] = np.mean(stress_data_worn[bootstrap_indices, :], axis=0)
                bootstrap_vars_worn[i] = np.mean(np.var(stress_data_worn[bootstrap_indices, :], axis=0, ddof=1)) if n_samples > 1 else 0
            bootstrap_means_diff = bootstrap_means_new - bootstrap_means_worn
            bootstrap_vars_diff = bootstrap_vars_new + bootstrap_vars_worn
            bootstrap_z_scores = (bootstrap_means_diff - mean_diff) / np.sqrt(bootstrap_vars_diff)
            bootstrap_z_scores_max = np.max(np.abs(bootstrap_z_scores), axis=1)
            bootstrap_ci_z_score = np.percentile(bootstrap_z_scores_max, [95])
            ci_min = -bootstrap_ci_z_score * np.sqrt(var_diff) + np.min(mean_diff)
            ci_max =  bootstrap_ci_z_score * np.sqrt(var_diff) + np.max(mean_diff)
            ci_min_all[mode, region] = ci_min
            ci_max_all[mode, region] = ci_max
            print(f"\tFor region {regions[region]} and mode {modes[mode]}:")
            print(f"\t\tCI: {ci_min} kPa - {ci_max} kPa")
    
    ci_output_dir = "./Results/RawData"
    # Create plots and tables from statistical tests
    plot_difference_ci(
        all_data_stress,
        ci_min_all,
        ci_max_all,
        stretch_ten,
        stretch_com,
        strain_shr,
        regions,
        modes,
        output_dir=ci_output_dir,
    )
    save_anova_table(anova_p_values, ci_output_dir)
    save_confidence_interval_table(ci_min_all, ci_max_all, regions, ci_output_dir)

    output_dir = "./Results/RawData"

    ## Save hysteresis and stiffness tables
    save_hysteresis_table(
        hysteresis_ten,
        hysteresis_com,
        hysteresis_shear,
        hysteresis_ten_samples,
        hysteresis_com_samples,
        hysteresis_shear_samples,
        n_materials,
        output_dir,
    )
    save_stiffness_table(
        stiffness_ten,
        stiffness_com,
        stiffness_shear,
        stiffness_ten_std,
        stiffness_com_std,
        stiffness_shear_std,
        output_dir,
    )

    ## Interpolate stress data to fewer points for table
    stretch_ten_table = np.linspace(np.min(stretch_ten), np.max(stretch_ten), n_pts_table)
    stress_ten_table = np.stack([np.interp(stretch_ten_table, stretch_ten[:, foam_idx], stress_ten[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stress_ten_std_table = np.stack([np.interp(stretch_ten_table, stretch_ten[:, foam_idx], stress_ten_std[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    transverse_stretch_ten_table = np.stack([np.interp(stretch_ten_table, stretch_ten[:, foam_idx], transverse_stretch_ten[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    transverse_stretch_ten_std_table = np.stack([np.interp(stretch_ten_table, stretch_ten[:, foam_idx], transverse_stretch_ten_std[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stretch_com_table = np.linspace(np.min(stretch_com), np.max(stretch_com), n_pts_table)
    stress_com_table = np.stack([np.interp(stretch_com_table, stretch_com[::-1, foam_idx], stress_com[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stress_com_std_table = np.stack([np.interp(stretch_com_table, stretch_com[::-1, foam_idx], stress_com_std[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    transverse_stretch_com_table = np.stack([np.interp(stretch_com_table, stretch_com[::-1, foam_idx], transverse_stretch_com[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    transverse_stretch_com_std_table = np.stack([np.interp(stretch_com_table, stretch_com[::-1, foam_idx], transverse_stretch_com_std[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    strain_shr_table = np.linspace(np.min(strain_shr), np.max(strain_shr), n_pts_table)
    stress_shr_table = np.stack([np.interp(strain_shr_table, strain_shr[:, foam_idx], stress_shr[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    stress_shr_std_table = np.stack([np.interp(strain_shr_table, strain_shr[:, foam_idx], stress_shr_std[:, foam_idx]) for foam_idx in range(n_materials)], axis=1)
    if not worn_shoe:
        stretch_conf_com_table = np.linspace(np.min(stretch_conf_com), np.max(stretch_conf_com), n_pts_table)
        stress_conf_com_table = np.stack([np.interp(stretch_conf_com_table, stretch_conf_com[::-1, foam_idx], stress_conf_com[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)
        stress_conf_com_std_table = np.stack([np.interp(stretch_conf_com_table, stretch_conf_com[::-1, foam_idx], stress_conf_com_std[::-1, foam_idx]) for foam_idx in range(n_materials)], axis=1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create stress plots (axial, transverse, and shear)
    show_error_bars = not worn_shoe
    plot_stress(LoadingMode.TENSION, show_error_bars, stretch_ten, stress_ten, stress_ten_std, stretch_ten_table, stress_ten_table, n_materials, output_dir)
    plot_stress(LoadingMode.COMPRESSION, show_error_bars, stretch_com, stress_com, stress_com_std, stretch_com_table, stress_com_table, n_materials, output_dir)
    plot_transverse_stretch(
        LoadingMode.COMPRESSION, show_error_bars, stretch_com, transverse_stretch_com, transverse_stretch_com_std, n_materials, output_dir
    )
    plot_transverse_stretch(
        LoadingMode.TENSION, show_error_bars, stretch_ten, transverse_stretch_ten, transverse_stretch_ten_std, n_materials, output_dir
    )
    if not worn_shoe:
        plot_stress(
            LoadingMode.CONFINED_COMPRESSION,
            show_error_bars,
            stretch_conf_com,
            stress_conf_com,
            stress_conf_com_std,
            stretch_conf_com_table,
            stress_conf_com_table,
            n_materials,
            output_dir,
        )
    plot_stress(LoadingMode.SHEAR, show_error_bars, strain_shr, stress_shr, stress_shr_std, strain_shr_table, stress_shr_table, n_materials, output_dir)


    ## Compute linearity of shear data
    for foam_idx in range(n_materials):
        strain = strain_shr[:, foam_idx]
        stress = stress_shr[:, foam_idx]
        # Compute r2 using scipy
        r2_shr = pearsonr(strain, stress)
        print(f"R2 of shear data for {foam_types_title[foam_idx]}: {r2_shr[0]}")


    # Create individual sample plots (tension, compression, shear, confined compression)
    for foam_idx in range(n_materials):
        plot_individual_samples(
            foam_idx,
            individual_samples_tension,
            individual_samples_compression,
            individual_samples_shear,
            individual_samples_conf_compression,
            output_dir,
        )

    # Create stress tables
    save_stress_tables(
        n_materials,
        stretch_ten_table,
        stress_ten_table,
        stress_ten_std_table,
        transverse_stretch_ten_table,
        transverse_stretch_ten_std_table,
        stretch_com_table,
        stress_com_table,
        stress_com_std_table,
        transverse_stretch_com_table,
        transverse_stretch_com_std_table,
        strain_shr_table,
        stress_shr_table,
        stress_shr_std_table,
        stiffness_ten,
        stiffness_ten_std,
        stiffness_com,
        stiffness_com_std,
        stiffness_shear,
        stiffness_shear_std,
        hysteresis_ten_samples,
        hysteresis_com_samples,
        hysteresis_shear_samples,
        output_dir,
    )

    save_stress_excel(
        stretch_ten,
        stress_ten,
        stress_ten_std,
        stretch_com,
        stress_com,
        stress_com_std,
        transverse_stretch_ten,
        transverse_stretch_ten_std,
        transverse_stretch_com,
        transverse_stretch_com_std,
        strain_shr,
        stress_shr,
        stress_shr_std,
        excel_filename="WornFoamData.xlsx" if worn_shoe else "FoamData.xlsx",
    )

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
