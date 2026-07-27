"""
process_foam_data.py

Port of the provided MATLAB script to Python using numpy, pandas, matplotlib.
Ensure your data folder structure matches the MATLAB script expectations.
"""

import os
import shutil
import subprocess
from enum import StrEnum
from zipfile import BadZipFile
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks
from scipy.stats import f, pearsonr, t, ttest_ind
from skfda.inference.anova import oneway_anova
from skfda.representation.grid import FDataGrid
from tabulate import tabulate

# ---------- Settings (match MATLAB) ----------
worn_shoe = True
should_process = False

root_folder = "./input/raw_data/asics/worn-shoe/" if worn_shoe else "./input/raw_data/asics/final-tcs/"
transverse_folder = "./input/data_mm/"
out_dir = "./input/"

colors = ["r", "g", "b"] * 2 if worn_shoe else ["r", "b"]
linestyles = (["-"] * 3 + ["--"] * 3) if worn_shoe else ["-", "-"]
foam_types = ["new-toe", "new-mid", "new-heel",  "worn-toe", "worn-mid", "worn-heel"] if worn_shoe else ["leap", "turbo"]
foam_types_title = [x.replace("-", " ").title() for x in foam_types] if worn_shoe else ["FF LEAP\u2122", "FF TURBO\u2122 PLUS"] 

header = 0 if worn_shoe else None

n_pts_table = 13
n_pts_plt = 101

# Strain window for initial stiffness / Poisson's ratio fits
MAX_STRAIN_LINEAR = 0.2

# Global font size for all plot text elements
FONT_SIZE = 20
DIFFERENCE_CI_TICK_LABEL_FONT_SIZE = 20

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
    max_digits_std = max(1, max_digits - 1)
    std_phantom = r"\phantom{0}" * max(0, max_digits_std - std_digits)

    sign_str = "-" if val_sign < 0 else ""
    val_str = rf"{sign_str}{val_phantom}{val_int}.{val_frac:0{decimal_places}d}"

    sign_str = "-" if std_sign < 0 else ""
    std_str = rf"{sign_str}{std_phantom}{std_int}.{std_frac:0{decimal_places}d}"

    return rf"{val_str}\hspace{{0.5em}}$\pm$ {std_str}"


def fit_stiffness(strain, stress, max_strain=MAX_STRAIN_LINEAR, n_pts=101):
    """Piola stress vs engineering strain slope through origin over [0, max_strain]."""
    x_data = np.linspace(0, max_strain, n_pts)
    y_data = np.interp(x_data, strain, stress)
    denom = np.dot(x_data, x_data)
    if denom <= 0:
        return 0.0
    return np.dot(x_data, y_data) / denom


def fit_poissons_ratio(axial_strain, transverse_stretch, loading_mode, max_strain=MAX_STRAIN_LINEAR, n_pts=101):
    """Poisson's ratio from transverse vs axial engineering strain over [0, max_strain]."""
    x_data = np.linspace(0, max_strain, n_pts)
    lam2 = np.interp(x_data, axial_strain, transverse_stretch)
    eps_transverse = lam2 - 1.0
    denom = np.dot(x_data, x_data)
    if denom <= 0:
        return 0.0
    slope = np.dot(x_data, eps_transverse) / denom
    if loading_mode == "ten":
        return -slope
    if loading_mode == "com":
        return slope
    raise ValueError(f"loading_mode must be 'ten' or 'com', got {loading_mode!r}")


def pool_poissons_samples(sample_lists):
    """Concatenate per-sample Poisson's ratios across selected materials."""
    pooled = np.concatenate([np.asarray(sl, dtype=float) for sl in sample_lists])
    return pooled[~np.isnan(pooled)]


def welch_anova(*sample_groups):
    """
    Welch's one-way ANOVA (heteroscedastic), for two or more groups.

    Returns (F_statistic, p_value). Assumes normality but not equal variances.
    """
    groups = []
    for g in sample_groups:
        vals = np.asarray(g, dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        groups.append(vals)
    k = len(groups)
    if k < 2:
        return np.nan, np.nan
    ns = np.array([len(g) for g in groups], dtype=float)
    if np.any(ns < 2):
        return np.nan, np.nan
    means = np.array([g.mean() for g in groups], dtype=float)
    variances = np.array([g.var(ddof=1) for g in groups], dtype=float)
    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        # Degenerate equal-within-group case: fall back to classical comparison of means
        if np.allclose(means, means[0]):
            return 0.0, 1.0
        return np.inf, 0.0
    weights = ns / variances
    weight_sum = weights.sum()
    grand_mean = (weights * means).sum() / weight_sum
    numerator = (weights * (means - grand_mean) ** 2).sum() / (k - 1)
    lambda_term = (((1.0 - weights / weight_sum) ** 2) / (ns - 1)).sum()
    denom = 1.0 + (2.0 * (k - 2) / (k ** 2 - 1.0)) * lambda_term
    f_stat = numerator / denom
    df1 = k - 1
    df2 = (k ** 2 - 1.0) / (3.0 * lambda_term) if lambda_term > 0 else np.inf
    p_value = float(f.sf(f_stat, df1, df2))
    return float(f_stat), p_value


def scalar_oneway_anova(sample_lists, foam_indices=None):
    """Welch one-way ANOVA for scalar per-sample data grouped by material."""
    if foam_indices is None:
        foam_indices = range(len(sample_lists))
    groups = []
    for i in foam_indices:
        vals = np.asarray(sample_lists[i], dtype=float)
        groups.append(vals[~np.isnan(vals)])
    return welch_anova(*groups)


def energy_return_from_hysteresis(hysteresis_samples):
    """Per-sample energy return (%) from hysteresis samples."""
    h = np.asarray(hysteresis_samples, dtype=float)
    return (2.0 - h) / (2.0 + h) * 100.0


def _as_sample_lists(mode_samples, n_materials=6):
    """Normalize mode samples to a list of length n_materials of 1D float arrays."""
    return [np.asarray(mode_samples[i], dtype=float).ravel() for i in range(n_materials)]


def compute_scalar_anova_p_values(mode_sample_lists, n_materials=6):
    """
    Scalar Welch ANOVA p-values matching the stress ANOVA table layout.

    mode_sample_lists: sequence over modes; each entry is length-n_materials sample lists.
    Returns array shaped (8, n_modes).
    """
    # Material indices: new-toe=0, new-mid=1, new-heel=2, worn-toe=3, worn-mid=4, worn-heel=5
    region_materials = [
        [0, 3],  # toe
        [1, 4],  # mid
        [2, 5],  # heel
    ]
    # Row labels: Toe vs Heel, Toe vs Mid, Mid vs Heel
    region_pairs = [(0, 2), (0, 1), (1, 2)]
    # 3-way table order: toe, heel, mid (matches group_data row 4)
    three_way_region_order = [0, 2, 1]
    # New vs worn rows: toe, heel, mid (matches group_data rows 1-3)
    new_vs_worn_region_order = [0, 2, 1]

    n_modes = len(mode_sample_lists)
    p_values = np.full((8, n_modes), np.nan)
    for mode_idx, mode_samples in enumerate(mode_sample_lists):
        samples = _as_sample_lists(mode_samples, n_materials)

        _, p_values[0, mode_idx] = scalar_oneway_anova(samples)

        for row_idx, region_idx in enumerate(new_vs_worn_region_order):
            groups = [samples[i] for i in region_materials[region_idx]]
            _, p_values[1 + row_idx, mode_idx] = scalar_oneway_anova(groups)

        three_way_groups = [
            np.concatenate([samples[i] for i in region_materials[region_idx]])
            for region_idx in three_way_region_order
        ]
        three_way_groups = [g[~np.isnan(g)] for g in three_way_groups]
        _, p_values[4, mode_idx] = welch_anova(*three_way_groups)

        for pair_idx, (r1, r2) in enumerate(region_pairs):
            g1 = np.concatenate([samples[i] for i in region_materials[r1]])
            g2 = np.concatenate([samples[i] for i in region_materials[r2]])
            g1 = g1[~np.isnan(g1)]
            g2 = g2[~np.isnan(g2)]
            _, p_values[5 + pair_idx, mode_idx] = welch_anova(g1, g2)

    return p_values


def compute_energy_return_anova_p_values(
    hysteresis_ten_samples,
    hysteresis_com_samples,
    hysteresis_shear_samples,
    n_materials,
):
    """Scalar Welch ANOVA p-values for energy return (same tests as stress ANOVA table)."""
    mode_sample_lists = [
        [energy_return_from_hysteresis(hysteresis_ten_samples[i]) for i in range(n_materials)],
        [energy_return_from_hysteresis(hysteresis_com_samples[i]) for i in range(n_materials)],
        [energy_return_from_hysteresis(hysteresis_shear_samples[i]) for i in range(n_materials)],
    ]
    return compute_scalar_anova_p_values(mode_sample_lists, n_materials)


def scalar_mean_diff_ttest_ci(group_a, group_b, alpha=0.05):
    """
    Welch (unpaired, heteroscedastic) t CI for mean(group_a) - mean(group_b).

    Assumes normality but not equal variances (Welch–Satterthwaite df).
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    var1 = np.var(a, ddof=1)
    var2 = np.var(b, ddof=1)
    se2 = var1 / n1 + var2 / n2
    if not np.isfinite(se2) or se2 <= 0:
        diff = a.mean() - b.mean()
        if np.isclose(diff, 0.0):
            return 0.0, 0.0
        return np.nan, np.nan
    se = np.sqrt(se2)
    df = se2 ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
    if not np.isfinite(df) or df <= 0:
        return np.nan, np.nan
    diff = a.mean() - b.mean()
    t_crit = float(t.ppf(1.0 - alpha / 2.0, df))
    half = t_crit * se
    return float(diff - half), float(diff + half)


def scalar_mean_vs_zero_ttest_p_ci(vals, alpha=0.05):
    """
    Two-sided one-sample t-test vs 0 with CI for mean(vals).

    Assumes normality; does not involve heteroscedasticity because it's
    within-sample variance only.
    """
    x = np.asarray(vals, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd < 0:
        return np.nan, np.nan, np.nan
    if sd == 0.0:
        # If variance is 0, the mean is known exactly under the model.
        if np.isclose(mean, 0.0):
            return 1.0, 0.0, 0.0
        return 0.0, mean, mean
    se = sd / np.sqrt(n)
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan, np.nan
    df = n - 1
    t_stat = mean / se
    p = float(2.0 * (1.0 - t.cdf(abs(t_stat), df)))
    t_crit = float(t.ppf(1.0 - alpha / 2.0, df))
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    return p, float(lo), float(hi)


def compute_scalar_new_worn_cis(mode_sample_lists, n_materials=6, alpha=0.05):
    """
    Welch t CIs for mean(new) - mean(worn) by mode and region.

    Returns (ci_min, ci_max) each shaped (n_modes, 3) with region order toe, mid, heel.
    """
    region_materials = [
        [0, 3],  # toe
        [1, 4],  # mid
        [2, 5],  # heel
    ]
    n_modes = len(mode_sample_lists)
    ci_min = np.full((n_modes, 3), np.nan)
    ci_max = np.full((n_modes, 3), np.nan)
    for mode_idx, mode_samples in enumerate(mode_sample_lists):
        samples = _as_sample_lists(mode_samples, n_materials)
        for region_idx, (idx_new, idx_worn) in enumerate(region_materials):
            lo, hi = scalar_mean_diff_ttest_ci(
                samples[idx_new], samples[idx_worn], alpha=alpha
            )
            ci_min[mode_idx, region_idx] = lo
            ci_max[mode_idx, region_idx] = hi
    return ci_min, ci_max


def compute_scalar_region_cis(mode_sample_lists, n_materials=6, alpha=0.05):
    """
    Welch t CIs for mean(region_a) - mean(region_b), pooling new and worn.

    Returns (ci_min, ci_max) each shaped (n_modes, 3) with pair order
    toe-mid, toe-heel, mid-heel.
    """
    region_materials = [
        [0, 3],  # toe
        [1, 4],  # mid
        [2, 5],  # heel
    ]
    # Pair order matches region summary table rows
    region_pairs = [(0, 1), (0, 2), (1, 2)]
    n_modes = len(mode_sample_lists)
    ci_min = np.full((n_modes, 3), np.nan)
    ci_max = np.full((n_modes, 3), np.nan)
    for mode_idx, mode_samples in enumerate(mode_sample_lists):
        samples = _as_sample_lists(mode_samples, n_materials)
        region_vals = [
            np.concatenate([samples[i] for i in mats]) for mats in region_materials
        ]
        for pair_idx, (r1, r2) in enumerate(region_pairs):
            lo, hi = scalar_mean_diff_ttest_ci(
                region_vals[r1], region_vals[r2], alpha=alpha
            )
            ci_min[mode_idx, pair_idx] = lo
            ci_max[mode_idx, pair_idx] = hi
    return ci_min, ci_max


def stiffness_samples_from_curves(
    stretch_ten,
    stretch_com,
    strain_shr,
    all_data_stress,
    n_materials=6,
    max_strain=MAX_STRAIN_LINEAR,
):
    """Recompute per-sample stiffness from stored mean-cycle stress curves."""
    stiffness_ten_samples = []
    stiffness_com_samples = []
    stiffness_shear_samples = []
    for foam_idx in range(n_materials):
        region = foam_idx % 3
        new_worn = foam_idx // 3
        strain_ten = stretch_ten[:, foam_idx] - 1.0
        strain_com = 1.0 - stretch_com[:, foam_idx]
        strain_shear = strain_shr[:, foam_idx]
        ten = []
        com = []
        shr = []
        for sample_idx in range(all_data_stress.shape[3]):
            ten.append(
                fit_stiffness(
                    strain_ten,
                    all_data_stress[0, region, new_worn, sample_idx, :],
                    max_strain=max_strain,
                )
            )
            com.append(
                fit_stiffness(
                    strain_com,
                    all_data_stress[1, region, new_worn, sample_idx, :],
                    max_strain=max_strain,
                )
            )
            shr.append(
                fit_stiffness(
                    strain_shear,
                    all_data_stress[2, region, new_worn, sample_idx, :],
                    max_strain=max_strain,
                )
            )
        stiffness_ten_samples.append(np.asarray(ten, dtype=float))
        stiffness_com_samples.append(np.asarray(com, dtype=float))
        stiffness_shear_samples.append(np.asarray(shr, dtype=float))
    return stiffness_ten_samples, stiffness_com_samples, stiffness_shear_samples


def poisson_samples_from_curves(
    stretch_ten,
    stretch_com,
    all_data_transverse,
    n_materials=6,
    max_strain=MAX_STRAIN_LINEAR,
):
    """Recompute per-sample Poisson's ratio from stored mean-cycle transverse curves."""
    poissons_ten_samples = []
    poissons_com_samples = []
    for foam_idx in range(n_materials):
        region = foam_idx % 3
        new_worn = foam_idx // 3
        strain_ten = stretch_ten[:, foam_idx] - 1.0
        strain_com = 1.0 - stretch_com[:, foam_idx]
        ten = []
        com = []
        for sample_idx in range(all_data_transverse.shape[3]):
            ten.append(
                fit_poissons_ratio(
                    strain_ten,
                    all_data_transverse[0, region, new_worn, sample_idx, :],
                    "ten",
                    max_strain=max_strain,
                )
            )
            com.append(
                fit_poissons_ratio(
                    strain_com,
                    all_data_transverse[1, region, new_worn, sample_idx, :],
                    "com",
                    max_strain=max_strain,
                )
            )
        poissons_ten_samples.append(np.asarray(ten, dtype=float))
        poissons_com_samples.append(np.asarray(com, dtype=float))
    return poissons_ten_samples, poissons_com_samples


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
    except (BadZipFile, EOFError, OSError, KeyError, ValueError) as exc:
        if os.path.exists(excel_path):
            os.remove(excel_path)
        print(f"Corrupt workbook detected and removed ({type(exc).__name__}): {excel_path}")
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def oneway_anova_np(first, *rest, grid_points=None, n_reps=100000, return_dist=False,
                    random_state=None, p=2, equal_var=False):
    """
    One-way functional ANOVA for numpy curve samples.

    Each input array should have shape (n_samples, n_points), where each row is
    one functional observation on a common grid. Wraps skfda's oneway_anova by
    converting inputs to FDataGrid objects. Default equal_var=False uses the
    heteroscedastic (Welch-style) functional ANOVA.
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


UNIAXIAL_STRETCH_MIN = 0.4
UNIAXIAL_STRETCH_MAX = 1.3


def uniaxial_stretch_grid():
    return np.linspace(UNIAXIAL_STRETCH_MIN, UNIAXIAL_STRETCH_MAX, n_pts_plt) ## Hard code so stretch and stress always align


def combine_uniaxial_stress_curves(ten_stretch, ten_stress, com_stretch, com_stress):
    """Resample tension and compression onto a single stretch grid from 0.4 to 1.3."""
    com_stretch_inc = com_stretch[::-1]
    com_stress_inc = com_stress[::-1]
    ut_stretch_src = np.concatenate([com_stretch_inc, ten_stretch[1:]])
    ut_stress_src = np.concatenate([com_stress_inc, ten_stress[1:]])
    ut_grid = uniaxial_stretch_grid()
    return np.interp(ut_grid, ut_stretch_src, ut_stress_src)


def build_combined_uniaxial_stress(all_data_stress, stretch_ten, stretch_com):
    """Combine tension/compression samples into shape (3, 2, 5, n_pts_plt)."""
    combined = np.zeros((3, 2, 5, n_pts_plt))
    for region in range(3):
        for new_worn in range(2):
            foam_idx = region + 3 * new_worn
            for sample in range(5):
                combined[region, new_worn, sample, :] = combine_uniaxial_stress_curves(
                    stretch_ten[:, foam_idx],
                    all_data_stress[0, region, new_worn, sample, :],
                    stretch_com[:, foam_idx],
                    -all_data_stress[1, region, new_worn, sample, :],
                )
    return combined


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

    ## Create hysteresis plot
    loading = np.mean(y_interp_all[0::2], axis=0)
    unloading = np.mean(y_interp_all[1::2], axis=0)
    plt.plot(x_interp, loading, label='Loading Stress', color='blue')
    plt.plot(x_interp, unloading, label='Unloading Stress', color='red')
    ## Shade area under loading with crosshatch 
    plt.fill_between(x_interp, loading, loading * 0.0, hatch='\\\\', alpha= 0.1, color = 'blue', edgecolor='black', label="Area under loading")
    ## Shade area under unloading with crosshatch pattern
    plt.fill_between(x_interp, unloading, unloading * 0.0, hatch='//', alpha= 0.1, color = 'red', edgecolor='black', label="Area under unloading")
    plt.legend()
    plt.xlabel('Stretch [-]')
    plt.ylabel('Stress [kPa]')
    plt.savefig('hysteresis.png')
    plt.close()

        

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
    # Per-sample transverse stretch on the same grids as all_data_stress (tension/compression only)
    all_data_transverse = np.full((2, 3, 2, 5, n_pts_plt), np.nan)
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
    poissons_ratio_ten = np.zeros((n_materials))
    poissons_ratio_ten_std = np.zeros((n_materials))
    poissons_ratio_ten_samples = [[] for _ in range(n_materials)]
    poissons_ratio_com = np.zeros((n_materials))
    poissons_ratio_com_std = np.zeros((n_materials))
    poissons_ratio_com_samples = [[] for _ in range(n_materials)]

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
        all_data_transverse[0, foam_idx % 3, foam_idx // 3, :transverse_stretch_all_plt.shape[0], :] = (
            transverse_stretch_all_plt
        )
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

        ### Stiffness and Poisson's ratio per sample
        for sample_idx in range(stress_all_plt.shape[0]):
            stress_curve = stress_all_plt[sample_idx, :]
            transverse_curve = transverse_stretch_all_plt[sample_idx, :]
            stiffness_ten_samples[foam_idx].append(fit_stiffness(x_interp_plt, stress_curve))
            poissons_ratio_ten_samples[foam_idx].append(
                fit_poissons_ratio(x_interp_plt, transverse_curve, "ten")
            )
        stiffness_ten[foam_idx] = np.mean(np.array(stiffness_ten_samples[foam_idx]))
        stiffness_ten_std[foam_idx] = np.std(np.array(stiffness_ten_samples[foam_idx]), ddof=0)
        poissons_ratio_ten[foam_idx] = np.mean(np.array(poissons_ratio_ten_samples[foam_idx]))
        poissons_ratio_ten_std[foam_idx] = np.std(np.array(poissons_ratio_ten_samples[foam_idx]), ddof=0)

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


        stress_all_plt = np.array(stress_all_plt)
        all_data_stress[1, foam_idx % 3, foam_idx // 3, :, :] = stress_all_plt
        transverse_stretch_all_plt = np.array(transverse_stretch_all_plt)  # shape (n_samples, n_pts_plt)
        all_data_transverse[1, foam_idx % 3, foam_idx // 3, :transverse_stretch_all_plt.shape[0], :] = (
            transverse_stretch_all_plt
        )
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

        ### Stiffness and Poisson's ratio per sample
        for sample_idx in range(stress_all_plt.shape[0]):
            stress_curve = stress_all_plt[sample_idx, :]
            transverse_curve = transverse_stretch_all_plt[sample_idx, :]
            stiffness_com_samples[foam_idx].append(fit_stiffness(x_interp_plt, stress_curve))
            poissons_ratio_com_samples[foam_idx].append(
                fit_poissons_ratio(x_interp_plt, transverse_curve, "com")
            )
        stiffness_com[foam_idx] = np.mean(np.array(stiffness_com_samples[foam_idx]))
        stiffness_com_std[foam_idx] = np.std(np.array(stiffness_com_samples[foam_idx]), ddof=0)
        poissons_ratio_com[foam_idx] = np.mean(np.array(poissons_ratio_com_samples[foam_idx]))
        poissons_ratio_com_std[foam_idx] = np.std(
            np.array(poissons_ratio_com_samples[foam_idx]), ddof=0
        )

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
            stiffness_shear_samples[foam_idx].append(
                fit_stiffness(strain_interp_plt, stress_all_plt[sample_idx, :])
            )
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
        all_data_transverse=all_data_transverse,
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
        stiffness_ten_samples=stiffness_ten_samples,
        stiffness_com=stiffness_com,
        stiffness_com_std=stiffness_com_std,
        stiffness_com_samples=stiffness_com_samples,
        stiffness_shear=stiffness_shear,
        stiffness_shear_std=stiffness_shear_std,
        stiffness_shear_samples=stiffness_shear_samples,
        poissons_ratio_ten=poissons_ratio_ten,
        poissons_ratio_ten_std=poissons_ratio_ten_std,
        poissons_ratio_ten_samples=poissons_ratio_ten_samples,
        poissons_ratio_com=poissons_ratio_com,
        poissons_ratio_com_std=poissons_ratio_com_std,
        poissons_ratio_com_samples=poissons_ratio_com_samples,
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
    fig, axes = plt.subplots(3, 3, figsize=(12, 14))
    x_data_by_mode = [stretch_ten, stretch_com, strain_shr]
    y_labels = [
        r"$P_{11,\mathrm{new}} - P_{11,\mathrm{worn}}$ [kPa]",
        r"$|P_{11,\mathrm{new}}| - |P_{11,\mathrm{worn}}|$ [kPa]",
        r"$P_{12,\mathrm{new}} - P_{12,\mathrm{worn}}$ [kPa]",
    ]
    x_labels = ["Stretch [-]", "Stretch [-]", "Shear Strain [-]"]
    row_ylim = []
    for mode in range(3):
        y_min, y_max = np.inf, -np.inf
        x_all = x_data_by_mode[mode]
        for region in range(3):
            stress_data_new = all_data_stress[mode, region, 0, :, :]
            stress_data_worn = all_data_stress[mode, region, 1, :, :]
            mean_diff = np.mean(stress_data_new, axis=0) - np.mean(stress_data_worn, axis=0)
            ci_min = ci_min_all[mode, region]
            ci_max = ci_max_all[mode, region]
            y_min = min(y_min, np.min(mean_diff + ci_min))
            y_max = max(y_max, np.max(mean_diff + ci_max))
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        row_ylim.append((y_min - pad, y_max + pad))

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
            ax.fill_between(x, mean_diff + ci_min, mean_diff + ci_max, alpha=0.2, label="95% SCB")
            ax.set_ylim(row_ylim[mode])
            ax.set_title(f"{modes[mode].capitalize()} - {regions[region].replace("worn-", "").capitalize()}", fontsize=FONT_SIZE)
            ax.grid(True)
            ax.tick_params(labelsize=DIFFERENCE_CI_TICK_LABEL_FONT_SIZE)
            if region == 0:
                ax.set_ylabel(y_labels[mode], fontsize=FONT_SIZE)
            ax.set_xlabel(x_labels[mode], fontsize=FONT_SIZE)
            if mode == 0 and region == 0:
                ax.legend(fontsize=FONT_SIZE, loc='lower left')
            if mode == 1: 
                ax.invert_xaxis()
    fig.suptitle("New - Worn Stress Difference (95% SCB)", fontsize=FONT_SIZE, fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(7, 5) if mode != LoadingMode.SHEAR else (10, 5))
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
    xlabel = "Shear Strain [-]" if mode == LoadingMode.SHEAR else "Stretch [-]"
    ylabel = "Shear Stress [kPa]" if mode == LoadingMode.SHEAR else "Stress [kPa]"
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    if mode == LoadingMode.SHEAR:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=FONT_SIZE)
    if mode == LoadingMode.SHEAR:
        ax.legend(fontsize=FONT_SIZE, bbox_to_anchor=(1.05, 0.5), loc="center left")
    ax.grid(True)
    plt.tight_layout()
    save_figure(fig, output_dir, filename)
    plt.close(fig)


def plot_stress_region_mode_grid(
    stretch_ten,
    stress_ten,
    stress_ten_std,
    stretch_ten_table,
    stress_ten_table,
    stretch_com,
    stress_com,
    stress_com_std,
    stretch_com_table,
    stress_com_table,
    strain_shr,
    stress_shr,
    stress_shr_std,
    strain_shr_table,
    stress_shr_table,
    n_materials,
    output_dir="./Results/RawData",
):
    """
    3x3 stress curves: rows heel/mid/toe, columns tension/compression/shear.
    New = dark blue mean + light blue ±std band; worn = dark red + light red.
    Table points (n_pts_table) are white circles with black edges.
    """
    if n_materials != 6:
        print("Skipping stress region/mode grid (requires 6 worn-shoe materials).")
        return

    # foam_types: new-toe, new-mid, new-heel, worn-toe, worn-mid, worn-heel
    region_specs = [
        ("Heel", 2, 5),
        ("Mid", 1, 4),
        ("Toe", 0, 3),
    ]
    mode_data = [
        (
            "Tension",
            stretch_ten,
            stress_ten,
            stress_ten_std,
            stretch_ten_table,
            stress_ten_table,
            False,
        ),
        (
            "Compression",
            stretch_com,
            stress_com,
            stress_com_std,
            stretch_com_table,
            stress_com_table,
            True,
        ),
        (
            "Shear",
            strain_shr,
            stress_shr,
            stress_shr_std,
            strain_shr_table,
            stress_shr_table,
            False,
        ),
    ]
    color_new = "#0000ff"
    color_worn = "#ff0000"
    shade_new = "#0000ff"
    shade_worn = "#ff0000"
    marker_size = 11  # ~2.75× the markersize=4 used in plot_stress

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0), sharex="col", sharey="col")
    for row, (region_name, idx_new, idx_worn) in enumerate(region_specs):
        for col, (mode_name, x, stress, stress_std, x_table, stress_table, invert) in enumerate(mode_data):
            ax = axes[row, col]
            for foam_idx, line_color, shade_color, label in (
                (idx_new, color_new, shade_new, "New"),
                (idx_worn, color_worn, shade_worn, "Worn"),
            ):
                x_curve = x[:, foam_idx]
                y_curve = stress[:, foam_idx]
                y_std = stress_std[:, foam_idx]
                ax.fill_between(
                    x_curve,
                    y_curve - y_std,
                    y_curve + y_std,
                    color=shade_color,
                    alpha=0.3,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    x_curve,
                    y_curve,
                    color=line_color,
                    linewidth=2.0,
                    label=label if row == 0 and col == 0 else None,
                    zorder=2,
                )
                x_pts = x_table if np.ndim(x_table) == 1 else x_table[:, foam_idx]
                ax.plot(
                    x_pts,
                    stress_table[:, foam_idx],
                    linestyle="none",
                    marker="o",
                    markersize=marker_size,
                    markerfacecolor=line_color,
                    markeredgecolor=line_color,
                    markeredgewidth=0.0,
                    zorder=3,
                )
            if invert:
                ax.invert_xaxis()
                ax.invert_yaxis()
            if row == 0:
                ax.set_title(mode_name, fontsize=FONT_SIZE)
            if col == 0:
                ax.set_ylabel(f"{region_name}\nStress [kPa]", fontsize=FONT_SIZE - 4)
            if row == 2:
                xlabel = "Shear Strain [-]" if mode_name == "Shear" else "Stretch [-]"
                ax.set_xlabel(xlabel, fontsize=FONT_SIZE - 4)
            if mode_name == "Shear":
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=FONT_SIZE - 6)
            ax.grid(True, alpha=0.35)

    legend_handles = [
        plt.Line2D([0], [0], color=color_new, linewidth=2.0, label="New mean"),
        plt.Line2D([0], [0], color=color_worn, linewidth=2.0, label="Worn mean"),
        plt.Rectangle((0, 0), 1, 1, facecolor=shade_new, edgecolor="none", alpha=0.3, label=r"New $\pm$ std"),
        plt.Rectangle((0, 0), 1, 1, facecolor=shade_worn, edgecolor="none", alpha=0.3, label=r"Worn $\pm$ std"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=FONT_SIZE - 4,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    plt.tight_layout()
    save_figure(fig, output_dir, "StressRegionModeGrid.pdf")
    plt.close(fig)
    print(f"Stress region/mode grid saved to: {os.path.join(output_dir, 'StressRegionModeGrid.pdf')}")


def plot_transverse_stretch_region_mode_grid(
    stretch_ten,
    transverse_stretch_ten,
    transverse_stretch_ten_std,
    stretch_ten_table,
    transverse_stretch_ten_table,
    stretch_com,
    transverse_stretch_com,
    transverse_stretch_com_std,
    stretch_com_table,
    transverse_stretch_com_table,
    n_materials,
    output_dir="./Results/RawData",
):
    """
    3x2 transverse-stretch curves: rows heel/mid/toe, columns tension/compression.
    Same new/worn styling as the stress region/mode grid.
    """
    if n_materials != 6:
        print("Skipping transverse stretch region/mode grid (requires 6 worn-shoe materials).")
        return

    region_specs = [
        ("Heel", 2, 5),
        ("Mid", 1, 4),
        ("Toe", 0, 3),
    ]
    # (title, axial x, transverse y, y std, x_table, y_table, invert_x, invert_y)
    mode_data = [
        (
            "Tension",
            stretch_ten,
            transverse_stretch_ten,
            transverse_stretch_ten_std,
            stretch_ten_table,
            transverse_stretch_ten_table,
            False,
            True,
        ),
        (
            "Compression",
            stretch_com,
            transverse_stretch_com,
            transverse_stretch_com_std,
            stretch_com_table,
            transverse_stretch_com_table,
            True,
            False,
        ),
    ]
    color_new = "#0000ff"
    color_worn = "#ff0000"
    shade_new = "#0000ff"
    shade_worn = "#ff0000"
    marker_size = 11

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.0), sharex="col", sharey="col")
    for row, (region_name, idx_new, idx_worn) in enumerate(region_specs):
        for col, (
            mode_name,
            x,
            y,
            y_std,
            x_table,
            y_table,
            invert_x,
            invert_y,
        ) in enumerate(mode_data):
            ax = axes[row, col]
            for foam_idx, line_color, shade_color in (
                (idx_new, color_new, shade_new),
                (idx_worn, color_worn, shade_worn),
            ):
                x_curve = x[:, foam_idx]
                y_curve = y[:, foam_idx]
                y_err = y_std[:, foam_idx]
                ax.fill_between(
                    x_curve,
                    y_curve - y_err,
                    y_curve + y_err,
                    color=shade_color,
                    alpha=0.3,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    x_curve,
                    y_curve,
                    color=line_color,
                    linewidth=2.0,
                    zorder=2,
                )
                x_pts = x_table if np.ndim(x_table) == 1 else x_table[:, foam_idx]
                ax.plot(
                    x_pts,
                    y_table[:, foam_idx],
                    linestyle="none",
                    marker="o",
                    markersize=marker_size,
                    markerfacecolor=line_color,
                    markeredgecolor=line_color,
                    markeredgewidth=0.0,
                    zorder=3,
                )
            if invert_x:
                ax.invert_xaxis()
            if invert_y:
                ax.invert_yaxis()
            if row == 0:
                ax.set_title(mode_name, fontsize=FONT_SIZE)
            if col == 0:
                ax.set_ylabel(f"{region_name}\nTransverse Stretch [-]", fontsize=FONT_SIZE - 4)
            if row == 2:
                ax.set_xlabel("Axial Stretch [-]", fontsize=FONT_SIZE - 4)
            ax.tick_params(labelsize=FONT_SIZE - 6)
            ax.grid(True, alpha=0.35)

    legend_handles = [
        plt.Line2D([0], [0], color=color_new, linewidth=2.0, label="New mean"),
        plt.Line2D([0], [0], color=color_worn, linewidth=2.0, label="Worn mean"),
        plt.Rectangle((0, 0), 1, 1, facecolor=shade_new, edgecolor="none", alpha=0.3, label=r"New $\pm$ std"),
        plt.Rectangle((0, 0), 1, 1, facecolor=shade_worn, edgecolor="none", alpha=0.3, label=r"Worn $\pm$ std"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=FONT_SIZE - 4,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    plt.tight_layout()
    save_figure(fig, output_dir, "TransverseStretchRegionModeGrid.pdf")
    plt.close(fig)
    print(
        f"Transverse stretch region/mode grid saved to: "
        f"{os.path.join(output_dir, 'TransverseStretchRegionModeGrid.pdf')}"
    )


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

    fig, ax = plt.subplots(figsize=(7.5, 5) if mode == LoadingMode.TENSION else (10, 5))
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
    if mode == LoadingMode.COMPRESSION:
        ax.legend(fontsize=FONT_SIZE, bbox_to_anchor=(1.05, 0.5), loc="center left")
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


def render_latex_table_to_png(table_stem, output_dir="./Results/RawData", dpi=200):
    """
    Compile a tabular-only .tex fragment to PDF and PNG (standalone + pdflatex).

    Expects ``{table_stem}.tex`` in ``output_dir``. Writes ``{table_stem}.pdf``
    and ``{table_stem}.png``.
    """
    os.makedirs(output_dir, exist_ok=True)
    fragment_path = os.path.join(output_dir, f"{table_stem}.tex")
    if not os.path.isfile(fragment_path):
        print(f"Skipping table render; missing {fragment_path}")
        return

    render_stem = f"{table_stem}_render"
    render_tex_path = os.path.join(output_dir, f"{render_stem}.tex")
    render_doc = (
        r"\documentclass[border=8pt]{standalone}"
        "\n"
        r"\usepackage[table]{xcolor}"
        "\n"
        r"\usepackage{makecell}"
        "\n"
        r"\usepackage{booktabs}"
        "\n"
        r"\renewcommand{\arraystretch}{1.25}"
        "\n"
        r"\begin{document}"
        "\n"
        rf"\input{{{table_stem}.tex}}"
        "\n"
        r"\end{document}"
        "\n"
    )
    with open(render_tex_path, "w") as f:
        f.write(render_doc)

    pdflatex_cmd = shutil.which("pdflatex") or "/Library/TeX/texbin/pdflatex"

    result = subprocess.run(
        [pdflatex_cmd, "-interaction=nonstopmode", f"{render_stem}.tex"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    render_pdf_path = os.path.join(output_dir, f"{render_stem}.pdf")
    if result.returncode != 0 or not os.path.isfile(render_pdf_path):
        print(f"pdflatex failed for {table_stem} (exit {result.returncode})")
        return

    pdf_path = os.path.join(output_dir, f"{table_stem}.pdf")
    shutil.copyfile(render_pdf_path, pdf_path)

    pdftoppm_cmd = shutil.which("pdftoppm")
    if pdftoppm_cmd is None:
        print(f"pdftoppm not found; PDF only at {pdf_path}")
        return

    png_prefix = os.path.join(output_dir, table_stem)
    ppm_result = subprocess.run(
        [pdftoppm_cmd, "-png", "-r", str(dpi), render_pdf_path, png_prefix],
        capture_output=True,
        text=True,
        check=False,
    )
    if ppm_result.returncode != 0:
        print(f"pdftoppm failed for {table_stem} (exit {ppm_result.returncode})")
        return

    png_candidates = [
        os.path.join(output_dir, f"{table_stem}-1.png"),
        os.path.join(output_dir, f"{table_stem}-01.png"),
        os.path.join(output_dir, f"{table_stem}.png"),
    ]
    png_path = os.path.join(output_dir, f"{table_stem}.png")
    for candidate in png_candidates:
        if os.path.isfile(candidate) and candidate != png_path:
            shutil.move(candidate, png_path)
            break

    if os.path.isfile(png_path):
        print(f"Table render saved to: {pdf_path} and {png_path}")
    else:
        print(f"Table PDF saved to: {pdf_path} (PNG conversion did not produce output)")


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
        r"\makecell{$p$ Value \\ Uniaxial}",
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
    table = _latex_tabular_from_tabulate(data, headers, r"|l||c|c|c|c|c|c||l|l|")
    os.makedirs(output_dir, exist_ok=True)
    anova_table_path = os.path.join(output_dir, "anova_table.tex")
    with open(anova_table_path, "w") as f:
        f.write(table)
    print(f"ANOVA table saved to: {anova_table_path}")


def save_scalar_anova_table(anova_p_values, mode_names, filename, output_dir="./Results/RawData"):
    """Write ANOVA p-value table for scalar quantities (same layout as anova_table.tex)."""
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
    ] + [rf"\makecell{{$p$ Value \\ {name}}}" for name in mode_names]
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
    p_values_formatted = np.vectorize(fmt_p_value)(anova_p_values)
    data = np.concatenate([test_names, group_data_formatted, p_values_formatted], axis=1)
    p_cols = "l|" * len(mode_names)
    colspec = rf"|l||c|c|c|c|c|c||{p_cols}"
    table = _latex_tabular_from_tabulate(data, headers, colspec)
    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, filename)
    with open(table_path, "w") as f:
        f.write(table)
    print(f"ANOVA table saved to: {table_path}")


def save_energy_return_anova_table(energy_return_anova_p_values, output_dir="./Results/RawData"):
    save_scalar_anova_table(
        energy_return_anova_p_values,
        ["Tension", "Compression", "Shear"],
        "energy_return_anova_table.tex",
        output_dir=output_dir,
    )


def fmt_ci_interval(lo, hi):
    """Format a CI as $[lo, hi]$ for LaTeX tables."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    lo_s = fmt_ci_value(lo).strip("$")
    hi_s = fmt_ci_value(hi).strip("$")
    return rf"$[{lo_s},\,{hi_s}]$"


def save_scalar_summary_table(
    stiffness_anova,
    energy_anova,
    poisson_anova,
    stiffness_ci_min,
    stiffness_ci_max,
    energy_ci_min,
    energy_ci_max,
    poisson_ci_min,
    poisson_ci_max,
    output_dir="./Results/RawData",
):
    """
    Combined scalar summary: 6-way and new-vs-worn p-values/CIs.

    Columns: stiffness (ten/com/shr), energy return (ten/com/shr), Poisson (ten/com).
    Poisson has no shear mode; the last column is compression.
    """
    headers = [
        r"\makecell{Test}",
        r"\makecell{Stiffness \\ Tension}",
        r"\makecell{Stiffness \\ Compression}",
        r"\makecell{Stiffness \\ Shear}",
        r"\makecell{Energy return \\ Tension}",
        r"\makecell{Energy return \\ Compression}",
        r"\makecell{Energy return \\ Shear}",
        r"\makecell{Poisson ratio \\ Tension}",
        r"\makecell{Poisson ratio \\ Compression}",
    ]
    # ANOVA rows: 0=6-way, 1=toe, 2=heel, 3=mid
    # CI region order: 0=toe, 1=mid, 2=heel
    row_specs = [
        ("6 way ANOVA $p$ value", "p", 0, None),
        ("Worn vs new toe $p$ value", "p", 1, None),
        ("Worn vs new toe CI", "ci", None, 0),
        ("Worn vs new mid $p$ value", "p", 3, None),
        ("Worn vs new mid CI", "ci", None, 1),
        ("Worn vs new heel $p$ value", "p", 2, None),
        ("Worn vs new heel CI", "ci", None, 2),
    ]

    def p_cell(anova, mode_idx, row_idx):
        if mode_idx >= anova.shape[1]:
            return ""
        return fmt_p_value(anova[row_idx, mode_idx])

    def ci_cell(ci_min, ci_max, mode_idx, region_idx):
        if mode_idx >= ci_min.shape[0]:
            return ""
        return fmt_ci_interval(ci_min[mode_idx, region_idx], ci_max[mode_idx, region_idx])

    rows = []
    for label, kind, anova_row, region_idx in row_specs:
        if kind == "p":
            cells = [
                p_cell(stiffness_anova, 0, anova_row),
                p_cell(stiffness_anova, 1, anova_row),
                p_cell(stiffness_anova, 2, anova_row),
                p_cell(energy_anova, 0, anova_row),
                p_cell(energy_anova, 1, anova_row),
                p_cell(energy_anova, 2, anova_row),
                p_cell(poisson_anova, 0, anova_row),
                p_cell(poisson_anova, 1, anova_row),
            ]
        else:
            cells = [
                ci_cell(stiffness_ci_min, stiffness_ci_max, 0, region_idx),
                ci_cell(stiffness_ci_min, stiffness_ci_max, 1, region_idx),
                ci_cell(stiffness_ci_min, stiffness_ci_max, 2, region_idx),
                ci_cell(energy_ci_min, energy_ci_max, 0, region_idx),
                ci_cell(energy_ci_min, energy_ci_max, 1, region_idx),
                ci_cell(energy_ci_min, energy_ci_max, 2, region_idx),
                ci_cell(poisson_ci_min, poisson_ci_max, 0, region_idx),
                ci_cell(poisson_ci_min, poisson_ci_max, 1, region_idx),
            ]
        rows.append([label, *cells])

    table = _latex_tabular_from_tabulate(rows, headers, r"|l||c|c|c||c|c|c||c|c|")
    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, "scalar_summary_table.tex")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Scalar summary table saved to: {table_path}")
    render_latex_table_to_png("scalar_summary_table", output_dir=output_dir)


def save_scalar_region_summary_table(
    stiffness_anova,
    energy_anova,
    poisson_anova,
    stiffness_ci_min,
    stiffness_ci_max,
    energy_ci_min,
    energy_ci_max,
    poisson_ci_min,
    poisson_ci_max,
    output_dir="./Results/RawData",
):
    """
    Region comparison scalar summary: pairwise p-values/CIs with new and worn pooled.

    Columns match save_scalar_summary_table. CIs are mean(region_a) - mean(region_b).
    """
    headers = [
        r"\makecell{Test}",
        r"\makecell{Stiffness \\ Tension}",
        r"\makecell{Stiffness \\ Compression}",
        r"\makecell{Stiffness \\ Shear}",
        r"\makecell{Energy return \\ Tension}",
        r"\makecell{Energy return \\ Compression}",
        r"\makecell{Energy return \\ Shear}",
        r"\makecell{Poisson ratio \\ Tension}",
        r"\makecell{Poisson ratio \\ Compression}",
    ]
    # ANOVA rows: 4=3-way, 5=toe vs heel, 6=toe vs mid, 7=mid vs heel
    # CI pair order: 0=toe-mid, 1=toe-heel, 2=mid-heel
    row_specs = [
        ("3 way ANOVA $p$ value", "p", 4, None),
        ("Toe vs mid $p$ value", "p", 6, None),
        ("Toe vs mid CI", "ci", None, 0),
        ("Toe vs heel $p$ value", "p", 5, None),
        ("Toe vs heel CI", "ci", None, 1),
        ("Mid vs heel $p$ value", "p", 7, None),
        ("Mid vs heel CI", "ci", None, 2),
    ]

    def p_cell(anova, mode_idx, row_idx):
        if mode_idx >= anova.shape[1]:
            return ""
        return fmt_p_value(anova[row_idx, mode_idx])

    def ci_cell(ci_min, ci_max, mode_idx, pair_idx):
        if mode_idx >= ci_min.shape[0]:
            return ""
        return fmt_ci_interval(ci_min[mode_idx, pair_idx], ci_max[mode_idx, pair_idx])

    rows = []
    for label, kind, anova_row, pair_idx in row_specs:
        if kind == "p":
            cells = [
                p_cell(stiffness_anova, 0, anova_row),
                p_cell(stiffness_anova, 1, anova_row),
                p_cell(stiffness_anova, 2, anova_row),
                p_cell(energy_anova, 0, anova_row),
                p_cell(energy_anova, 1, anova_row),
                p_cell(energy_anova, 2, anova_row),
                p_cell(poisson_anova, 0, anova_row),
                p_cell(poisson_anova, 1, anova_row),
            ]
        else:
            cells = [
                ci_cell(stiffness_ci_min, stiffness_ci_max, 0, pair_idx),
                ci_cell(stiffness_ci_min, stiffness_ci_max, 1, pair_idx),
                ci_cell(stiffness_ci_min, stiffness_ci_max, 2, pair_idx),
                ci_cell(energy_ci_min, energy_ci_max, 0, pair_idx),
                ci_cell(energy_ci_min, energy_ci_max, 1, pair_idx),
                ci_cell(energy_ci_min, energy_ci_max, 2, pair_idx),
                ci_cell(poisson_ci_min, poisson_ci_max, 0, pair_idx),
                ci_cell(poisson_ci_min, poisson_ci_max, 1, pair_idx),
            ]
        rows.append([label, *cells])

    table = _latex_tabular_from_tabulate(rows, headers, r"|l||c|c|c||c|c|c||c|c|")
    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, "scalar_region_summary_table.tex")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Scalar region summary table saved to: {table_path}")
    render_latex_table_to_png("scalar_region_summary_table", output_dir=output_dir)


def format_ci_region_name(region_name):
    """Strip worn- prefix and capitalize for confidence interval table rows."""
    return region_name.removeprefix("worn-").capitalize()


def _pool_region_mode_samples(mode_samples_by_material, region_idx, n_materials=6):
    """Pool new+worn samples for a given region (toe/mid/heel)."""
    region_materials = [
        [0, 3],  # toe
        [1, 4],  # mid
        [2, 5],  # heel
    ]
    mats = region_materials[region_idx]
    pooled = np.concatenate(
        [np.asarray(mode_samples_by_material[i], dtype=float).ravel() for i in mats]
    )
    return pooled[~np.isnan(pooled)]


def save_scalar_mode_comparison_table(
    stiffness_mode_samples,
    energy_return_mode_samples,
    poisson_mode_samples,
    regions,
    output_dir="./Results/RawData",
):
    """
    Pool new+worn per region and compare modes with Welch t-tests.

    Table columns (9 comparisons total):
      1-3: Stiffness pairwise differences within region
      4-6: Energy-return pairwise differences within region
      7  : Poisson tension - compression
      8  : Poisson tension vs 0 (one-sample)
      9  : Poisson compression vs 0 (one-sample)
    Rows:
      Toe p value / Toe CI, Mid p value / Mid CI, Heel p value / Heel CI.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pooled new+worn per region uses fixed material indices for this project layout.
    n_regions = len(regions)
    if n_regions != 3:
        print(f"Warning: expected 3 regions (toe/mid/heel), got {n_regions}")

    # Two-sample Welch comparisons: (A_vals, B_vals) -> mean(A) - mean(B)
    def welch_diff_p_ci(a_vals, b_vals):
        a = np.asarray(a_vals, dtype=float).ravel()
        b = np.asarray(b_vals, dtype=float).ravel()
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            return "", ""
        p_val = float(ttest_ind(a, b, equal_var=False).pvalue)
        lo, hi = scalar_mean_diff_ttest_ci(a, b, alpha=0.05)
        return fmt_p_value(p_val), fmt_ci_interval(lo, hi)

    def one_sample_vs_zero_p_ci(x_vals):
        p_val, lo, hi = scalar_mean_vs_zero_ttest_p_ci(x_vals, alpha=0.05)
        return fmt_p_value(p_val), fmt_ci_interval(lo, hi)

    # Column specs: (column_header, kind, getter)
    # kind="diff": two-sample difference A-B
    # kind="one": one-sample mean vs 0
    col_specs = [
        (r"Stiffness Ten - Comp", "diff", ("stiff", 0, 1)),
        (r"Stiffness Ten - Shear", "diff", ("stiff", 0, 2)),
        (r"Stiffness Comp - Shear", "diff", ("stiff", 1, 2)),
        (r"Energy Ten - Comp", "diff", ("energy", 0, 1)),
        (r"Energy Ten - Shear", "diff", ("energy", 0, 2)),
        (r"Energy Comp - Shear", "diff", ("energy", 1, 2)),
        (r"Poisson Ten - Comp", "diff", ("poisson", 0, 1)),
        (r"Poisson Ten vs 0", "one", ("poisson", 0, None)),
        (r"Poisson Comp vs 0", "one", ("poisson", 1, None)),
    ]

    headers = [r"\makecell{Region}"] + [rf"\makecell{{{h}}}" for h, _, _ in col_specs]

    rows = []
    for region_idx in range(3):
        region_name = format_ci_region_name(regions[region_idx])
        pooled_stiff = [
            _pool_region_mode_samples(stiffness_mode_samples[mode_idx], region_idx)
            for mode_idx in range(3)
        ]
        pooled_energy = [
            _pool_region_mode_samples(energy_return_mode_samples[mode_idx], region_idx)
            for mode_idx in range(3)
        ]
        pooled_poisson = [
            _pool_region_mode_samples(poisson_mode_samples[mode_idx], region_idx)
            for mode_idx in range(2)
        ]

        # Build p-value cells
        p_cells = []
        ci_cells = []
        for _, kind, spec in col_specs:
            src = spec[0]
            if kind == "diff":
                _, a_i, b_i = spec
                if src == "stiff":
                    a_vals = pooled_stiff[a_i]
                    b_vals = pooled_stiff[b_i]
                elif src == "energy":
                    a_vals = pooled_energy[a_i]
                    b_vals = pooled_energy[b_i]
                elif src == "poisson":
                    a_vals = pooled_poisson[a_i]
                    b_vals = pooled_poisson[b_i]
                else:
                    a_vals, b_vals = [], []
                p_s, ci_s = welch_diff_p_ci(a_vals, b_vals)
                p_cells.append(p_s)
                ci_cells.append(ci_s)
            else:
                _, x_i, _ = spec
                if src == "poisson":
                    x_vals = pooled_poisson[x_i]
                else:
                    x_vals = []
                p_s, ci_s = one_sample_vs_zero_p_ci(x_vals)
                p_cells.append(p_s)
                ci_cells.append(ci_s)

        rows.append([rf"{region_name} p value", *p_cells])
        rows.append([rf"{region_name} CI", *ci_cells])

    colspec = rf"|l|{'|'.join(['c'] * len(col_specs))}|"
    table = _latex_tabular_from_tabulate(
        rows,
        headers,
        colspec,
    )
    table_path = os.path.join(output_dir, "scalar_mode_comparison_table.tex")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Scalar mode comparison table saved to: {table_path}")
    render_latex_table_to_png("scalar_mode_comparison_table", output_dir=output_dir)


def save_confidence_interval_table(ci_min_all, ci_max_all, regions, output_dir="./Results/RawData"):
    ci_headers = [
        r"\makecell{Region}",
        r"\makecell{Tension \\ Lower \\ {[}kPa]}",
        r"\makecell{Tension \\ Upper \\ {[}kPa]}",
        r"\makecell{Compression \\ Lower \\ {[}kPa]}",
        r"\makecell{Compression \\ Upper \\ {[}kPa]}",
        r"\makecell{Shear \\ Lower \\ {[}kPa]}",
        r"\makecell{Shear \\ Upper \\ {[}kPa]}",
    ]
    ci_rows = []
    for region in range(3):
        row = [format_ci_region_name(regions[region])]
        for mode in range(3):
            row.append(fmt_ci_value(ci_min_all[mode, region]))
            row.append(fmt_ci_value(ci_max_all[mode, region]))
        ci_rows.append(row)
    ci_table = _latex_tabular_from_tabulate(ci_rows, ci_headers, r"|l||c|c||c|c||c|c|")
    os.makedirs(output_dir, exist_ok=True)
    ci_table_path = os.path.join(output_dir, "confidence_interval_table.tex")
    with open(ci_table_path, "w") as f:
        f.write(ci_table)
    print(f"Confidence interval table saved to: {ci_table_path}")


def save_scalar_ci_table(
    ci_min_all,
    ci_max_all,
    regions,
    mode_names,
    filename,
    unit_label="",
    output_dir="./Results/RawData",
):
    """CI table for mean(new)-mean(worn) of a scalar quantity by region and mode."""
    unit_tex = rf" \\ {{{unit_label}}}" if unit_label else ""
    ci_headers = [r"\makecell{Region}"]
    for name in mode_names:
        ci_headers.append(rf"\makecell{{{name} \\ Lower{unit_tex}}}")
        ci_headers.append(rf"\makecell{{{name} \\ Upper{unit_tex}}}")
    ci_rows = []
    for region in range(len(regions)):
        row = [format_ci_region_name(regions[region])]
        for mode in range(len(mode_names)):
            row.append(fmt_ci_value(ci_min_all[mode, region]))
            row.append(fmt_ci_value(ci_max_all[mode, region]))
        ci_rows.append(row)
    mode_colspec = "||".join(["c|c"] * len(mode_names))
    colspec = rf"|l|{mode_colspec}|"
    ci_table = _latex_tabular_from_tabulate(ci_rows, ci_headers, colspec)
    os.makedirs(output_dir, exist_ok=True)
    ci_table_path = os.path.join(output_dir, filename)
    with open(ci_table_path, "w") as f:
        f.write(ci_table)
    print(f"Scalar CI table saved to: {ci_table_path}")


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
    poissons_ratio_ten,
    poissons_ratio_ten_std,
    poissons_ratio_com,
    poissons_ratio_com_std,
    hysteresis_ten_samples,
    hysteresis_com_samples,
    hysteresis_shear_samples,
):
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
    nu_ten = poissons_ratio_ten[mat]
    nu_ten_std = poissons_ratio_ten_std[mat]
    nu_com = poissons_ratio_com[mat]
    nu_com_std = poissons_ratio_com_std[mat]
    energy_return_ten = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_ten_samples[mat]])) * 100.0
    energy_return_ten_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_ten_samples[mat]]), ddof=0) * 100.0
    energy_return_com = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_com_samples[mat]])) * 100.0
    energy_return_com_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_com_samples[mat]]), ddof=0) * 100.0
    energy_return_shr = np.mean(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_shear_samples[mat]])) * 100.0
    energy_return_shr_std = np.std(np.array([(2.0 - h) / (2.0 + h) for h in hysteresis_shear_samples[mat]]), ddof=0) * 100.0

    lines = []
    lines.append(r"\begin{tabular}{|ccc||ccc||cc|}")
    lines.append(r"\hline")
    lines.append(r"  \multicolumn{3}{|c||}{\sffamily{\bfseries{uniaxial tension}}}")
    lines.append(r"& \multicolumn{3}{c||} {\sffamily{\bfseries{uniaxial compression}}}")
    lines.append(r"& \multicolumn{2}{c|}  {\sffamily{\bfseries{simple shear}}} \\")
    lines.append(r"  \multicolumn{3}{|c||}{$n=5$}")
    lines.append(r"& \multicolumn{3}{c||}{$n=5$}")
    lines.append(r"& \multicolumn{2}{c|}{$n=5$} \\ \hline")
    lines.append(r"$\lambda_1$ & $P_{11}$ & $\lambda_2$ & $\lambda_1$ & $P_{11}$ & $\lambda_2$ & $\gamma$ & $P_{12}$  \\")
    lines.append(r"\,[-] & [kPa] & [-] & [-] & [kPa] & [-] & [-] & [kPa]  \\")
    lines.append(r"\hline \hline")

    for i in range(n_pts_table):
        ten_str = format_with_phantoms(ten_stress[i], ten_std[i], max_digits=3)
        ten_trans_str = format_with_phantoms(ten_trans[i], ten_trans_std[i], decimal_places=3, max_digits=1)
        com_str = format_with_phantoms(com_stress[i], com_std[i], max_digits=3)
        com_trans_str = format_with_phantoms(com_trans[i], com_trans_std[i], decimal_places=3, max_digits=1)
        shr_str = format_with_phantoms(shr_stress[i], shr_std[i], max_digits=2)
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
    lines.append(rf"& \multicolumn{{2}}{{c|}}  {{$\textsf{{G}} = {G_shr:.2f} \pm {G_shr_std:.2f}$\,kPa}} \\")
    lines.append(r"\hline \hline")
    lines.append(r"  \multicolumn{3}{|c||}{\sffamily{\bfseries{tensile Poisson's ratio}}}")
    lines.append(r"& \multicolumn{3}{c||} {\sffamily{\bfseries{compressive Poisson's ratio}}}")
    lines.append(r"& \multicolumn{2}{c|}{} \\")
    lines.append(rf"  \multicolumn{{3}}{{|c||}}{{$\nu_{{\rm{{ten}}}} = {nu_ten:.3f} \pm {nu_ten_std:.3f}$}}")
    lines.append(rf"& \multicolumn{{3}}{{c||}} {{$\nu_{{\rm{{com}}}} = {nu_com:.3f} \pm {nu_com_std:.3f}$}}")
    lines.append(r"& \multicolumn{2}{c|}{} \\")
    lines.append(r"\hline \hline")
    lines.append(r"  \multicolumn{3}{|c||}{\sffamily{\bfseries{energy return}}}")
    lines.append(r"& \multicolumn{3}{c||} {\sffamily{\bfseries{energy return}}}")
    lines.append(r"& \multicolumn{2}{c|}  {\sffamily{\bfseries{energy return}}} \\")
    lines.append(rf"  \multicolumn{{3}}{{|c||}}{{$\eta_{{\rm{{ten}}}}  = {energy_return_ten:.1f} \pm {energy_return_ten_std:.1f} \%$}}")
    lines.append(rf"& \multicolumn{{3}}{{c||}} {{$\eta_{{\rm{{com}}}}  = {energy_return_com:.1f} \pm {energy_return_com_std:.1f}\%$}}")
    lines.append(rf"& \multicolumn{{2}}{{c|}}  {{$\eta_{{\rm{{shr}}}}  = {energy_return_shr:.1f} \pm {energy_return_shr_std:.1f} \%$}} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _hsl_to_rgb(h_deg, s_pct, l_pct):
    """Convert HSL (hue deg, sat %, light %) to an RGB tuple for matplotlib."""
    return colorsys.hls_to_rgb(h_deg / 360.0, l_pct / 100.0, s_pct / 100.0)


def _bar_series_mean_err(samples_by_material, material_indices, error="std"):
    """Mean and ±std / ±SEM for one quantity across selected materials."""
    means = []
    errs = []
    for mat in material_indices:
        vals = np.asarray(samples_by_material[mat], dtype=float)
        vals = vals[~np.isnan(vals)]
        means.append(float(np.mean(vals)))
        if len(vals) <= 1:
            errs.append(0.0)
            continue
        std = float(np.std(vals, ddof=0))
        errs.append(std / np.sqrt(len(vals)) if error == "sem" else std)
    return np.asarray(means), np.asarray(errs)


def plot_stiffness_energy_return_bars(
    n_materials,
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
    error="std",
):
    """
    2x3 bar figure: stiffness (row 0) and energy return (row 1)
    vs tension | compression | shear, with 6 bars ordered as
    New/Worn Heel, Mid, Toe (blue / red shade by region).
    """
    if n_materials != 6:
        print("Skipping stiffness/energy-return bar plot (requires 6 worn-shoe materials).")
        return

    # heel / mid / toe × new (blue, hue 240°) then worn (red, hue 0°);
    # lightness 30% / 50% / 80% for dark / mid / light (saturation 100%)
    bar_specs = [
        (2, "New Heel", _hsl_to_rgb(240, 100, 30)),   # dark blue
        (5, "Worn Heel", _hsl_to_rgb(0, 100, 30)),    # dark red
        (1, "New Mid", _hsl_to_rgb(240, 100, 50)),    # mid blue
        (4, "Worn Mid", _hsl_to_rgb(0, 100, 50)),     # mid red
        (0, "New Toe", _hsl_to_rgb(240, 100, 80)),    # light blue
        (3, "Worn Toe", _hsl_to_rgb(0, 100, 80)),     # light red
    ]
    material_indices = [spec[0] for spec in bar_specs]
    labels = [spec[1] for spec in bar_specs]
    colors_bar = [spec[2] for spec in bar_specs]
    x = np.arange(len(bar_specs))

    energy_return_by_mode = [
        [energy_return_from_hysteresis(hysteresis_ten_samples[i]) for i in range(n_materials)],
        [energy_return_from_hysteresis(hysteresis_com_samples[i]) for i in range(n_materials)],
        [energy_return_from_hysteresis(hysteresis_shear_samples[i]) for i in range(n_materials)],
    ]
    stiffness_means = [
        np.asarray(stiffness_ten, dtype=float)[material_indices],
        np.asarray(stiffness_com, dtype=float)[material_indices],
        np.asarray(stiffness_shear, dtype=float)[material_indices],
    ]
    stiffness_stds = [
        np.asarray(stiffness_ten_std, dtype=float)[material_indices],
        np.asarray(stiffness_com_std, dtype=float)[material_indices],
        np.asarray(stiffness_shear_std, dtype=float)[material_indices],
    ]
    # Tables store population std; SEM needs n. Energy return supplies sample counts for SEM.
    if error == "sem":
        for mode_idx, samples in enumerate(energy_return_by_mode):
            for j, mat in enumerate(material_indices):
                n = max(len(np.asarray(samples[mat], dtype=float)), 1)
                stiffness_stds[mode_idx][j] = stiffness_stds[mode_idx][j] / np.sqrt(n)

    mode_titles = ["Tension", "Compression", "Shear"]
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6), sharex="col")
    err_label = r"$\pm$ SEM" if error == "sem" else r"$\pm$ std"
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.6, label=label)
        for label, color in zip(labels, colors_bar)
    ]

    for col, mode_title in enumerate(mode_titles):
        ax_stiff = axes[0, col]
        ax_stiff.bar(
            x,
            stiffness_means[col],
            yerr=stiffness_stds[col],
            color=colors_bar,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax_stiff.set_title(mode_title, fontsize=FONT_SIZE)
        if col == 0:
            ax_stiff.set_ylabel(f"Stiffness [kPa]\n({err_label})", fontsize=FONT_SIZE - 4)
        ax_stiff.tick_params(labelsize=FONT_SIZE - 6, labelbottom=False)
        ax_stiff.set_xticks(x)
        ax_stiff.grid(axis="y", alpha=0.35)

        er_means, er_errs = _bar_series_mean_err(
            energy_return_by_mode[col], material_indices, error=error
        )
        ax_er = axes[1, col]
        ax_er.bar(
            x,
            er_means,
            yerr=er_errs,
            color=colors_bar,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        if col == 0:
            ax_er.set_ylabel(f"Energy return [%]\n({err_label})", fontsize=FONT_SIZE - 4)
        ax_er.set_ylim((50.0, 80.0) if col == 2 else (80.0, 100.0))
        ax_er.set_xticks(x)
        ax_er.set_xticklabels([])
        ax_er.tick_params(labelsize=FONT_SIZE - 6)
        ax_er.grid(axis="y", alpha=0.35)

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=FONT_SIZE - 4,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    plt.tight_layout()
    save_figure(fig, output_dir, "StiffnessEnergyReturnBars.pdf")
    plt.close(fig)
    print(f"Stiffness / energy-return bar figure saved to: {os.path.join(output_dir, 'StiffnessEnergyReturnBars.pdf')}")


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
    poissons_ratio_ten,
    poissons_ratio_ten_std,
    poissons_ratio_com,
    poissons_ratio_com_std,
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
            poissons_ratio_ten,
            poissons_ratio_ten_std,
            poissons_ratio_com,
            poissons_ratio_com_std,
            hysteresis_ten_samples,
            hysteresis_com_samples,
            hysteresis_shear_samples,
        )
        table_path = os.path.join(output_dir, f"{foam_name}_stress_table.tex")
        with open(table_path, "w") as f:
            f.write(tbl)
        print(f"Stress table saved to: {table_path}\n")

    plot_stiffness_energy_return_bars(
        n_materials,
        stiffness_ten,
        stiffness_ten_std,
        stiffness_com,
        stiffness_com_std,
        stiffness_shear,
        stiffness_shear_std,
        hysteresis_ten_samples,
        hysteresis_com_samples,
        hysteresis_shear_samples,
        output_dir=output_dir,
        error="std",
    )


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
    # poissons_ratio_ten = all_data["poissons_ratio_ten"]
    # poissons_ratio_ten_std = all_data["poissons_ratio_ten_std"]
    poissons_ratio_ten_samples = all_data["poissons_ratio_ten_samples"]
    poissons_ratio_ten = np.nanmean(np.array(poissons_ratio_ten_samples), axis=1)
    poissons_ratio_ten_std = np.nanstd(np.array(poissons_ratio_ten_samples), axis=1, ddof=1)
    # poissons_ratio_com = all_data["poissons_ratio_com"]
    # poissons_ratio_com_std = all_data["poissons_ratio_com_std"]
    poissons_ratio_com_samples = all_data["poissons_ratio_com_samples"]
    poissons_ratio_com = np.nanmean(np.array(poissons_ratio_com_samples), axis=1)
    poissons_ratio_com_std = np.nanstd(np.array(poissons_ratio_com_samples), axis=1, ddof=1)

    

    # --- Perform statistical tests ---
    regions = foam_types[3:]
    modes = ["tension", "compression", "shear"]
    experiments = ["uniaxial", "shear"]
    # Rows: 0=six-way, 1-3=new vs worn (toe/heel/mid), 4=three-way,
    #       5-7=region pairs (toe-heel, toe-mid, heel-mid); cols=uniaxial, shear
    anova_p_values = np.full((8, 2), np.nan)
    region_pairs = [(0, 1), (0, 2), (1, 2)]
    uniaxial_stress = build_combined_uniaxial_stress(all_data_stress, stretch_ten, stretch_com)
    shear_stress = all_data_stress[2, :, :, :, :]
    experiment_stress = [uniaxial_stress, shear_stress]
    uniaxial_grid = uniaxial_stretch_grid()

    print("6 way FDA ANOVA across all data:")
    for exp_idx, experiment in enumerate(experiments):
        data_reshape = experiment_stress[exp_idx].reshape(6, 5, -1)
        data_list = [data_reshape[i, :, :] for i in range(data_reshape.shape[0])]
        # Perform FDA ANOVA
        _, p_val = oneway_anova_np(*data_list)
        anova_p_values[0, exp_idx] = p_val
        print(f"\tp value for {experiment}: {p_val}")

    print("Pairwise FDA ANOVA comparing new vs worn for each region and experiment:")
    for exp_idx, experiment in enumerate(experiments):
        for region in range(3):
            data_reshape = experiment_stress[exp_idx][region, :, :, :]
            _, p_val = oneway_anova_np(
                data_reshape[0, :, :], data_reshape[1, :, :]
            )
            anova_p_values[1 + region, exp_idx] = p_val
            print(f"\tp value for {experiment} and {regions[region]}: {p_val}")

    print("3 way FDA ANOVA ignoring worn vs new:")
    for exp_idx, experiment in enumerate(experiments):
        data_reshape = experiment_stress[exp_idx].reshape(3, 10, -1)
        _, p_val = oneway_anova_np(
            data_reshape[0, :, :], data_reshape[1, :, :], data_reshape[2, :, :]
        )
        anova_p_values[4, exp_idx] = p_val
        print(f"\tp value for {experiment}: {p_val}")

    print("2 way FDA ANOVA ignoring worn vs new:")
    for exp_idx, experiment in enumerate(experiments):
        data_reshape = experiment_stress[exp_idx].reshape(3, 10, -1)
        for pair_idx, (region1, region2) in enumerate(region_pairs):
            _, p_val = oneway_anova_np(
                data_reshape[region1, :, :], data_reshape[region2, :, :]
            )
            anova_p_values[5 + pair_idx, exp_idx] = p_val
            print(
                f"\tp value for {experiment} comparing {regions[region1]} and {regions[region2]}: {p_val}"
            )

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

    energy_return_anova_p_values = compute_energy_return_anova_p_values(
        hysteresis_ten_samples,
        hysteresis_com_samples,
        hysteresis_shear_samples,
        n_materials,
    )
    save_energy_return_anova_table(energy_return_anova_p_values, ci_output_dir)

    ## Scalar statistical tests: stiffness, Poisson's ratio, energy return
    print("\nScalar statistical tests (stiffness, Poisson's ratio, energy return):")
    print(f"\tUsing linear-fit strain window max_strain = {MAX_STRAIN_LINEAR}")
    # Always recompute stiffness from curves so the strain window matches MAX_STRAIN_LINEAR
    (
        stiffness_ten_samples,
        stiffness_com_samples,
        stiffness_shear_samples,
    ) = stiffness_samples_from_curves(
        stretch_ten, stretch_com, strain_shr, all_data_stress, n_materials
    )
    # Refresh material means/stds used later if needed
    stiffness_ten = np.array([np.nanmean(s) for s in stiffness_ten_samples])
    stiffness_ten_std = np.array([np.nanstd(s, ddof=0) for s in stiffness_ten_samples])
    stiffness_com = np.array([np.nanmean(s) for s in stiffness_com_samples])
    stiffness_com_std = np.array([np.nanstd(s, ddof=0) for s in stiffness_com_samples])
    stiffness_shear = np.array([np.nanmean(s) for s in stiffness_shear_samples])
    stiffness_shear_std = np.array([np.nanstd(s, ddof=0) for s in stiffness_shear_samples])

    if "all_data_transverse" in all_data.files:
        all_data_transverse = all_data["all_data_transverse"]
        poissons_ratio_ten_samples, poissons_ratio_com_samples = poisson_samples_from_curves(
            stretch_ten, stretch_com, all_data_transverse, n_materials
        )
        poissons_ratio_ten = np.array([np.nanmean(s) for s in poissons_ratio_ten_samples])
        poissons_ratio_ten_std = np.array([np.nanstd(s, ddof=0) for s in poissons_ratio_ten_samples])
        poissons_ratio_com = np.array([np.nanmean(s) for s in poissons_ratio_com_samples])
        poissons_ratio_com_std = np.array([np.nanstd(s, ddof=0) for s in poissons_ratio_com_samples])
    else:
        print(
            "\tWarning: all_data_transverse missing; Poisson samples use values stored in npz "
            f"(may not match max_strain={MAX_STRAIN_LINEAR}). Re-run process_data() to refresh."
        )

    stiffness_mode_samples = [
        stiffness_ten_samples,
        stiffness_com_samples,
        stiffness_shear_samples,
    ]
    poisson_mode_samples = [
        poissons_ratio_ten_samples,
        poissons_ratio_com_samples,
    ]
    energy_return_mode_samples = [
        [energy_return_from_hysteresis(hysteresis_ten_samples[i]) for i in range(n_materials)],
        [energy_return_from_hysteresis(hysteresis_com_samples[i]) for i in range(n_materials)],
        [energy_return_from_hysteresis(hysteresis_shear_samples[i]) for i in range(n_materials)],
    ]

    stiffness_anova = compute_scalar_anova_p_values(stiffness_mode_samples, n_materials)
    poisson_anova = compute_scalar_anova_p_values(poisson_mode_samples, n_materials)
    # energy_return_anova_p_values already computed above

    save_scalar_anova_table(
        stiffness_anova,
        ["Tension", "Compression", "Shear"],
        "stiffness_anova_table.tex",
        output_dir=ci_output_dir,
    )
    save_scalar_anova_table(
        poisson_anova,
        ["Tension", "Compression"],
        "poisson_anova_table.tex",
        output_dir=ci_output_dir,
    )

    stiffness_ci_min, stiffness_ci_max = compute_scalar_new_worn_cis(
        stiffness_mode_samples, n_materials
    )
    poisson_ci_min, poisson_ci_max = compute_scalar_new_worn_cis(
        poisson_mode_samples, n_materials
    )
    energy_ci_min, energy_ci_max = compute_scalar_new_worn_cis(
        energy_return_mode_samples, n_materials
    )

    save_scalar_ci_table(
        stiffness_ci_min,
        stiffness_ci_max,
        regions,
        ["Tension", "Compression", "Shear"],
        "stiffness_confidence_interval_table.tex",
        unit_label="[kPa]",
        output_dir=ci_output_dir,
    )
    save_scalar_ci_table(
        poisson_ci_min,
        poisson_ci_max,
        regions,
        ["Tension", "Compression"],
        "poisson_confidence_interval_table.tex",
        unit_label="[-]",
        output_dir=ci_output_dir,
    )
    save_scalar_ci_table(
        energy_ci_min,
        energy_ci_max,
        regions,
        ["Tension", "Compression", "Shear"],
        "energy_return_confidence_interval_table.tex",
        unit_label="[\\%]",
        output_dir=ci_output_dir,
    )
    save_scalar_summary_table(
        stiffness_anova,
        energy_return_anova_p_values,
        poisson_anova,
        stiffness_ci_min,
        stiffness_ci_max,
        energy_ci_min,
        energy_ci_max,
        poisson_ci_min,
        poisson_ci_max,
        output_dir=ci_output_dir,
    )

    stiffness_region_ci_min, stiffness_region_ci_max = compute_scalar_region_cis(
        stiffness_mode_samples, n_materials
    )
    poisson_region_ci_min, poisson_region_ci_max = compute_scalar_region_cis(
        poisson_mode_samples, n_materials
    )
    energy_region_ci_min, energy_region_ci_max = compute_scalar_region_cis(
        energy_return_mode_samples, n_materials
    )
    save_scalar_region_summary_table(
        stiffness_anova,
        energy_return_anova_p_values,
        poisson_anova,
        stiffness_region_ci_min,
        stiffness_region_ci_max,
        energy_region_ci_min,
        energy_region_ci_max,
        poisson_region_ci_min,
        poisson_region_ci_max,
        output_dir=ci_output_dir,
    )

    save_scalar_mode_comparison_table(
        stiffness_mode_samples,
        energy_return_mode_samples,
        poisson_mode_samples,
        regions,
        output_dir=ci_output_dir,
    )

    region_pair_labels = ["toe-mid", "toe-heel", "mid-heel"]
    for quantity, anova_p, mode_names, ci_lo, ci_hi, region_ci_lo, region_ci_hi in (
        (
            "Stiffness",
            stiffness_anova,
            modes,
            stiffness_ci_min,
            stiffness_ci_max,
            stiffness_region_ci_min,
            stiffness_region_ci_max,
        ),
        (
            "Poisson's ratio",
            poisson_anova,
            modes[:2],
            poisson_ci_min,
            poisson_ci_max,
            poisson_region_ci_min,
            poisson_region_ci_max,
        ),
        (
            "Energy return",
            energy_return_anova_p_values,
            modes,
            energy_ci_min,
            energy_ci_max,
            energy_region_ci_min,
            energy_region_ci_max,
        ),
    ):
        print(f"\n{quantity}:")
        test_labels = [
            "6 way",
            "New vs Worn (toe)",
            "New vs Worn (heel)",
            "New vs Worn (mid)",
            "Toe vs Mid vs Heel",
            "Toe vs Heel",
            "Toe vs Mid",
            "Mid vs Heel",
        ]
        for row_idx, test_name in enumerate(test_labels):
            vals = ", ".join(
                f"{mode_names[m]} p={anova_p[row_idx, m]:.4g}" for m in range(len(mode_names))
            )
            print(f"\t{test_name}: {vals}")
        for mode_idx, mode_name in enumerate(mode_names):
            for region_idx, region_name in enumerate(regions):
                print(
                    f"\tCI new-worn {mode_name} {region_name}: "
                    f"{ci_lo[mode_idx, region_idx]:.4g} to {ci_hi[mode_idx, region_idx]:.4g}"
                )
            for pair_idx, pair_name in enumerate(region_pair_labels):
                print(
                    f"\tCI region {mode_name} {pair_name}: "
                    f"{region_ci_lo[mode_idx, pair_idx]:.4g} to "
                    f"{region_ci_hi[mode_idx, pair_idx]:.4g}"
                )

    ## Statistical tests for Poisson's ratio (tension vs compression)
    print("\nPoisson's ratio t-tests (tension vs compression):")
    ten_vals = pool_poissons_samples(poissons_ratio_ten_samples)
    com_vals = pool_poissons_samples(poissons_ratio_com_samples)
    t_stat, p_val = ttest_ind(ten_vals, com_vals, equal_var=False)
    print(
        f"\tt = {t_stat:.4f}, p = {p_val:.4g} "
        f"(ν_ten = {ten_vals.mean():.4f}, ν_com = {com_vals.mean():.4f})"
    )
    if p_val < 0.05:
        print(f"\tSignificant difference between tension and compression")
        print(f"\t\tPoisson's ratio tension: {ten_vals.mean():.4f}")
        print(f"\t\tPoisson's ratio compression: {com_vals.mean():.4f}")
        print(f"\t\tPoisson's ratio tension std: {ten_vals.std():.4f}")
        print(f"\t\tPoisson's ratio compression std: {com_vals.std():.4f}")
        print(f"\t\tPoisson's ratio tension min: {ten_vals.min():.4f}")
        print(f"\t\tPoisson's ratio compression min: {com_vals.min():.4f}")
        print(f"\t\tPoisson's ratio tension max: {ten_vals.max():.4f}")

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
    plot_stress_region_mode_grid(
        stretch_ten,
        stress_ten,
        stress_ten_std,
        stretch_ten_table,
        stress_ten_table,
        stretch_com,
        stress_com,
        stress_com_std,
        stretch_com_table,
        stress_com_table,
        strain_shr,
        stress_shr,
        stress_shr_std,
        strain_shr_table,
        stress_shr_table,
        n_materials,
        output_dir,
    )
    plot_transverse_stretch_region_mode_grid(
        stretch_ten,
        transverse_stretch_ten,
        transverse_stretch_ten_std,
        stretch_ten_table,
        transverse_stretch_ten_table,
        stretch_com,
        transverse_stretch_com,
        transverse_stretch_com_std,
        stretch_com_table,
        transverse_stretch_com_table,
        n_materials,
        output_dir,
    )


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
        poissons_ratio_ten,
        poissons_ratio_ten_std,
        poissons_ratio_com,
        poissons_ratio_com_std,
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
