"""
Statistical functions for the preprocessing pipeline.

Scalar ANOVA, Welch tests, confidence intervals, sample extraction helpers.
"""

import numpy as np
from scipy.stats import f, t, ttest_ind
from skfda.inference.anova import oneway_anova
from skfda.representation.grid import FDataGrid


max_strain_linear = 0.2

def fit_initial_slope(x, y, max_x, n_pts=101):
    """Computes stiffness via least-squares slope over [0, max_strain]."""
    x_data = np.linspace(0, max_x, n_pts)
    y_data = np.interp(x_data, x, y)
    denom = np.dot(x_data, x_data)
    if denom <= 0:
        return 0.0
    return np.dot(x_data, y_data) / denom


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

def _as_sample_lists(mode_samples, n_materials=6):
    """Normalize mode samples to a list of length n_materials of 1D float arrays."""
    return [np.asarray(mode_samples[i], dtype=float).ravel() for i in range(n_materials)]


def pool_region_mode_samples(mode_samples_by_material, region_idx, n_materials=6):
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


def _mode_pair_indices(n_modes):
    """Pairwise mode indices: (0,1), (0,2), (1,2) for 3 modes; (0,1) for 2 modes."""
    if n_modes >= 3:
        return [(0, 1), (0, 2), (1, 2)]
    if n_modes == 2:
        return [(0, 1)]
    return []


def compute_scalar_p_values(mode_sample_lists, n_materials=6):
    """
    Scalar Welch p-values for new-vs-worn, region, and mode comparisons.

    mode_sample_lists: sequence over modes; each entry is length-n_materials sample lists.

    Returns
    -------
    new_vs_worn : (3, n_modes)
        New vs worn within region; region order toe, mid, heel.
    region_pairs : (3, n_modes)
        Pairwise region comparisons (new+worn pooled); pair order
        toe-mid, toe-heel, mid-heel.
    mode_pairs : (3, n_mode_pairs)
        Pairwise mode comparisons within region (new+worn pooled);
        region order toe, mid, heel. Mode-pair order is (0,1), (0,2), (1,2)
        when n_modes >= 3, or just (0,1) when n_modes == 2.
    """
    # Material indices: new-toe=0, new-mid=1, new-heel=2, worn-toe=3, worn-mid=4, worn-heel=5
    region_materials = [
        [0, 3],  # toe
        [1, 4],  # mid
        [2, 5],  # heel
    ]
    region_pairs = [(0, 1), (0, 2), (1, 2)]  # toe-mid, toe-heel, mid-heel

    n_modes = len(mode_sample_lists)
    mode_pair_idx = _mode_pair_indices(n_modes)
    n_mode_pairs = len(mode_pair_idx)

    new_vs_worn = np.full((3, n_modes), np.nan)
    region_p = np.full((3, n_modes), np.nan)
    mode_p = np.full((3, n_mode_pairs), np.nan)

    # Per-mode: new vs worn and region pairwise
    for mode_idx, mode_samples in enumerate(mode_sample_lists):
        samples = _as_sample_lists(mode_samples, n_materials)

        for region_idx, mats in enumerate(region_materials):
            groups = [samples[i] for i in mats]
            _, new_vs_worn[region_idx, mode_idx] = scalar_oneway_anova(groups)

        region_vals = [
            np.concatenate([samples[i] for i in mats]) for mats in region_materials
        ]
        region_vals = [g[~np.isnan(g)] for g in region_vals]
        for pair_idx, (r1, r2) in enumerate(region_pairs):
            _, region_p[pair_idx, mode_idx] = welch_anova(region_vals[r1], region_vals[r2])

    # Across modes within each region (new+worn pooled)
    for region_idx in range(3):
        pooled = [
            pool_region_mode_samples(mode_sample_lists[m], region_idx, n_materials)
            for m in range(n_modes)
        ]
        for pair_idx, (m1, m2) in enumerate(mode_pair_idx):
            _, mode_p[region_idx, pair_idx] = welch_anova(pooled[m1], pooled[m2])

    return new_vs_worn, region_p, mode_p


def compute_scalar_mode_cis(mode_sample_lists, n_materials=6, alpha=0.05):
    """
    Welch t CIs for mean(mode_a) - mean(mode_b) within each region (new+worn pooled).

    Returns (ci_min, ci_max) each shaped (3, n_mode_pairs) with region order
    toe, mid, heel and the same mode-pair order as compute_scalar_p_values.
    """
    n_modes = len(mode_sample_lists)
    mode_pair_idx = _mode_pair_indices(n_modes)
    n_mode_pairs = len(mode_pair_idx)
    ci_min = np.full((3, n_mode_pairs), np.nan)
    ci_max = np.full((3, n_mode_pairs), np.nan)
    for region_idx in range(3):
        pooled = [
            pool_region_mode_samples(mode_sample_lists[m], region_idx, n_materials)
            for m in range(n_modes)
        ]
        for pair_idx, (m1, m2) in enumerate(mode_pair_idx):
            lo, hi = scalar_mean_diff_ttest_ci(pooled[m1], pooled[m2], alpha=alpha)
            ci_min[region_idx, pair_idx] = lo
            ci_max[region_idx, pair_idx] = hi
    return ci_min, ci_max

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


def compute_p_threshold(p_values, alpha=0.05):
    """
    Compute the p-value threshold for a given alpha.
    """
    n_tests = p_values.shape[0]
    p_values_sorted = np.sort(p_values)
    for i in range(n_tests):
        threshold = alpha / (n_tests - i)
        if p_values_sorted[i] > threshold:
            return threshold
    return alpha