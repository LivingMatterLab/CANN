"""
Plotting and table-generation functions for the preprocessing pipeline.
"""

import os
from enum import StrEnum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tabulate import tabulate

from util import *
from stats_functions import *

import pandas as pd


# Define loading modes as enum
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

# Transverse stretch is only relevant in tension and compression
_TRANSVERSE_LOADING_MODES = frozenset({LoadingMode.TENSION, LoadingMode.COMPRESSION})

### Define results class
class PreprocessingResults: 
    def __init__(self, foam_types, foam_types_title, colors, linestyles, worn_shoe, n_pts_table, n_pts_plt, FONT_SIZE, out_dir, max_strain_linear) -> None:

        # Initialize settings that affect plotting
        self.foam_types = foam_types
        self.foam_types_title = foam_types_title
        self.colors = colors
        self.linestyles = linestyles
        self.worn_shoe = worn_shoe
        self.n_pts_table = n_pts_table
        self.n_pts_plt = n_pts_plt
        self.FONT_SIZE = FONT_SIZE
        self.out_dir = out_dir
        self.max_strain_linear = max_strain_linear
        # Load data from numpy save file
        self.load_from_file(os.path.join(out_dir, "all_data.npz"))
        

        # --- Perform statistical tests ---
        self.regions = self.foam_types[3:]
        self.modes = ["tension", "compression", "shear"]

        # Compute stiffness and poisson ratio for each sample
        self.compute_stiffness_samples()
        self.compute_poisson_samples()

        # Compute means / std deviations for stiffness and poisson ratio
        self.stiffness_ten = np.array([np.nanmean(s) for s in self.stiffness_ten_samples])
        self.stiffness_ten_std = np.array([np.nanstd(s, ddof=0) for s in self.stiffness_ten_samples])
        self.stiffness_com = np.array([np.nanmean(s) for s in self.stiffness_com_samples])
        self.stiffness_com_std = np.array([np.nanstd(s, ddof=0) for s in self.stiffness_com_samples])
        self.stiffness_shear = np.array([np.nanmean(s) for s in self.stiffness_shear_samples])
        self.stiffness_shear_std = np.array([np.nanstd(s, ddof=0) for s in self.stiffness_shear_samples])

        self.poissons_ratio_ten = np.array([np.nanmean(s) for s in self.poissons_ten_samples])
        self.poissons_ratio_ten_std = np.array([np.nanstd(s, ddof=0) for s in self.poissons_ten_samples])
        self.poissons_ratio_com = np.array([np.nanmean(s) for s in self.poissons_com_samples])
        self.poissons_ratio_com_std = np.array([np.nanstd(s, ddof=0) for s in self.poissons_com_samples])
        
        # Assemble all stiffness, poisson ratios, and energy returns into arrays
        self.stiffness_mode_samples = [
            self.stiffness_ten_samples,
            self.stiffness_com_samples,
            self.stiffness_shear_samples,
        ]
        self.poisson_mode_samples = [
            self.poissons_ten_samples,
            self.poissons_com_samples,
        ]
        self.energy_return_mode_samples = [
            [self.energy_return_ten_samples[i] for i in range(self.n_materials)],
            [self.energy_return_com_samples[i] for i in range(self.n_materials)],
            [self.energy_return_shear_samples[i] for i in range(self.n_materials)],
        ]

        # Compute p values for comparison of scalars (stiffness, poisson ratio, energy return)
        (
            self.stiffness_new_worn_p,
            self.stiffness_region_p,
            self.stiffness_mode_p,
        ) = compute_scalar_p_values(self.stiffness_mode_samples, self.n_materials)
        (
            self.poisson_new_worn_p,
            self.poisson_region_p,
            self.poisson_mode_p,
        ) = compute_scalar_p_values(self.poisson_mode_samples, self.n_materials)
        (
            self.energy_new_worn_p,
            self.energy_region_p,
            self.energy_mode_p,
        ) = compute_scalar_p_values(self.energy_return_mode_samples, self.n_materials)

        # Compute confidence intervals for difference between scalar values in new and worn foam
        self.stiffness_ci_min, self.stiffness_ci_max = compute_scalar_new_worn_cis(
            self.stiffness_mode_samples, self.n_materials
        )
        self.poisson_ci_min, self.poisson_ci_max = compute_scalar_new_worn_cis(
            self.poisson_mode_samples, self.n_materials
        )
        self.energy_ci_min, self.energy_ci_max = compute_scalar_new_worn_cis(
            self.energy_return_mode_samples, self.n_materials
        )

        # Compute confidence intervals for difference among scalar values in different regions
        self.stiffness_region_ci_min, self.stiffness_region_ci_max = compute_scalar_region_cis(
            self.stiffness_mode_samples, self.n_materials
        )
        self.poisson_region_ci_min, self.poisson_region_ci_max = compute_scalar_region_cis(
            self.poisson_mode_samples, self.n_materials
        )
        self.energy_region_ci_min, self.energy_region_ci_max = compute_scalar_region_cis(
            self.energy_return_mode_samples, self.n_materials
        )

        # Compute confidence intervals for difference among modes within each region
        self.stiffness_mode_ci_min, self.stiffness_mode_ci_max = compute_scalar_mode_cis(
            self.stiffness_mode_samples, self.n_materials
        )
        self.poisson_mode_ci_min, self.poisson_mode_ci_max = compute_scalar_mode_cis(
            self.poisson_mode_samples, self.n_materials
        )
        self.energy_mode_ci_min, self.energy_mode_ci_max = compute_scalar_mode_cis(
            self.energy_return_mode_samples, self.n_materials
        )

        # Compute p value significance threshold based on number of positive tests
        p_values_all = np.concatenate([self.stiffness_new_worn_p.flatten(), self.poisson_new_worn_p.flatten(), self.energy_new_worn_p.flatten(), self.stiffness_region_p.flatten(), self.poisson_region_p.flatten(), self.energy_region_p.flatten(), self.stiffness_mode_p.flatten(), self.poisson_mode_p.flatten(), self.energy_mode_p.flatten()])
        self.p_threshold = compute_p_threshold(p_values_all, alpha=0.05)

        # Compute stress & trans stretch means / stds from per sample data
        all_means_stress = np.mean(self.all_data_stress, axis=3).reshape((3, 6, -1))
        all_stds_stress = np.std(self.all_data_stress, axis=3).reshape((3, 6, -1))
        self.stress_ten = all_means_stress[0, :, :].T
        self.stress_com = all_means_stress[1, :, :].T
        self.stress_shr = all_means_stress[2, :, :].T   
        self.stress_ten_std = all_stds_stress[0, :, :].T    
        self.stress_com_std = all_stds_stress[1, :, :].T    
        self.stress_shr_std = all_stds_stress[2, :, :].T

        all_means_transverse = np.nanmean(self.all_data_transverse, axis=3).reshape((2, 6, -1))
        all_stds_transverse = np.nanstd(self.all_data_transverse, axis=3).reshape((2, 6, -1))
        self.transverse_stretch_ten = all_means_transverse[0, :, :].T
        self.transverse_stretch_com = all_means_transverse[1, :, :].T
        self.transverse_stretch_ten_std = all_stds_transverse[0, :, :].T    
        self.transverse_stretch_com_std = all_stds_transverse[1, :, :].T    

        # Resample original data to include in tables and in plots as discrete points
        self.stretch_ten_table, self.stress_ten_table, self.stress_ten_std_table, self.transverse_stretch_ten_table, self.transverse_stretch_ten_std_table = resample_table(
            self.stretch_ten, n_pts_table,
            self.stress_ten, self.stress_ten_std, self.transverse_stretch_ten, self.transverse_stretch_ten_std
        )
        self.stretch_com_table, self.stress_com_table, self.stress_com_std_table, self.transverse_stretch_com_table, self.transverse_stretch_com_std_table = resample_table(
            self.stretch_com, n_pts_table,
            self.stress_com, self.stress_com_std, self.transverse_stretch_com, self.transverse_stretch_com_std,
            is_compression=True
        )
        self.strain_shr_table, self.stress_shr_table, self.stress_shr_std_table = resample_table(
            self.strain_shr, n_pts_table, self.stress_shr, self.stress_shr_std
        )
        if not worn_shoe:
            self.stretch_conf_com_table, self.stress_conf_com_table, self.stress_conf_com_std_table = resample_table(
                self.stretch_conf_com, n_pts_table,
                self.stress_conf_com, self.stress_conf_com_std,
                is_compression=True
            )

    def load_from_file(self, path):
        self.n_materials = len(self.foam_types)
        all_data = np.load(path, allow_pickle=True)
        required_keys = [
            "all_data_stress", "all_data_transverse", 
            "stretch_ten", "stretch_com", "strain_shr", "stretch_conf_com", 
            "energy_return_ten_samples", "energy_return_com_samples", "energy_return_shear_samples"
        ]
        missing_keys = [key for key in required_keys if key not in all_data.files]
        if missing_keys:
            raise KeyError(
                f"Missing keys in all_data.npz: {missing_keys}. "
                "Run process_data() once (set should_process=True) to regenerate."
            )
        self.all_data_stress = all_data["all_data_stress"]
        self.all_data_transverse = all_data["all_data_transverse"]
        self.stretch_ten = all_data["stretch_ten"]
        self.stretch_com = all_data["stretch_com"]
        self.strain_shr = all_data["strain_shr"]
        self.stretch_conf_com = all_data["stretch_conf_com"]
        self.energy_return_ten_samples = all_data["energy_return_ten_samples"]
        self.energy_return_com_samples = all_data["energy_return_com_samples"]
        self.energy_return_shear_samples = all_data["energy_return_shear_samples"]
        self.stress_ten_samples = all_data["stress_ten_samples"]
        self.stress_com_samples = all_data["stress_com_samples"]
        self.stress_shear_samples = all_data["stress_shear_samples"]
        self.stress_confcom_samples = all_data["stress_confcom_samples"]


    def save_plots_and_tables(self, output_dir="./Results/RawData"):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.save_scalar_summary_table(output_dir)
        self.save_scalar_region_summary_table(output_dir)
        self.save_scalar_mode_comparison_table(output_dir)

        # Create stress plots (axial, transverse, and shear)
        show_error_bars = not self.worn_shoe
        self.plot_stress(LoadingMode.TENSION, show_error_bars, self.stretch_ten, self.stress_ten, self.stress_ten_std, self.stretch_ten_table, self.stress_ten_table, self.n_materials, output_dir)
        self.plot_stress(LoadingMode.COMPRESSION, show_error_bars, self.stretch_com, self.stress_com, self.stress_com_std, self.stretch_com_table, self.stress_com_table, self.n_materials, output_dir)
        self.plot_transverse_stretch(
            LoadingMode.COMPRESSION, show_error_bars, self.stretch_com, self.transverse_stretch_com, self.transverse_stretch_com_std, self.n_materials, output_dir
        )
        self.plot_transverse_stretch(
            LoadingMode.TENSION, show_error_bars, self.stretch_ten, self.transverse_stretch_ten, self.transverse_stretch_ten_std, self.n_materials, output_dir
        )
        if not self.worn_shoe:
            self.plot_stress(
                LoadingMode.CONFINED_COMPRESSION,
                show_error_bars,
                self.stretch_conf_com,
                self.stress_conf_com,
                self.stress_conf_com_std,
                self.stretch_conf_com_table,
                self.stress_conf_com_table,
                self.n_materials,
                output_dir,
            )
        self.plot_stress(LoadingMode.SHEAR, show_error_bars, self.strain_shr, self.stress_shr, self.stress_shr_std, self.strain_shr_table, self.stress_shr_table, self.n_materials, output_dir)
        self.plot_stress_region_mode_grid(output_dir)
        self.plot_transverse_stretch_region_mode_grid(output_dir)


        # Create individual sample plots (tension, compression, shear, confined compression)
        for foam_idx in range(self.n_materials):
            self.plot_individual_samples(
                foam_idx,
                self.stress_ten_samples,
                self.stress_com_samples,
                self.stress_shear_samples,
                self.stress_confcom_samples,
                output_dir,
            )

        # Create stress tables
        self.save_stress_tables(output_dir)
        self.save_stress_excel(excel_filename="WornFoamData.xlsx" if self.worn_shoe else "FoamData.xlsx")

    
    def compute_stiffness_samples(self):
        """Recompute per-sample stiffness from stored mean stress curves."""
        self.stiffness_ten_samples = []
        self.stiffness_com_samples = []
        self.stiffness_shear_samples = []
        for foam_idx in range(self.n_materials):
            region = foam_idx % 3
            new_worn = foam_idx // 3
            strain_ten = self.stretch_ten[:, foam_idx] - 1.0
            strain_com = 1.0 - self.stretch_com[:, foam_idx]
            strain_shear = self.strain_shr[:, foam_idx]
            ten = []
            com = []
            shr = []
            for sample_idx in range(self.all_data_stress.shape[3]):
                ten.append(
                    fit_initial_slope(
                        strain_ten,
                        self.all_data_stress[0, region, new_worn, sample_idx, :],
                        max_x=self.max_strain_linear,
                    )
                )
                com.append(
                    fit_initial_slope(
                        strain_com,
                        self.all_data_stress[1, region, new_worn, sample_idx, :],
                        max_x=self.max_strain_linear,
                    )
                )
                shr.append(
                    fit_initial_slope(
                        strain_shear,
                        self.all_data_stress[2, region, new_worn, sample_idx, :],
                        max_x=self.max_strain_linear,
                    )
                )
            self.stiffness_ten_samples.append(np.asarray(ten, dtype=float))
            self.stiffness_com_samples.append(np.asarray(com, dtype=float))
            self.stiffness_shear_samples.append(np.asarray(shr, dtype=float))


    def compute_poisson_samples(self):
        """Recompute per-sample Poisson's ratio from stored transverse stretch curves."""
        self.poissons_ten_samples = []
        self.poissons_com_samples = []
        for foam_idx in range(self.n_materials):
            region = foam_idx % 3
            new_worn = foam_idx // 3
            strain_ten = self.stretch_ten[:, foam_idx] - 1.0
            strain_com = 1.0 - self.stretch_com[:, foam_idx]
            ten = []
            com = []
            for sample_idx in range(self.all_data_transverse.shape[3]):
                ten.append(
                    -fit_initial_slope(
                        strain_ten,
                        self.all_data_transverse[0, region, new_worn, sample_idx, :] - 1,
                        max_x=self.max_strain_linear,
                    )
                )
                com.append(
                    fit_initial_slope(
                        strain_com,
                        self.all_data_transverse[1, region, new_worn, sample_idx, :] - 1,
                        max_x=self.max_strain_linear,
                    )
                )
            self.poissons_ten_samples.append(np.asarray(ten, dtype=float))
            self.poissons_com_samples.append(np.asarray(com, dtype=float))
    
    def save_scalar_summary_table(self, output_dir="./Results/RawData"):
        """
        Combined scalar summary: new-vs-worn p-values/CIs.

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
        # Region order for both p-values and CIs: 0=toe, 1=mid, 2=heel
        row_specs = [
            ("Worn vs new toe $p$ value", "p", 0),
            ("Worn vs new toe CI", "ci", 0),
            ("Worn vs new mid $p$ value", "p", 1),
            ("Worn vs new mid CI", "ci", 1),
            ("Worn vs new heel $p$ value", "p", 2),
            ("Worn vs new heel CI", "ci", 2),
        ]

        def p_cell(p_arr, mode_idx, region_idx):
            if mode_idx >= p_arr.shape[1]:
                return ""
            return fmt_p_value(p_arr[region_idx, mode_idx], self.p_threshold)

        def ci_cell(ci_min, ci_max, mode_idx, region_idx):
            if mode_idx >= ci_min.shape[0]:
                return ""
            return fmt_ci_interval(ci_min[mode_idx, region_idx], ci_max[mode_idx, region_idx])

        rows = []
        for label, kind, region_idx in row_specs:
            if kind == "p":
                cells = [
                    p_cell(self.stiffness_new_worn_p, 0, region_idx),
                    p_cell(self.stiffness_new_worn_p, 1, region_idx),
                    p_cell(self.stiffness_new_worn_p, 2, region_idx),
                    p_cell(self.energy_new_worn_p, 0, region_idx),
                    p_cell(self.energy_new_worn_p, 1, region_idx),
                    p_cell(self.energy_new_worn_p, 2, region_idx),
                    p_cell(self.poisson_new_worn_p, 0, region_idx),
                    p_cell(self.poisson_new_worn_p, 1, region_idx),
                ]
            else:
                cells = [
                    ci_cell(self.stiffness_ci_min, self.stiffness_ci_max, 0, region_idx),
                    ci_cell(self.stiffness_ci_min, self.stiffness_ci_max, 1, region_idx),
                    ci_cell(self.stiffness_ci_min, self.stiffness_ci_max, 2, region_idx),
                    ci_cell(self.energy_ci_min, self.energy_ci_max, 0, region_idx),
                    ci_cell(self.energy_ci_min, self.energy_ci_max, 1, region_idx),
                    ci_cell(self.energy_ci_min, self.energy_ci_max, 2, region_idx),
                    ci_cell(self.poisson_ci_min, self.poisson_ci_max, 0, region_idx),
                    ci_cell(self.poisson_ci_min, self.poisson_ci_max, 1, region_idx),
                ]
            rows.append([label, *cells])

        table = latex_tabular_from_tabulate(rows, headers, r"|l||c|c|c||c|c|c||c|c|")
        table_name = "scalar_new_worn_table"
        table_dir = table_output_dir(output_dir, table_name)
        table_path = os.path.join(table_dir, f"{table_name}.tex")
        with open(table_path, "w") as fh:
            fh.write(table)
        print(f"Scalar summary table saved to: {table_path}")
        render_latex_table_to_png(table_name, output_dir=output_dir)


    def save_scalar_region_summary_table(self, output_dir="./Results/RawData"):
        """
        Region comparison scalar summary: pairwise region p-values/CIs with new and worn pooled.

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
        # Pair order for both p-values and CIs: 0=toe-mid, 1=toe-heel, 2=mid-heel
        row_specs = [
            ("Toe vs mid $p$ value", "p", 0),
            ("Toe vs mid CI", "ci", 0),
            ("Toe vs heel $p$ value", "p", 1),
            ("Toe vs heel CI", "ci", 1),
            ("Mid vs heel $p$ value", "p", 2),
            ("Mid vs heel CI", "ci", 2),
        ]

        def p_cell(p_arr, mode_idx, pair_idx):
            if mode_idx >= p_arr.shape[1]:
                return ""
            return fmt_p_value(p_arr[pair_idx, mode_idx], self.p_threshold)

        def ci_cell(ci_min, ci_max, mode_idx, pair_idx):
            if mode_idx >= ci_min.shape[0]:
                return ""
            return fmt_ci_interval(ci_min[mode_idx, pair_idx], ci_max[mode_idx, pair_idx])

        rows = []
        for label, kind, pair_idx in row_specs:
            if kind == "p":
                cells = [
                    p_cell(self.stiffness_region_p, 0, pair_idx),
                    p_cell(self.stiffness_region_p, 1, pair_idx),
                    p_cell(self.stiffness_region_p, 2, pair_idx),
                    p_cell(self.energy_region_p, 0, pair_idx),
                    p_cell(self.energy_region_p, 1, pair_idx),
                    p_cell(self.energy_region_p, 2, pair_idx),
                    p_cell(self.poisson_region_p, 0, pair_idx),
                    p_cell(self.poisson_region_p, 1, pair_idx),
                ]
            else:
                cells = [
                    ci_cell(self.stiffness_region_ci_min, self.stiffness_region_ci_max, 0, pair_idx),
                    ci_cell(self.stiffness_region_ci_min, self.stiffness_region_ci_max, 1, pair_idx),
                    ci_cell(self.stiffness_region_ci_min, self.stiffness_region_ci_max, 2, pair_idx),
                    ci_cell(self.energy_region_ci_min, self.energy_region_ci_max, 0, pair_idx),
                    ci_cell(self.energy_region_ci_min, self.energy_region_ci_max, 1, pair_idx),
                    ci_cell(self.energy_region_ci_min, self.energy_region_ci_max, 2, pair_idx),
                    ci_cell(self.poisson_region_ci_min, self.poisson_region_ci_max, 0, pair_idx),
                    ci_cell(self.poisson_region_ci_min, self.poisson_region_ci_max, 1, pair_idx),
                ]
            rows.append([label, *cells])

        table = latex_tabular_from_tabulate(rows, headers, r"|l||c|c|c||c|c|c||c|c|")
        table_name = "scalar_region_table"
        table_dir = table_output_dir(output_dir, table_name)
        table_path = os.path.join(table_dir, f"{table_name}.tex")
        with open(table_path, "w") as fh:
            fh.write(table)
        print(f"Scalar region summary table saved to: {table_path}")
        render_latex_table_to_png(table_name, output_dir=output_dir)


    def save_scalar_mode_comparison_table(self, output_dir="./Results/RawData"):
        """
        Pool new+worn per region and compare modes with Welch tests.

        Table columns:
        1-3: Stiffness pairwise differences within region (ten-com, ten-shear, com-shear)
        4-6: Energy-return pairwise differences within region
        7  : Poisson tension - compression
        8  : Poisson tension vs 0 (one-sample)
        9  : Poisson compression vs 0 (one-sample)
        Rows: Toe / Mid / Heel p value and CI.
        """
        os.makedirs(output_dir, exist_ok=True)

        n_regions = len(self.regions)
        if n_regions != 3:
            print(f"Warning: expected 3 regions (toe/mid/heel), got {n_regions}")

        # Mode-pair columns use precomputed p/CI arrays; poisson vs 0 is one-sample.
        # Mode-pair order: 0=ten-com, 1=ten-shear, 2=com-shear (poisson only has pair 0).
        col_specs = [
            (r"Stiffness Ten - Comp", "mode", "stiff", 0),
            (r"Stiffness Ten - Shear", "mode", "stiff", 1),
            (r"Stiffness Comp - Shear", "mode", "stiff", 2),
            (r"Energy Ten - Comp", "mode", "energy", 0),
            (r"Energy Ten - Shear", "mode", "energy", 1),
            (r"Energy Comp - Shear", "mode", "energy", 2),
            (r"Poisson Ten - Comp", "mode", "poisson", 0),
            (r"Poisson Ten vs 0", "one", "poisson", 0),
            (r"Poisson Comp vs 0", "one", "poisson", 1),
        ]

        mode_p = {
            "stiff": self.stiffness_mode_p,
            "energy": self.energy_mode_p,
            "poisson": self.poisson_mode_p,
        }
        mode_ci_min = {
            "stiff": self.stiffness_mode_ci_min,
            "energy": self.energy_mode_ci_min,
            "poisson": self.poisson_mode_ci_min,
        }
        mode_ci_max = {
            "stiff": self.stiffness_mode_ci_max,
            "energy": self.energy_mode_ci_max,
            "poisson": self.poisson_mode_ci_max,
        }
        poisson_pooled = [
            [pool_region_mode_samples(self.poisson_mode_samples[m], r) for m in range(2)]
            for r in range(3)
        ]

        headers = [r"\makecell{Region}"] + [rf"\makecell{{{h}}}" for h, _, _, _ in col_specs]

        rows = []
        for region_idx in range(3):
            region_name = format_ci_region_name(self.regions[region_idx])
            p_cells = []
            ci_cells = []
            for _, kind, src, idx in col_specs:
                if kind == "mode":
                    p_cells.append(fmt_p_value(mode_p[src][region_idx, idx], self.p_threshold))
                    ci_cells.append(
                        fmt_ci_interval(
                            mode_ci_min[src][region_idx, idx],
                            mode_ci_max[src][region_idx, idx],
                        )
                    )
                else:
                    p_val, lo, hi = scalar_mean_vs_zero_ttest_p_ci(
                        poisson_pooled[region_idx][idx], alpha=0.05
                    )
                    p_cells.append(fmt_p_value(p_val, self.p_threshold))
                    ci_cells.append(fmt_ci_interval(lo, hi))

            rows.append([rf"{region_name} p value", *p_cells])
            rows.append([rf"{region_name} CI", *ci_cells])

        colspec = rf"|l|{'|'.join(['c'] * len(col_specs))}|"
        table = latex_tabular_from_tabulate(rows, headers, colspec)
        table_name = "scalar_mode_comparison_table"
        table_dir = table_output_dir(output_dir, table_name)
        table_path = os.path.join(table_dir, f"{table_name}.tex")
        with open(table_path, "w") as fh:
            fh.write(table)
        print(f"Scalar mode comparison table saved to: {table_path}")
        render_latex_table_to_png(table_name, output_dir=output_dir)

    def plot_stress_region_mode_grid(self, output_dir):
        """
        3x3 stress curves: rows heel/mid/toe, columns tension/compression/shear.
        New = dark blue mean + light blue ±std band; worn = dark red + light red.
        Table points (n_pts_table) are white circles with black edges.
        """
        if self.n_materials != 6:
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
                self.stretch_ten,
                self.stress_ten,
                self.stress_ten_std,
                self.stretch_ten_table,
                self.stress_ten_table,
                False,
            ),
            (
                "Compression",
                self.stretch_com,
                self.stress_com,
                self.stress_com_std,
                self.stretch_com_table,
                self.stress_com_table,
                True,
            ),
            (
                "Shear",
                self.strain_shr,
                self.stress_shr,
                self.stress_shr_std,
                self.strain_shr_table,
                self.stress_shr_table,
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
                    ax.set_title(mode_name, fontsize=self.FONT_SIZE)
                if col == 0:
                    ax.set_ylabel(f"{region_name}\nStress [kPa]", fontsize=self.FONT_SIZE - 4)
                if row == 2:
                    xlabel = "Shear Strain [-]" if mode_name == "Shear" else "Stretch [-]"
                    ax.set_xlabel(xlabel, fontsize=self.FONT_SIZE - 4)
                if mode_name == "Shear":
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                ax.tick_params(labelsize=self.FONT_SIZE - 6)
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
            fontsize=self.FONT_SIZE - 4,
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


    def plot_transverse_stretch_region_mode_grid(self, output_dir):
        """
        3x2 transverse-stretch curves: rows heel/mid/toe, columns tension/compression.
        Same new/worn styling as the stress region/mode grid.
        """
        if self.n_materials != 6:
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
                self.stretch_ten,
                self.transverse_stretch_ten,
                self.transverse_stretch_ten_std,
                self.stretch_ten_table,
                self.transverse_stretch_ten_table,
                False,
                True,
            ),
            (
                "Compression",
                self.stretch_com,
                self.transverse_stretch_com,
                self.transverse_stretch_com_std,
                self.stretch_com_table,
                self.transverse_stretch_com_table,
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
                    ax.set_title(mode_name, fontsize=self.FONT_SIZE)
                if col == 0:
                    ax.set_ylabel(f"{region_name}\nTransverse Stretch [-]", fontsize=self.FONT_SIZE - 4)
                if row == 2:
                    ax.set_xlabel("Axial Stretch [-]", fontsize=self.FONT_SIZE - 4)
                ax.tick_params(labelsize=self.FONT_SIZE - 6)
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
            fontsize=self.FONT_SIZE - 4,
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
    def plot_stiffness_energy_return_bars(self, output_dir, error="std"):
        """
        2x3 bar figure: stiffness (row 0) and energy return (row 1)
        vs tension | compression | shear, with 6 bars ordered as
        New/Worn Heel, Mid, Toe (blue / red shade by region).
        """
        if self.n_materials != 6:
            print("Skipping stiffness/energy-return bar plot (requires 6 worn-shoe materials).")
            return

        # heel / mid / toe × new (blue, hue 240°) then worn (red, hue 0°);
        # lightness 30% / 50% / 80% for dark / mid / light (saturation 100%)
        bar_specs = [
            (2, "New Heel", hsl_to_rgb(240, 100, 30)),   # dark blue
            (5, "Worn Heel", hsl_to_rgb(0, 100, 30)),    # dark red
            (1, "New Mid", hsl_to_rgb(240, 100, 50)),    # mid blue
            (4, "Worn Mid", hsl_to_rgb(0, 100, 50)),     # mid red
            (0, "New Toe", hsl_to_rgb(240, 100, 80)),    # light blue
            (3, "Worn Toe", hsl_to_rgb(0, 100, 80)),     # light red
        ]
        material_indices = [spec[0] for spec in bar_specs]
        labels = [spec[1] for spec in bar_specs]
        colors_bar = [spec[2] for spec in bar_specs]
        x = np.arange(len(bar_specs))

        stiffness_means = [
            np.asarray(self.stiffness_ten, dtype=float)[material_indices],
            np.asarray(self.stiffness_com, dtype=float)[material_indices],
            np.asarray(self.stiffness_shear, dtype=float)[material_indices],
        ]
        stiffness_stds = [
            np.asarray(self.stiffness_ten_std, dtype=float)[material_indices],
            np.asarray(self.stiffness_com_std, dtype=float)[material_indices],
            np.asarray(self.stiffness_shear_std, dtype=float)[material_indices],
        ]
        # Tables store population std; SEM needs n. Energy return supplies sample counts for SEM.
        if error == "sem":
            for mode_idx, samples in enumerate(self.energy_return_mode_samples):
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
            ax_stiff.set_title(mode_title, fontsize=self.FONT_SIZE)
            if col == 0:
                ax_stiff.set_ylabel(f"Stiffness [kPa]\n({err_label})", fontsize=self.FONT_SIZE - 4)
            ax_stiff.tick_params(labelsize=self.FONT_SIZE - 6, labelbottom=False)
            ax_stiff.set_xticks(x)
            ax_stiff.grid(axis="y", alpha=0.35)

            er_means, er_errs = bar_series_mean_err(
                self.energy_return_mode_samples[col], material_indices, error=error
            )
            ax_er = axes[1, col]
            ax_er.bar(
                x,
                er_means * 100.0, # Convert to pct
                yerr=er_errs * 100.0, # convert to pct
                color=colors_bar,
                edgecolor="black",
                linewidth=0.6,
                capsize=4,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
            )
            if col == 0:
                ax_er.set_ylabel(f"Energy return [%]\n({err_label})", fontsize=self.FONT_SIZE - 4)
            ax_er.set_ylim((50.0, 80.0) if col == 2 else (80.0, 100.0))
            ax_er.set_xticks(x)
            ax_er.set_xticklabels([])
            ax_er.tick_params(labelsize=self.FONT_SIZE - 6)
            ax_er.grid(axis="y", alpha=0.35)

        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=self.FONT_SIZE - 4,
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


    def save_stress_tables(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for mat in range(self.n_materials):
            foam_name = self.foam_types[mat]
            table_stem = f"{foam_name}_stress_table"
            tbl = self.build_stress_table_latex(mat)
            table_dir = table_output_dir(output_dir, "stress_tables")
            table_path = os.path.join(table_dir, f"{table_stem}.tex")
            with open(table_path, "w") as fh:
                fh.write(tbl)
            print(f"Stress table saved to: {table_path}\n")

        self.plot_stiffness_energy_return_bars(output_dir, error="std")


    def save_stress_excel(self, excel_dir=None, excel_filename="WornFoamData.xlsx"):
        """Write combined tension/compression/shear columns to Excel (MATLAB-style layout)."""

        if excel_dir is None:
            excel_dir = self.out_dir

        stretch_ut = np.vstack([np.flipud(self.stretch_com), self.stretch_ten[1:, :]])
        stress_ut = np.vstack([np.flipud(self.stress_com), self.stress_ten[1:, :]])
        stress_ut_std = np.vstack([np.flipud(self.stress_com_std), self.stress_ten_std[1:, :]])
        transverse_stretch_ut = np.vstack([np.flipud(self.transverse_stretch_com), self.transverse_stretch_ten[1:, :]])
        transverse_stretch_ut_std = np.vstack(
            [np.flipud(self.transverse_stretch_com_std), self.transverse_stretch_ten_std[1:, :]]
        )

        strain_ss = np.vstack([-np.flipud(self.strain_shr), self.strain_shr[1:, :]])
        stress_ss = np.vstack([-np.flipud(self.stress_shr), self.stress_shr[1:, :]])
        stress_ss_std = np.vstack([np.flipud(self.stress_shr_std), self.stress_shr_std[1:, :]])

        data_cols = []
        headings = []
        for i, foam in enumerate(self.foam_types):
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


    def build_stress_table_latex(self, mat):
        ten_stretch = self.stretch_ten_table
        ten_stress = self.stress_ten_table[:, mat]
        ten_std = self.stress_ten_std_table[:, mat]
        ten_trans = self.transverse_stretch_ten_table[:, mat]
        ten_trans_std = self.transverse_stretch_ten_std_table[:, mat]
        com_stretch = self.stretch_com_table[::-1]
        com_stress = -self.stress_com_table[::-1, mat]
        com_std = self.stress_com_std_table[::-1, mat]
        com_trans = self.transverse_stretch_com_table[::-1, mat]
        com_trans_std = self.transverse_stretch_com_std_table[::-1, mat]
        shr_strain = self.strain_shr_table
        shr_stress = self.stress_shr_table[:, mat]
        shr_std = self.stress_shr_std_table[:, mat]

        E_ten = self.stiffness_ten[mat]
        E_ten_std = self.stiffness_ten_std[mat]
        E_com = self.stiffness_com[mat]
        E_com_std = self.stiffness_com_std[mat]
        G_shr = self.stiffness_shear[mat]
        G_shr_std = self.stiffness_shear_std[mat]
        nu_ten = self.poissons_ratio_ten[mat]
        nu_ten_std = self.poissons_ratio_ten_std[mat]
        nu_com = self.poissons_ratio_com[mat]
        nu_com_std = self.poissons_ratio_com_std[mat]
        energy_return_ten = np.mean(self.energy_return_ten_samples[mat]) * 100.0
        energy_return_ten_std = np.std(self.energy_return_ten_samples[mat], ddof=0) * 100.0
        energy_return_com = np.mean(self.energy_return_com_samples[mat]) * 100.0
        energy_return_com_std = np.std(self.energy_return_com_samples[mat], ddof=0) * 100.0
        energy_return_shr = np.mean(self.energy_return_shear_samples[mat]) * 100.0
        energy_return_shr_std = np.std(self.energy_return_shear_samples[mat], ddof=0) * 100.0

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

        for i in range(self.n_pts_table):
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



    ## Plotting methods
    def plot_stress(
        self,
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
            ax.plot(x[:, foam_idx], stress[:, foam_idx], self.colors[foam_idx] + self.linestyles[foam_idx], label=f"{self.foam_types_title[foam_idx]}")
            ax.plot(x_table, stress_table[:, foam_idx], self.colors[foam_idx] + "o", markersize=4)
            if show_error_bars:
                ax.fill_between(
                    x[:, foam_idx],
                    stress[:, foam_idx] - stress_std[:, foam_idx],
                    stress[:, foam_idx] + stress_std[:, foam_idx],
                    color=self.colors[foam_idx],
                    alpha=0.25,
                )
        if mode in (LoadingMode.COMPRESSION, LoadingMode.CONFINED_COMPRESSION):
            ax.invert_xaxis()
            # ax.invert_yaxis()
        xlabel = "Shear Strain [-]" if mode == LoadingMode.SHEAR else "Stretch [-]"
        ylabel = "Shear Stress [kPa]" if mode == LoadingMode.SHEAR else "Stress [kPa]"
        ax.set_xlabel(xlabel, fontsize=self.FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=self.FONT_SIZE)
        ax.set_title(title, fontsize=self.FONT_SIZE)
        if mode == LoadingMode.SHEAR:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=self.FONT_SIZE)
        if mode == LoadingMode.SHEAR:
            ax.legend(fontsize=self.FONT_SIZE, bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True)
        plt.tight_layout()
        save_figure(fig, output_dir, filename)
        plt.close(fig)


    def plot_transverse_stretch(
        self,
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
                self.colors[foam_idx] + self.linestyles[foam_idx],
                label=f"{self.foam_types_title[foam_idx]}",
            )
            if show_error_bars:
                ax.fill_between(
                    axial_stretch[:, foam_idx],
                    transverse_stretch[:, foam_idx] - transverse_stretch_std[:, foam_idx],
                    transverse_stretch[:, foam_idx] + transverse_stretch_std[:, foam_idx],
                    color=self.colors[foam_idx],
                    alpha=0.25,
                )
        if mode == LoadingMode.COMPRESSION:
            ax.invert_xaxis()
        if mode == LoadingMode.TENSION:
            ax.invert_yaxis()
        ax.set_xlabel("Axial Stretch [-]", fontsize=self.FONT_SIZE)
        ax.set_ylabel("Transverse Stretch [-]", fontsize=self.FONT_SIZE)
        ax.set_title(title, fontsize=self.FONT_SIZE)
        ax.tick_params(labelsize=self.FONT_SIZE)
        if mode == LoadingMode.COMPRESSION:
            ax.legend(fontsize=self.FONT_SIZE, bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True)
        plt.tight_layout()
        save_figure(fig, output_dir, filename)
        plt.close(fig)


    def plot_individual_samples(
        self,
        foam_idx,
        individual_samples_tension,
        individual_samples_compression,
        individual_samples_shear,
        individual_samples_conf_compression,
        output_dir,
    ):
        """5x4 subplot figure: individual tension/compression/shear/confined samples for one material."""
        fig, axes = plt.subplots(5, 4, figsize=(14, 16))
        fig.suptitle(f"{self.foam_types_title[foam_idx]} - Individual Samples", fontsize=self.FONT_SIZE, fontweight="bold")

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
        if not self.worn_shoe:
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
                ax.plot(sample_data["stretch"], sample_data["stress"], self.colors[foam_idx], linewidth=1.5)
            ax.set_xlabel("Stretch [-]", fontsize=self.FONT_SIZE)
            ax.set_ylabel("Stress [kPa]", fontsize=self.FONT_SIZE)
            ax.set_title(f"Tension \n Sample {sample_idx + 1}", fontsize=self.FONT_SIZE)
            ax.tick_params(labelsize=self.FONT_SIZE)
            ax.grid(True, alpha=0.3)
            if ten_x_max > ten_x_min and ten_y_max > ten_y_min:
                ax.set_xlim(1.0, 1.3)
                ax.set_ylim(ten_y_min, ten_y_max)

        for sample_idx in range(5):
            ax = axes[sample_idx, 1]
            if sample_idx < len(individual_samples_compression[foam_idx]):
                sample_data = individual_samples_compression[foam_idx][sample_idx]
                ax.plot(sample_data["stretch"], sample_data["stress"], self.colors[foam_idx], linewidth=1.5)
            ax.set_xlabel("Stretch [-]", fontsize=self.FONT_SIZE)
            ax.set_ylabel("Stress [kPa]", fontsize=self.FONT_SIZE)
            ax.set_title(f"Compression \n Sample {sample_idx + 1}", fontsize=self.FONT_SIZE)
            ax.tick_params(labelsize=self.FONT_SIZE)
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
                ax.plot(sample_data["strain"], sample_data["stress"], self.colors[foam_idx], linewidth=1.5)
            ax.set_xlabel("Shear Strain [-]", fontsize=self.FONT_SIZE)
            ax.set_ylabel("Stress [kPa]", fontsize=self.FONT_SIZE)
            ax.set_title(f"Shear \n Sample {sample_idx + 1}", fontsize=self.FONT_SIZE)
            ax.tick_params(labelsize=self.FONT_SIZE)
            ax.grid(True, alpha=0.3)
            if shr_x_max > shr_x_min and shr_y_max > shr_y_min:
                ax.set_xlim(shr_x_min, shr_x_max)
                ax.set_ylim(shr_y_min, shr_y_max)

        if not self.worn_shoe:
            for sample_idx in range(5):
                ax = axes[sample_idx, 3]
                if sample_idx < len(individual_samples_conf_compression[foam_idx]):
                    sample_data = individual_samples_conf_compression[foam_idx][sample_idx]
                    ax.plot(sample_data["stretch"], sample_data["stress"], self.colors[foam_idx], linewidth=1.5)
                ax.set_xlabel("Stretch [-]", fontsize=self.FONT_SIZE)
                ax.set_ylabel("Stress [kPa]", fontsize=self.FONT_SIZE)
                ax.set_title(f"Confined Compression\nSample {sample_idx + 1}", fontsize=self.FONT_SIZE)
                ax.tick_params(labelsize=self.FONT_SIZE)
                ax.grid(True, alpha=0.3)
                if conf_com_x_max > conf_com_x_min and conf_com_y_max > conf_com_y_min:
                    ax.set_xlim(conf_com_x_min, conf_com_x_max)
                    ax.set_ylim(conf_com_y_min, conf_com_y_max)
                ax.invert_xaxis()
                ax.invert_yaxis()

        plt.tight_layout()
        filename = f"{self.foam_types[foam_idx]}_individual_samples.pdf"
        save_figure(fig, output_dir, filename)
        plt.close(fig)
