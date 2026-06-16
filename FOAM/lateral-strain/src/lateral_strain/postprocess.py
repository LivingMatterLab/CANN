"""Experiment start (t0) and pixel-to-mm calibration."""

from __future__ import annotations

import numpy as np
from lateral_strain.segment import SegmentResult
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import shutil


def plot_series_px(
    results: list[SegmentResult],
    out_dir: Path
) -> None:
    t = np.array([x.time for x in results])
    h = np.array([x.height for x in results])
    w = np.array([x.width for x in results])
    out_csv = out_dir / f"data_px.csv"
    out_png = out_dir / f"plot_px.png"
    out_pdf = out_dir / f"plot_px.pdf"
    plot_series(
        t, w, h, out_png, out_csv, "px",
        show_title=False,
        legend_loc="upper left",
        figsize=(7, 5),
        font_scale=1.5,
        capitalize_labels=True,
        pad_for_legend=True,
        out_pdf=out_pdf,
    )

def plot_series_mm(t: np.ndarray, w_mm: np.ndarray, h_mm: np.ndarray, 
        out_dir: Path) -> None:
    out_csv = out_dir / f"data_mm.csv"
    out_png = out_dir / f"plot_mm.png"
    plot_series(t, w_mm, h_mm, out_png, out_csv, "mm", True)

def _axis_limits_with_padding(
    values: np.ndarray,
    *,
    top_frac: float = 0.2,
    bottom_frac: float = 0.05,
) -> tuple[float, float]:
    vmin, vmax = float(np.min(values)), float(np.max(values))
    span = max(vmax - vmin, 1e-6)
    return vmin - span * bottom_frac, vmax + span * top_frac


def plot_series(
    t: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    out_png: Path,
    out_csv: Path,
    units: str = "",
    qc: bool = False,
    *,
    show_title: bool = True,
    legend_loc: str = "best",
    figsize: tuple[float, float] = (10, 5),
    font_scale: float = 1.0,
    capitalize_labels: bool = False,
    pad_for_legend: bool = False,
    out_pdf: Path | None = None,
) -> None:
    qc_reject = False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    time_label = "Time (s)" if capitalize_labels else "time (s)"
    width_label = f"Width [{units}]" if capitalize_labels else f"width [{units}]"
    height_label = f"Height [{units}]" if capitalize_labels else f"height [{units}]"
    width_legend = "Width" if capitalize_labels else "width"
    height_legend = "Height" if capitalize_labels else "height"

    base_font = plt.rcParams["font.size"] * font_scale
    with plt.rc_context(
        {
            "font.size": base_font,
            "axes.labelsize": base_font,
            "axes.titlesize": base_font,
            "xtick.labelsize": base_font,
            "ytick.labelsize": base_font,
            "legend.fontsize": base_font,
        }
    ):
        fig, ax = plt.subplots(figsize=figsize)
        secax = ax.twinx()
        ax.plot(t, w, label=width_legend, color="r")
        secax.plot(t, h, label=height_legend, color="b")
        ax.set_xlabel(time_label)
        ax.set_ylabel(width_label)
        secax.set_ylabel(height_label)
        if show_title:
            ax.set_title(f"Data {units}")

        if pad_for_legend:
            ax.set_ylim(_axis_limits_with_padding(w, top_frac=0.25, bottom_frac=0.05))
            secax.set_ylim(_axis_limits_with_padding(h, top_frac=0.25, bottom_frac=0.05))
            t_span = max(float(np.max(t) - np.min(t)), 1e-6)
            ax.set_xlim(float(np.min(t)), float(np.max(t)) + t_span * 0.03)

        if qc:
            w_noise = np.mean((w[0:-1] - w[1:])**2)
            w_signal = np.var(w)
            w_noise_ratio = w_noise / w_signal
            ax.text(
                0.05, 0.95, f"NR: {w_noise_ratio:.2f}",
                transform=ax.transAxes, verticalalignment="top", horizontalalignment="left",
            )
            if w_noise_ratio > 0.03:
                qc_reject = True

        lines = ax.get_lines() + secax.get_lines()
        labels = [line.get_label() for line in lines]
        if pad_for_legend:
            ax.legend(lines, labels, loc=legend_loc)
        else:
            fig.legend(lines, labels, loc=legend_loc)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        if out_pdf is not None:
            fig.savefig(out_pdf)
        plt.close(fig)



        

    ## Save to CSV
    with open(out_csv, "w") as f:
        f.write("time,width,height\n")
        for i in range(len(t)):
            f.write(f"{t[i]},{w[i]},{h[i]}\n")
    if qc_reject:
        ## Delete out_csv
        out_csv.unlink()
        return

def postprocess(out: Path, w0_mm: float, is_comp: bool):
    data_path = out / "data_px.csv"
    data = pd.read_csv(data_path)
    t = data["time"].to_numpy()
    w = data["width"].to_numpy()
    h = data["height"].to_numpy()
    t, w_mm, h_mm = scale_shift_wh(t, w, h, w0_mm, is_comp)
    plot_series_mm(t, w_mm, h_mm, out)

from scipy.signal import find_peaks

def find_t0(t: np.ndarray, w: np.ndarray, h: np.ndarray, is_comp: bool) -> float:
    ## Ignore w to not make any assumptions about poisson ratio
    min_prominence = h[0] * 0.1
    peaks_idx, _ = find_peaks((np.max(h)-h) if is_comp else h, distance=50, height=0, prominence=min_prominence)
    peak_width = t[peaks_idx[1]] - t[peaks_idx[0]]
    return t[peaks_idx[0]] - peak_width/2


def scale_shift_wh(t: np.ndarray, w: np.ndarray, h: np.ndarray, w0_mm:float, is_comp: bool):
    t0 = find_t0(t, w, h, is_comp)
    t_max = t0 + (35.0 if is_comp else 15.0)
    start_idx = np.argmax(t * (t < t0))
    end_idx = np.argmax(t * (t < t_max)) + 2

    t_shift = t[start_idx:end_idx] - t0
    w_shift = w[start_idx:end_idx]
    h_shift = h[start_idx:end_idx]
    
    scale_factor = w0_mm / w_shift[0]
    w_shift = w_shift * scale_factor
    h_shift = h_shift * scale_factor
    return t_shift, w_shift, h_shift

def export(input_dir: Path, output_dir: Path) -> None:
    ## 
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in input_dir.glob("*"):
        if folder.is_dir():
            csv_file = folder / "data_mm.csv"
            dst = output_dir / f"{folder.name}.csv"
            if csv_file.exists():
                shutil.copyfile(csv_file, dst)

                