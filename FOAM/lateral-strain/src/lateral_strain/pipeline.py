"""Batch run: all videos in a folder → CSV, plots, overlay videos."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lateral_strain.ingest import discover_videos
from lateral_strain.postprocess import postprocess, plot_series_px
from lateral_strain.visual_checks import (
    save_segment_debug_rows,
)
import pandas as pd


def run_folder(
    input_dir: Path,
    out: Path,
    measurements_dir: Path,
    is_comp: bool
) -> None:
    vids = discover_videos(input_dir)
    print(vids)
    for vp in vids:
        # try:
        run_file(vp, measurements_dir, out, is_comp)
        # except:
        #     print("error occurred")    

def run_file(vp:Path, measurements_dir:Path, out:Path, is_comp:bool):
    if not vp.is_file():
            raise SystemExit(f"Video not found: {vp}")
    name = vp.stem
    print(name)
    ## Load w0_mm
    root = name[0:-2].replace("-", "_")
    measurements_path = measurements_dir / f"{root}_measurements.csv"
    meas = pd.read_csv(measurements_path, header=0).values
    widths_mm = np.mean(meas[:, 0:3], axis=1)
    w0_mm = widths_mm[int(name[-1])-1]
    results = save_segment_debug_rows(vp, out / name, is_comp)
    plot_series_px(results, out / name)
    postprocess(out / name, w0_mm, is_comp)
