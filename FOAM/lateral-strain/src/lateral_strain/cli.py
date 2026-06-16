"""CLI: ingest, segment, run, visual."""

from __future__ import annotations

## TODO: 
# 1. Document entire codebase so others can use including readme
# 2. Handle edge cases where error occurs in automated pipeline
# 3. Create final script to collect csvs into single excel file
# 4. Update requirements.txt
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from lateral_strain import postprocess
from lateral_strain.ingest import IngestConfig, discover_videos
from lateral_strain.pipeline import run_folder
from lateral_strain.postprocess import (export, postprocess, plot_series_px)
from lateral_strain.visual_checks import (
    save_ingest_montage,
    save_segment_debug_rows,
)


def _add_common_video_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-dir", type=Path, default=Path("input/compression/new"))
    p.add_argument("--output", type=Path, default=Path("output"))

def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = IngestConfig()
    out = Path(args.output) / "debug" / "ingest"
    for vp in discover_videos(Path(args.input_dir)):
        save_ingest_montage(vp, out / f"{vp.stem}_montage.png", cfg)


def cmd_segment(args: argparse.Namespace) -> None:
    for vp in discover_videos(Path(args.input_dir)):
        name = vp.stem
        results = save_segment_debug_rows(vp, args.output / name, args.compression)
        plot_series_px(results, args.output / name)

def cmd_run(args: argparse.Namespace) -> None:
    run_folder(
        Path(args.input_dir),
        Path(args.output), 
        Path(args.measurements_dir),
        args.compression
    )

def cmd_export(args: argparse.Namespace) -> None:
    export(
        Path(args.input_dir),
        Path(args.output_dir)
    )

def cmd_postproc(args: argparse.Namespace) -> None:
    vids = discover_videos(args.input_dir)
    for vp in vids:
        name = vp.stem
        print(name)
        ## Load w0_mm
        root = name[0:-2].replace("-", "_")
        measurements_path = args.measurements_dir / f"{root}_measurements.csv"
        meas = pd.read_csv(measurements_path, header=0).values
        widths_mm = np.mean(meas[:, 0:3], axis=1)
        w0_mm = widths_mm[int(name[-1])-1]
        try:
            postprocess(args.output / name, w0_mm, args.compression)
        except:
            print(f"Error processing {name}")

def cmd_run(args: argparse.Namespace) -> None:
    run_folder(
        Path(args.input_dir),
        Path(args.output), 
        Path(args.measurements_dir),
        args.compression
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lateral strain / sample tracking from experiment videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_in = sub.add_parser("ingest", help="Montage PNGs for 1 Hz (or custom) sampling.")
    _add_common_video_args(p_in)
    p_in.set_defaults(func=cmd_ingest)

    p_seg = sub.add_parser("segment", help="Mask + overlay debug rows for several frames per video.")
    _add_common_video_args(p_seg)
    p_seg.add_argument("-c", "--compression", action='store_true')
    p_seg.set_defaults(func=cmd_segment)

    p_run = sub.add_parser("run", help="Full pipeline for all videos in input dir.")
    _add_common_video_args(p_run)
    p_run.add_argument("--measurements_dir", type=Path, default=Path("input/measurements"))
    p_run.add_argument("-c", "--compression", action='store_true')
    p_run.set_defaults(func=cmd_run)

    p_postproc = sub.add_parser("postproc", help="Convert pixel measurements to mm and find t0.")

    _add_common_video_args(p_postproc)
    p_postproc.add_argument("--measurements_dir", type=Path, default=Path("input/measurements"))
    p_postproc.add_argument("-c", "--compression", action='store_true')
    p_postproc.set_defaults(func=cmd_postproc)

    p_export = sub.add_parser("export", help="Aggregate data and export to FoamData.csv")
    p_export.add_argument("--input-dir", type=Path, default=Path("output"))
    p_export.add_argument("--output-dir", type=Path, default=Path("output/data_mm"))
    p_export.set_defaults(func=cmd_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
