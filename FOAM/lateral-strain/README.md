# lateral-strain

Analyze lab videos to track a white rectangular sample: width/height vs. time, optional mm calibration and experiment start (`t0`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Montage: verify 1 Hz frame sampling
python -m lateral_strain ingest --input-dir input/compression/new --output output/debug

# Segment debug PNGs
python -m lateral_strain segment --input-dir input/compression/new --output output/debug

# Full pipeline (CSV + plots + overlay video)
python -m lateral_strain run --input-dir input/compression/new --output output

# Visual checks (ingest + segment + measure plots for one video)
python -m lateral_strain visual --video input/compression/new/new_comp_1.mov --output output/debug
```

Tuning: adjust LAB/HSV thresholds and ROI in `segment.py` or via CLI `--roi-fraction`, `--l-min`, etc.

## Tests

```bash
pytest
```

Synthetic geometry tests only; real videos are validated via `output/debug/` images.
