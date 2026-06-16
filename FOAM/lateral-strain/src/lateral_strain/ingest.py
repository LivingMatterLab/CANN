"""Video open, time-based sampling (e.g. 1 Hz), and file discovery."""
## Input = directory
## Output = np array with images at given frame rate

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np

VIDEO_EXTENSIONS = (".mov", ".MOV", ".mp4", ".MP4", ".avi", ".AVI")


class SampledFrame(NamedTuple):
    video_path: Path
    video_id: str
    t_video_sec: float
    frame_index: int
    image_bgr: np.ndarray


@dataclass
class IngestConfig:
    """Frame sampling: target_hz=1 means approximately one frame per second of video time."""

    target_hz: float = 30.0
    max_seconds: float = 60.0 ## 60 for compression, 30 for tension

## Find all videos with appropriate extension
def discover_videos(input_dir: Path) -> list[Path]:
    """Return sorted video paths under input_dir (non-recursive)."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        if input_dir.suffix in VIDEO_EXTENSIONS: 
            return [input_dir]
        else:
            raise FileNotFoundError(f"Not a directory or video: {input_dir}")
    paths_recursive = [discover_videos(p) for p in input_dir.iterdir() if p.is_dir()]
    
    paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix in VIDEO_EXTENSIONS]
    paths = paths + [p for subdir in paths_recursive for p in subdir]
    return sorted(paths, key=lambda p: p.name.lower())

## Open video with open cv
def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap

## Get frame rate
def video_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3 or np.isnan(fps):
        return 30.0
    return float(fps)

## Iterator that returns frames at given sample rate
def iter_sampled_frames(
    video_path: Path,
    config: IngestConfig | None = None,
) -> Iterator[SampledFrame]:
    """
    Yield frames at ~target_hz samples per second of source timeline.
    Uses time-based stepping: next sample at t >= prev_t + 1/target_hz.
    """
    cfg = config or IngestConfig()
    cap = open_video(video_path)
    try:
        fps = video_fps(cap)
        dt = 1.0 / cfg.target_hz
        video_id = video_path.stem
        max_t = cfg.max_seconds

        frame_index = 0
        next_sample_t = 0.0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = frame_index / fps


            if max_t is not None and t > max_t + 1e-6:
                break
            # print(t)
            # print(max_t)
            ## Crop frame to be only center third of image
            sz = frame.shape
            # frame = frame[:, (sz[1] // 3):(2 * sz[1] // 3)]

            if t + 1e-9 >= next_sample_t:
                yield SampledFrame(
                    video_path=video_path,
                    video_id=video_id,
                    t_video_sec=t,
                    frame_index=frame_index,
                    image_bgr=frame,
                )
                next_sample_t += dt

            frame_index += 1
    finally:
        cap.release()
