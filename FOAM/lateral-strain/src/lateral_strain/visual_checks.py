"""Save PNG montages, segment overlays, preview video, and matplotlib plots."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageFont

from lateral_strain.ingest import IngestConfig, iter_sampled_frames
from lateral_strain.segment import (
    segment_sample,
    _seg_cfg
)


_ANN_COLOR = (0, 0, 255)
_LABEL_TEXT_COLOR = (0, 0, 0)
_LABEL_BG_ALPHA = 0.6
_LABEL_PAD = 10
_LABEL_FONT_PX = 60
_DEJAVU_FONT_PATH = fm.findfont(fm.FontProperties(family="DejaVu Sans"))


@lru_cache(maxsize=1)
def _label_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_DEJAVU_FONT_PATH, _LABEL_FONT_PX)


def _text_bbox(text: str) -> tuple[int, int, int, int, ImageFont.FreeTypeFont]:
    font = _label_font()
    left, top, right, bottom = font.getbbox(text)
    return left, top, right - left, bottom - top, font


def _draw_text(
    img: np.ndarray,
    text: str,
    x: int,
    y_top: int,
    *,
    with_bg: bool = False,
) -> None:
    left, top, tw, th, font = _text_bbox(text)
    x_draw = x - left
    y_draw = y_top - top
    pad = _LABEL_PAD
    if with_bg:
        x0 = max(0, x_draw - pad)
        y0 = max(0, y_draw - pad)
        x1 = min(img.shape[1], x_draw + tw + pad)
        y1 = min(img.shape[0], y_draw + th + pad)
        roi = img[y0:y1, x0:x1]
        white_bg = np.full_like(roi, 255)
        cv2.addWeighted(white_bg, _LABEL_BG_ALPHA, roi, 1.0 - _LABEL_BG_ALPHA, 0, roi)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil_img).text((x_draw, y_draw), text, font=font, fill=_LABEL_TEXT_COLOR[::-1])
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y_baseline: int,
) -> None:
    _, top, _, th, _ = _text_bbox(text)
    y_top = y_baseline - th - top
    _draw_text(img, text, x, y_top, with_bg=True)


def _label_panel(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    _, top, _, th, _ = _text_bbox(title)
    y_top = out.shape[0] - th - top - 10
    _draw_text(out, title, 10, y_top, with_bg=True)
    return out


def _mask_to_bgr(mask: np.ndarray | None, invert: bool = False) -> np.ndarray:
    if mask is None:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    gray = 255 - mask if invert else mask
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    pos = 0.0
    draw = True
    while pos < dist:
        end = min(pos + (dash_len if draw else gap_len), dist)
        if draw:
            cv2.line(
                img,
                (int(x1 + dx * pos), int(y1 + dy * pos)),
                (int(x1 + dx * end), int(y1 + dy * end)),
                color,
                thickness,
                cv2.LINE_AA,
            )
        pos = end
        draw = not draw


def _annotate_intersection_final(
    mask: np.ndarray,
    contour: np.ndarray | None,
    height_px: float,
) -> np.ndarray:
    out = _mask_to_bgr(mask, invert=True)
    if contour is not None:
        x0, y0, w, h = cv2.boundingRect(contour)
        cv2.rectangle(out, (x0, y0), (x0 + w, y0 + h), _ANN_COLOR, 2)
        label = f"h={height_px:.0f}px"
        _draw_label(out, label, x0 + 24, y0 + h // 2)
    return _label_panel(out, "Final Contour")


def _perpendicular_width_endpoints(
    rect_fit: np.ndarray,
    width_px: float,
    angle: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Width line along the same direction used to measure width in segment.py."""
    bot_right, bot_left, top_left, top_right = rect_fit.reshape(-1, 2).astype(float)
    width_dir = np.array([np.cos(angle), -np.sin(angle)], dtype=float)
    left_mid = (bot_left + top_left) / 2
    right_mid = (bot_right + top_right) / 2
    if np.dot(width_dir, right_mid - left_mid) < 0:
        width_dir = -width_dir
    left_pt = left_mid
    right_pt = left_mid + width_dir * width_px
    return (float(left_pt[0]), float(left_pt[1])), (float(right_pt[0]), float(right_pt[1]))


def _annotate_parallelogram_width(
    overlay: np.ndarray,
    rect_fit: np.ndarray,
    width_px: float,
    angle: float,
) -> np.ndarray:
    out = overlay.copy()
    left_pt, right_pt = _perpendicular_width_endpoints(rect_fit, width_px, angle)
    _draw_dashed_line(out, left_pt, right_pt, _ANN_COLOR, thickness=2)
    label = f"w={width_px:.0f}px"
    label_x = int((left_pt[0] + right_pt[0]) / 2 - 150)
    label_y = int((left_pt[1] + right_pt[1]) / 2 - 5)
    _, top, _, th, _ = _text_bbox(label)
    _draw_text(out, label, label_x, label_y - th - top)
    return out


def save_ingest_montage(
    video_path: Path,
    out_png: Path,
    ingest_cfg: IngestConfig | None = None,
    max_cells: int = 12,
) -> None:
    """Grid of first N sampled frames with timestamp labels."""
    ingest_cfg = ingest_cfg or IngestConfig()
    frames = list(iter_sampled_frames(video_path, ingest_cfg))[:max_cells]
    if not frames:
        raise RuntimeError(f"No frames sampled from {video_path}")
    n = len(frames)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    cell_h, cell_w = frames[0].image_bgr.shape[:2]
    thumb_h, thumb_w = 360, int(360 * cell_w / max(cell_h, 1))
    montage = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for i, sf in enumerate(frames):
        r, c = i // cols, i % cols
        im = cv2.resize(sf.image_bgr, (thumb_w, thumb_h))
        cv2.putText(
            im,
            f"t={sf.t_video_sec:.1f}s",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        montage[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = im
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), montage)

def save_segment_debug_rows(
    video_path: Path,
    out_dir: Path,
    is_comp: bool
) -> None:
    """Write comprehensive debug outputs for segmentation stages and candidates."""
    start_time = 1.0 if is_comp else 0.0 # Skip first 1 second of compression video since it is blurry and will result in less accurate angle calculation
    ingest_cfg = IngestConfig()
    seg_cfg = _seg_cfg(is_comp)
    out_mp4 = out_dir / "overlay.mp4"
    # all_frames = list(iter_sampled_frames(video_path, ingest_cfg))
    # indices = _pick_frame_indices(len(all_frames), 5)
    out_dir = out_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    first = True
    results = []
    angle = None
    for sf in iter_sampled_frames(video_path, ingest_cfg):
        if sf.t_video_sec < start_time:
            continue
        idx += 1
        res = segment_sample(sf, seg_cfg, angle)
        if seg_cfg.fixed_angle:
            angle = res.angle
        bgr = sf.image_bgr
        overlay_all = bgr.copy()
        overlay_best = bgr.copy()
        overlay_skew = bgr.copy()
        for j, c in enumerate(res.candidates):
            color = (0, 200, 255) if j > 0 else (0, 255, 0)
            thick = 1 if j > 0 else 2
            cv2.drawContours(overlay_all, [c], -1, color, thick)
        if res.contour is not None:
            cv2.drawContours(overlay_best, [res.contour], -1, (255, 0, 0), 2)
        cv2.drawContours(overlay_best, [res.rect_fit.astype(np.int32)], -1, (0, 0, 255), 2)
        if res.rect_fit is not None and res.width is not None and res.angle is not None:
            overlay_best = _annotate_parallelogram_width(
                overlay_best, res.rect_fit, float(res.width), float(res.angle)
            )
        og_white = _mask_to_bgr(res.original_white_mask, invert=True)
        p_original = _label_panel(og_white, "Threshold Pixels")
        p_eroded = _label_panel(_mask_to_bgr(res.eroded_mask, invert=True), "Erode")
        p_dilated = _label_panel(_mask_to_bgr(res.dilated_main_mask, invert=True), "Dilate")
        p_final = _annotate_intersection_final(res.mask, res.contour, float(res.height or 0))
        p_bgr = _label_panel(bgr, "Original Image")
        p_best = _label_panel(overlay_best, "Segmentation")

        row1 = np.hstack([p_bgr, p_original, p_eroded])
        row2 = np.hstack([p_dilated, p_final, p_best])
        row = np.vstack([row1, row2])
        path = out_dir / f"t_{sf.t_video_sec:.2f}s_idx_{sf.frame_index}.png"
        cv2.imwrite(str(path), row)

        cv2.imwrite(str(out_dir / f"mask_original.png"), _mask_to_bgr(res.original_white_mask))
        cv2.imwrite(str(out_dir / f"mask_eroded.png"), _mask_to_bgr(res.eroded_mask))
        cv2.imwrite(str(out_dir / f"mask_main_component.png"), _mask_to_bgr(res.main_component_mask))
        cv2.imwrite(str(out_dir / f"mask_dilated_main.png"), _mask_to_bgr(res.dilated_main_mask))
        cv2.imwrite(str(out_dir / f"mask_intersection_final.png"), _mask_to_bgr(res.mask))
        cv2.imwrite(str(out_dir / f"overlay_candidates.png"), overlay_all)
        cv2.imwrite(str(out_dir / f"overlay_best.png"), overlay_best)
        cv2.imwrite(str(out_dir / f"overlay_skew.png"), overlay_skew)

        areas = [float(cv2.contourArea(c)) for c in res.candidates]
        areas_text = ",".join(f"{a:.1f}" for a in areas)
        diag_txt = "\n".join(
            [
                f"video={sf.video_path.name}",
                f"t_sec={sf.t_video_sec:.4f}",
                f"frame_index={sf.frame_index}",
                f"candidate_count={len(res.candidates)}",
                f"candidate_areas_px={areas_text}",
                # f"skew_rect_iou={res.skew_rect_iou}",
                # f"skew_rect_params={res.skew_rect_params}",
            ]
        )
        (out_dir / f"diagnostics.txt").write_text(diag_txt, encoding="utf-8")

        h, w = overlay_best.shape[:2]
        if first:
            vw, vh = w, h
            fps = max(1.0, ingest_cfg.target_hz)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (vw, vh))
            first = False
        assert writer is not None
        writer.write(overlay_best)
        results.append(res)
    if writer is not None:
        writer.release()
    return results

# def measurements_to_csv(measurements: list[FrameMeasurement], path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     lines = ["t_sec,frame_index,width_px,height_px,angle_deg,ok\n"]
#     for m in measurements:
#         lines.append(
#             f"{m.t_video_sec:.6f},{m.frame_index},{m.width_px:.4f},{m.height_px:.4f},"
#             f"{m.angle_deg:.4f},{int(m.ok)}\n"
#         )
#     path.write_text("".join(lines), encoding="utf-8")


# def augmented_csv_with_mm(
#     measurements: list[FrameMeasurement],
#     path: Path,
#     post_cfg: PostprocessConfig,
# ) -> None:
#     t = np.array([m.t_video_sec for m in measurements])
#     w = np.array([m.width_px for m in measurements])
#     h = np.array([m.height_px for m in measurements])
#     ok = np.array([m.ok for m in measurements])
#     t0 = resolve_t0(t[ok], h[ok], post_cfg) if np.any(ok) else 0.0
#     cal = compute_calibration(t, w, h, t0, post_cfg)
#     wmm, hmm = apply_mm(w, h, cal.mm_per_px)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     lines = [
#         "t_sec,frame_index,width_px,height_px,angle_deg,ok,width_mm,height_mm,t0_sec,mm_per_px\n",
#     ]
#     mpp = cal.mm_per_px if cal.mm_per_px is not None else float("nan")
#     for i, m in enumerate(measurements):
#         lines.append(
#             f"{m.t_video_sec:.6f},{m.frame_index},{m.width_px:.4f},{m.height_px:.4f},"
#             f"{m.angle_deg:.4f},{int(m.ok)},{wmm[i]:.6f},{hmm[i]:.6f},{t0:.6f},{mpp:.8f}\n"
#         )
#     path.write_text("".join(lines), encoding="utf-8")
