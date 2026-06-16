"""White-ish rectangular sample near frame center: mask + best contour."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from lateral_strain.ingest import SampledFrame


@dataclass
class SegmentConfig:
    """
    Tuning when mask leaks or misses the sample:
    - ab_max_delta: ...
    - morph_kernel: ...
    - erode_pixels: erosion iterations to suppress spurious white speckles.
    - dilate_pixels: dilation iterations after selecting main connected mass.
    """

    ab_max_delta: int = 25
    morph_kernel: int = 5 
    erode_pixels: int = 10 
    dilate_pixels: int = 20 
    l_min_left: int = 190
    l_min_right: int = 190
    fixed_angle: bool = False
    use_mean: bool = False
    preserve_corners: bool = False


@dataclass
class SegmentResult:
    mask: np.ndarray
    contour: np.ndarray | None
    candidates: list[np.ndarray] = field(default_factory=list)
    original_white_mask: np.ndarray | None = None
    eroded_mask: np.ndarray | None = None
    main_component_mask: np.ndarray | None = None
    dilated_main_mask: np.ndarray | None = None
    roi_mask: np.ndarray | None = None
    width: np.float | None = 0
    height: np.float | None = 0
    time: np.float | None = 0
    rect_fit: np.ndarray | None = None
    rect_bbox: np.ndarray | None = None
    angle: float | None = None
    # (x0, y0, x1, y1, w) maximizing IOU vs. clipped contour; None if no contour.
    # skew_rect_params: tuple[float, float, float, float, float] | None = None
    # skew_rect_iou: float | None = None

def _seg_cfg(is_compression: bool) -> SegmentConfig:
    if is_compression:
        return SegmentConfig(
            l_min_right=180, # was 190 for original videos
            l_min_left=180,
            ab_max_delta=50,
            erode_pixels=25,
            dilate_pixels=25,
            fixed_angle=True,
            use_mean=True,
            preserve_corners=True
        )
    else:
        return SegmentConfig()

def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep largest foreground component whose centroid is in middle third of image height."""
    ## Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask) # Return empty array if no connected components
    h = mask.shape[0]
    y_min = h / 3.0
    y_max = h#(2.0 * h) / 3.0
    eligible_labels: list[int] = []
    # Identify which components are centered in the ~middle third~ bottom two thirds vertically
    for label in range(1, num_labels):
        cy = float(centroids[label][1])
        if y_min <= cy <= y_max:
            eligible_labels.append(label)

    # If no eligible components, consider all components
    labels_for_selection = eligible_labels if eligible_labels else list(range(1, num_labels))
    # Get eligible label with largest area
    largest_label = max(labels_for_selection, key=lambda lb: int(stats[lb, cv2.CC_STAT_AREA]))
    # Return mask with all pixels in the largest connected component
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def _lab_white_mask(bgr: np.ndarray, cfg: SegmentConfig) -> dict[str, np.ndarray]:
    ## Convert to LAB color space 
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    mid_a, mid_b = 128, 128
    # Get white pixels
    neutral = np.logical_and(
        np.abs(a_ch.astype(np.int16) - mid_a) <= cfg.ab_max_delta,
        np.abs(b_ch.astype(np.int16) - mid_b) <= cfg.ab_max_delta,
    )
    left_mask = np.zeros_like(l_ch)
    w = l_ch.shape[1]
    left_mask[:, 0:int(w/2)] = 1

    bright_left = np.logical_and(l_ch >= cfg.l_min_left, left_mask > 0)
    bright_right = np.logical_and(l_ch >= cfg.l_min_right, left_mask == 0)
    bright = np.logical_or(bright_left, bright_right)#
    original_white_mask = (bright & neutral).astype(np.uint8) * 255

    ## Erode by number of pixels specified 
    m = original_white_mask.copy()
    if cfg.erode_pixels > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.erode(m, erode_kernel, iterations=cfg.erode_pixels)
    eroded_mask = m.copy()
    ## Keep only largest connected component
    m = _largest_connected_component(m)
    main_component_mask = m.copy()
    ## Dilate by specified number of pixels
    if cfg.dilate_pixels > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.dilate(m, dilate_kernel, iterations=cfg.dilate_pixels)
    dilated_main_mask = m.copy()
    # Keep only pixels that are both in the grown main mass and original white mask.
    m_bbox = cv2.boundingRect(m)
    if cfg.preserve_corners: 
        # For compression the sample gets very short, so the standard method of eroding and dilating results in an octogon instead of a rectangle
        # Cropping to the 
        m[m_bbox[1]:m_bbox[1]+m_bbox[3], m_bbox[0]:m_bbox[0]+m_bbox[2]] = 255
    m = cv2.bitwise_and(original_white_mask, m)
    k = max(3, cfg.morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return {
        "original_white_mask": original_white_mask,
        "eroded_mask": eroded_mask,
        "main_component_mask": main_component_mask,
        "dilated_main_mask": dilated_main_mask,
        "final_mask": m,
        "rect_bbox": rect_to_cont(m_bbox)
    }



def segment_sample(frame: SampledFrame, cfg: SegmentConfig | None = None, angle: float | None = None) -> SegmentResult:
    """
    Build mask and pick single best contour for the white sample.
    Contour coordinates are in full-frame space (ROI offset applied).
    """
    cfg = cfg or SegmentConfig()
    ## Get image and dimension
    bgr = frame.image_bgr
    h, w = bgr.shape[:2]
    ## Get largest connected component after eroding and dilating white pixels
    masks = _lab_white_mask(bgr, cfg)
    full_mask = masks["final_mask"]

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_idx = np.argmax(np.array([cv2.contourArea(c) for c in contours]))
    # if len(contours) > 1: 
    #     print("Too many contours found")
    best = contours[best_idx]

    bx0, by0, bw, bh = cv2.boundingRect(best)

    ## Find bottom of contour and permute
    start_idx = np.argmin(best[:, 0, 1]) # index of min y coord
    best_permuted = np.concatenate([best[start_idx:, :, :], best[0:start_idx, :, :]], axis=0)
    # Split into left and right wall based on index of top point
    top_idx = np.argmax(best_permuted[:, 0, 1])
    left_wall = best_permuted[0:top_idx, 0, :]
    right_wall = best_permuted[top_idx:, 0, :]
    right_wall = right_wall[::-1, :]
    # Crop each wall so only middle third / fifth is included (depending on tension vs compression)
    if angle is None:
        y_min = by0 + bh / 3
        y_max = by0 + 2 * bh / 3
    else: 
        y_min = by0 + 2 * bh / 5
        y_max = by0 + 3 * bh / 5

    # Crop to only include y coordinates between y_min and y_max (plus one on each side so we can interpolate successfully)
    start_idx = np.argmax(left_wall[:, 1] > y_min) - 1
    end_idx = np.argmax(left_wall[:, 1] > y_max) + 1
    left_wall_cropped = left_wall[start_idx:end_idx, :]
    start_idx = np.argmax(right_wall[:, 1] > y_min) - 1
    end_idx = np.argmax(right_wall[:, 1] > y_max) + 1
    right_wall_cropped = right_wall[start_idx:end_idx, :]

    # Sort each wall by y coordinate
    left_wall_sorted = left_wall_cropped[np.argsort(left_wall_cropped[:, 1]), :]
    right_wall_sorted = right_wall_cropped[np.argsort(right_wall_cropped[:, 1]), :]
 
    # Interpolate to standard y grid
    y_grid = np.linspace(y_min, y_max, 100)
    left_wall_interpolated_x = np.interp(y_grid, left_wall_sorted[:, 1], left_wall_sorted[:, 0])
    right_wall_interpolated_x = np.interp(y_grid, right_wall_sorted[:, 1], right_wall_sorted[:, 0])
    # Find average of left and right wall and fit a line to it
    centerline_x = (left_wall_interpolated_x + right_wall_interpolated_x) / 2
    coeffs = np.polyfit(y_grid, centerline_x, 1)
    # Find angle to rotate this line to be flat (positive angle means rotate CCW)
    angle = angle or np.atan(coeffs[0]) # if angle is provided use it instead of calculating it
    rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    # Create x, y array for interpolated wall
    left_wall_interpolated = np.column_stack((left_wall_interpolated_x, y_grid))
    right_wall_interpolated = np.column_stack((right_wall_interpolated_x, y_grid))
    # Rotate each wall by this angle
    left_wall_rotated = np.dot(left_wall_interpolated, rot_mat)
    right_wall_rotated = np.dot(right_wall_interpolated, rot_mat)
    # Get median x coordinate of each wall
    left_wall_avg_x = np.mean(left_wall_rotated[:, 0]) if cfg.use_mean else np.median(left_wall_rotated[:, 0])
    right_wall_avg_x = np.mean(right_wall_rotated[:, 0]) if cfg.use_mean else np.median(right_wall_rotated[:, 0])
    avg_width = right_wall_avg_x - left_wall_avg_x 
    # For debugging, transform line x = left_wall_median_x to orig coordinates
    # Want to solve problem (X, y_min) -> (left_wall_med_x, ~)
    # [X, y] * [A11, A12; A21, A22]  = [left_wall_med_x, ~]
    # X * A11 + y * A21 = left_wall_med_x
    # X = (left_wall_med_x - A21 * y_min) / A11

    # Return plottable bounding box / parallelogram
    x_bot_left = (left_wall_avg_x - rot_mat[1, 0] * y_min) / rot_mat[0, 0]
    x_bot_right = (right_wall_avg_x - rot_mat[1, 0] * y_min) / rot_mat[0, 0]
    x_top_left = (left_wall_avg_x - rot_mat[1, 0] * y_max) / rot_mat[0, 0]
    x_top_right = (right_wall_avg_x - rot_mat[1, 0] * y_max) / rot_mat[0, 0]
    rect_fit = np.reshape(np.array([[x_bot_right, y_min], [x_bot_left, y_min], [x_top_left, y_max], [x_top_right, y_max]]), shape=(-1, 1 ,2))

    return SegmentResult(
        mask=full_mask,
        contour=best,
        original_white_mask=masks["original_white_mask"],
        eroded_mask=masks["eroded_mask"],
        main_component_mask=masks["main_component_mask"],
        dilated_main_mask=masks["dilated_main_mask"],
        width=avg_width,
        height=bh,
        angle=angle,
        time=frame.t_video_sec,
        rect_fit=rect_fit,
        rect_bbox=masks["rect_bbox"]
    )

def rect_to_cont(rect: [int]):
    x, y, w, h = rect
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape((-1, 1, 2))
