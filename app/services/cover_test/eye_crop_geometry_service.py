"""Eye Crop Geometry Extractor for Hirschberg Analysis.

Designed specifically for periocular / eye-crop images (e.g. 224x224 crops containing
OD and OS) where full-face landmarks (MediaPipe) fail.
Extracts:
- Iris center (x, y) & radius for OD and OS
- Corneal light reflex (Purkinje-I glint) (x, y)
- Corneal reflex displacement dx_mm, dy_mm
- Eye corners (inner / nasal canthus, outer / temporal canthus)
- Eye width (palpebral fissure width)
- Scleral ratio (nasal distance / temporal distance)
- Bilateral asymmetry (symmetry deviation, vertical asymmetry)
- Head roll angle from inter-ocular line
- Blur variance and quality flags
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def segment_iris_pupil(eye_bgr: np.ndarray) -> Tuple[float, float, float, str]:
    """Segment iris / pupil in a single eye crop (e.g. 112x224).
    
    Returns (cx, cy, radius, method_status).
    """
    h, w = eye_bgr.shape[:2]
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Dark component segmentation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    p15 = float(np.percentile(blurred, 15))
    dark_thresh = min(95.0, max(30.0, p15))
    dark_mask = (blurred <= dark_thresh).astype(np.uint8) * 255
    
    # Restrict to plausible eye region ROI (avoid outer boundary skin/eyebrow)
    roi = np.zeros_like(dark_mask)
    roi[int(h * 0.12):int(h * 0.88), int(w * 0.05):int(w * 0.95)] = 255
    dark_mask = cv2.bitwise_and(dark_mask, roi)
    
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dark_mask)
    candidates = []
    crop_center = np.array([w / 2.0, h / 2.0])
    
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 30 or area > 10000:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 3.0:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        dist = float(np.linalg.norm(np.array([cx, cy]) - crop_center))
        # Circularity estimate
        equiv_rad = math.sqrt(area / math.pi)
        score = dist - equiv_rad * 0.8
        candidates.append((score, cx, cy, equiv_rad * 1.15, "dark_component"))
        
    if candidates:
        _, cx, cy, rad, method = min(candidates, key=lambda c: c[0])
        # clamp radius
        rad = float(np.clip(rad, min(w, h) * 0.10, min(w, h) * 0.45))
        return cx, cy, rad, method
        
    # 2. Hough Circles fallback
    eq = cv2.equalizeHist(gray)
    hough_blur = cv2.medianBlur(eq, 5)
    circles = cv2.HoughCircles(
        hough_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, w // 3),
        param1=50,
        param2=20,
        minRadius=max(8, int(min(w, h) * 0.10)),
        maxRadius=max(15, int(min(w, h) * 0.45)),
    )
    if circles is not None and len(circles[0]) > 0:
        center = np.array([w / 2.0, h / 2.0])
        best = min(circles[0], key=lambda c: float(np.linalg.norm(np.array([c[0], c[1]]) - center)))
        return float(best[0]), float(best[1]), float(best[2]), "hough"
        
    # 3. Default centroid fallback
    inv = 255.0 - gray.astype(np.float32)
    yy, xx = np.indices(gray.shape)
    tot = float(np.sum(inv))
    if tot > 1e-5:
        cx = float(np.sum(xx * inv) / tot)
        cy = float(np.sum(yy * inv) / tot)
        return cx, cy, float(min(w, h) * 0.25), "dark_centroid"
        
    return w / 2.0, h / 2.0, float(min(w, h) * 0.25), "default"


def detect_corneal_reflex(
    eye_bgr: np.ndarray,
    iris_cx: float,
    iris_cy: float,
    iris_rad: float,
) -> Tuple[Optional[float], Optional[float], float, str]:
    """Detect corneal light reflection (glint) within the iris region.
    
    Returns (reflex_x, reflex_y, reflex_area, status).
    """
    h, w = eye_bgr.shape[:2]
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    
    # Search mask around iris center
    search_rad = max(10.0, iris_rad * 1.25)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(iris_cx)), int(round(iris_cy))), int(round(search_rad)), 255, -1)
    
    pixels = gray[mask > 0]
    if len(pixels) == 0:
        return None, None, 0.0, "MASK_EMPTY"
        
    peak = float(np.max(pixels))
    p90 = float(np.percentile(pixels, 90))
    
    # Specular reflections are noticeably brighter than the surrounding iris
    # In real flash photos, peak is usually >= 140
    if peak < 120.0:
        return None, None, 0.0, "LOW_PEAK_BRIGHTNESS"
        
    thresh = max(140.0, min(peak - 20.0, p90 + 10.0))
    _, binary = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, mask)
    
    num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates = []
    
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 1 or area > 180:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 3.5:
            continue
        rx, ry = float(centroids[i][0]), float(centroids[i][1])
        dist = math.hypot(rx - iris_cx, ry - iris_cy)
        if dist > search_rad:
            continue
        
        comp_mask = comp_labels == i
        comp_peak = float(np.max(gray[comp_mask])) if np.any(comp_mask) else peak
        # Score favors proximity to iris center and high brightness
        score = dist * 1.0 + area * 0.1 - (comp_peak - thresh) * 0.05
        candidates.append((score, rx, ry, float(area)))
        
    if not candidates:
        # Fallback to local maximum location inside the iris circle
        masked_gray = gray.copy()
        masked_gray[mask == 0] = 0
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(masked_gray)
        if max_v >= 135:
            return float(max_l[0]), float(max_l[1]), 2.0, "LOCAL_MAX_FALLBACK"
        return None, None, 0.0, "NO_REFLEX"
        
    candidates.sort(key=lambda c: c[0])
    _, rx, ry, area = candidates[0]
    return rx, ry, area, "DETECTED"


def detect_eye_corners(
    eye_bgr: np.ndarray,
    iris_cx: float,
    iris_cy: float,
    iris_rad: float,
    is_od: bool,
) -> Tuple[float, float, float, float, float, float]:
    """Detect inner and outer canthi, eye width, and scleral ratio.
    
    For OD (patient right eye, image left half):
      - Outer (temporal) canthus is on the LEFT (x < iris_cx)
      - Inner (nasal) canthus is on the RIGHT (x > iris_cx)
    For OS (patient left eye, image right half):
      - Inner (nasal) canthus is on the LEFT (x < iris_cx)
      - Outer (temporal) canthus is on the RIGHT (x > iris_cx)
      
    Returns (temporal_x, temporal_y, nasal_x, nasal_y, eye_width, scleral_ratio).
    """
    h, w = eye_bgr.shape[:2]
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    
    # Analyze horizontal profile around the eye axis
    cy_int = int(round(np.clip(iris_cy, 5, h - 6)))
    band = gray[max(0, cy_int - 6):min(h, cy_int + 7), :]
    profile = band.mean(axis=0)
    
    # Find boundary edges using gradient
    grad = np.gradient(profile)
    
    # Left corner search (x < iris_cx - iris_rad * 0.5)
    left_limit = int(max(2, iris_cx - iris_rad * 0.5))
    if left_limit > 5:
        left_corner_x = float(np.argmin(profile[:left_limit]))
        # Clamp to realistic anatomical range
        left_corner_x = float(np.clip(left_corner_x, 1.0, max(2.0, iris_cx - iris_rad * 0.8)))
    else:
        left_corner_x = float(max(1.0, iris_cx - iris_rad * 1.8))
        
    # Right corner search (x > iris_cx + iris_rad * 0.5)
    right_start = int(min(w - 3, iris_cx + iris_rad * 0.5))
    if right_start < w - 5:
        right_corner_x = float(right_start + np.argmin(profile[right_start:]))
        right_corner_x = float(np.clip(right_corner_x, min(w - 2.0, iris_cx + iris_rad * 0.8), w - 1.0))
    else:
        right_corner_x = float(min(w - 1.0, iris_cx + iris_rad * 1.8))
        
    eye_width = max(5.0, right_corner_x - left_corner_x)
    
    if is_od:
        # OD: left is temporal, right is nasal
        temporal_x = left_corner_x
        nasal_x = right_corner_x
        dist_temporal = max(1.0, iris_cx - temporal_x)
        dist_nasal = max(1.0, nasal_x - iris_cx)
    else:
        # OS: left is nasal, right is temporal
        nasal_x = left_corner_x
        temporal_x = right_corner_x
        dist_nasal = max(1.0, iris_cx - nasal_x)
        dist_temporal = max(1.0, temporal_x - iris_cx)
        
    scleral_ratio = float(np.clip(dist_nasal / dist_temporal, 0.1, 10.0))
    
    return temporal_x, iris_cy, nasal_x, iris_cy, eye_width, scleral_ratio


def extract_eye_crop_geometric_features(
    bgr_image: np.ndarray, iris_diameter_mm: Optional[float] = None
) -> Dict[str, Optional[float]]:
    """Extract complete 19-dimensional geometric feature vector from an eye crop image.
    
    The input image is assumed to contain both eyes (left half = OD, right half = OS).
    """
    h, w = bgr_image.shape[:2]
    half_w = w // 2
    od_bgr = bgr_image[:, :half_w]
    os_bgr = bgr_image[:, half_w:]
    
    # 1. Iris segmentation
    od_cx, od_cy, od_rad, od_iris_st = segment_iris_pupil(od_bgr)
    os_cx, os_cy, os_rad, os_iris_st = segment_iris_pupil(os_bgr)
    
    # OS coordinates in full-image reference frame
    os_cx_full = os_cx + half_w
    
    # 2. Corneal reflex detection
    od_rx, od_ry, od_area, od_refl_st = detect_corneal_reflex(od_bgr, od_cx, od_cy, od_rad)
    os_rx, os_ry, os_area, os_refl_st = detect_corneal_reflex(os_bgr, os_cx, os_cy, os_rad)
    
    # Millimeter outputs require a subject-specific measured iris diameter.
    od_rad_safe = max(2.0, od_rad)
    os_rad_safe = max(2.0, os_rad)
    if iris_diameter_mm is not None and iris_diameter_mm <= 0:
        raise ValueError("iris_diameter_mm must be a positive measured value")
    px_to_mm = iris_diameter_mm / (od_rad_safe + os_rad_safe) if iris_diameter_mm else None
    dx_right_mm = ((od_rx - od_cx) * px_to_mm) if od_rx is not None and px_to_mm else None
    dy_right_mm = ((od_ry - od_cy) * px_to_mm) if od_ry is not None and px_to_mm else None
    dx_left_mm = ((os_rx - os_cx) * px_to_mm) if os_rx is not None and px_to_mm else None
    dy_left_mm = ((os_ry - os_cy) * px_to_mm) if os_ry is not None and px_to_mm else None
    abs_dx_left_mm = abs(dx_left_mm) if dx_left_mm is not None else None
    abs_dx_right_mm = abs(dx_right_mm) if dx_right_mm is not None else None
    symmetry_deviation_mm = (
        abs(dx_left_mm + dx_right_mm)
        if dx_left_mm is not None and dx_right_mm is not None else None
    )
    vertical_asymmetry_mm = (
        abs(dy_left_mm - dy_right_mm)
        if dy_left_mm is not None and dy_right_mm is not None else None
    )
    
    # 4. Eye corners and scleral ratio
    _, _, od_nasal_x, _, od_width, scleral_ratio_right = detect_eye_corners(od_bgr, od_cx, od_cy, od_rad, is_od=True)
    _, _, os_nasal_x, _, os_width, scleral_ratio_left = detect_eye_corners(os_bgr, os_cx, os_cy, os_rad, is_od=False)
    
    scleral_ratio_min = min(scleral_ratio_left, scleral_ratio_right)
    scleral_ratio_diff = abs(scleral_ratio_left - scleral_ratio_right)
    
    # Intercanthal distance: distance between OD nasal corner and OS nasal corner
    # In OD crop, nasal corner x is near right edge (half_w)
    # In OS crop, nasal corner x is near left edge (0 + half_w)
    intercanthal_px = float(np.hypot((os_nasal_x + half_w) - od_nasal_x, os_cy - od_cy))
    intercanthal_distance_mm = intercanthal_px * px_to_mm if px_to_mm else None
    
    # 5. Pseudostrabismus indicator
    # Large intercanthal distance or epicanthal fold with low reflex displacement
    is_likely_pseudostrabismus_flag = None
    
    # 6. Quality & pose metrics
    gray_full = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())
    
    # Head roll from inter-ocular tilt
    delta_x = max(1.0, os_cx_full - od_cx)
    delta_y = os_cy - od_cy
    head_roll_deg = abs(math.atan2(delta_y, delta_x) * 180.0 / math.pi)
    
    # Head yaw estimate from relative eye width ratio
    yaw_ratio = max(od_width, os_width) / max(1.0, min(od_width, os_width))
    head_yaw_deg = float(np.clip((yaw_ratio - 1.0) * 45.0, 0.0, 60.0))
    head_pitch_deg = 0.0 # Eye crop has limited vertical face cues
    
    quality_acceptable_flag = 1.0 if blur_variance >= 15.0 and min(od_rad, os_rad) >= 5.0 else 0.0
    
    features = {
        "dx_left_mm": dx_left_mm,
        "dy_left_mm": dy_left_mm,
        "dx_right_mm": dx_right_mm,
        "dy_right_mm": dy_right_mm,
        "abs_dx_left_mm": abs_dx_left_mm,
        "abs_dx_right_mm": abs_dx_right_mm,
        "symmetry_deviation_mm": symmetry_deviation_mm,
        "vertical_asymmetry_mm": vertical_asymmetry_mm,
        "scleral_ratio_left": scleral_ratio_left,
        "scleral_ratio_right": scleral_ratio_right,
        "scleral_ratio_min": scleral_ratio_min,
        "scleral_ratio_diff": scleral_ratio_diff,
        "intercanthal_distance_mm": intercanthal_distance_mm,
        "is_likely_pseudostrabismus_flag": is_likely_pseudostrabismus_flag,
        "blur_variance": blur_variance,
        "head_pitch_abs": head_pitch_deg,
        "head_yaw_abs": head_yaw_deg,
        "head_roll_abs": head_roll_deg,
        "quality_acceptable_flag": quality_acceptable_flag,
    }
    return features


GEOMETRIC_FEATURE_NAMES = [
    "dx_left_mm",
    "dy_left_mm",
    "dx_right_mm",
    "dy_right_mm",
    "abs_dx_left_mm",
    "abs_dx_right_mm",
    "symmetry_deviation_mm",
    "vertical_asymmetry_mm",
    "scleral_ratio_left",
    "scleral_ratio_right",
    "scleral_ratio_min",
    "scleral_ratio_diff",
    "intercanthal_distance_mm",
    "is_likely_pseudostrabismus_flag",
    "blur_variance",
    "head_pitch_abs",
    "head_yaw_abs",
    "head_roll_abs",
    "quality_acceptable_flag",
]
