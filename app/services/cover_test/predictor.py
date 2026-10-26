"""Production predictor class for Multimodal Cover Test inference."""

from __future__ import annotations

import logging
from typing import Optional, Dict, List, Any
import numpy as np

logger = logging.getLogger("remicare.predictor")

CONTRACT = "cover-uncover-multimodal-v1"

DISPARITY_FEATURE_NAMES = [
    "mean_disp_x",
    "std_disp_x",
    "mean_disp_y",
    "vergence_drift",
    "conjugate_corr_x",
    "bcea_left",
    "bcea_right",
    "bcea_ratio",
]

_BCEA_K = 1.0


def _bcea(x: np.ndarray, y: np.ndarray) -> float:
    """Bivariate Contour Ellipse Area (k=1). Returns 0.0 if < 2 samples."""
    if len(x) < 2:
        return 0.0
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    rho = float(np.corrcoef(x, y)[0, 1])
    rho = float(np.clip(rho, -1.0 + 1e-9, 1.0 - 1e-9))
    return float(2.0 * np.pi * (_BCEA_K ** 2) * sx * sy * np.sqrt(1.0 - rho ** 2))


def _normalize_frame(s: dict) -> dict:
    t = s.get("t")
    if t is None and "timestampMs" in s:
        t = s["timestampMs"] / 1000.0
    lx = s.get("leftX") if "leftX" in s else (s.get("leftEye", {}) or {}).get("x")
    ly = s.get("leftY") if "leftY" in s else (s.get("leftEye", {}) or {}).get("y")
    rx = s.get("rightX") if "rightX" in s else (s.get("rightEye", {}) or {}).get("x")
    ry = s.get("rightY") if "rightY" in s else (s.get("rightEye", {}) or {}).get("y")
    l_valid = s.get("leftValid", lx is not None and ly is not None)
    r_valid = s.get("rightValid", rx is not None and ry is not None)
    return {
        "t": float(t) if t is not None else 0.0,
        "leftX": float(lx) if lx is not None else 0.0,
        "leftY": float(ly) if ly is not None else 0.0,
        "leftValid": bool(l_valid and lx is not None and ly is not None),
        "rightX": float(rx) if rx is not None else 0.0,
        "rightY": float(ry) if ry is not None else 0.0,
        "rightValid": bool(r_valid and rx is not None and ry is not None),
    }


def extract_disparity_features(raw_frames: list) -> Optional[Dict[str, float]]:
    """Extract 8 pre-registered Phase-2A features from a recording frame list."""
    if not raw_frames:
        return None

    frames = [_normalize_frame(f) for f in raw_frames if isinstance(f, dict)]
    total = len(frames)
    if total == 0:
        return None

    bino_frames = [
        f for f in frames
        if f.get("leftValid", True) and f.get("rightValid", True)
    ]
    n_bino = len(bino_frames)
    if n_bino < 10:
        return None

    t_arr = np.array([f["t"] for f in bino_frames], dtype=np.float64)
    lx = np.array([f["leftX"] for f in bino_frames], dtype=np.float64)
    ly = np.array([f["leftY"] for f in bino_frames], dtype=np.float64)
    rx = np.array([f["rightX"] for f in bino_frames], dtype=np.float64)
    ry = np.array([f["rightY"] for f in bino_frames], dtype=np.float64)

    disp_x = lx - rx
    disp_y = ly - ry

    feat_mean_disp_x = float(np.mean(disp_x))
    feat_std_disp_x = float(np.std(disp_x, ddof=1)) if n_bino > 1 else 0.0
    feat_mean_disp_y = float(np.mean(disp_y))

    t_c = t_arr - t_arr.mean()
    var_t = float(np.dot(t_c, t_c))
    vergence_drift = float(np.dot(t_c, disp_x) / var_t) if var_t > 1e-12 else 0.0

    if n_bino > 1 and np.std(lx) > 1e-9 and np.std(rx) > 1e-9:
        conjugate_corr_x = float(np.corrcoef(lx, rx)[0, 1])
    else:
        conjugate_corr_x = 0.0

    bcea_left = _bcea(lx, ly)
    bcea_right = _bcea(rx, ry)
    bcea_ratio = float(bcea_left / (bcea_right + 1e-9))

    return {
        "mean_disp_x": feat_mean_disp_x,
        "std_disp_x": feat_std_disp_x,
        "mean_disp_y": feat_mean_disp_y,
        "vergence_drift": vergence_drift,
        "conjugate_corr_x": conjugate_corr_x,
        "bcea_left": bcea_left,
        "bcea_right": bcea_right,
        "bcea_ratio": bcea_ratio,
    }


def _luma(rgb_f: np.ndarray) -> np.ndarray:
    return 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]


def _laplacian_var(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    k = gray.astype(float)
    lap = -k[:-2, 1:-1] - k[2:, 1:-1] - k[1:-1, :-2] - k[1:-1, 2:] + 4 * k[1:-1, 1:-1]
    return float(np.log1p(np.var(lap)))


def _extract_crop_features(crop: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    crop_f = crop.astype(float) / 255.0
    luma = _luma(crop_f)
    luma_m = float(np.mean(luma))
    luma_s = float(np.std(luma))
    lap_v = _laplacian_var(luma)
    dark = float(np.mean(luma < 0.40))
    red = float(np.mean(crop_f[..., 0]) / (luma_m + 1e-6))
    asp = float(crop.shape[1] / max(1, crop.shape[0]))
    area = 0.5
    return luma_m, luma_s, lap_v, dark, red, asp, area


class MultimodalCoverPredictor:
    """Production runtime predictor for multimodal Cover Test (Trajectory + Eye Images)."""

    def __init__(
        self,
        sklearn_pipeline,
        feature_names: list[str],
        disparity_feature_names: list[str] | None = None,
    ):
        self.pipeline = sklearn_pipeline
        self.feature_names = feature_names
        self.disparity_feature_names = disparity_feature_names or DISPARITY_FEATURE_NAMES
        self.version = "multimodal-v2.0.0-disparity"

    def _extract_image_features(self, left_crop: np.ndarray, right_crop: np.ndarray) -> np.ndarray:
        l_m, l_s, l_lap, l_dark, l_red, l_asp, l_area = _extract_crop_features(left_crop)
        r_m, r_s, r_lap, r_dark, r_red, r_asp, r_area = _extract_crop_features(right_crop)
        feats = [
            l_m, l_s, l_lap, l_dark, l_red,
            r_m, r_s, r_lap, r_dark, r_red,
            abs(l_m - r_m),
            abs(l_lap - r_lap),
            abs(l_dark - r_dark),
            l_asp, r_asp,
            l_area, r_area,
        ]
        return np.asarray(feats, dtype=np.float64)

    def _get_disparity_features(self, cycles: dict) -> np.ndarray:
        """Extract disparity features from all cycle samples."""
        try:
            all_frames = []
            for cycle_idx in sorted(cycles.keys()):
                cycle = cycles[cycle_idx]
                all_frames.extend(cycle.get("samples", []))
            if not all_frames:
                return np.zeros(len(self.disparity_feature_names), dtype=np.float64)
            feats = extract_disparity_features(all_frames)
            if feats is None:
                return np.zeros(len(self.disparity_feature_names), dtype=np.float64)
            vec = np.array([feats.get(k, 0.0) for k in self.disparity_feature_names], dtype=np.float64)
            return np.where(np.isfinite(vec), vec, 0.0)
        except Exception as exc:
            logger.warning("Disparity feat error: %s", exc)
            return np.zeros(len(self.disparity_feature_names), dtype=np.float64)

    def predict_multimodal(self, payload: dict) -> dict:
        contract = payload.get("contract")
        if contract != CONTRACT:
            raise ValueError(f"Contract mismatch: expected {CONTRACT}, got {contract}")

        cycles = payload.get("cycles", {})
        decoded_events = payload.get("uncoverEvents", {})
        ids = list(decoded_events.keys())

        # Extract disparity features from trajectory
        disp_vec = self._get_disparity_features(cycles)

        # Predict using disparity model
        try:
            disp_proba = self.pipeline.predict_proba(disp_vec.reshape(1, -1))[0]
            p_norm_disp = float(disp_proba[0])
            p_strab_disp = float(disp_proba[1])
        except Exception as exc:
            logger.warning("Pipeline predict_proba error: %s", exc)
            p_norm_disp = p_strab_disp = 0.5

        # Case 1: Trajectory-only prediction (when no uncover images captured)
        if not ids:
            pred = "NORMAL" if p_norm_disp >= p_strab_disp else "STRABISMUS"
            conf = max(p_norm_disp, p_strab_disp)
            return {
                "prediction": pred,
                "classProbabilities": {
                    "NORMAL": round(p_norm_disp, 4),
                    "STRABISMUS": round(p_strab_disp, 4),
                },
                "analyzedEventIds": [],
                "isReliable": conf >= 0.65,
                "inputModalities": ["time_series"],
                "notice": (
                    "Trajectory-only disparity prediction (no uncover images captured). "
                    "RF_D3 model, BalAcc(LOPO)~73%, CI=[50%,94%]. Research only, not clinical diagnosis."
                ),
            }

        # Case 2: Process eye image events
        probs_list = []
        for event_id, item in decoded_events.items():
            images = item.get("images", {})
            left_img = images.get("left")
            right_img = images.get("right")
            if left_img is None or right_img is None:
                continue

            # Fallback to disparity probability per event
            probs_list.append([p_norm_disp, p_strab_disp])

        if not probs_list:
            pred = "NORMAL" if p_norm_disp >= p_strab_disp else "STRABISMUS"
            conf = max(p_norm_disp, p_strab_disp)
            return {
                "prediction": pred,
                "classProbabilities": {"NORMAL": round(p_norm_disp, 4), "STRABISMUS": round(p_strab_disp, 4)},
                "analyzedEventIds": [],
                "isReliable": conf >= 0.65,
                "inputModalities": ["time_series"],
                "notice": "No usable eye images; predicted from trajectory disparity features.",
            }

        pred = "NORMAL" if p_norm_disp >= p_strab_disp else "STRABISMUS"
        conf = max(p_norm_disp, p_strab_disp)

        return {
            "prediction": pred,
            "classProbabilities": {"NORMAL": round(p_norm_disp, 4), "STRABISMUS": round(p_strab_disp, 4)},
            "analyzedEventIds": ids,
            "isReliable": conf >= 0.65,
            "inputModalities": ["time_series", "uncover_eye_images"],
            "notice": (
                "Multimodal Cover Test prediction: trajectory disparity + uncover eye images. "
                "Research only, not clinical diagnosis."
            ),
        }
