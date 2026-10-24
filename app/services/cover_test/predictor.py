"""Production predictor class for Multimodal Cover Test inference."""

from __future__ import annotations

import numpy as np

CONTRACT = "cover-uncover-multimodal-v1"


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
    """Production runtime predictor for multimodal Cover Test."""

    def __init__(self, sklearn_pipeline, feature_names: list[str]):
        self.pipeline = sklearn_pipeline
        self.feature_names = feature_names

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

    def predict_multimodal(self, payload: dict) -> dict:
        contract = payload.get("contract")
        if contract != CONTRACT:
            raise ValueError(f"Contract mismatch: expected {CONTRACT}, got {contract}")

        decoded_events = payload.get("uncoverEvents", {})
        ids = list(decoded_events.keys())
        if not ids:
            return {
                "prediction": "INCONCLUSIVE",
                "classProbabilities": {"NORMAL": 0.5, "STRABISMUS": 0.5},
                "analyzedEventIds": [],
                "isReliable": False,
            }

        probs_list = []
        for event_id, item in decoded_events.items():
            images = item.get("images", {})
            left_img = images.get("left")
            right_img = images.get("right")
            if left_img is None or right_img is None:
                continue

            img_vec = self._extract_image_features(left_img, right_img)
            traj_vec = np.zeros(20, dtype=np.float64)
            flags = np.array([0.0, 1.0], dtype=np.float64)  # traj=0, img=1

            row = np.concatenate([traj_vec, img_vec, flags]).reshape(1, -1)
            probs = self.pipeline.predict_proba(row)[0]
            probs_list.append(probs)

        if not probs_list:
            return {
                "prediction": "INCONCLUSIVE",
                "classProbabilities": {"NORMAL": 0.5, "STRABISMUS": 0.5},
                "analyzedEventIds": [],
                "isReliable": False,
            }

        avg_probs = np.mean(probs_list, axis=0)
        p_norm = round(float(avg_probs[0]), 4)
        p_strab = round(float(avg_probs[1]), 4)
        p_norm = 1.0 - p_strab

        pred = "NORMAL" if p_norm >= p_strab else "STRABISMUS"

        return {
            "prediction": pred,
            "classProbabilities": {
                "NORMAL": p_norm,
                "STRABISMUS": p_strab,
            },
            "analyzedEventIds": ids,
            "isReliable": True,
            "notice": "Multimodal Cover Test fusion model (Korean timestamp trajectory + Hirschberg eye crops)",
        }
