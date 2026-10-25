"""Unified AI inference service for Cover Test.

Combines two complementary branches:
  Branch A - Cover Test Multimodal (trajectory features + raw pixel features)
              using cover_multimodal_model.joblib / MultimodalCoverPredictor
  Branch B - Hirschberg v0.6 geometry branch (OpenCV eye-crop geometry -> 35-dim
              sklearn pipeline) using hirschberg_v06_ensemble.joblib branch_v02_pipeline

Fusion: weighted confidence vote. Falls back to whichever single branch succeeds.

Class mapping (Hirschberg -> Cover Test):
  esotropia  -> STRABISMUS
  exotropia  -> STRABISMUS
  normal     -> NORMAL
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger("remicare.unified_inference")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MULTIMODAL_PATH = _PROJECT_ROOT / "models" / "cover_multimodal_model.joblib"
_HIRSCHBERG_PATH = _PROJECT_ROOT / "models" / "hirschberg_v06_ensemble.joblib"

WEIGHT_MULTIMODAL = 0.35
WEIGHT_HIRSCHBERG = 0.65
MIN_BRANCH_CONFIDENCE = 0.40

_HIRSCHBERG_CLASS_MAP: Dict[str, str] = {
    "esotropia": "STRABISMUS",
    "exotropia": "STRABISMUS",
    "normal": "NORMAL",
}

GEO_ORDER = [
    "dx_left_mm", "dy_left_mm", "dx_right_mm", "dy_right_mm",
    "abs_dx_left_mm", "abs_dx_right_mm", "symmetry_deviation_mm",
    "vertical_asymmetry_mm", "scleral_ratio_left", "scleral_ratio_right",
    "scleral_ratio_min", "scleral_ratio_diff", "intercanthal_distance_mm",
    "is_likely_pseudostrabismus_flag", "blur_variance",
    "head_pitch_abs", "head_yaw_abs", "head_roll_abs", "quality_acceptable_flag",
]


def _pixel_stats(rgb: np.ndarray) -> List[float]:
    """8 pixel statistics for one eye crop (RGB numpy array)."""
    import cv2
    f = rgb.astype(float) / 255.0
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    luma_m = float(np.mean(luma))
    luma_s = float(np.std(luma))
    dark = float(np.mean(luma < 0.40))
    red = float(np.mean(f[..., 0]) / (luma_m + 1e-6))
    if luma.shape[0] >= 3 and luma.shape[1] >= 3:
        k = luma
        lap = -k[:-2, 1:-1] - k[2:, 1:-1] - k[1:-1, :-2] - k[1:-1, 2:] + 4 * k[1:-1, 1:-1]
        lap_v = float(np.log1p(np.var(lap)))
    else:
        lap_v = 0.0
    asp = float(rgb.shape[1] / max(1, rgb.shape[0]))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return [luma_m, luma_s, lap_v, dark, red, asp, blur_v, float(luma_m - luma_s)]


def _extract_geometry_features(
    left_rgb: np.ndarray, right_rgb: np.ndarray
) -> Optional[np.ndarray]:
    """Build 35-dim feature vector: 19 geo + 8 pixel-L + 8 pixel-R."""
    try:
        import cv2
        from app.services.cover_test.eye_crop_geometry_service import (
            extract_eye_crop_geometric_features,
        )

        target_h = 112

        def _resize(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            nw = max(1, int(w * target_h / max(h, 1)))
            return cv2.resize(img, (nw, target_h), interpolation=cv2.INTER_AREA)

        # Unmirrored camera: OD (patient right) appears on image LEFT half
        od_bgr = _resize(cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))  # patient right eye
        os_bgr = _resize(cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))   # patient left eye

        tw = od_bgr.shape[1] + os_bgr.shape[1]
        bilateral = np.zeros((target_h, tw, 3), dtype=np.uint8)
        bilateral[:, : od_bgr.shape[1]] = od_bgr
        bilateral[:, od_bgr.shape[1] :] = os_bgr

        geo_feats = extract_eye_crop_geometric_features(bilateral)
        geo_vec = [float(geo_feats.get(k) or 0.0) for k in GEO_ORDER]  # 19

        pix_l = _pixel_stats(left_rgb)   # 8
        pix_r = _pixel_stats(right_rgb)  # 8

        vec = np.array(geo_vec + pix_l + pix_r, dtype=np.float64)  # 35
        vec = np.where(np.isfinite(vec), vec, 0.0)
        return vec
    except Exception as exc:
        logger.warning("Geometry feature extraction failed: %s", exc)
        return None


class UnifiedCoverInferenceService:
    """Loads both model artifacts at startup and exposes a single predict()."""

    def __init__(
        self,
        multimodal_path: Optional[Path] = None,
        hirschberg_path: Optional[Path] = None,
    ) -> None:
        self._multimodal_artifact: Optional[Dict[str, Any]] = None
        self._hirschberg_branch_pipeline = None
        self._hirschberg_classes: List[str] = []
        self._load_errors: Dict[str, str] = {}
        self._load_multimodal(multimodal_path or _MULTIMODAL_PATH)
        self._load_hirschberg(hirschberg_path or _HIRSCHBERG_PATH)

    def _load_multimodal(self, path: Path) -> None:
        try:
            artifact = joblib.load(path)
            if artifact.get("contract") != "cover-uncover-multimodal-v1":
                raise ValueError("Contract mismatch")
            if not callable(getattr(artifact.get("model"), "predict_multimodal", None)):
                raise ValueError("Model missing predict_multimodal()")
            self._multimodal_artifact = artifact
            logger.info("[Unified] Multimodal model loaded: %s", artifact.get("version"))
        except Exception as exc:
            self._load_errors["multimodal"] = str(exc)
            logger.warning("[Unified] Multimodal not loaded: %s", exc)

    def _load_hirschberg(self, path: Path) -> None:
        try:
            artifact = joblib.load(path)
            pipeline = artifact.get("branch_v02_pipeline")
            if pipeline is None or not hasattr(pipeline, "predict_proba"):
                raise ValueError("branch_v02_pipeline missing")
            n_in = getattr(pipeline, "n_features_in_", None)
            if n_in is not None and int(n_in) != 35:
                raise ValueError(f"branch expects {n_in} features, not 35")
            self._hirschberg_branch_pipeline = pipeline
            self._hirschberg_classes = list(
                artifact.get("classes", ["esotropia", "exotropia", "normal"])
            )
            logger.info("[Unified] Hirschberg branch loaded, classes=%s", self._hirschberg_classes)
        except Exception as exc:
            self._load_errors["hirschberg"] = str(exc)
            logger.warning("[Unified] Hirschberg branch not loaded: %s", exc)

    def health(self) -> Dict[str, Any]:
        return {
            "loaded": self._multimodal_artifact is not None or self._hirschberg_branch_pipeline is not None,
            "version": "unified-v1.0.0",
            "multimodal": {
                "loaded": self._multimodal_artifact is not None,
                "version": (self._multimodal_artifact or {}).get("version"),
                "error": self._load_errors.get("multimodal"),
            },
            "hirschberg_branch": {
                "loaded": self._hirschberg_branch_pipeline is not None,
                "classes": self._hirschberg_classes,
                "error": self._load_errors.get("hirschberg"),
            },
        }

    def _run_branch_a(
        self, trajectories: Dict, decoded_events: Dict
    ) -> Optional[Dict[str, Any]]:
        if self._multimodal_artifact is None:
            return None
        try:
            result = self._multimodal_artifact["model"].predict_multimodal({
                "contract": "cover-uncover-multimodal-v1",
                "cycles": trajectories,
                "uncoverEvents": decoded_events,
            })
            pred = result.get("prediction", "INCONCLUSIVE")
            probs = result.get("classProbabilities", {})
            if not isinstance(probs, dict) or not probs:
                return None
            analyzed = result.get("analyzedEventIds", [])
            if not analyzed:
                return None
            conf = float(probs.get(pred, 0.0))
            return {
                "branch": "multimodal",
                "prediction": pred,
                "classProbabilities": {
                    "NORMAL": round(float(probs.get("NORMAL", 0.0)), 4),
                    "STRABISMUS": round(float(probs.get("STRABISMUS", 0.0)), 4),
                },
                "confidence": round(conf, 4),
                "analyzedEventIds": list(analyzed),
                "coverage": round(len(analyzed) / max(len(decoded_events), 1), 3),
                "notice": result.get("notice", ""),
            }
        except Exception as exc:
            logger.warning("[BranchA] Failed: %s", exc)
            return None

    def _run_branch_b(self, decoded_events: Dict) -> Optional[Dict[str, Any]]:
        if self._hirschberg_branch_pipeline is None:
            return None
        probs_list: List[np.ndarray] = []
        analyzed: List[str] = []
        for event_id, entry in decoded_events.items():
            images = entry.get("images", {})
            left_rgb = images.get("left")
            right_rgb = images.get("right")
            if left_rgb is None or right_rgb is None:
                continue
            vec = _extract_geometry_features(left_rgb, right_rgb)
            if vec is None:
                continue
            try:
                proba = self._hirschberg_branch_pipeline.predict_proba(vec.reshape(1, -1))[0]
                probs_list.append(proba)
                analyzed.append(event_id)
            except Exception as exc:
                logger.warning("[BranchB] event %s: %s", event_id, exc)

        if not probs_list:
            return None

        avg_probs = np.mean(probs_list, axis=0)
        binary: Dict[str, float] = {"NORMAL": 0.0, "STRABISMUS": 0.0}
        for i, cls in enumerate(self._hirschberg_classes):
            mapped = _HIRSCHBERG_CLASS_MAP.get(str(cls).lower(), "STRABISMUS")
            binary[mapped] += float(avg_probs[i])

        total = binary["NORMAL"] + binary["STRABISMUS"]
        if total > 0:
            binary = {k: round(v / total, 4) for k, v in binary.items()}

        pred = "NORMAL" if binary["NORMAL"] >= binary["STRABISMUS"] else "STRABISMUS"
        conf = float(binary[pred])
        return {
            "branch": "hirschberg_geometry",
            "prediction": pred,
            "classProbabilities": binary,
            "confidence": round(conf, 4),
            "analyzedEventIds": analyzed,
            "coverage": round(len(analyzed) / max(len(decoded_events), 1), 3),
            "notice": "Hirschberg v0.6 geometry branch (OpenCV iris/reflex features).",
        }

    def _fuse(
        self,
        branch_a: Optional[Dict],
        branch_b: Optional[Dict],
        decoded_events: Dict,
    ) -> Dict[str, Any]:
        active = []
        if branch_a:
            w = WEIGHT_MULTIMODAL if branch_a["confidence"] >= MIN_BRANCH_CONFIDENCE else WEIGHT_MULTIMODAL * 0.5
            active.append((w, branch_a))
        if branch_b:
            w = WEIGHT_HIRSCHBERG if branch_b["confidence"] >= MIN_BRANCH_CONFIDENCE else WEIGHT_HIRSCHBERG * 0.5
            active.append((w, branch_b))

        if not active:
            return self._inconclusive("BOTH_BRANCHES_FAILED", list(decoded_events.keys()), branch_a, branch_b)

        total_w = sum(w for w, _ in active)
        fused: Dict[str, float] = {"NORMAL": 0.0, "STRABISMUS": 0.0}
        for w, br in active:
            nw = w / total_w
            for cls in ("NORMAL", "STRABISMUS"):
                fused[cls] += nw * br["classProbabilities"].get(cls, 0.0)

        fused = {k: round(v, 4) for k, v in fused.items()}
        pred = "NORMAL" if fused["NORMAL"] >= fused["STRABISMUS"] else "STRABISMUS"
        conf = round(fused[pred], 4)
        is_reliable = conf >= 0.60 and len(active) == 2

        all_analyzed = list(decoded_events.keys())

        return {
            "status": "COMPLETED",
            "prediction": pred,
            "classProbabilities": fused,
            "confidence": conf,
            "isReliable": is_reliable,
            "inputModalities": ["time_series", "uncover_eye_images"],
            "receivedModalities": ["time_series", "uncover_eye_images"],
            "analyzedEventIds": all_analyzed,
            "branchesUsed": [br["branch"] for _, br in active],
            "branchDetails": {"multimodal": branch_a, "hirschberg_geometry": branch_b},
            "model": {"name": "cover-test-unified", "version": "unified-v1.0.0", "loaded": True},
            "notice": (
                "Unified Cover Test AI: trajectory + eye-crop geometry fusion. "
                "Research only, not a clinical diagnosis."
            ),
        }

    @staticmethod
    def _inconclusive(
        reason: str, event_ids: List[str],
        branch_a: Optional[Dict] = None, branch_b: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "INCONCLUSIVE", "prediction": "INCONCLUSIVE",
            "reason": reason, "classProbabilities": None, "confidence": None,
            "isReliable": False,
            "inputModalities": ["time_series", "uncover_eye_images"],
            "receivedModalities": ["time_series", "uncover_eye_images"],
            "analyzedEventIds": event_ids or [],
            "branchesUsed": [],
            "branchDetails": {"multimodal": branch_a, "hirschberg_geometry": branch_b},
            "model": {"name": "cover-test-unified", "version": "unified-v1.0.0", "loaded": True},
            "notice": f"Unified inference inconclusive: {reason}.",
        }

    def predict(
        self,
        trajectories: Dict[int, Dict[str, Any]],
        decoded_events: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run both branches and fuse results."""
        if not decoded_events:
            return self._inconclusive("NO_UNCOVER_EVENTS", [])
        branch_a = self._run_branch_a(trajectories, decoded_events)
        branch_b = self._run_branch_b(decoded_events)
        logger.info(
            "[Unified] A=%s(%.2f) B=%s(%.2f)",
            branch_a["prediction"] if branch_a else "FAIL",
            branch_a["confidence"] if branch_a else 0.0,
            branch_b["prediction"] if branch_b else "FAIL",
            branch_b["confidence"] if branch_b else 0.0,
        )
        return self._fuse(branch_a, branch_b, decoded_events)


@lru_cache(maxsize=1)
def get_unified_inference_service() -> UnifiedCoverInferenceService:
    return UnifiedCoverInferenceService()
