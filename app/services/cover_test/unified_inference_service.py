"""Unified AI inference service for Cover Test.

Combines two complementary branches:
  Branch A - Legacy Cover Test trajectory features (does NOT consume image pixels)
              using cover_multimodal_model.joblib / MultimodalCoverPredictor
  Branch B - Hirschberg v0.6 geometry branch (OpenCV eye-crop geometry -> 35-dim
              sklearn pipeline) using hirschberg_v06_ensemble.joblib branch_v02_pipeline

Legacy fusion remains research-only. V3 adds strict source timing/crop gates,
an actual trained pixel-transfer research branch and an optional paired model
whose clinical promotion requires independently held-out paired data.

Class mapping (Hirschberg -> Cover Test):
  esotropia  -> STRABISMUS
  exotropia  -> STRABISMUS
  normal     -> NORMAL
"""
from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
import joblib
import numpy as np

# Ensure project root is in sys.path for joblib unpickling
_ROOT_STR = str(Path(__file__).resolve().parents[3])
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

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
        self._paired_artifact = None
        self._image_transfer = None
        transfer_path = Path(os.getenv("COVER_IMAGE_TRANSFER_MODEL_PATH", str(_PROJECT_ROOT / "models" / "cover_image_transfer_candidate.joblib")))
        if transfer_path.is_file():
            try:
                transfer = joblib.load(transfer_path)
                if transfer.get("contract") != "hirschberg-paired-pixel-transfer-research-v1" or transfer.get("featureVersion") != "hog64x32-rgb-source-rois-v1":
                    raise ValueError("Image transfer contract mismatch")
                self._image_transfer = transfer
            except Exception as exc:
                self._load_errors["image_transfer"] = str(exc)
        paired_path = os.getenv("COVER_PAIRED_MODEL_PATH")
        if paired_path:
            try:
                from .paired_crop_model import CONTRACT, FEATURE_VERSION, PairedCropPredictor
                artifact = joblib.load(paired_path)
                if artifact.get("pairedContract") != CONTRACT or artifact.get("featureVersion") != FEATURE_VERSION or not isinstance(artifact.get("model"), PairedCropPredictor):
                    raise ValueError("Paired artifact contract mismatch")
                self._paired_artifact = artifact
            except Exception as exc:
                self._load_errors["paired"] = str(exc)

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
            "version": "unified-v2.1.0",
            "paired_crop": {"loaded": self._paired_artifact is not None,
                            "configured": bool(os.getenv("COVER_PAIRED_MODEL_PATH")),
                            "validation": (self._paired_artifact or {}).get("validation"),
                            "error": self._load_errors.get("paired")},
            "image_transfer": {"loaded": self._image_transfer is not None,
                               "researchOnly": True, "error": self._load_errors.get("image_transfer")},
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
            if pred == "INCONCLUSIVE":
                return None
            probs = result.get("classProbabilities", {})
            if not isinstance(probs, dict) or not probs:
                return None
            analyzed = result.get("analyzedEventIds", [])
            input_modalities = result.get("inputModalities", [])
            if not analyzed and "time_series" not in input_modalities:
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
                "coverage": round(len(analyzed) / max(len(decoded_events), 1), 3) if decoded_events else 1.0,
                "notice": result.get("notice", ""),
                "inputModalities": input_modalities,
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
        is_reliable = False  # legacy fusion has no independent paired-crop validation

        all_analyzed = sorted({event for _, branch in active for event in branch.get("analyzedEventIds", [])})
        modalities = sorted({modality for _, branch in active for modality in branch.get("inputModalities", [])})
        if branch_b:
            modalities = sorted(set(modalities) | {"uncover_eye_images"})

        return {
            "status": "COMPLETED",
            "prediction": pred,
            "classProbabilities": fused,
            "confidence": conf,
            "isReliable": is_reliable,
            "inputModalities": modalities,
            "receivedModalities": ["time_series", "uncover_eye_images"],
            "analyzedEventIds": all_analyzed,
            "branchesUsed": [br["branch"] for _, br in active],
            "branchDetails": {"multimodal": branch_a, "hirschberg_geometry": branch_b},
            "model": {"name": "cover-test-unified", "version": "unified-v2.1.0", "loaded": True},
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
            "inputModalities": [],
            "receivedModalities": ["time_series", "uncover_eye_images"],
            "analyzedEventIds": [], "receivedEventIds": event_ids or [],
            "branchesUsed": [],
            "branchDetails": {"multimodal": branch_a, "hirschberg_geometry": branch_b},
            "model": {"name": "cover-test-unified", "version": "unified-v2.1.0", "loaded": True},
            "notice": f"Unified inference inconclusive: {reason}.",
        }

    def _predict_legacy(
        self,
        trajectories: Dict[int, Dict[str, Any]],
        decoded_events: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run both branches and fuse results. Supports trajectory fallback when no uncover images exist."""
        if not decoded_events:
            if not trajectories or not any(c.get("samples") for c in trajectories.values()):
                return self._inconclusive("NO_TRAJECTORY_DATA", [])
            # Trajectory fallback via Branch A when no uncover images were captured
            branch_a = self._run_branch_a(trajectories, {})
            if branch_a and branch_a.get("prediction") in ("NORMAL", "STRABISMUS"):
                pred = branch_a["prediction"]
                conf = branch_a["confidence"]
                return {
                    "status": "COMPLETED",
                    "prediction": pred,
                    "classProbabilities": branch_a["classProbabilities"],
                    "confidence": conf,
                    "isReliable": False,
                    "inputModalities": ["time_series"],
                    "receivedModalities": ["time_series"],
                    "analyzedEventIds": [],
                    "branchesUsed": ["multimodal_trajectory_disparity"],
                    "branchDetails": {"multimodal": branch_a, "hirschberg_geometry": None},
                    "model": {"name": "cover-test-unified", "version": "unified-v2.1.0-trajectory", "loaded": True},
                    "notice": (
                        "Trajectory disparity inference (no uncover images captured). "
                        "RF_D3 model, LOPO BalAcc~73%, CI=[50%,94%]. Research only, not clinical diagnosis."
                    ),
                }
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

    def predict(self, trajectories, decoded_events):
        from .paired_crop_model import assess_timing, validate_source_event
        quality, rejected = {}, {}
        for key, cycle in trajectories.items():
            try:
                quality[str(key)] = assess_timing(cycle.get("samples", []))
            except (ValueError, TypeError) as exc:
                quality[str(key)] = {"eligible30fps": False, "reason": str(exc)}
        for event_id, item in decoded_events.items():
            try:
                event = item["event"]
                cycle = trajectories.get(event["cycle"], trajectories.get(str(event["cycle"]), {}))
                validate_source_event(event, cycle.get("samples", []))
                if not quality.get(str(event["cycle"]), {}).get("eligible30fps"):
                    raise ValueError("PROCESSED_CADENCE_BELOW_30FPS_POLICY")
            except (ValueError, KeyError, TypeError) as exc:
                rejected[event_id] = str(exc)
        suitable = bool(quality) and all(q["eligible30fps"] for q in quality.values())
        if rejected or (trajectories and not suitable):
            result = self._inconclusive("CROP_OR_TIMESTAMP_QUALITY_FAILED", list(decoded_events))
            result["rejectedEvents"] = rejected
        elif os.getenv("COVER_PAIRED_MODEL_PATH"):
            if self._paired_artifact is None:
                result = self._inconclusive("PAIRED_MODEL_LOAD_FAILED", list(decoded_events))
            else:
                result = self._paired_artifact["model"].predict_multimodal({
                    "contract": "cover-uncover-multimodal-v1", "cycles": trajectories,
                    "uncoverEvents": decoded_events})
                result["status"] = "INCONCLUSIVE" if result["prediction"] == "INCONCLUSIVE" else "COMPLETED"
                result["model"] = {"name": "cover-paired-crops", "version": self._paired_artifact["version"], "loaded": True}
        else:
            result = self._predict_legacy(trajectories, decoded_events)
            result["accuracy"] = None
            result["validationStatus"] = "NO_INDEPENDENT_PAIRED_CROP_VALIDATION"
        if decoded_events and not rejected and suitable and self._image_transfer is not None:
            from .paired_crop_model import crop_features
            evidence, failures = {}, {}
            for event_id, item in decoded_events.items():
                try:
                    vector = np.r_[crop_features(item["images"]["left"]), crop_features(item["images"]["right"])]
                    pipeline = self._image_transfer["pipeline"]
                    proba = pipeline.predict_proba(vector.reshape(1, -1))[0]
                    classes = list(pipeline.classes_)
                    evidence[event_id] = {"NORMAL": float(proba[classes.index(0)]),
                                          "STRABISMUS": float(proba[classes.index(1)])}
                except (ValueError, TypeError, KeyError) as exc:
                    failures[event_id] = str(exc)
            result["imageTransferResearch"] = {
                "status": "RESEARCH_ONLY", "eventProbabilities": evidence,
                "analyzedEventIds": list(evidence), "rejectedEvents": failures,
                "sourceEvaluation": self._image_transfer["validation"].get("evaluation"),
                "coverTestAccuracy": None, "promoted": False,
                "notice": "Static Hirschberg transfer scores. Source identity QC pending; not used to claim validated Cover Test accuracy or to change fusion weights."}
        result.setdefault("accuracy", None)
        result["timingQuality"] = quality
        result["deviceSuitable30fps"] = suitable
        return result


@lru_cache(maxsize=1)
def get_unified_inference_service() -> UnifiedCoverInferenceService:
    return UnifiedCoverInferenceService()
