"""Cover Test numeric trajectory consensus service.

The previous Korean-transfer 10-15 FPS artifact has been retired from runtime.
Until a RemiCare-trained Cover Test model is available, this service preserves
the same interface and returns safe INCONCLUSIVE results when no approved model
artifact is loaded.
Features:
- Evaluates 14 canonical spatial stability & dispersion features
- Uses TemporalConsensusAggregator to run multi-frame consensus across temporal sliding windows
- Rejects blinks, occlusions, and momentary saccadic jitters
- Produces aggregated consensus prediction only when an approved artifact exists
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from app.services.cv.temporal_aggregator import (
    FrameInferenceCandidate,
    TemporalAggregationResult,
    TemporalConsensusAggregator,
)
from app.services.shared_feature_contract import (
    ALL_SHARED_FEATURES,
    extract_shared_features_vector,
)

logger = logging.getLogger("remicare.fps_model_service")

# Paths for a future RemiCare-trained Cover Test model artifact.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_15FPS_MODEL_PATH = _PROJECT_ROOT / "models" / "remicare_15fps_candidate.joblib"
FALLBACK_15FPS_MODEL_PATH = _PROJECT_ROOT.parent / "models" / "candidates" / "remicare_15fps_candidate.joblib"

DEFAULT_14_FEATURES = [
    "leftValidRatio",
    "rightValidRatio",
    "bothValidRatio",
    "meanDeltaY",
    "medianDeltaY",
    "stdDeltaY",
    "minDeltaY",
    "maxDeltaY",
    "rangeDeltaY",
    "meanAbsDeltaY",
    "stdLeftX",
    "stdLeftY",
    "stdRightX",
    "stdRightY",
]


@dataclass
class FpsModelResult:
    """Structured result of 10-15 FPS model inference after temporal aggregation."""
    prediction: str                    # 'NORMAL', 'STRABISMUS', 'INCONCLUSIVE'
    class_probabilities: Dict[str, float]  # {'NORMAL': 0.xx, 'STRABISMUS': 0.xx}
    confidence: float                  # [0.0, 1.0]
    model_name: str                    # 'remicare-fps-10-15'
    model_version: str                 # '10-15fps-v1.1.0'
    model_source: str                  # 'fps_10_15_model'
    features_snapshot: Dict[str, Any]  # Key features dictionary
    status: str                        # 'COMPLETED' or 'INCONCLUSIVE'
    agreement_ratio: float
    is_reliable: bool
    total_frames_evaluated: int
    valid_frames_count: int
    notice: str


class FeatureContractMismatchError(ValueError):
    """Raised when extracted features do not satisfy the loaded model contract."""


class FpsModelService:
    """Singleton service executing 10-15 FPS model inference with temporal consensus."""

    _instance: Optional["FpsModelService"] = None

    def __init__(self, model_path: Optional[str] = None):
        self.artifact: Optional[Dict[str, Any]] = None
        self.model = None
        self.feature_names: List[str] = DEFAULT_14_FEATURES
        self.model_name = "remicare-fps-10-15"
        self.model_version = "10-15fps-v1.1.0"
        self.aggregator = TemporalConsensusAggregator(
            min_valid_frames=3,
            min_consensus_ratio=0.60,
        )
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None) -> None:
        """Loads an approved Cover Test model artifact when available."""
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.append(PRIMARY_15FPS_MODEL_PATH)
        candidates.append(FALLBACK_15FPS_MODEL_PATH)

        loaded_file = None
        for p in candidates:
            if p.exists():
                loaded_file = p
                break

        if loaded_file is None:
            logger.warning("[FpsModelInit] No approved Cover Test model artifact found. Returning INCONCLUSIVE.")
            return

        try:
            self.artifact = joblib.load(loaded_file)
            self.model = self.artifact.get("model")
            if "feature_names" in self.artifact:
                self.feature_names = list(self.artifact["feature_names"])
            self.model_name = self.artifact.get("name", "remicare-fps-10-15")
            self.model_version = self.artifact.get("version", "10-15fps-v1.1.0")
            self._validate_loaded_model_contract()
            logger.info(
                "[FpsModelLoaded] Loaded 10-15 FPS candidate: %s (version: %s, features: %d) from %s",
                self.model_name,
                self.model_version,
                len(self.feature_names),
                loaded_file,
            )
        except Exception as e:
            logger.error("[FpsModelLoadError] Failed to load 10-15 FPS model from %s: %s", loaded_file, e)
            self.model = None

    def _validate_loaded_model_contract(self) -> None:
        """Validate model input dimension against artifact feature_names before inference."""
        if self.model is None:
            return

        expected_dim = getattr(self.model, "n_features_in_", None)
        if expected_dim is None:
            logger.warning(
                "[FpsModelContract] Model %s lacks n_features_in_; runtime feature count checks will use artifact order only.",
                self.model_version,
            )
            return

        actual_dim = len(self.feature_names)
        if int(expected_dim) != actual_dim:
            raise FeatureContractMismatchError(
                f"Model expects {expected_dim} features but artifact declares {actual_dim} feature_names."
            )

    def predict_window(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes inference on a single window/slice of 10-15 FPS samples."""
        features = extract_shared_features_vector(samples)
        feature_names = self.feature_names or DEFAULT_14_FEATURES

        if self.model is None:
            return {
                "prediction": "INCONCLUSIVE",
                "classProbability": {"NORMAL": 0.0, "STRABISMUS": 0.0},
                "confidence": 0.0,
                "features": features,
                "reason": "MODEL_NOT_LOADED",
            }

        vec: List[float] = []
        missing_or_invalid: List[str] = []
        for name in feature_names:
            val = features.get(name)
            if val is None or not np.isfinite(val):
                missing_or_invalid.append(name)
                continue
            vec.append(float(val))

        if missing_or_invalid:
            raise FeatureContractMismatchError(
                "Cannot run 10-15 FPS model because required features are missing or non-finite: "
                f"{missing_or_invalid}"
            )

        expected_dim = getattr(self.model, "n_features_in_", None)
        if expected_dim is not None and int(expected_dim) != len(vec):
            raise FeatureContractMismatchError(
                f"Model expects {expected_dim} features but received {len(vec)}."
            )

        X = np.asarray([vec], dtype=float)
        pred_idx = int(self.model.predict(X)[0])
        label = "NORMAL" if pred_idx == 0 else "STRABISMUS"

        class_probs = {"NORMAL": 0.5, "STRABISMUS": 0.5}
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(X)[0]
                class_probs = {
                    "NORMAL": round(float(probs[0]), 4),
                    "STRABISMUS": round(float(probs[1]), 4),
                }
            except Exception:
                class_probs[label] = 1.0

        confidence = float(class_probs.get(label, 0.5))
        return {
            "prediction": label,
            "classProbability": class_probs,
            "confidence": confidence,
            "features": features,
        }

    def aggregate_and_predict(
        self,
        raw_trajectories: Dict[int, Dict[str, Any]],
    ) -> FpsModelResult:
        """Collects 10-15 FPS samples from all cycles, aggregates consensus, and returns final result.
        
        Follows the required flow:
        Frame processing 10-15 FPS -> Analyze frames -> Temporal consensus aggregation -> Final result.
        """
        all_samples: List[Dict[str, Any]] = []
        sample_idx = 0
        for c_num in sorted(raw_trajectories.keys()):
            c_dict = raw_trajectories[c_num]
            for s in c_dict.get("samples", []):
                s_copy = dict(s)
                s_copy["index"] = sample_idx
                all_samples.append(s_copy)
                sample_idx += 1

        total_samples = len(all_samples)
        if total_samples == 0:
            return FpsModelResult(
                prediction="INCONCLUSIVE",
                class_probabilities={"NORMAL": 0.0, "STRABISMUS": 0.0},
                confidence=0.0,
                model_name=self.model_name,
                model_version=self.model_version,
                model_source="fps_10_15_model",
                features_snapshot={},
                status="INCONCLUSIVE",
                agreement_ratio=0.0,
                is_reliable=False,
                total_frames_evaluated=0,
                valid_frames_count=0,
                notice="No valid frame samples available for 10-15 FPS model inference.",
            )

        # Global features snapshot across entire session
        global_features = extract_shared_features_vector(all_samples)
        key_features = {
            k: round(float(global_features[k]), 4) if global_features.get(k) is not None else None
            for k in self.feature_names
        }

        # Sub-window consensus aggregation across 10-15 FPS stream
        # Dynamically size sliding window to ensure appropriate consensus coverage
        if total_samples >= 20:
            window_size = 12
            step_size = max(2, (total_samples - window_size) // 5)
            min_valid = 3
        elif total_samples >= 8:
            window_size = max(4, total_samples // 2)
            step_size = max(1, (total_samples - window_size) // 3)
            min_valid = 2
        else:
            window_size = total_samples
            step_size = 1
            min_valid = 1

        aggregator = TemporalConsensusAggregator(
            min_valid_frames=min_valid,
            min_consensus_ratio=0.60,
        )

        aggregation_result: TemporalAggregationResult = aggregator.aggregate_subwindow_inferences(
            samples=all_samples,
            inference_fn=self.predict_window,
            window_size=window_size,
            step_size=step_size,
        )

        if aggregation_result.valid_frames_count == 0:
            return FpsModelResult(
                prediction="INCONCLUSIVE",
                class_probabilities={"NORMAL": 0.0, "STRABISMUS": 0.0},
                confidence=0.0,
                model_name=self.model_name,
                model_version=self.model_version,
                model_source="fps_10_15_model",
                features_snapshot=key_features,
                status="INCONCLUSIVE",
                agreement_ratio=0.0,
                is_reliable=False,
                total_frames_evaluated=aggregation_result.total_frames_evaluated,
                valid_frames_count=0,
                notice=(
                    "10-15 FPS model inference was inconclusive because the model contract was not "
                    "satisfied or no valid inference windows were available."
                ),
            )

        final_prediction = aggregation_result.consensus_status
        if final_prediction not in ("NORMAL", "STRABISMUS", "INCONCLUSIVE"):
            final_prediction = "INCONCLUSIVE"

        probs = aggregation_result.class_probabilities
        clean_probs = {
            "NORMAL": round(float(probs.get("NORMAL", 0.0)), 4),
            "STRABISMUS": round(float(probs.get("STRABISMUS", 0.0)), 4),
        }

        # Normalize probabilities sum to 1.0 if both non-zero
        p_norm = clean_probs["NORMAL"]
        p_strab = clean_probs["STRABISMUS"]
        p_sum = p_norm + p_strab
        if p_sum > 0:
            clean_probs["NORMAL"] = round(p_norm / p_sum, 4)
            clean_probs["STRABISMUS"] = round(p_strab / p_sum, 4)

        confidence = round(float(clean_probs.get(final_prediction, aggregation_result.confidence)), 4)
        status_str = "COMPLETED" if final_prediction in ("NORMAL", "STRABISMUS") else "INCONCLUSIVE"

        notice = (
            "Screening result derived from 10-15 FPS model consensus aggregation. "
            "Preliminary screening support only — not a medical diagnosis."
        )

        return FpsModelResult(
            prediction=final_prediction,
            class_probabilities=clean_probs,
            confidence=confidence,
            model_name=self.model_name,
            model_version=self.model_version,
            model_source="fps_10_15_model",
            features_snapshot=key_features,
            status=status_str,
            agreement_ratio=aggregation_result.agreement_ratio,
            is_reliable=aggregation_result.is_reliable,
            total_frames_evaluated=aggregation_result.total_frames_evaluated,
            valid_frames_count=aggregation_result.valid_frames_count,
            notice=notice,
        )

    @classmethod
    def get_instance(cls) -> "FpsModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


def get_fps_model_service() -> FpsModelService:
    """Convenience getter for singleton FpsModelService."""
    return FpsModelService.get_instance()
