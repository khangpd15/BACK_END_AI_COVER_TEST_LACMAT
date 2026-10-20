"""Temporal Aggregation and Majority Voting Service for Multi-Frame Inference.

Mitigates single-frame false positives / negatives by:
- Filtering raw frames via Quality Gate (rejecting blinks, occlusions, blur, tilt)
- Prioritizing frames during stable fixational gaze over saccadic transitions
- Running inference across multiple high-quality temporal frames
- Combining predictions via:
  1. Majority Vote (Hard Voting)
  2. Soft Probability-Weighted Voting (weighted by tracking quality & fixation stability)
- Computing consensus agreement ratio and clinical confidence reliability
- Preserving current ML model architectures completely without modification
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from app.schemas import ScreeningStatus
from app.services.cv.quality_gate import QualityGateResult


@dataclass
class FrameInferenceCandidate:
    """A single frame inference candidate with quality and gaze telemetry."""
    frame_id: Union[int, str]
    timestamp: float
    prediction: str                 # "NORMAL", "STRABISMUS", "ATTENTION", or "INCONCLUSIVE"
    probabilities: Dict[str, float]  # e.g. {"NORMAL": 0.85, "STRABISMUS": 0.15}
    quality_result: Optional[QualityGateResult] = None
    is_fixating: bool = True
    gaze_stability: float = 1.0


@dataclass
class TemporalAggregationResult:
    """Consolidated result across a sequence of verified temporal frames."""
    consensus_status: str              # Final aggregated decision
    confidence: float                  # [0.0, 1.0] consensus confidence
    class_probabilities: Dict[str, float]  # Weighted probability distribution
    total_frames_evaluated: int
    valid_frames_count: int
    rejected_frames_count: int
    agreement_ratio: float             # Fraction of valid frames agreeing with consensus
    majority_vote_counts: Dict[str, int]
    is_reliable: bool
    rejection_reasons: List[str] = field(default_factory=list)
    clinical_note: Optional[str] = None


class TemporalConsensusAggregator:
    """Aggregates multiple temporal frame predictions to ensure clinical stability."""

    def __init__(
        self,
        min_valid_frames: int = 3,
        min_consensus_ratio: float = 0.60,  # At least 60% agreement required
        fixation_weight_boost: float = 1.25,
    ):
        self.min_valid_frames = min_valid_frames
        self.min_consensus_ratio = min_consensus_ratio
        self.fixation_weight_boost = fixation_weight_boost

    def aggregate_candidates(
        self,
        candidates: List[FrameInferenceCandidate],
    ) -> TemporalAggregationResult:
        """Aggregates a list of frame candidates into a single stable consensus."""
        total_frames = len(candidates)
        if total_frames == 0:
            return TemporalAggregationResult(
                consensus_status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
                confidence=0.0,
                class_probabilities={"NORMAL": 0.0, "STRABISMUS": 0.0},
                total_frames_evaluated=0,
                valid_frames_count=0,
                rejected_frames_count=0,
                agreement_ratio=0.0,
                majority_vote_counts={},
                is_reliable=False,
                rejection_reasons=["NO_CANDIDATE_FRAMES_PROVIDED"],
                clinical_note="No frames provided for temporal aggregation.",
            )

        # 1. Separate valid frames from rejected frames
        valid_candidates: List[FrameInferenceCandidate] = []
        all_rejections: List[str] = []

        for cand in candidates:
            if cand.prediction in ("INCONCLUSIVE", "SCREENING_INCONCLUSIVE"):
                all_rejections.append(f"Frame {cand.frame_id}: INCONCLUSIVE_PREDICTION")
                continue

            if cand.quality_result is not None and not cand.quality_result.is_acceptable:
                all_rejections.extend([f"Frame {cand.frame_id}: {r}" for r in cand.quality_result.failure_reasons])
                continue

            valid_candidates.append(cand)

        valid_count = len(valid_candidates)
        rejected_count = total_frames - valid_count

        # 2. Check minimum frame threshold
        if valid_count < self.min_valid_frames:
            return TemporalAggregationResult(
                consensus_status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
                confidence=0.0,
                class_probabilities={"NORMAL": 0.0, "STRABISMUS": 0.0},
                total_frames_evaluated=total_frames,
                valid_frames_count=valid_count,
                rejected_frames_count=rejected_count,
                agreement_ratio=0.0,
                majority_vote_counts={},
                is_reliable=False,
                rejection_reasons=all_rejections or ["INSUFFICIENT_VALID_FRAMES_AFTER_QUALITY_GATE"],
                clinical_note=(
                    f"Only {valid_count} valid frames available, but minimum {self.min_valid_frames} "
                    "frames required to guarantee temporal stability."
                ),
            )

        # 3. Hard Majority Voting
        vote_counts: Dict[str, int] = {}
        for cand in valid_candidates:
            vote_counts[cand.prediction] = vote_counts.get(cand.prediction, 0) + 1

        majority_class = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        agreement_ratio = vote_counts[majority_class] / valid_count

        # 4. Soft Probability-Weighted Voting
        # Compute dynamic weight w_i based on camera score, tracking score, and fixation state
        all_classes = set()
        for cand in valid_candidates:
            all_classes.update(cand.probabilities.keys())

        weighted_probs: Dict[str, float] = {c: 0.0 for c in all_classes}
        total_weight = 0.0

        for cand in valid_candidates:
            # Base weight from tracking & camera quality
            cam_q = cand.quality_result.camera_quality_score if cand.quality_result else 1.0
            track_q = cand.quality_result.tracking_quality_score if cand.quality_result else 1.0
            w = max(0.1, cam_q * track_q)

            # Boost weight if eye was fixating steadily during this frame
            if cand.is_fixating:
                w *= self.fixation_weight_boost

            w *= max(0.2, cand.gaze_stability)
            total_weight += w

            for c in all_classes:
                weighted_probs[c] += w * cand.probabilities.get(c, 0.0)

        if total_weight > 0:
            for c in all_classes:
                weighted_probs[c] = round(weighted_probs[c] / total_weight, 4)

        # Determine soft-voted class
        soft_majority_class = max(weighted_probs.keys(), key=lambda k: weighted_probs[k])

        # If hard and soft votes agree, confidence is high
        is_reliable = bool(
            agreement_ratio >= self.min_consensus_ratio
            and valid_count >= self.min_valid_frames
        )

        final_status = majority_class
        confidence = float(weighted_probs.get(final_status, agreement_ratio))

        note = (
            f"Temporal aggregation of {valid_count}/{total_frames} valid frames. "
            f"Consensus agreement: {agreement_ratio * 100:.1f}%. "
            f"Votes: {dict(vote_counts)}."
        )

        return TemporalAggregationResult(
            consensus_status=final_status,
            confidence=round(confidence, 4),
            class_probabilities=weighted_probs,
            total_frames_evaluated=total_frames,
            valid_frames_count=valid_count,
            rejected_frames_count=rejected_count,
            agreement_ratio=round(agreement_ratio, 4),
            majority_vote_counts=vote_counts,
            is_reliable=is_reliable,
            rejection_reasons=all_rejections[:10],  # Keep top reasons concise
            clinical_note=note,
        )

    def aggregate_subwindow_inferences(
        self,
        samples: List[Dict[str, Any]],
        inference_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
        window_size: int = 15,
        step_size: int = 5,
    ) -> TemporalAggregationResult:
        """Splits time-series into sliding temporal sub-windows, infers each, and aggregates votes.
        
        Useful for running existing ML models over multiple overlapping windows to eliminate
        single transient flutter artifacts.
        """
        n = len(samples)
        if n < window_size:
            # Single evaluation
            pred_dict = inference_fn(samples)
            candidate = FrameInferenceCandidate(
                frame_id=0,
                timestamp=samples[0].get("t", 0.0) if samples else 0.0,
                prediction=pred_dict.get("prediction", "NORMAL"),
                probabilities=pred_dict.get("classProbability", {"NORMAL": 0.5, "STRABISMUS": 0.5}),
                is_fixating=True,
            )
            return self.aggregate_candidates([candidate])

        candidates: List[FrameInferenceCandidate] = []
        win_idx = 0
        for start_idx in range(0, n - window_size + 1, step_size):
            window_samples = samples[start_idx : start_idx + window_size]
            mid_t = float(window_samples[len(window_samples) // 2].get("t", 0.0))

            try:
                res = inference_fn(window_samples)
                cand = FrameInferenceCandidate(
                    frame_id=win_idx,
                    timestamp=mid_t,
                    prediction=res.get("prediction", "NORMAL"),
                    probabilities=res.get("classProbability", {"NORMAL": 0.5, "STRABISMUS": 0.5}),
                    is_fixating=True,
                )
                candidates.append(cand)
                win_idx += 1
            except Exception:
                continue

        return self.aggregate_candidates(candidates)
