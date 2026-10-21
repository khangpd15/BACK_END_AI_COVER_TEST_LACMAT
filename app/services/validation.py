"""Validation service / Data Integrity Gate for RemiCare Strabismus AI.

Verifies raw Cover Test time-series data integrity before any preprocessing
or feature extraction. Ensures the AI backend rejects malformed, corrupted,
or clinically insufficient time-series with clear, safe inconclusive outcomes.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.schemas import CoverCycle, EyeSample, ScreeningRequest

logger = logging.getLogger("remicare.validation")

# Reasonable normalized coordinate range with generous tolerance for edge-of-frame tracking
COORD_MIN = -0.2
COORD_MAX = 1.2

# Minimum valid samples required to compute reliable statistics
MIN_BASELINE_VALID_SAMPLES = 3
MIN_TRACKING_VALID_SAMPLES = 3
MIN_OVERALL_VALID_RATIO = 0.40
MIN_MEAN_TRACKING_QUALITY = 0.35


@dataclass
class CycleValidationStats:
    cycle: int
    coveredEye: str
    trackedEye: str
    totalSamples: int = 0
    validSamples: int = 0
    validRatio: float = 0.0
    meanTrackingQuality: float = 0.0
    baselineValidCount: int = 0
    trackingValidCount: int = 0
    issues: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    cycle_stats: List[CycleValidationStats] = field(default_factory=list)
    total_samples: int = 0
    valid_samples: int = 0
    valid_ratio: float = 0.0
    mean_tracking_quality: float = 0.0

    def to_quality_dict(self) -> Dict[str, Any]:
        return {
            "status": "PASS" if self.is_valid else "FAIL",
            "reason": self.reason,
            "sampleCount": self.total_samples,
            "validSampleCount": self.valid_samples,
            "validRatio": round(self.valid_ratio, 4) if self.total_samples > 0 else 0.0,
            "meanTrackingQuality": round(self.mean_tracking_quality, 4) if self.total_samples > 0 else 0.0,
            "issues": self.issues,
        }


def validate_sample_integrity(
    sample: EyeSample,
    prev_t: Optional[float],
    tracked_eye: str,
    cycle_idx: int,
    sample_idx: int,
) -> tuple[bool, List[str]]:
    """Validate integrity of an individual eye sample."""
    issues = []

    # 1. Timestamp validation
    if not isinstance(sample.t, (int, float)) or not math.isfinite(sample.t):
        issues.append(f"Cycle {cycle_idx} sample {sample_idx}: non-finite or invalid timestamp t={sample.t}")
        return False, issues

    if sample.t < 0:
        issues.append(f"Cycle {cycle_idx} sample {sample_idx}: negative timestamp t={sample.t}")
        return False, issues

    if prev_t is not None and sample.t <= prev_t:
        issues.append(
            f"Cycle {cycle_idx} sample {sample_idx}: non-monotonic timestamp t={sample.t} <= prev_t={prev_t}"
        )
        return False, issues

    # 2. Tracking quality
    if not isinstance(sample.trackingQuality, (int, float)) or not math.isfinite(sample.trackingQuality):
        issues.append(f"Cycle {cycle_idx} sample {sample_idx}: non-finite trackingQuality")
        return False, issues

    if not (0.0 <= sample.trackingQuality <= 1.0):
        issues.append(
            f"Cycle {cycle_idx} sample {sample_idx}: trackingQuality {sample.trackingQuality} outside [0.0, 1.0]"
        )
        return False, issues

    # 3. Phase check
    valid_phases = {"BASELINE", "COVER", "UNCOVER", "TRACKING"}
    if sample.phase not in valid_phases:
        issues.append(f"Cycle {cycle_idx} sample {sample_idx}: invalid phase '{sample.phase}'")
        return False, issues

    # 4. Coordinate finite & range check
    coords = [
        ("leftX", sample.leftX),
        ("leftY", sample.leftY),
        ("rightX", sample.rightX),
        ("rightY", sample.rightY),
    ]

    for coord_name, coord_val in coords:
        if coord_val is not None:
            if not isinstance(coord_val, (int, float)) or not math.isfinite(coord_val):
                issues.append(
                    f"Cycle {cycle_idx} sample {sample_idx}: {coord_name} is non-finite or NaN ({coord_val})"
                )
                return False, issues
            if not (COORD_MIN <= coord_val <= COORD_MAX):
                issues.append(
                    f"Cycle {cycle_idx} sample {sample_idx}: {coord_name} ({coord_val}) outside valid range [{COORD_MIN}, {COORD_MAX}]"
                )
                return False, issues

    return True, issues


def is_sample_valid_for_tracked_eye(sample: EyeSample, tracked_eye: str) -> bool:
    """Check if the sample has valid, finite coordinates for the eye being tracked."""
    if tracked_eye == "LEFT":
        return bool(
            sample.leftValid
            and sample.leftX is not None
            and sample.leftY is not None
            and math.isfinite(sample.leftX)
            and math.isfinite(sample.leftY)
        )
    elif tracked_eye == "RIGHT":
        return bool(
            sample.rightValid
            and sample.rightX is not None
            and sample.rightY is not None
            and math.isfinite(sample.rightX)
            and math.isfinite(sample.rightY)
        )
    return False


def validate_screening_request(request: ScreeningRequest) -> ValidationResult:
    """Run full Data Integrity Gate on a ScreeningRequest.
    
    Returns ValidationResult with is_valid=True if all integrity criteria pass,
    or is_valid=False with specific failure reason and diagnostic issues.
    """
    all_issues: List[str] = []

    # 1. Check sampleId
    if not request.sampleId or not str(request.sampleId).strip():
        return ValidationResult(
            is_valid=False,
            reason="MISSING_SAMPLE_ID",
            issues=["sampleId is missing or empty"],
        )

    # 2. Check test identifier
    if request.test != "COVER_TEST":
        return ValidationResult(
            is_valid=False,
            reason="INVALID_TEST_TYPE",
            issues=[f"test must be 'COVER_TEST', got '{request.test}'"],
        )

    # 3. Check cycles
    if not request.cycles or len(request.cycles) == 0:
        return ValidationResult(
            is_valid=False,
            reason="EMPTY_CYCLES",
            issues=["cycles array is empty"],
        )

    total_samples = 0
    total_valid_samples = 0
    total_quality_sum = 0.0
    cycle_stats_list: List[CycleValidationStats] = []

    for cycle in request.cycles:
        c_stats = CycleValidationStats(
            cycle=cycle.cycle,
            coveredEye=cycle.coveredEye,
            trackedEye=cycle.trackedEye,
        )

        # Eye configuration validation
        valid_eyes = {"LEFT", "RIGHT"}
        if cycle.coveredEye not in valid_eyes:
            all_issues.append(f"Cycle {cycle.cycle}: invalid coveredEye '{cycle.coveredEye}'")
            return ValidationResult(
                is_valid=False,
                reason="INVALID_EYE_CONFIGURATION",
                issues=all_issues,
            )
        if cycle.trackedEye not in valid_eyes:
            all_issues.append(f"Cycle {cycle.cycle}: invalid trackedEye '{cycle.trackedEye}'")
            return ValidationResult(
                is_valid=False,
                reason="INVALID_EYE_CONFIGURATION",
                issues=all_issues,
            )
        if cycle.coveredEye == cycle.trackedEye:
            all_issues.append(f"Cycle {cycle.cycle}: coveredEye '{cycle.coveredEye}' cannot equal trackedEye '{cycle.trackedEye}'")
            return ValidationResult(
                is_valid=False,
                reason="INVALID_EYE_CONFIGURATION",
                issues=all_issues,
            )

        if not cycle.samples or len(cycle.samples) == 0:
            all_issues.append(f"Cycle {cycle.cycle}: empty samples list")
            return ValidationResult(
                is_valid=False,
                reason="EMPTY_CYCLE_SAMPLES",
                issues=all_issues,
            )

        prev_t: Optional[float] = None
        has_baseline_phase = False
        has_tracking_phase = False
        baseline_valid_samples = 0
        tracking_valid_samples = 0
        cycle_valid_samples = 0
        cycle_quality_sum = 0.0

        for s_idx, sample in enumerate(cycle.samples):
            # Detailed sample integrity checks
            sample_ok, sample_issues = validate_sample_integrity(
                sample=sample,
                prev_t=prev_t,
                tracked_eye=cycle.trackedEye,
                cycle_idx=cycle.cycle,
                sample_idx=s_idx,
            )

            if not sample_ok:
                all_issues.extend(sample_issues)
                # Map specific issue to clean reason code
                first_issue = sample_issues[0] if sample_issues else "INVALID_SAMPLE"
                if "negative timestamp" in first_issue:
                    reason_code = "NEGATIVE_TIMESTAMP"
                elif "non-monotonic" in first_issue:
                    reason_code = "NON_MONOTONIC_TIMESTAMPS"
                elif "non-finite or NaN" in first_issue:
                    reason_code = "COORDINATES_NAN_OR_INFINITE"
                elif "outside valid range" in first_issue:
                    reason_code = "COORDINATES_OUT_OF_BOUNDS"
                elif "trackingQuality" in first_issue:
                    reason_code = "INVALID_TRACKING_QUALITY"
                else:
                    reason_code = "INSUFFICIENT_OR_INVALID_TIME_SERIES"

                return ValidationResult(
                    is_valid=False,
                    reason=reason_code,
                    issues=all_issues,
                )

            prev_t = sample.t
            cycle_quality_sum += sample.trackingQuality

            # Phase tracking
            if sample.phase == "BASELINE":
                has_baseline_phase = True
                if is_sample_valid_for_tracked_eye(sample, cycle.trackedEye):
                    baseline_valid_samples += 1
            elif sample.phase in ("UNCOVER", "TRACKING"):
                has_tracking_phase = True
                if is_sample_valid_for_tracked_eye(sample, cycle.trackedEye):
                    tracking_valid_samples += 1

            if is_sample_valid_for_tracked_eye(sample, cycle.trackedEye):
                cycle_valid_samples += 1

        c_samples_count = len(cycle.samples)
        c_stats.totalSamples = c_samples_count
        c_stats.validSamples = cycle_valid_samples
        c_stats.validRatio = cycle_valid_samples / c_samples_count if c_samples_count > 0 else 0.0
        c_stats.meanTrackingQuality = cycle_quality_sum / c_samples_count if c_samples_count > 0 else 0.0
        c_stats.baselineValidCount = baseline_valid_samples
        c_stats.trackingValidCount = tracking_valid_samples

        total_samples += c_samples_count
        total_valid_samples += cycle_valid_samples
        total_quality_sum += cycle_quality_sum

        # Check phase presence
        if not has_baseline_phase:
            all_issues.append(f"Cycle {cycle.cycle}: missing BASELINE phase")
            return ValidationResult(
                is_valid=False,
                reason="MISSING_BASELINE_PHASE",
                issues=all_issues,
            )

        if not has_tracking_phase:
            all_issues.append(f"Cycle {cycle.cycle}: missing UNCOVER or TRACKING phase")
            return ValidationResult(
                is_valid=False,
                reason="MISSING_TRACKING_PHASE",
                issues=all_issues,
            )

        # Check minimum valid samples in critical phases
        if baseline_valid_samples < MIN_BASELINE_VALID_SAMPLES:
            all_issues.append(
                f"Cycle {cycle.cycle}: insufficient valid BASELINE samples ({baseline_valid_samples} < {MIN_BASELINE_VALID_SAMPLES})"
            )
            return ValidationResult(
                is_valid=False,
                reason="INSUFFICIENT_VALID_BASELINE_SAMPLES",
                issues=all_issues,
            )

        if tracking_valid_samples < MIN_TRACKING_VALID_SAMPLES:
            all_issues.append(
                f"Cycle {cycle.cycle}: insufficient valid TRACKING samples ({tracking_valid_samples} < {MIN_TRACKING_VALID_SAMPLES})"
            )
            return ValidationResult(
                is_valid=False,
                reason="INSUFFICIENT_VALID_TRACKING_SAMPLES",
                issues=all_issues,
            )

        cycle_stats_list.append(c_stats)

    overall_valid_ratio = total_valid_samples / total_samples if total_samples > 0 else 0.0
    overall_mean_quality = total_quality_sum / total_samples if total_samples > 0 else 0.0

    if overall_valid_ratio < MIN_OVERALL_VALID_RATIO:
        all_issues.append(
            f"Overall valid ratio ({overall_valid_ratio:.2f}) is below minimum threshold ({MIN_OVERALL_VALID_RATIO})"
        )
        return ValidationResult(
            is_valid=False,
            reason="LOW_VALID_SAMPLE_RATIO",
            issues=all_issues,
            cycle_stats=cycle_stats_list,
            total_samples=total_samples,
            valid_samples=total_valid_samples,
            valid_ratio=overall_valid_ratio,
            mean_tracking_quality=overall_mean_quality,
        )

    if overall_mean_quality < MIN_MEAN_TRACKING_QUALITY:
        all_issues.append(
            f"Overall mean tracking quality ({overall_mean_quality:.2f}) is below minimum threshold ({MIN_MEAN_TRACKING_QUALITY})"
        )
        return ValidationResult(
            is_valid=False,
            reason="LOW_TRACKING_QUALITY",
            issues=all_issues,
            cycle_stats=cycle_stats_list,
            total_samples=total_samples,
            valid_samples=total_valid_samples,
            valid_ratio=overall_valid_ratio,
            mean_tracking_quality=overall_mean_quality,
        )

    # All integrity gate criteria PASSED
    return ValidationResult(
        is_valid=True,
        reason=None,
        issues=[],
        cycle_stats=cycle_stats_list,
        total_samples=total_samples,
        valid_samples=total_valid_samples,
        valid_ratio=overall_valid_ratio,
        mean_tracking_quality=overall_mean_quality,
    )
