"""Shared Feature Contract for Phase 4 Cross-Domain Transfer.

Freezes the explicit mathematical definitions for features shared between:
- Korean Eye Tracking Research Data (10–15 FPS transfer pipeline)
- RemiCare Webcam + MediaPipe (30FPS Cover Test)

Explicitly handles:
- Invariant Shared Features (Disparity, dispersion, velocities, valid ratios)
- Potential Domain Shift Features (Raw viewport positions: meanLeftX, meanLeftY, meanRightX, meanRightY)
- Excluded Features (Hardware-dependent sampling rates and protocol durations)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass(frozen=True)
class SharedFeatureSpec:
    name: str
    formula: str
    source: str
    dtype: str
    description: str
    domain_shift_risk: str  # 'LOW_DOMAIN_SHIFT', 'POTENTIAL_DOMAIN_SHIFT', 'HIGH_DOMAIN_SHIFT'
    notes: str


SHARED_FEATURE_CONTRACT_VERSION = "shared-v1.0.0"


# =============================================================================
# FEATURE SPECIFICATIONS (30 Shared Features)
# =============================================================================

SHARED_FEATURE_SPECS: Dict[str, SharedFeatureSpec] = {
    # -------------------------------------------------------------------------
    # 1. Validity Ratios (3)
    # -------------------------------------------------------------------------
    "leftValidRatio": SharedFeatureSpec(
        name="leftValidRatio",
        formula="count(leftValid == True) / totalSamples",
        source="leftValid",
        dtype="float64",
        description="Ratio of frames with valid left eye detection",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Computed on all recorded frames.",
    ),
    "rightValidRatio": SharedFeatureSpec(
        name="rightValidRatio",
        formula="count(rightValid == True) / totalSamples",
        source="rightValid",
        dtype="float64",
        description="Ratio of frames with valid right eye detection",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Computed on all recorded frames.",
    ),
    "bothValidRatio": SharedFeatureSpec(
        name="bothValidRatio",
        formula="count(leftValid == True and rightValid == True) / totalSamples",
        source="leftValid, rightValid",
        dtype="float64",
        description="Ratio of frames with both eyes simultaneously tracked",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Both-valid frames form the basis for all disparity calculations.",
    ),

    # -------------------------------------------------------------------------
    # 2. Horizontal Inter-Ocular Disparity (7)
    # -------------------------------------------------------------------------
    "meanDeltaX": SharedFeatureSpec(
        name="meanDeltaX",
        formula="mean(rightX - leftX) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Mean horizontal distance between pupils/irises",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Primary static indicator of horizontal alignment.",
    ),
    "medianDeltaX": SharedFeatureSpec(
        name="medianDeltaX",
        formula="median(rightX - leftX) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Median horizontal disparity (outlier/blink resistant)",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Robust measure of central tendency.",
    ),
    "stdDeltaX": SharedFeatureSpec(
        name="stdDeltaX",
        formula="std(rightX - leftX, ddof=0) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Standard deviation of horizontal disparity",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Captures fixation instability and refixation movement.",
    ),
    "minDeltaX": SharedFeatureSpec(
        name="minDeltaX",
        formula="min(rightX - leftX) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Minimum horizontal disparity observed",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Lower bound of horizontal alignment range.",
    ),
    "maxDeltaX": SharedFeatureSpec(
        name="maxDeltaX",
        formula="max(rightX - leftX) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Maximum horizontal disparity observed",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Upper bound of horizontal alignment range.",
    ),
    "rangeDeltaX": SharedFeatureSpec(
        name="rangeDeltaX",
        formula="maxDeltaX - minDeltaX",
        source="maxDeltaX, minDeltaX",
        dtype="float64",
        description="Peak-to-peak horizontal disparity excursion",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Amplitude of horizontal movement disparity.",
    ),
    "meanAbsDeltaX": SharedFeatureSpec(
        name="meanAbsDeltaX",
        formula="mean(abs(rightX - leftX)) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Mean magnitude of horizontal distance",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Non-negative distance indicator.",
    ),

    # -------------------------------------------------------------------------
    # 3. Vertical Inter-Ocular Disparity (7)
    # -------------------------------------------------------------------------
    "meanDeltaY": SharedFeatureSpec(
        name="meanDeltaY",
        formula="mean(rightY - leftY) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Mean vertical inter-ocular disparity",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Indicates vertical misalignment or head tilt.",
    ),
    "medianDeltaY": SharedFeatureSpec(
        name="medianDeltaY",
        formula="median(rightY - leftY) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Median vertical disparity",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Robust vertical offset metric.",
    ),
    "stdDeltaY": SharedFeatureSpec(
        name="stdDeltaY",
        formula="std(rightY - leftY, ddof=0) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Standard deviation of vertical disparity",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Vertical dispersion / tremor.",
    ),
    "minDeltaY": SharedFeatureSpec(
        name="minDeltaY",
        formula="min(rightY - leftY) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Minimum vertical disparity observed",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Minimum vertical offset.",
    ),
    "maxDeltaY": SharedFeatureSpec(
        name="maxDeltaY",
        formula="max(rightY - leftY) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Maximum vertical disparity observed",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Maximum vertical offset.",
    ),
    "rangeDeltaY": SharedFeatureSpec(
        name="rangeDeltaY",
        formula="maxDeltaY - minDeltaY",
        source="maxDeltaY, minDeltaY",
        dtype="float64",
        description="Peak-to-peak vertical disparity range",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Vertical amplitude excursion.",
    ),
    "meanAbsDeltaY": SharedFeatureSpec(
        name="meanAbsDeltaY",
        formula="mean(abs(rightY - leftY)) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Mean absolute vertical disparity",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Magnitude of vertical misalignment.",
    ),

    # -------------------------------------------------------------------------
    # 4. Viewport Coordinates & Dispersion (8)
    # -------------------------------------------------------------------------
    # Note: Mean positions are marked POTENTIAL_DOMAIN_SHIFT (Step 2)
    "meanLeftX": SharedFeatureSpec(
        name="meanLeftX",
        formula="mean(leftX) on leftValid frames",
        source="leftX",
        dtype="float64",
        description="Mean horizontal position of left eye",
        domain_shift_risk="POTENTIAL_DOMAIN_SHIFT",
        notes="Affected by head position, camera crop, and distance to camera.",
    ),
    "stdLeftX": SharedFeatureSpec(
        name="stdLeftX",
        formula="std(leftX, ddof=0) on leftValid frames",
        source="leftX",
        dtype="float64",
        description="Horizontal dispersion of left eye",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Translation-invariant metric of left eye tremor/movement.",
    ),
    "meanLeftY": SharedFeatureSpec(
        name="meanLeftY",
        formula="mean(leftY) on leftValid frames",
        source="leftY",
        dtype="float64",
        description="Mean vertical position of left eye",
        domain_shift_risk="POTENTIAL_DOMAIN_SHIFT",
        notes="Affected by head tilt, webcam tilt, and user height.",
    ),
    "stdLeftY": SharedFeatureSpec(
        name="stdLeftY",
        formula="std(leftY, ddof=0) on leftValid frames",
        source="leftY",
        dtype="float64",
        description="Vertical dispersion of left eye",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Translation-invariant vertical movement metric.",
    ),
    "meanRightX": SharedFeatureSpec(
        name="meanRightX",
        formula="mean(rightX) on rightValid frames",
        source="rightX",
        dtype="float64",
        description="Mean horizontal position of right eye",
        domain_shift_risk="POTENTIAL_DOMAIN_SHIFT",
        notes="Affected by user position in webcam viewport.",
    ),
    "stdRightX": SharedFeatureSpec(
        name="stdRightX",
        formula="std(rightX, ddof=0) on rightValid frames",
        source="rightX",
        dtype="float64",
        description="Horizontal dispersion of right eye",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Translation-invariant right eye dispersion.",
    ),
    "meanRightY": SharedFeatureSpec(
        name="meanRightY",
        formula="mean(rightY) on rightValid frames",
        source="rightY",
        dtype="float64",
        description="Mean vertical position of right eye",
        domain_shift_risk="POTENTIAL_DOMAIN_SHIFT",
        notes="Affected by viewport positioning.",
    ),
    "stdRightY": SharedFeatureSpec(
        name="stdRightY",
        formula="std(rightY, ddof=0) on rightValid frames",
        source="rightY",
        dtype="float64",
        description="Vertical dispersion of right eye",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Translation-invariant vertical stability.",
    ),

    # -------------------------------------------------------------------------
    # 5. Kinematics and Velocities (5)
    # -------------------------------------------------------------------------
    "meanLeftVelocity": SharedFeatureSpec(
        name="meanLeftVelocity",
        formula="mean(hypot(dx, dy) / dt) for dt > 0.001s",
        source="leftX, leftY, t (leftValid)",
        dtype="float64",
        description="Mean frame-to-frame velocity of left eye (coord/s)",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Normalized for sampling interval dt.",
    ),
    "peakLeftVelocity": SharedFeatureSpec(
        name="peakLeftVelocity",
        formula="max(hypot(dx, dy) / dt) for dt > 0.001s",
        source="leftX, leftY, t (leftValid)",
        dtype="float64",
        description="Peak velocity of left eye (coord/s)",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Reflects peak saccade/refixation speed.",
    ),
    "meanRightVelocity": SharedFeatureSpec(
        name="meanRightVelocity",
        formula="mean(hypot(dx, dy) / dt) for dt > 0.001s",
        source="rightX, rightY, t (rightValid)",
        dtype="float64",
        description="Mean frame-to-frame velocity of right eye (coord/s)",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Normalized for sampling interval dt.",
    ),
    "peakRightVelocity": SharedFeatureSpec(
        name="peakRightVelocity",
        formula="max(hypot(dx, dy) / dt) for dt > 0.001s",
        source="rightX, rightY, t (rightValid)",
        dtype="float64",
        description="Peak velocity of right eye (coord/s)",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Reflects peak saccade/refixation speed.",
    ),
    "velocityDisparity": SharedFeatureSpec(
        name="velocityDisparity",
        formula="abs(meanRightVelocity - meanLeftVelocity)",
        source="meanRightVelocity, meanLeftVelocity",
        dtype="float64",
        description="Asymmetry between right and left eye mean velocities",
        domain_shift_risk="LOW_DOMAIN_SHIFT",
        notes="Key physiological indicator of asymmetric ocular motility.",
    ),
}

# The canonical ordered list of all 30 shared features
ALL_SHARED_FEATURES: List[str] = list(SHARED_FEATURE_SPECS.keys())

# Raw viewport position features marked with POTENTIAL_DOMAIN_SHIFT
VIEWPORT_POSITION_FEATURES: List[str] = [
    name for name, spec in SHARED_FEATURE_SPECS.items()
    if spec.domain_shift_risk == "POTENTIAL_DOMAIN_SHIFT"
]
POTENTIAL_DOMAIN_SHIFT_FEATURES = VIEWPORT_POSITION_FEATURES

# The 26 translation-invariant shared features
INVARIANT_SHARED_FEATURES: List[str] = [
    name for name, spec in SHARED_FEATURE_SPECS.items()
    if spec.domain_shift_risk != "POTENTIAL_DOMAIN_SHIFT"
]

# Explicitly excluded features (Hardware or protocol specific)
EXCLUDED_FEATURES: List[str] = [
    "sampleCount",     # Korean ~1500 vs RemiCare ~33 (protocol length)
    "durationMs",      # Korean ~16000ms vs RemiCare ~1000-3000ms (protocol length)
    "estimatedHz",     # Korean ~60Hz vs Webcam ~30FPS (hardware sensor)
    "meanIntervalMs",  # Korean ~16.6ms vs Webcam ~33.3ms (hardware frame rate)
    "sampleId",        # Metadata string
    "label_name",      # Target label string
    "label",           # Target integer
]


def extract_shared_features_vector(samples: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Extract strictly the 30 shared features without faking missing values.
    
    If source data has no valid frames for a metric, returns None (null).
    """
    total_samples = len(samples)
    if total_samples == 0:
        return {f: None for f in ALL_SHARED_FEATURES}

    # Normalize timestamps to seconds for velocity calculation
    raw_ts = [s["t"] for s in samples]
    if len(raw_ts) > 1 and (raw_ts[-1] - raw_ts[0]) > 200:
        ts_sec = [t / 1000.0 for t in raw_ts]
    else:
        ts_sec = [float(t) for t in raw_ts]

    left_valid_samples = []
    right_valid_samples = []
    both_valid_samples = []

    for i, s in enumerate(samples):
        t_s = ts_sec[i]
        lx, ly = s.get("leftX"), s.get("leftY")
        rx, ry = s.get("rightX"), s.get("rightY")
        lv = bool(s.get("leftValid", False) and lx is not None and ly is not None and math.isfinite(lx) and math.isfinite(ly))
        rv = bool(s.get("rightValid", False) and rx is not None and ry is not None and math.isfinite(rx) and math.isfinite(ry))

        if lv:
            left_valid_samples.append((t_s, float(lx), float(ly)))
        if rv:
            right_valid_samples.append((t_s, float(rx), float(ry)))
        if lv and rv:
            both_valid_samples.append((t_s, float(lx), float(ly), float(rx), float(ry)))

    both_count = len(both_valid_samples)
    left_count = len(left_valid_samples)
    right_count = len(right_valid_samples)

    result: Dict[str, Optional[float]] = {}

    # Validity ratios
    result["leftValidRatio"] = round(left_count / total_samples, 4)
    result["rightValidRatio"] = round(right_count / total_samples, 4)
    result["bothValidRatio"] = round(both_count / total_samples, 4)

    # Disparity metrics
    if both_count > 0:
        dxs = np.array([r[3] - r[1] for r in both_valid_samples], dtype=float)
        dys = np.array([r[4] - r[2] for r in both_valid_samples], dtype=float)

        result["meanDeltaX"] = round(float(np.mean(dxs)), 6)
        result["medianDeltaX"] = round(float(np.median(dxs)), 6)
        result["stdDeltaX"] = round(float(np.std(dxs)), 6)
        result["minDeltaX"] = round(float(np.min(dxs)), 6)
        result["maxDeltaX"] = round(float(np.max(dxs)), 6)
        result["rangeDeltaX"] = round(float(np.max(dxs) - np.min(dxs)), 6)
        result["meanAbsDeltaX"] = round(float(np.mean(np.abs(dxs))), 6)

        result["meanDeltaY"] = round(float(np.mean(dys)), 6)
        result["medianDeltaY"] = round(float(np.median(dys)), 6)
        result["stdDeltaY"] = round(float(np.std(dys)), 6)
        result["minDeltaY"] = round(float(np.min(dys)), 6)
        result["maxDeltaY"] = round(float(np.max(dys)), 6)
        result["rangeDeltaY"] = round(float(np.max(dys) - np.min(dys)), 6)
        result["meanAbsDeltaY"] = round(float(np.mean(np.abs(dys))), 6)
    else:
        for f in ["meanDeltaX", "medianDeltaX", "stdDeltaX", "minDeltaX", "maxDeltaX", "rangeDeltaX", "meanAbsDeltaX",
                  "meanDeltaY", "medianDeltaY", "stdDeltaY", "minDeltaY", "maxDeltaY", "rangeDeltaY", "meanAbsDeltaY"]:
            result[f] = None

    # Left eye position and dispersion
    if left_count > 0:
        l_xs = [r[1] for r in left_valid_samples]
        l_ys = [r[2] for r in left_valid_samples]
        result["meanLeftX"] = round(float(np.mean(l_xs)), 6)
        result["stdLeftX"] = round(float(np.std(l_xs)), 6)
        result["meanLeftY"] = round(float(np.mean(l_ys)), 6)
        result["stdLeftY"] = round(float(np.std(l_ys)), 6)
    else:
        result["meanLeftX"] = result["stdLeftX"] = result["meanLeftY"] = result["stdLeftY"] = None

    # Right eye position and dispersion
    if right_count > 0:
        r_xs = [r[1] for r in right_valid_samples]
        r_ys = [r[2] for r in right_valid_samples]
        result["meanRightX"] = round(float(np.mean(r_xs)), 6)
        result["stdRightX"] = round(float(np.std(r_xs)), 6)
        result["meanRightY"] = round(float(np.mean(r_ys)), 6)
        result["stdRightY"] = round(float(np.std(r_ys)), 6)
    else:
        result["meanRightX"] = result["stdRightX"] = result["meanRightY"] = result["stdRightY"] = None

    # Velocities
    l_vels: List[float] = []
    for i in range(len(left_valid_samples) - 1):
        dt = left_valid_samples[i + 1][0] - left_valid_samples[i][0]
        if dt > 0.001:
            d = math.hypot(
                left_valid_samples[i + 1][1] - left_valid_samples[i][1],
                left_valid_samples[i + 1][2] - left_valid_samples[i][2],
            )
            l_vels.append(d / dt)

    r_vels: List[float] = []
    for i in range(len(right_valid_samples) - 1):
        dt = right_valid_samples[i + 1][0] - right_valid_samples[i][0]
        if dt > 0.001:
            d = math.hypot(
                right_valid_samples[i + 1][1] - right_valid_samples[i][1],
                right_valid_samples[i + 1][2] - right_valid_samples[i][2],
            )
            r_vels.append(d / dt)

    mean_l_v = float(np.mean(l_vels)) if l_vels else None
    peak_l_v = float(np.max(l_vels)) if l_vels else None
    mean_r_v = float(np.mean(r_vels)) if r_vels else None
    peak_r_v = float(np.max(r_vels)) if r_vels else None

    result["meanLeftVelocity"] = round(mean_l_v, 6) if mean_l_v is not None else None
    result["peakLeftVelocity"] = round(peak_l_v, 6) if peak_l_v is not None else None
    result["meanRightVelocity"] = round(mean_r_v, 6) if mean_r_v is not None else None
    result["peakRightVelocity"] = round(peak_r_v, 6) if peak_r_v is not None else None

    if mean_l_v is not None and mean_r_v is not None:
        result["velocityDisparity"] = round(abs(mean_r_v - mean_l_v), 6)
    else:
        result["velocityDisparity"] = None

    return result


def extract_remicare_shared_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute pipeline:
    sample.json -> validation -> preprocessing -> shared feature extraction -> feature vector

    Adheres strictly to Phase 4 format:
    {
      "sampleId": "...",
      "source": "REMICARE_WEBCAM_MEDIAPIPE",
      "featureContractVersion": "shared-v1.0.0",
      "features": { ... },
      "missingFeatures": [ ... ]
    }

    If source data cannot compute a feature, sets feature to None (null) and lists in missingFeatures.
    Never synthesizes or imputes missing values.
    """
    from app.schemas import ScreeningRequest
    from app.services.validation import validate_screening_request
    from app.services.preprocessing import preprocess_screening_request

    request_obj = ScreeningRequest.model_validate(raw_payload)
    val_result = validate_screening_request(request_obj)
    if not val_result.is_valid:
        raise ValueError(f"Sample failed Data Integrity Gate: {val_result.reason}")

    proc_data = preprocess_screening_request(request_obj)
    all_samples: List[Dict[str, Any]] = []
    for cycle in proc_data.cycles:
        for s in cycle.samples:
            all_samples.append({
                "t": s.t,
                "leftX": s.leftX,
                "leftY": s.leftY,
                "leftValid": s.leftValid,
                "rightX": s.rightX,
                "rightY": s.rightY,
                "rightValid": s.rightValid,
            })

    extracted_features = extract_shared_features_vector(all_samples)
    missing = [f for f in ALL_SHARED_FEATURES if extracted_features.get(f) is None]

    return {
        "sampleId": request_obj.sampleId,
        "source": "REMICARE_WEBCAM_MEDIAPIPE",
        "featureContractVersion": SHARED_FEATURE_CONTRACT_VERSION,
        "features": extracted_features,
        "missingFeatures": missing,
    }


def run_shared_transfer_inference(
    shared_sample_payload: Dict[str, Any],
    model_artifact: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute transfer experiment using a Korean-trained shared feature model on a RemiCare sample.

    Checks contract compatibility:
    - If required model features are missing or null, returns MODEL_INPUT_INCOMPATIBLE without imputation.
    - If compatible, runs inference and attaches explicit DOMAIN SHIFT and CLINICAL DISCLAIMER warnings.
    """
    features = shared_sample_payload.get("features", {})
    required_features = model_artifact.get("feature_names", ALL_SHARED_FEATURES)

    missing_required = [
        f for f in required_features
        if f not in features or features[f] is None
    ]

    if missing_required:
        return {
            "status": "MODEL_INPUT_INCOMPATIBLE",
            "inputCompatible": False,
            "missingFeatures": missing_required,
            "domainShiftWarning": True,
            "domainShift": {
                "sourceDomain": "KOREAN_INFRARED_EYE_TRACKER",
                "targetDomain": "REMICARE_WEBCAM_MEDIAPIPE",
            },
            "notice": "Research transfer experiment only — not a diagnosis.",
        }

    # Build input feature vector in exact order
    feat_vector = [features[f] for f in required_features]
    X = np.array([feat_vector], dtype=float)

    clf = model_artifact["model"]
    pred_idx = int(clf.predict(X)[0])
    prediction_label = "NORMAL" if pred_idx == 0 else "STRABISMUS"

    probs = None
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba(X)[0]
            probs = {"NORMAL": round(float(p[0]), 4), "STRABISMUS": round(float(p[1]), 4)}
        except Exception:
            probs = None

    potential_shift = [
        f for f in required_features
        if f in SHARED_FEATURE_SPECS and SHARED_FEATURE_SPECS[f].domain_shift_risk == "POTENTIAL_DOMAIN_SHIFT"
    ]

    return {
        "status": "TRANSFER_EXPERIMENT",
        "inputCompatible": True,
        "model": model_artifact.get("name", "Korean Shared Feature Model"),
        "prediction": prediction_label,
        "classProbability": round(float(probs[prediction_label]), 4) if probs is not None else None,
        "probabilities": probs,
        "clinicalMeaning": "None",
        "clinicalRiskDisclaimer": (
            "This probability represents the Random Forest classification score on an out-of-domain "
            "sample. It is NOT clinical risk, NOT diagnostic confidence, and NOT probability of disease."
        ),
        "domainShiftWarning": True,
        "domainShift": {
            "sourceDomain": "KOREAN_INFRARED_EYE_TRACKER",
            "targetDomain": "REMICARE_WEBCAM_MEDIAPIPE",
            "riskLevel": "HIGH",
            "potentialShiftFeatures": potential_shift,
            "warning": (
                "Source domain is Korean eye-tracking research data. "
                "Target domain is RemiCare webcam 10–15 FPS. "
                "Feature distributions may diverge significantly."
            ),
        },
        "notice": "Research transfer experiment only — not a diagnosis.",
        "productionConstraint": (
            "Transfer experiment output cannot be used directly as SCREENING_CLEAR or "
            "SCREENING_ATTENTION in production."
        ),
    }
