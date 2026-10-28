"""Versioned paired-eye pixel + event-time model; no inferred clock units.

Training and serving MUST import this same extractor. Artifacts are trusted local
joblib files. Scores are probabilities, never estimates of clinical accuracy.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

CONTRACT = "cover-paired-crops-ms-v3"
FEATURE_VERSION = "hog64x32-canthal-seconds-v1"
CLASSES = ["NORMAL", "STRABISMUS"]


def finite(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number)) and math.isfinite(value)


def timestamp_ms(sample, unit="milliseconds"):
    # Source clock takes precedence over cycle-relative time. Never compare these
    # two clocks directly and never guess seconds from the magnitude of a value.
    value = sample.get("timestampMs")
    if value is None:
        value = sample.get("t")
        if unit not in {"milliseconds", "seconds"}:
            raise ValueError("TIMESTAMP_UNIT_REQUIRED")
        if finite(value) and unit == "seconds":
            value *= 1000
    if not finite(value) or value < 0:
        raise ValueError("INVALID_TIMESTAMP")
    return float(value)


def assess_timing(samples, unit="milliseconds"):
    """Assess processed SOURCE frames, including invalid/occluded eye samples.

    29.5 Hz tolerates nominal 29.97/30 encoding. A median alone cannot mask
    dropped frames: also require effective rate, p95 and maximum interval.
    """
    if len(samples) < 16:
        raise ValueError("INSUFFICIENT_SOURCE_FRAMES")
    times = np.asarray([timestamp_ms(s, unit) for s in samples])
    delta = np.diff(times)
    if np.any(delta <= 0):
        raise ValueError("NON_MONOTONIC_OR_DUPLICATE_TIMESTAMP")
    ids = [s.get("frameId") for s in samples]
    if any(i is not None for i in ids):
        if not all(type(i) is int and i >= 0 for i in ids) or any(b <= a for a, b in zip(ids, ids[1:])):
            raise ValueError("NON_MONOTONIC_OR_MISSING_FRAME_ID")
    effective = float(1000 * (len(times) - 1) / (times[-1] - times[0]))
    median = float(1000 / np.median(delta))
    p95, maximum = float(np.percentile(delta, 95)), float(max(delta))
    eligible = effective >= 29.5 and median >= 29.5 and p95 <= 50 and maximum <= 100
    return {"eligible30fps": bool(eligible), "effectiveFps": effective,
            "medianFps": median, "p95IntervalMs": p95, "maxIntervalMs": maximum,
            "sourceFrames": len(samples), "durationMs": float(times[-1] - times[0]),
            "reason": None if eligible else "PROCESSED_CADENCE_BELOW_30FPS_POLICY"}


def coordinates(sample, side):
    if sample.get("coordinateSpace") != "CANTHAL_LOCAL_V1":
        return None
    eye = sample.get(side + "EyeLocal") or {}
    x, y, valid = eye.get("x"), eye.get("y"), eye.get("valid", False)
    if sample.get(side + "Valid") is False or sample.get("headPose", {}).get("isValid") is False or sample.get("blink") is True:
        return None
    if valid is not True or not finite(x) or not finite(y):
        return None
    return float(x), float(y)


def trajectory_features(samples, event_ms=None, unit="milliseconds"):
    # Work within ONE cycle/event; no derivatives across cycle resets or gaps.
    rows = []
    for sample in samples:
        t = timestamp_ms(sample, unit)
        if event_ms is not None and not event_ms <= t <= event_ms + 1500:
            continue
        left, right = coordinates(sample, "left"), coordinates(sample, "right")
        if left is not None and right is not None:
            rows.append([t / 1000, *left, *right])
    if len(rows) < 15:
        raise ValueError("INSUFFICIENT_VALID_BINOCULAR_FRAMES")
    a = np.asarray(rows, dtype=float)
    dt = np.diff(a[:, 0])
    if np.any(dt <= 0) or np.ptp(a[:, 0]) < 1.0:
        raise ValueError("INVALID_EVENT_TIME_COVERAGE")
    anchor = event_ms/1000 if event_ms is not None else a[0, 0]
    early_mask = a[:, 0] <= anchor + .5 + 1e-9
    early_times = a[early_mask, 0]
    if len(early_times) < 5 or np.ptp(early_times) < .3-1e-9 or np.max(np.diff(early_times)) > .250001:
        raise ValueError("INSUFFICIENT_EARLY_EVENT_COVERAGE")
    large_gaps = np.flatnonzero(dt > .250001)
    if len(large_gaps) and np.any(a[large_gaps,0] < anchor + .5):
        raise ValueError("EARLY_EVENT_GAP")
    tail_start = max(a[-1,0]-.5, a[large_gaps[-1]+1,0] if len(large_gaps) else a[0,0])
    tail_mask = a[:,0] >= tail_start-1e-9
    if tail_mask.sum() < 5 or np.ptp(a[tail_mask,0]) < .2-1e-9:
        raise ValueError("INSUFFICIENT_CONTIGUOUS_SETTLING_TAIL")
    disp = a[:, 1:3] - a[:, 3:5]
    early, tail = disp[early_mask].mean(axis=0), disp[tail_mask].mean(axis=0)
    connected = dt <= .100001
    if connected.sum() < 5:
        raise ValueError("INSUFFICIENT_CONNECTED_KINEMATICS")
    velocity = np.diff(disp,axis=0)[connected] / dt[connected,None]
    drift = np.average(velocity,axis=0,weights=dt[connected])
    # Twelve kinematic features, per second, in canthus-aligned eye coordinates.
    return np.r_[disp.mean(axis=0), disp.std(axis=0), drift, tail - early,
                 np.max(np.abs(velocity), axis=0), disp[tail_mask].std(axis=0)]


def crop_features(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("EXPECTED_RGB_UINT8_CROP")
    h, w = rgb.shape[:2]
    if not (16 <= h <= 1024 and 32 <= w <= 1024):
        raise ValueError("CROP_TOO_SMALL_OR_LARGE")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if gray.std() < 4 or np.mean((gray < 4) | (gray > 251)) > .8:
        raise ValueError("LOW_INFORMATION_EYE_CROP")
    small = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor((64, 32), (16, 16), (8, 8), (8, 8), 9).compute(small).ravel()
    # Preserve eye shape; never swap/mirror eyes or use filename/device as input.
    return np.r_[hog, gray.mean() / 255, gray.std() / 255, w / h]


def paired_features(left_rgb, right_rgb, samples, event_ms=None, unit="milliseconds"):
    return np.r_[crop_features(left_rgb), crop_features(right_rgb),
                 trajectory_features(samples, event_ms, unit)]


def validate_source_event(event, samples):
    ts = timestamp_ms(event)
    matches = [s for s in samples if s.get("frameId") == event.get("frameId") and s.get("timestampMs") == ts]
    if type(event.get("frameId")) is not int or len(matches) != 1:
        raise ValueError("CROP_SOURCE_FRAME_MISMATCH")
    if event.get("mirrored") is not False:
        raise ValueError("UNVERIFIED_CROP_ORIENTATION")
    if event.get("cropGeometryVersion") != "source-eye-contour-v1":
        raise ValueError("UNVERIFIED_CROP_GEOMETRY")
    interval = event.get("onsetIntervalMs")
    if not isinstance(interval, list) or len(interval) != 2 or not all(finite(v) for v in interval):
        raise ValueError("MISSING_ONSET_INTERVAL")
    uncertainty = ts - interval[0]
    if interval[1] != ts or not 0 < uncertainty <= 100 or not finite(event.get("timingUncertaintyMs")) or abs(event["timingUncertaintyMs"] - uncertainty) > .002:
        raise ValueError("UNCERTAIN_ONSET_OVER_100MS")
    if not finite(event.get("confirmedAtMs")) or event["confirmedAtMs"] < ts:
        raise ValueError("INVALID_CONFIRMATION_TIMESTAMP")
    for side in ("left", "right"):
        crop = event.get("crops", {}).get(side, {})
        rect = crop.get("cropRegion", {})
        values = [rect.get(k) for k in ("x", "y", "width", "height")]
        if not all(type(v) is int for v in values):
            raise ValueError("INVALID_CROP_RECTANGLE")
        x, y, w, h = values
        sw, sh = event.get("sourceWidth"), event.get("sourceHeight")
        if not all(type(v) is int and v > 0 for v in (sw, sh)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > sw or y+h > sh or (w, h) != (crop.get("width"), crop.get("height")):
            raise ValueError("CROP_OUTSIDE_SOURCE_OR_RESIZED")
    return ts


class PairedCropPredictor:
    def __init__(self, pipeline, validation=None):
        self.pipeline = pipeline
        self.validation = validation or {}
        self.version = CONTRACT

    def predict_multimodal(self, payload):
        if payload.get("contract") != "cover-uncover-multimodal-v1":
            raise ValueError("CONTRACT_MISMATCH")
        probabilities, analyzed, rejected, timing, analyzed_cycles = [], [], {}, {}, set()
        cycles = payload.get("cycles", {})
        for event_id, item in payload.get("uncoverEvents", {}).items():
            try:
                event = item["event"]
                cycle = cycles.get(event["cycle"], cycles.get(str(event["cycle"]), {}))
                samples = cycle.get("samples", [])
                if cycle.get("primaryOcclusionVerified") is not True or cycle.get("occlusionVerified") is not True:
                    raise ValueError("UNVERIFIED_COVER_UNCOVER_PROCEDURE")
                quality = assess_timing(samples)
                timing[event_id] = quality
                if not quality["eligible30fps"]:
                    raise ValueError(quality["reason"])
                ts = validate_source_event(event, samples)
                images = item["images"]
                for side in ("left", "right"):
                    if images[side].shape[:2][::-1] != (event["crops"][side]["width"], event["crops"][side]["height"]):
                        raise ValueError("CROP_IMAGE_SIZE_MISMATCH")
                vector = paired_features(images["left"], images["right"], samples, ts)
                proba = self.pipeline.predict_proba(vector.reshape(1, -1))[0]
                classes = list(self.pipeline.classes_)
                if set(classes) != set(CLASSES) or not np.all(np.isfinite(proba)):
                    raise ValueError("INVALID_MODEL_CLASSES_OR_PROBABILITIES")
                probabilities.append([float(proba[classes.index(c)]) for c in CLASSES])
                analyzed.append(event_id)
                analyzed_cycles.add(event["cycle"])
            except (KeyError, ValueError, TypeError) as exc:
                rejected[event_id] = str(exc)
        result = {"prediction": "INCONCLUSIVE", "classProbabilities": None,
                  "confidence": None, "isReliable": False, "inputModalities": [],
                  "analyzedEventIds": analyzed, "rejectedEvents": rejected, "timingQuality": timing,
                  "validation": self.validation, "accuracy": self.validation.get("heldoutAccuracy"),
                  "notice": "Paired-eye research model; confidence is not accuracy."}
        if probabilities:
            avg = np.mean(probabilities, axis=0)
            result.update(classProbabilities=dict(zip(CLASSES, map(float, avg))),
                          confidence=float(max(avg)), inputModalities=["time_series", "uncover_eye_images"])
            # Incomplete sessions and unvalidated candidates remain abstentions.
            if not rejected and len(analyzed) >= 4 and len(analyzed_cycles) >= 2 and self.validation.get("promoted") is True:
                result.update(prediction=CLASSES[int(np.argmax(avg))], isReliable=True)
        return result
