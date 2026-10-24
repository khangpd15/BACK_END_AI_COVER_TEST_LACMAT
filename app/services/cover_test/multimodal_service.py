"""Frame-linked eye images and an explicit contract for a trained multimodal model.

The numeric-only classifier is never presented as an image+trajectory classifier.
Only a configured, trusted server-side artifact can produce a combined prediction.
"""
from io import BytesIO
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 1024 * 1024
CONTRACT = "cover-uncover-multimodal-v1"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "cover_multimodal_model.joblib"


def reject(message):
    raise HTTPException(status_code=422, detail=message)


def number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_uncover_bundle(manifest, trajectories, images, session_id):
    """Validate all links and decode images before storage/inference side effects."""
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != "2.0.0":
        reject("Unsupported uncover manifest")
    if manifest.get("sessionId") != session_id:
        reject("Uncover manifest session mismatch")
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) > 6:
        reject("Expected at most six uncover events")
    if events != [event for c in trajectories.values() for event in c.get("uncoverEvents", [])]:
        reject("Cycle and manifest events disagree")
    decoded, seen, used = {}, set(), set()
    for event in events:
        if not isinstance(event, dict):
            reject("Invalid uncover event")
        cycle, eye = event.get("cycle"), event.get("uncoveredEye")
        if type(cycle) is not int or cycle not in trajectories or not isinstance(eye, str) or eye not in {"LEFT", "RIGHT"} or (cycle, eye) in seen:
            reject("Invalid or repeated uncover cycle/eye")
        seen.add((cycle, eye))
        if event.get("status") in {"NOT_CAPTURED", "FAILED"}:
            continue
        if event.get("status") != "CAPTURED":
            reject("Invalid uncover status")
        event_id = f"c{cycle}_{eye.lower()}_uncover"
        if event.get("eventId") != event_id:
            reject("Invalid event ID")
        ts, frame_id = event.get("timestampMs"), event.get("frameId")
        if not number(ts) or ts < 0 or type(frame_id) is not int or frame_id < 1:
            reject("Invalid source frame identity")
        matches = [s for s in trajectories[cycle].get("samples", [])
                   if s.get("frameId") == frame_id and s.get("timestampMs") == ts]
        if len(matches) != 1:
            reject("Image must match exactly one trajectory frame")
        sample = matches[0]
        if not number(event.get("t")) or abs(event["t"] - sample.get("t", -1)) > 0.002:
            reject("Event and sample cycle-relative timestamps disagree")
        interval = event.get("onsetIntervalMs")
        confirmed = event.get("confirmedAtMs")
        if not isinstance(interval, list) or len(interval) != 2 or not all(map(number, interval)):
            reject("Missing onset uncertainty interval")
        if not number(confirmed) or not (0 <= interval[0] < ts == interval[1] <= confirmed):
            reject("Invalid onset/confirmation times")
        if not number(event.get("timingUncertaintyMs")) or abs(event["timingUncertaintyMs"] - (ts - interval[0])) > 0.002:
            reject("Timing uncertainty does not match source frames")
        if event.get("mirrored") is not False:
            reject("Uncover crops must retain unmirrored camera orientation")
        if not isinstance(event.get("crops"), dict):
            reject("Missing crop metadata")
        pair = {}
        for side in ("left", "right"):
            crop = event.get("crops", {}).get(side, {})
            field = f"{event_id}_{side}_eye"
            if not isinstance(crop, dict) or crop.get("field") != field or crop.get("file") != field + ".jpg":
                reject("Invalid crop field")
            content = images.get(field)
            if not isinstance(content, bytes) or not 0 < len(content) <= MAX_IMAGE_BYTES:
                reject("Missing or oversized JPEG eye crop")
            try:
                with Image.open(BytesIO(content)) as image:
                    if image.format != "JPEG" or not (8 <= image.width <= 1024 and 8 <= image.height <= 1024):
                        reject("Eye crops must be JPEG images of 8..1024 pixels")
                    if image.size != (crop.get("width"), crop.get("height")):
                        reject("Crop dimensions disagree with JPEG")
                    image.load()
                    pair[side] = np.asarray(image.convert("RGB")).copy()
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                reject("Invalid JPEG eye crop")
            used.add(field)
        decoded[event_id] = {"event": event, "images": pair}
    if used != set(images):
        reject("Unreferenced or duplicated eye image fields")
    return decoded


def inconclusive(reason, event_ids, loaded=False):
    return {
        "status": "INCONCLUSIVE", "prediction": "INCONCLUSIVE", "reason": reason,
        "classProbabilities": None, "confidence": None, "isReliable": False,
        "model": {"name": "cover-uncover-multimodal", "loaded": loaded},
        "receivedModalities": ["time_series", "uncover_eye_images"],
        "inputModalities": [], "analyzedEventIds": [],
        "receivedEventIds": event_ids,
        "notice": "Dữ liệu đã được nhận; chưa có kết quả mô hình kết hợp ảnh mắt và chuỗi thời gian.",
    }


class MultimodalCoverService:
    def __init__(self, artifact_path=None):
        self.artifact = None
        self.load_error = False
        # An explicit deployment path wins; otherwise load the artifact shipped
        # in the Docker image at /srv/remicare/models/.
        path = artifact_path or os.getenv("COVER_MULTIMODAL_MODEL_PATH") or DEFAULT_MODEL_PATH
        try:
            artifact = joblib.load(path)  # trusted administrator-configured artifact only
            if artifact.get("contract") != CONTRACT or not callable(getattr(artifact.get("model"), "predict_multimodal", None)):
                raise ValueError("Multimodal artifact contract mismatch")
            self.artifact = artifact
        except Exception:
            self.load_error = True

    def health(self):
        """Expose model availability without leaking artifact or file details."""
        if self.artifact is None:
            return {"loaded": False, "status": "load_failed" if self.load_error else "not_loaded"}
        return {
            "loaded": True,
            "name": self.artifact.get("name", CONTRACT),
            "version": self.artifact.get("version"),
        }

    def predict(self, trajectories, decoded):
        ids = list(decoded)
        if not ids:
            return inconclusive("UNCOVER_IMAGES_UNAVAILABLE", ids)
        if self.artifact is None:
            from .research_evidence import collect_research_evidence
            result = inconclusive("MULTIMODAL_MODEL_LOAD_FAILED" if self.load_error else "MULTIMODAL_MODEL_NOT_LOADED", ids)
            result["researchEvidence"] = collect_research_evidence(trajectories, decoded)
            return result
        try:
            result = self.artifact["model"].predict_multimodal({
                "contract": CONTRACT, "cycles": trajectories, "uncoverEvents": decoded,
            })
            probabilities = result.get("classProbabilities")
            labels = self.artifact.get("classes", ["NORMAL", "STRABISMUS"])
            if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
                raise ValueError("Invalid model output classes")
            if not all(number(p) and 0 <= p <= 1 for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 0.001:
                raise ValueError("Invalid model probabilities")
            # The adapter must acknowledge actual image consumption, not merely receipt.
            if set(result.get("analyzedEventIds", [])) != set(ids):
                raise ValueError("Model did not analyze all image pairs")
            prediction = result.get("prediction")
            if prediction not in labels and prediction != "INCONCLUSIVE":
                raise ValueError("Invalid prediction")
            return {
                "status": "INCONCLUSIVE" if prediction == "INCONCLUSIVE" else "COMPLETED",
                "prediction": prediction, "classProbabilities": probabilities,
                "confidence": probabilities.get(prediction), "isReliable": bool(result.get("isReliable", False)),
                "inputModalities": result.get("inputModalities", []), "analyzedEventIds": ids,
                "model": {"name": self.artifact.get("name", CONTRACT),
                          "version": self.artifact.get("version"), "loaded": True},
                "notice": result.get("notice", "Combined image and trajectory model output."),
            }
        except Exception:
            return inconclusive("MULTIMODAL_INFERENCE_FAILED", ids, loaded=True)


@lru_cache(maxsize=1)
def get_multimodal_cover_service():
    return MultimodalCoverService()
