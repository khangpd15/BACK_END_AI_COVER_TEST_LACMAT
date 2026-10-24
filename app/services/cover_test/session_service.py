"""Core business logic service for Cover Test session persistence, validation, storage, and AI inference."""

import logging
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
from app.config import ENABLE_AI_INFERENCE_ON_SAVE
from app.schemas.cover_test import (
    CoverTestSessionMetadataInput,
    CoverTestSessionResponse,
)
from app.services.cover_test.storage_service import (
    StorageService,
)

logger = logging.getLogger("remicare.session_service")

# Regular expression for strict RFC 4122 UUID v4 validation
UUID_V4_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Forbidden PII keys
FORBIDDEN_PII_KEYS = {
    "name",
    "phone",
    "phonenumber",
    "email",
    "cccd",
    "cmnd",
    "ssn",
    "dob",
    "dateofbirth",
    "address",
    "patientname",
}

VALID_PHASES = {
    "BASELINE",
    "COVER",
    "UNCOVER",
    "TRACKING",
    "COVER_LEFT",
    "COVER_RIGHT",
    "UNCOVER_LEFT",
    "UNCOVER_RIGHT",
    "CYCLE_COMPLETE",
    "NEXT_CYCLE",
    "FINISHED",
}

VALID_EYES = {"LEFT", "RIGHT"}


def validate_uuid_v4(val: str) -> uuid.UUID:
    """Validates that a string is a valid UUID format (v4, v7, etc.)."""
    if not val or not isinstance(val, str):
        raise HTTPException(
            status_code=422,
            detail="sessionId is required and must be a valid UUID string.",
        )
    clean_val = val.strip().strip("{}")
    try:
        return uuid.UUID(clean_val)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=422,
            detail=f"sessionId '{clean_val}' is not a valid UUID v4.",
        )


def normalize_covered_eye(eye_val: Any, cycle_num: int) -> str:
    """Normalizes covered eye string to strictly match database check constraints."""
    val = str(eye_val or "").strip().upper()
    if val in {"LEFT", "COVER_LEFT", "OS", "L"}:
        return "LEFT"
    if val in {"RIGHT", "COVER_RIGHT", "OD", "R"}:
        return "RIGHT"
    if val in {"ALTERNATING", "ALT", "BOTH"}:
        return "ALTERNATING"
    return "LEFT" if cycle_num % 2 == 1 else "RIGHT"


def normalize_tracked_eye(eye_val: Any, cycle_num: int) -> str:
    """Normalizes tracked eye string to strictly match database check constraints."""
    val = str(eye_val or "").strip().upper()
    if val in {"RIGHT", "TRACK_RIGHT", "OD", "R"}:
        return "RIGHT"
    if val in {"LEFT", "TRACK_LEFT", "OS", "L"}:
        return "LEFT"
    if val in {"BOTH", "OU"}:
        return "BOTH"
    return "RIGHT" if cycle_num % 2 == 1 else "LEFT"


def check_pii(data: Any, path: str = "") -> None:
    """Recursively checks for forbidden PII keys in dictionaries."""
    if isinstance(data, dict):
        for k, v in data.items():
            norm_k = k.lower().replace("_", "").replace("-", "")
            if norm_k in FORBIDDEN_PII_KEYS:
                raise HTTPException(
                    status_code=422,
                    detail=f"PII field '{k}' detected at '{path}'. PII is strictly forbidden.",
                )
            check_pii(v, f"{path}.{k}" if path else k)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            check_pii(item, f"{path}[{idx}]")


def validate_coordinate(val: Any, field_name: str, sample_idx: int) -> Optional[float]:
    """Validates that coordinate is finite and in range [0.0, 1.0] or None."""
    if val is None:
        return None
    if not isinstance(val, (int, float)):
        raise HTTPException(
            status_code=422,
            detail=f"Sample [{sample_idx}]: {field_name} must be numeric, got {type(val).__name__}",
        )
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval):
        raise HTTPException(
            status_code=422,
            detail=f"Sample [{sample_idx}]: {field_name} must be finite (not NaN/Infinity), got {fval}",
        )
    if fval < 0.0 or fval > 1.0:
        raise HTTPException(
            status_code=422,
            detail=f"Sample [{sample_idx}]: {field_name} out of bounds ({fval}). Must be between 0.0 and 1.0.",
        )
    return fval


def validate_cycle_samples(samples: List[Dict[str, Any]], cycle_num: int) -> Tuple[int, int, float, float]:
    """Validates time-series samples within a cycle.
    
    Checks timestamp monotonicity, coordinate bounds, trackingQuality bounds.
    Returns: (total_samples, valid_samples, valid_ratio, mean_tracking_quality)
    """
    if not isinstance(samples, list) or len(samples) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"Cycle {cycle_num}: samples must be a non-empty array.",
        )

    last_t: Optional[float] = None
    valid_count = 0
    total_quality = 0.0

    for idx, s in enumerate(samples):
        if not isinstance(s, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}] must be a JSON object.",
            )

        # Check PII inside sample
        check_pii(s, f"cycle_{cycle_num}.sample_{idx}")

        # Timestamp monotonicity check
        t_val = s.get("t")
        if t_val is None:
            t_val = s.get("timestamp")
        if t_val is None or not isinstance(t_val, (int, float)) or not math.isfinite(float(t_val)):
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: timestamp/t must be a finite valid number.",
            )
        f_t = float(t_val)
        if f_t < 0.0:
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: timestamp/t cannot be negative ({f_t}).",
            )
        if last_t is not None and f_t <= last_t:
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: non-monotonic or duplicate timestamp detected ({f_t} <= {last_t}).",
            )
        last_t = f_t

        # Tracking quality check
        tq = s.get("trackingQuality", 0.95)
        if not isinstance(tq, (int, float)) or math.isnan(float(tq)) or math.isinf(float(tq)):
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: trackingQuality must be finite numeric.",
            )
        f_tq = float(tq)
        if f_tq < 0.0 or f_tq > 1.0:
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: trackingQuality must be in [0.0, 1.0], got {f_tq}.",
            )
        total_quality += f_tq

        # Coordinates check
        lx = s.get("leftX") if "leftX" in s else (s.get("leftEye", {}) or {}).get("x")
        ly = s.get("leftY") if "leftY" in s else (s.get("leftEye", {}) or {}).get("y")
        rx = s.get("rightX") if "rightX" in s else (s.get("rightEye", {}) or {}).get("x")
        ry = s.get("rightY") if "rightY" in s else (s.get("rightEye", {}) or {}).get("y")

        validate_coordinate(lx, "leftX", idx)
        validate_coordinate(ly, "leftY", idx)
        validate_coordinate(rx, "rightX", idx)
        validate_coordinate(ry, "rightY", idx)
        if "x" in s:
            validate_coordinate(s.get("x"), "x", idx)
        if "y" in s:
            validate_coordinate(s.get("y"), "y", idx)

        # Validity count
        l_valid = s.get("leftValid", False)
        r_valid = s.get("rightValid", False)
        if l_valid or r_valid:
            valid_count += 1

    total = len(samples)
    v_ratio = round(valid_count / total, 4) if total > 0 else 0.0
    mean_q = round(total_quality / total, 4) if total > 0 else 0.0
    return total, valid_count, v_ratio, mean_q


class CoverTestSessionService:
    """Orchestrates validation, persistence, storage, and AI inference for a Cover Test session."""

    def __init__(
        self,
        db_session: Optional[Any] = None,
        storage_service: Optional[StorageService] = None,
    ):
        self.db_session = db_session
        self.repo = None
        self.storage = None

    async def persist_session(
        self,
        metadata_input: CoverTestSessionMetadataInput,
        raw_trajectories: Dict[int, Dict[str, Any]],
        images_data: Optional[Dict[Tuple[int, str], Tuple[bytes, str]]] = None,
        representative_image: Optional[Tuple[bytes, str]] = None,
        run_inference: bool = True,
        uncover_manifest: Optional[dict] = None,
        uncover_images: Optional[Dict[str, bytes]] = None,
    ) -> CoverTestSessionResponse:
        # 1. Validate canonical UUID v4
        session_uuid = validate_uuid_v4(metadata_input.sessionId)
        session_id_str = str(session_uuid)

        # 2. Validate PII in metadata
        check_pii(metadata_input.clientMetadata or {}, "clientMetadata")
        decoded_events = {}
        if uncover_manifest is not None:
            from app.services.cover_test.multimodal_service import validate_uncover_bundle
            check_pii(uncover_manifest, "uncoverManifest")
            decoded_events = validate_uncover_bundle(
                uncover_manifest, raw_trajectories, uncover_images or {}, session_id_str,
            )

        # Validate and infer entirely within this request. No repository, files,
        # or singleton memory store may retain the user's measurements or images.
        if not raw_trajectories:
            raise HTTPException(status_code=422, detail="No cycle samples supplied")
        for cycle_num, raw in raw_trajectories.items():
            validate_cycle_samples(raw.get("samples") or [], cycle_num)

        from app.services.cover_test.multimodal_service import get_multimodal_cover_service, inconclusive
        from starlette.concurrency import run_in_threadpool
        should_infer = run_inference and ENABLE_AI_INFERENCE_ON_SAVE
        if not should_infer:
            result = inconclusive("INFERENCE_DISABLED", list(decoded_events))
        elif uncover_manifest is not None:
            result = await run_in_threadpool(
                get_multimodal_cover_service().predict, raw_trajectories, decoded_events,
            )
        else:
            from app.services.fps_model_service import get_fps_model_service
            model = get_fps_model_service()
            if model.model is None:
                result = inconclusive("MODEL_NOT_LOADED", [])
            else:
                prediction = await run_in_threadpool(model.aggregate_and_predict, raw_trajectories)
                result = {
                    "status": prediction.status, "prediction": prediction.prediction,
                    "classProbabilities": prediction.class_probabilities,
                    "confidence": prediction.class_probabilities.get(prediction.prediction),
                    "isReliable": prediction.is_reliable,
                    "model": {"name": prediction.model_name, "version": prediction.model_version, "loaded": True},
                    "notice": prediction.notice,
                }
        return CoverTestSessionResponse(
            success=True, sessionId=session_id_str, saved=False,
            processingStatus="COMPLETED", storageRoot="", storageMode="STATELESS",
            cyclesSaved=0, imagesSaved=0, aiResult=result,
            message="Analysis completed without retaining user measurements or images.",
        )
