"""API endpoint and validation for saving raw Cover Test session sampling data.

Stores sessions under data/cover_test/sessions/<session_id>/ without overwriting old sessions.
Provides strict validation of timestamps, phases, coveredEye, and coordinates.
"""

from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("remicare.cover_test_session")

router = APIRouter(tags=["Cover Test Sessions"])

VALID_PHASES = {
    "BASELINE",
    "COVER",
    "UNCOVER",
    "TRACKING",
    "COVER_LEFT",
    "COVER_RIGHT",
    "UNCOVER_LEFT",
    "UNCOVER_RIGHT",
    "INTRO",
    "PREPARING",
    "CYCLE_COMPLETE",
    "NEXT_CYCLE",
    "COMPLETE",
    "FINISHED",
}

VALID_EYES = {"LEFT", "RIGHT"}

FORBIDDEN_PII_FIELDS = {
    "name",
    "phone",
    "phonenumber",
    "email",
    "address",
    "cccd",
    "cmnd",
    "ssn",
    "patientname",
}


def _validate_coordinate(val: Any, field_name: str, sample_idx: int) -> Optional[float]:
    if val is None:
        return None
    if not isinstance(val, (int, float)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Sample [{sample_idx}]: {field_name} must be numeric, got {type(val).__name__}",
        )
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Sample [{sample_idx}]: {field_name} must be finite, got {fval}",
        )
    if fval < 0.0 or fval > 1.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Sample [{sample_idx}]: {field_name} out of bounds ({fval}). Must be between 0.0 and 1.0.",
        )
    return fval


class CoverTestSessionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    sessionId: str = Field(..., description="Unique session ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    samples: List[Dict[str, Any]] = Field(..., description="List of raw time-series samples")


class CoverTestSessionResponse(BaseModel):
    success: bool
    sessionId: str
    saved: bool
    sessionPath: str
    sampleCount: int
    message: str = "Session sampling data saved successfully"


@router.post(
    "/sessions",
    response_model=CoverTestSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save Cover Test session sampling data",
    description=(
        "Validates and persists raw Cover Test time-series sampling data into data/cover_test/sessions/<session_id>/ "
        "with metadata.json and sampling.json for future AI research and training."
    ),
)
async def save_cover_test_session(payload: CoverTestSessionRequest) -> CoverTestSessionResponse:
    session_id = payload.sessionId.strip()
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sessionId must not be empty",
        )

    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', session_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid sessionId '{session_id}'. Must be alphanumeric, underscores, hyphens, or dots.",
        )

    # Validate samples list
    if not isinstance(payload.samples, list) or len(payload.samples) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="samples must be a non-empty array of sample objects",
        )

    # Validate PII in metadata
    meta = payload.metadata or {}
    for key in meta.keys():
        if key.lower().replace("_", "") in FORBIDDEN_PII_FIELDS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"PII field '{key}' is forbidden in session metadata for privacy compliance",
            )

    # Validate each sample
    validated_samples: List[Dict[str, Any]] = []
    last_timestamp: Optional[float] = None
    for idx, s in enumerate(payload.samples):
        if not isinstance(s, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sample [{idx}] must be a JSON object",
            )

        # Check PII in sample
        for key in s.keys():
            if key.lower().replace("_", "") in FORBIDDEN_PII_FIELDS:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"PII field '{key}' is forbidden in sample [{idx}] for privacy compliance",
                )

        # Timestamp validation
        timestamp = s.get("timestamp")
        if timestamp is None:
            timestamp = s.get("t")
        if timestamp is None or not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sample [{idx}]: timestamp must be a finite valid numeric value",
            )
        timestamp_float = float(timestamp)
        if timestamp_float < 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sample [{idx}]: timestamp cannot be negative",
            )
        if last_timestamp is not None and timestamp_float <= last_timestamp:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sample [{idx}]: timestamp must be strictly increasing ({timestamp_float} <= {last_timestamp})",
            )
        last_timestamp = timestamp_float

        # Phase validation
        phase = s.get("phase")
        if not phase or not isinstance(phase, str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sample [{idx}]: phase is required and must be a string",
            )
        phase_upper = phase.strip().upper()
        if phase_upper not in VALID_PHASES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sample [{idx}]: invalid phase '{phase}'. Valid phases: {sorted(list(VALID_PHASES))}",
            )

        # Covered eye validation
        covered_eye = s.get("coveredEye")
        if covered_eye is not None:
            if not isinstance(covered_eye, str):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Sample [{idx}]: coveredEye must be string or null",
                )
            ce_upper = covered_eye.strip().upper()
            if ce_upper not in VALID_EYES and ce_upper not in {"NONE", "NULL", ""}:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Sample [{idx}]: invalid coveredEye '{covered_eye}'. Must be 'LEFT', 'RIGHT', or null",
                )
            if ce_upper in {"NONE", "NULL", ""}:
                covered_eye = None
            else:
                covered_eye = ce_upper

        # Validate coordinates (leftEye / rightEye / left / right / leftX / leftY / rightX / rightY)
        left_eye_obj = s.get("leftEye") or s.get("left") or {}
        right_eye_obj = s.get("rightEye") or s.get("right") or {}

        # Coordinate extraction
        lx = left_eye_obj.get("x") if isinstance(left_eye_obj, dict) else s.get("leftX")
        ly = left_eye_obj.get("y") if isinstance(left_eye_obj, dict) else s.get("leftY")
        rx = right_eye_obj.get("x") if isinstance(right_eye_obj, dict) else s.get("rightX")
        ry = right_eye_obj.get("y") if isinstance(right_eye_obj, dict) else s.get("rightY")

        # Perform strict boundary checks
        _validate_coordinate(lx, "leftEye.x", idx)
        _validate_coordinate(ly, "leftEye.y", idx)
        _validate_coordinate(rx, "rightEye.x", idx)
        _validate_coordinate(ry, "rightEye.y", idx)

        # Also validate top-level leftX, leftY, rightX, rightY if present
        if "leftX" in s:
            _validate_coordinate(s.get("leftX"), "leftX", idx)
        if "leftY" in s:
            _validate_coordinate(s.get("leftY"), "leftY", idx)
        if "rightX" in s:
            _validate_coordinate(s.get("rightX"), "rightX", idx)
        if "rightY" in s:
            _validate_coordinate(s.get("rightY"), "rightY", idx)

        validated_samples.append(s)

    # Determine backend project root and storage directories
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sessions_base_dir = os.path.join(backend_root, "data", "cover_test", "sessions")
    os.makedirs(sessions_base_dir, exist_ok=True)

    # Prevent overwriting old sessions: if folder exists, create unique suffix
    session_folder_name = session_id
    target_session_dir = os.path.join(sessions_base_dir, session_folder_name)
    if os.path.exists(target_session_dir):
        unique_suffix = f"{int(time.time() * 1000)}"
        session_folder_name = f"{session_id}_{unique_suffix}"
        target_session_dir = os.path.join(sessions_base_dir, session_folder_name)
        logger.warning(
            f"[SessionCollisionAvoidance] Session folder '{session_id}' already exists. "
            f"Writing to new unique folder '{session_folder_name}' to prevent overwrite."
        )

    os.makedirs(target_session_dir, exist_ok=True)

    # Construct metadata.json
    created_at = meta.get("createdAt") or datetime.now(timezone.utc).isoformat()
    final_metadata = {
        "sessionId": session_id,
        "testType": meta.get("testType", "cover_test"),
        "createdAt": created_at,
        "samplingRate": meta.get("samplingRate", 15),
        "protocolVersion": meta.get("protocolVersion", "cover-test-v1"),
        "camera": meta.get("camera", {"mirrored": True}),
    }
    # Include any additional research non-PII metadata
    for k, v in meta.items():
        if k not in final_metadata:
            final_metadata[k] = v

    metadata_path = os.path.join(target_session_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2, ensure_ascii=False)

    # Construct sampling.json
    final_sampling = {
        "sessionId": session_id,
        "sampleCount": len(validated_samples),
        "samples": validated_samples,
    }
    sampling_path = os.path.join(target_session_dir, "sampling.json")
    with open(sampling_path, "w", encoding="utf-8") as f:
        json.dump(final_sampling, f, indent=2, ensure_ascii=False)

    rel_session_path = os.path.relpath(target_session_dir, backend_root).replace("\\", "/")
    logger.info(
        f"[SessionSavedSuccessfully] sessionId={session_id}, samples={len(validated_samples)}, path={rel_session_path}"
    )

    return CoverTestSessionResponse(
        success=True,
        sessionId=session_id,
        saved=True,
        sessionPath=rel_session_path,
        sampleCount=len(validated_samples),
        message="Session sampling data saved successfully",
    )
