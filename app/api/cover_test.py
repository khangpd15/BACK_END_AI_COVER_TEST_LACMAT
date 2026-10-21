
async def get_db_session():
    """Stateless dependency stub for backwards compatibility."""
    yield None

"""New API endpoint for RemiCare Cover Test persistent cloud storage and research inference."""

import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
from app.services.cover_test.multimodal_service import MAX_IMAGE_BYTES
from app.schemas.cover_test import (
    CoverTestSessionMetadataInput,
    CoverTestSessionResponse,
)
from app.services.cover_test.session_service import CoverTestSessionService

logger = logging.getLogger("remicare.api.cover_test")

router = APIRouter(prefix="/sessions", tags=["Cover Test Storage"])


@router.post(
    "",
    response_model=CoverTestSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Persist Cover Test 3-cycle session, trajectories, and eye crops to local storage & database",
    description=(
        "Receives multipart/form-data containing session metadata, 3 raw JSON trajectories, "
        "and frame-linked uncover eye pairs (up to 12 images). "
        "Guarantees raw data persistence in local storage and database before executing optional AI transfer inference."
    ),
)
async def persist_cover_test_session(
    request: Request,
    session_metadata: str = Form(..., description="JSON string of CoverTestSessionMetadataInput"),
    cycle_1_raw: Optional[UploadFile] = File(None),
    cycle_1_left_eye: Optional[UploadFile] = File(None),
    cycle_1_right_eye: Optional[UploadFile] = File(None),
    cycle_2_raw: Optional[UploadFile] = File(None),
    cycle_2_left_eye: Optional[UploadFile] = File(None),
    cycle_2_right_eye: Optional[UploadFile] = File(None),
    cycle_3_raw: Optional[UploadFile] = File(None),
    cycle_3_left_eye: Optional[UploadFile] = File(None),
    cycle_3_right_eye: Optional[UploadFile] = File(None),
    representative_image: Optional[UploadFile] = File(None, description="Deprecated/Ignored: Biometric images are not persisted to protect customer privacy"),
    uncover_manifest: Optional[str] = Form(None),
    run_inference: bool = Query(True, description="Whether to execute AI inference after storage"),
    ) -> CoverTestSessionResponse:
    # 1. Parse session_metadata JSON
    try:
        meta_dict = json.loads(session_metadata)
        metadata_obj = CoverTestSessionMetadataInput(**meta_dict)
    except Exception as e:
        logger.warning("Failed to parse session_metadata Form field: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid session_metadata JSON: {str(e)}",
        )

    # 2. Extract and parse raw cycle trajectories
    raw_files = [
        (1, cycle_1_raw),
        (2, cycle_2_raw),
        (3, cycle_3_raw),
    ]
    raw_trajectories = {}
    for c_num, raw_file in raw_files:
        if raw_file is not None and raw_file.filename:
            try:
                content_bytes = await raw_file.read()
                if content_bytes:
                    raw_trajectories[c_num] = json.loads(content_bytes.decode("utf-8"))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cycle {c_num} raw trajectory JSON corrupted: {str(e)}",
                )

    if not raw_trajectories:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one cycle raw trajectory (cycle_1_raw) is required.",
        )

    # Frame-linked pairs only. Old timer-triggered uploads retain legacy behavior.
    manifest = None
    eye_images = {}
    if uncover_manifest is not None:
        if len(uncover_manifest) > 128 * 1024:
            raise HTTPException(status_code=413, detail="Uncover manifest too large")
        try:
            manifest = json.loads(uncover_manifest)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid uncover manifest JSON")
        form = await request.form()
        for field, upload in form.multi_items():
            if field.startswith("c") and "_uncover_" in field:
                if not isinstance(upload, StarletteUploadFile) or field in eye_images or len(eye_images) >= 12:
                    raise HTTPException(status_code=422, detail="Invalid or duplicate uncover image")
                content = await upload.read(MAX_IMAGE_BYTES + 1)
                if len(content) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Eye image too large")
                eye_images[field] = content

    # 4. Delegate to Session Service
    try:
        session_service = CoverTestSessionService(db_session=None)
        return await session_service.persist_session(
            metadata_input=metadata_obj,
            raw_trajectories=raw_trajectories,
            run_inference=run_inference,
            uncover_manifest=manifest,
            uncover_images=eye_images,
        )
    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e)
        logger.error("[CoverTestPersistError] Unhandled session persistence error: %s", err_str, exc_info=True)
        err_detail = f"Cover test storage error: {err_str}"

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_detail,
        )

