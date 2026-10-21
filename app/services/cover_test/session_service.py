import asyncio
"""Core business logic service for Cover Test session persistence, validation, storage, and AI inference."""

from datetime import datetime, timezone
import json
import logging
import math
import re
import uuid
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ENABLE_AI_INFERENCE_ON_SAVE
from app.db.repositories.cover_test_repository import CoverTestRepository
from app.schemas.cover_test import (
    CoverTestSessionMetadataInput,
    CoverTestSessionResponse,
)
from app.services.cover_test.storage_service import (
    StorageService,
    get_storage_service,
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
        db_session: AsyncSession,
        storage_service: Optional[StorageService] = None,
    ):
        self.db_session = db_session
        self.repo = CoverTestRepository(db_session)
        self.storage = storage_service or get_storage_service()

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

        # 3. Determine storage paths
        now_utc = datetime.now(timezone.utc)
        year_str = now_utc.strftime("%Y")
        month_str = now_utc.strftime("%m")
        storage_root = f"cover-test-raw/{year_str}/{month_str}/{session_id_str}"
        storage_prefix = f"{year_str}/{month_str}/{session_id_str}"

        # 4. Upsert session initial state
        session_record = await self.repo.upsert_session(
            session_id=session_uuid,
            cycle_count=len(raw_trajectories),
            sampling_rate_hz=metadata_input.samplingRateHz,
            source_device=metadata_input.sourceDevice,
            tracker=metadata_input.tracker,
            raw_schema_version=metadata_input.rawSchemaVersion,
            storage_root=storage_root,
            processing_status="UPLOADING",
            client_metadata=metadata_input.clientMetadata or {},
        )

        # 5. Process and validate cycles
        cycles_to_save = []
        manifest_cycles = []
        raw_uploads = []
        img_uploads = []
        total_samples_saved = 0

        for cycle_num in sorted(raw_trajectories.keys()):
            raw_cycle = raw_trajectories[cycle_num]
            samples = raw_cycle.get("samples") or []
            total, valid_cnt, v_ratio, mean_q = validate_cycle_samples(samples, cycle_num)
            v_ratio = max(0.0, min(1.0, float(v_ratio)))
            mean_q = max(0.0, min(1.0, float(mean_q)))
            total_samples_saved += total

            covered_eye = normalize_covered_eye(raw_cycle.get("coveredEye"), cycle_num)
            tracked_eye = normalize_tracked_eye(raw_cycle.get("trackedEye"), cycle_num)

            raw_dur = raw_cycle.get("durationMs")
            if raw_dur is not None:
                try:
                    duration_ms = max(0, int(round(float(raw_dur))))
                except (ValueError, TypeError):
                    duration_ms = 0
            else:
                duration_ms = max(0, int(round(float(samples[-1].get("t", 0))))) if samples else 0

            # Upload raw.json to storage
            cycle_folder_name = f"cycle_{cycle_num:02d}"
            raw_json_storage_path = f"{storage_prefix}/{cycle_folder_name}/raw.json"
            clean_raw_payload = {
                "schemaVersion": metadata_input.rawSchemaVersion,
                "sessionId": session_id_str,
                "cycle": cycle_num,
                "coveredEye": covered_eye,
                "trackedEye": tracked_eye,
                "samplingRateHz": metadata_input.samplingRateHz,
                "durationMs": duration_ms,
                "samples": samples,
                "uncoverEvents": raw_cycle.get("uncoverEvents", []),
            }
            raw_uploads.append((raw_json_storage_path, clean_raw_payload))

            cycles_to_save.append({
                "cycle_number": cycle_num,
                "covered_eye": covered_eye,
                "tracked_eye": tracked_eye,
                "sample_count": total,
                "valid_sample_count": valid_cnt,
                "valid_ratio": v_ratio,
                "mean_tracking_quality": mean_q,
                "cycle_status": "COMPLETED",
                "raw_storage_path": raw_json_storage_path,
                "duration_ms": duration_ms,
            })
            manifest_cycles.append({
                "cycle": cycle_num,
                "storagePath": raw_json_storage_path,
                "sampleCount": total,
                "validRatio": v_ratio,
            })

        # Save cycles to database
        db_cycles = await self.repo.upsert_cycles(session_uuid, cycles_to_save)

        # 6. Save validated eye crops only, with frame/time links and checksums.
        # Never accept a client-controlled storage path or store full-face frames.
        stored_images = []
        for event_id, entry in decoded_events.items():
            event = entry["event"]
            for side in ("left", "right"):
                field = event["crops"][side]["field"]
                content = uncover_images[field]
                image_path = f"{storage_prefix}/cycle_{event['cycle']:02d}/{event_id}_{side}.jpg"
                img_uploads.append((image_path, content))
                stored_images.append({
                    "eventId": event_id, "eye": side.upper(), "frameId": event["frameId"],
                    "timestampMs": event["timestampMs"], "storagePath": image_path,
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                })

        # 7. Upload manifest.json and execute raw JSON storage uploads concurrently
        manifest_payload = {
            "sessionId": session_id_str,
            "createdAt": now_utc.isoformat(),
            "cycleCount": len(cycles_to_save),
            "samplingRateHz": metadata_input.samplingRateHz,
            "cycles": manifest_cycles,
            "imagesCount": len(stored_images),
            "images": stored_images,
            "uncoverManifest": uncover_manifest,
            "storageRoot": storage_root,
        }

        # Concurrently upload JSON trajectories and manifest
        upload_tasks = [
            self.storage.upload_raw_json(p, d) for p, d in raw_uploads
        ] + [
            self.storage.upload_eye_image(p, content) for p, content in img_uploads
        ] + [
            self.storage.upload_raw_json(f"{storage_prefix}/manifest.json", manifest_payload)
        ]
        if upload_tasks:
            await asyncio.gather(*upload_tasks)

        # Commit DB transaction so raw numeric trajectories and metadata are guaranteed persisted
        await self.db_session.commit()
        logger.info(
            "[CoverTestSessionPersisted] sessionId=%s, cycles=%d, storage=%s",
            session_id_str,
            len(cycles_to_save),
            storage_root,
        )

        # 8. 10-15 FPS Model Inference & Final Consensus Persistence (No Images Saved)
        ai_result_payload = None
        processing_status = "COMPLETED"

        should_infer = run_inference and ENABLE_AI_INFERENCE_ON_SAVE
        if uncover_manifest is not None:
            from app.services.cover_test.multimodal_service import get_multimodal_cover_service, inconclusive
            from starlette.concurrency import run_in_threadpool
            if should_infer:
                ai_result_payload = await run_in_threadpool(
                    get_multimodal_cover_service().predict, raw_trajectories, decoded_events,
                )
            else:
                ai_result_payload = inconclusive("INFERENCE_DISABLED", list(decoded_events))
            if ai_result_payload.get("model", {}).get("loaded") and ai_result_payload.get("inputModalities"):
                await self.repo.upsert_result(session_id=session_uuid, result_data={
                    "model_name": ai_result_payload["model"]["name"],
                    "model_version": ai_result_payload["model"].get("version") or "unversioned",
                    "model_source": "cover_multimodal",
                    "feature_schema_version": "cover-mm-v1",
                    "status": ai_result_payload["status"], "input_compatible": True,
                    "prediction": ai_result_payload["prediction"],
                    "class_probabilities": ai_result_payload["classProbabilities"],
                    "confidence": ai_result_payload["confidence"],
                    "features_snapshot": {"analyzedEventIds": ai_result_payload["analyzedEventIds"]},
                    "notice": ai_result_payload["notice"],
                })
            # No silent fall-through to the trajectory-only classifier.
            should_infer = False
        if should_infer:
            from app.services.fps_model_service import get_fps_model_service

            fps_svc = get_fps_model_service()
            if fps_svc.model is None:
                ai_result_payload = {
                    "status": "INCONCLUSIVE",
                    "prediction": "INCONCLUSIVE",
                    "reason": "MODEL_NOT_LOADED",
                    "reasonCodes": ["MODEL_NOT_LOADED"],
                    "model": {"name": None, "version": None, "loaded": False},
                    "classProbabilities": None,
                    "confidence": None,
                    "isReliable": False,
                    "notice": "Cover Test data saved; no approved AI model is loaded for this session.",
                }
                should_infer = False
        if should_infer:
            from app.services.fps_model_service import get_fps_model_service

            fps_svc = get_fps_model_service()

            # Step 1: Finalize result from 10-15 FPS Model with temporal consensus aggregation
            try:
                from starlette.concurrency import run_in_threadpool
                fps_res = await run_in_threadpool(fps_svc.aggregate_and_predict, raw_trajectories)
                logger.info(
                    "[MODEL]\nsource = fps_10_15\nprediction = %s\nprobabilities = %s",
                    fps_res.prediction,
                    fps_res.class_probabilities,
                )
            except Exception as model_err:
                logger.error("[MODEL]\nsource = fps_10_15\ninference = failed\nerror = %s", model_err)
                fps_res = None

            # If 10-15 FPS model inference fails -> DO NOT proceed to DB result insert
            if fps_res is None or (fps_res.prediction == "INCONCLUSIVE" and fps_res.total_frames_evaluated == 0):
                logger.warning(
                    "[AIInferenceFailed] sessionId=%s: 10-15 FPS model failed to evaluate frames. Preserving raw data with PARTIAL_SUCCESS.",
                    session_id_str,
                )
                processing_status = "PARTIAL_SUCCESS"
                ai_result_payload = None
            else:
                # Step 2: Korean transfer models have been retired from runtime.
                # Cover Test results must come from the active RemiCare-trained model only.
                comparison_models_list = []

                # Step 3: INSERT 1 record into cover_test_results with 10-15 FPS prediction (NO image stored)
                target_prediction = fps_res.prediction
                target_probabilities = fps_res.class_probabilities
                target_model_name = fps_res.model_name
                target_model_version = fps_res.model_version
                target_sampling_profile = "RemiCare Cover Test numeric trajectory consensus"
                target_notice = fps_res.notice

                try:
                    result_record = await self.repo.upsert_result(
                        session_id=session_uuid,
                        result_data={
                            "model_name": target_model_name,
                            "model_version": target_model_version,
                            "model_source": "fps_10_15_model",
                            "feature_schema_version": "shared-v1.0.0",
                            "status": fps_res.status,
                            "input_compatible": True,
                            "prediction": target_prediction,
                            "class_probabilities": target_probabilities,
                            "confidence": float(target_probabilities.get(target_prediction, 0.5)),
                            "domain_shift_warning": False,
                            "features_snapshot": fps_res.features_snapshot,
                            "comparison_models": comparison_models_list,
                            "notice": target_notice,
                        },
                    )
                    await self.db_session.commit()
                    logger.info("[DATABASE]\ninsert = success\nrecord_id = %s", str(result_record.result_id))

                    ai_result_payload = {
                        "status": fps_res.status,
                        "modelSource": "fps_10_15_model",
                        "prediction": target_prediction,
                        "classProbability": target_probabilities,
                        "classProbabilities": target_probabilities,
                        "confidence": float(target_probabilities.get(target_prediction, 0.5)),
                        "model": {
                            "name": target_model_name,
                            "version": target_model_version,
                            "source": "fps_10_15_model",
                        },
                        "samplingProfile": target_sampling_profile,
                        "agreementRatio": fps_res.agreement_ratio if fps_res else 1.0,
                        "isReliable": fps_res.is_reliable if fps_res else True,
                        "features": fps_res.features_snapshot if fps_res else {},
                        "comparisonModels": comparison_models_list,
                        "notice": target_notice,
                    }
                    processing_status = "COMPLETED"

                except Exception as db_insert_err:
                    logger.error("[DATABASE]\ninsert = failed\nerror = %s", db_insert_err)
                    raise db_insert_err

        # Update final processing status
        await self.repo.update_session_status(session_uuid, processing_status)
        await self.db_session.commit()

        return CoverTestSessionResponse(
            success=True,
            sessionId=session_id_str,
            saved=True,
            processingStatus=processing_status,
            storageRoot=storage_root,
            cyclesSaved=len(cycles_to_save),
            imagesSaved=len(stored_images),
            storageMode="CLOUD" if self.storage.is_live else "MEMORY",
            aiResult=ai_result_payload,
            message="Cover test data saved. Check aiResult for model availability and analyzed modalities.",
        )
