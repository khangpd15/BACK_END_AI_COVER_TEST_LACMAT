import asyncio
import json
from copy import deepcopy
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from PIL import Image

from app.services.cover_test.multimodal_service import validate_uncover_bundle, MultimodalCoverService

SESSION = "11111111-1111-4111-8111-111111111111"


def bundle():
    out = BytesIO()
    Image.new("RGB", (32, 24), (120, 100, 80)).save(out, "JPEG")
    event = {
        "eventId": "c1_left_uncover", "cycle": 1, "uncoveredEye": "LEFT",
        "status": "CAPTURED", "frameId": 12, "timestampMs": 400.25, "t": 300.25,
        "onsetIntervalMs": [367, 400.25], "confirmedAtMs": 434,
        "timingUncertaintyMs": 33.25, "mirrored": False, "crops": {},
    }
    images = {}
    for side in ("left", "right"):
        field = f"c1_left_uncover_{side}_eye"
        event["crops"][side] = {"field": field, "file": field + ".jpg", "width": 32, "height": 24}
        images[field] = out.getvalue()
    cycles = {1: {"uncoverEvents": [event], "samples": [
        {"t": 267, "timestampMs": 367, "frameId": 11},
        {"t": 300.25, "timestampMs": 400.25, "frameId": 12},
        {"t": 334, "timestampMs": 434, "frameId": 13},
    ]}}
    return {"schemaVersion": "2.0.0", "sessionId": SESSION, "events": [event]}, cycles, images


def test_pair_links_exactly_to_source_frame(monkeypatch):
    monkeypatch.delenv("HIRSCHBERG_RESEARCH_ROOT", raising=False)
    manifest, cycles, images = bundle()
    decoded = validate_uncover_bundle(manifest, cycles, images, SESSION)
    assert decoded["c1_left_uncover"]["images"]["right"].shape == (24, 32, 3)
    service = MultimodalCoverService()
    service.artifact = None
    result = service.predict(cycles, decoded)
    assert result["prediction"] == "INCONCLUSIVE"
    assert result["inputModalities"] == []
    assert result["researchEvidence"]["timing"][0]["observedMedianFps"] == pytest.approx(29.85, abs=.01)


@pytest.mark.parametrize("fault", ["timestamp", "missing_eye", "bad_jpeg", "mirrored", "duplicate", "extra"])
def test_reject_mismatched_or_incomplete_pairs(fault):
    manifest, cycles, images = bundle()
    if fault == "timestamp":
        cycles[1]["samples"][1]["timestampMs"] += 1
    elif fault == "missing_eye":
        images.pop("c1_left_uncover_right_eye")
    elif fault == "bad_jpeg":
        images["c1_left_uncover_left_eye"] = b"bad"
    elif fault == "mirrored":
        manifest["events"][0]["mirrored"] = True
    elif fault == "duplicate":
        cycles[1]["samples"].append(deepcopy(cycles[1]["samples"][1]))
    else:
        images["extra"] = b"bad"
    with pytest.raises(HTTPException) as error:
        validate_uncover_bundle(manifest, cycles, images, SESSION)
    assert error.value.status_code == 422


def test_persistence_saves_both_crops_and_frame_manifest(monkeypatch):
    from app.services.cover_test import session_service as module
    from app.schemas.cover_test import CoverTestSessionMetadataInput
    monkeypatch.setattr(module, "ENABLE_AI_INFERENCE_ON_SAVE", False)
    manifest, cycles, images = bundle()
    service = module.CoverTestSessionService.__new__(module.CoverTestSessionService)
    service.db_session, service.repo, service.storage = AsyncMock(), AsyncMock(), AsyncMock()
    service.storage.is_live = False
    result = asyncio.run(service.persist_session(
        CoverTestSessionMetadataInput(sessionId=SESSION), cycles,
        uncover_manifest=manifest, uncover_images=images))
    assert result.imagesSaved == 2
    assert result.aiResult["reason"] == "INFERENCE_DISABLED"
    assert service.storage.upload_eye_image.await_count == 2
    stored = service.storage.upload_raw_json.call_args_list[-1].args[1]
    assert stored["images"][0]["timestampMs"] == 400.25
    assert stored["uncoverManifest"] == manifest
    service.repo.upsert_result.assert_not_called()


def test_multipart_route_receives_dynamic_pair_fields(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api import cover_test
    from app.schemas.cover_test import CoverTestSessionResponse
    manifest, cycles, images = bundle()
    received = {}

    async def persist(self, **kwargs):
        received.update(kwargs)
        validate_uncover_bundle(kwargs["uncover_manifest"], kwargs["raw_trajectories"],
                                kwargs["uncover_images"], SESSION)
        return CoverTestSessionResponse(success=True, saved=True, sessionId=SESSION,
                                        processingStatus="COMPLETED", storageRoot="mock",
                                        cyclesSaved=1, imagesSaved=2, message="saved")

    async def database():
        yield None

    monkeypatch.setattr(cover_test.CoverTestSessionService, "persist_session", persist)
    app = FastAPI()
    app.include_router(cover_test.router)
    app.dependency_overrides[cover_test.get_db_session] = database
    files = {"cycle_1_raw": ("raw.json", json.dumps(cycles[1]), "application/json"),
             **{field: (field + ".jpg", content, "image/jpeg") for field, content in images.items()}}
    with TestClient(app) as client:
        response = client.post("/sessions", data={
            "session_metadata": json.dumps({"sessionId": SESSION}),
            "uncover_manifest": json.dumps(manifest)}, files=files)
    assert response.status_code == 201, response.text
    assert received["uncover_images"] == images
