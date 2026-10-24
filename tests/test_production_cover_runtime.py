import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.cover_test import CoverTestSessionMetadataInput


def test_stateless_analysis_with_explicit_missing_model(monkeypatch):
    from app.services.cover_test import session_service as module
    from app.services import fps_model_service
    monkeypatch.setattr(module, "ENABLE_AI_INFERENCE_ON_SAVE", True)
    monkeypatch.setattr(fps_model_service, "get_fps_model_service", lambda: SimpleNamespace(model=None))
    service = module.CoverTestSessionService.__new__(module.CoverTestSessionService)
    service.db_session = AsyncMock()
    service.repo = AsyncMock()
    service.storage = AsyncMock()
    metadata = CoverTestSessionMetadataInput(sessionId="11111111-1111-4111-8111-111111111111")
    samples = [{"t": i * 16.7, "phase": "BASELINE", "leftX": .4, "leftY": .5,
                "rightX": .6, "rightY": .5, "leftValid": True, "rightValid": True}
               for i in range(3)]
    result = asyncio.run(service.persist_session(metadata, {1: {"samples": samples}}))
    assert result.saved is False
    assert result.storageMode == "STATELESS"
    assert result.success is True
    assert result.aiResult["reason"] == "MODEL_NOT_LOADED"
    assert result.aiResult["prediction"] == "INCONCLUSIVE"
    assert result.aiResult["confidence"] is None
    assert result.aiResult["classProbabilities"] is None
    service.repo.upsert_result.assert_not_called()
    service.storage.upload_raw_json.assert_not_called()
    service.repo.upsert_session.assert_not_called()
    service.db_session.commit.assert_not_called()


def test_real_service_without_repository(monkeypatch):
    from app.services.cover_test import session_service as module
    monkeypatch.setattr(module, "ENABLE_AI_INFERENCE_ON_SAVE", False)
    service = module.CoverTestSessionService(db_session=None)
    metadata = CoverTestSessionMetadataInput(sessionId="11111111-1111-4111-8111-111111111111")
    result = asyncio.run(service.persist_session(metadata, {1: {"samples": [{"t": 0}]}}))
    assert result.success and not result.saved
    assert service.repo is None and service.storage is None
    assert result.aiResult["reason"] == "INFERENCE_DISABLED"
