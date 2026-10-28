"""Tests for UnifiedCoverInferenceService in Cover Test."""

import pytest
import numpy as np
from app.services.cover_test.unified_inference_service import (
    get_unified_inference_service,
    UnifiedCoverInferenceService,
)


def test_unified_service_health():
    svc = get_unified_inference_service()
    health = svc.health()
    assert health["loaded"] is True
    assert health["version"] == "unified-v2.1.0"
    assert health["multimodal"]["loaded"] is True
    assert health["hirschberg_branch"]["loaded"] is True
    assert health["multimodal"]["error"] is None
    assert health["hirschberg_branch"]["error"] is None


def test_unified_rejects_15fps_for_new_30fps_policy():
    """Nominal device capability cannot turn a 15fps processed stream into 30fps."""
    svc = get_unified_inference_service()

    # Generate synthetic 15 FPS trajectory with 50 frames
    samples = []
    for i in range(50):
        t = i * 0.0667
        samples.append({
            "t": t,
            "timestampMs": t * 1000.0,
            "leftX": 0.45 + 0.002 * np.sin(i * 0.2),
            "leftY": 0.50 + 0.001 * np.cos(i * 0.2),
            "rightX": 0.55 + 0.002 * np.sin(i * 0.2),
            "rightY": 0.50 + 0.001 * np.cos(i * 0.2),
            "leftValid": True,
            "rightValid": True,
            "trackingQuality": 0.95,
        })

    trajectories = {
        1: {"samples": samples},
        2: {"samples": samples},
        3: {"samples": samples},
    }

    # Empty decoded_events (no uncover images captured)
    decoded_events = {}

    result = svc.predict(trajectories, decoded_events)

    assert result["status"] == "INCONCLUSIVE"
    assert result["deviceSuitable30fps"] is False
    assert result["isReliable"] is False
    assert result["classProbabilities"] is None


def test_unified_inconclusive_when_empty_trajectories():
    """Verify clean inconclusive handling when no trajectory data exists."""
    svc = get_unified_inference_service()
    result = svc.predict({}, {})
    assert result["status"] == "INCONCLUSIVE"
    assert result["prediction"] == "INCONCLUSIVE"
    assert result["reason"] == "NO_TRAJECTORY_DATA"
