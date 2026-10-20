import pytest
from fastapi import HTTPException

pytest.importorskip("greenlet")

from app.services.cover_test.session_service import validate_cycle_samples


def test_duplicate_timestamps_are_rejected():
    samples = [
        {
            "t": 0,
            "phase": "BASELINE",
            "leftX": 0.4,
            "leftY": 0.5,
            "leftValid": True,
            "rightX": 0.6,
            "rightY": 0.5,
            "rightValid": True,
            "trackingQuality": 0.9,
        },
        {
            "t": 0,
            "phase": "BASELINE",
            "leftX": 0.41,
            "leftY": 0.5,
            "leftValid": True,
            "rightX": 0.61,
            "rightY": 0.5,
            "rightValid": True,
            "trackingQuality": 0.9,
        },
    ]

    with pytest.raises(HTTPException) as exc:
        validate_cycle_samples(samples, cycle_num=1)

    assert exc.value.status_code == 422
    assert "duplicate timestamp" in str(exc.value.detail)
