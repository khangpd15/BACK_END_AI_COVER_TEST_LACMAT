import pytest

from app.services.fps_model_service import (
    FeatureContractMismatchError,
    FpsModelService,
)


class DummyModel:
    n_features_in_ = 2

    def predict(self, x):
        return [0]


def test_predict_window_rejects_missing_required_feature():
    service = FpsModelService.__new__(FpsModelService)
    service.model = DummyModel()
    service.feature_names = ["leftValidRatio", "missingFeature"]

    samples = [
        {
            "t": 0,
            "leftX": 0.4,
            "leftY": 0.5,
            "leftValid": True,
            "rightX": 0.6,
            "rightY": 0.5,
            "rightValid": True,
        },
        {
            "t": 66,
            "leftX": 0.41,
            "leftY": 0.5,
            "leftValid": True,
            "rightX": 0.61,
            "rightY": 0.5,
            "rightValid": True,
        },
    ]

    with pytest.raises(FeatureContractMismatchError):
        service.predict_window(samples)


def test_predict_window_returns_inconclusive_when_model_not_loaded():
    service = FpsModelService.__new__(FpsModelService)
    service.model = None
    service.feature_names = ["leftValidRatio"]

    result = service.predict_window([])

    assert result["prediction"] == "INCONCLUSIVE"
    assert result["reason"] == "MODEL_NOT_LOADED"

