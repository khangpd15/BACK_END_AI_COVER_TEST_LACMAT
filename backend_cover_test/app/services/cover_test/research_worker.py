"""Isolated, opt-in eye-pair research scoring; never a clinical fusion model."""
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    import cv2
    import numpy as np
    import joblib
    from app.services.hirschberg_ai_service import _features_for_bundle
    from app.services.onnx_runtime import get_cpu_session

    bundle = joblib.load(root / "app/models/research/hirschberg_candidate_v0.5_safe.joblib")
    session = get_cpu_session(root / "app/models/best_model.onnx")
    request = json.load(sys.stdin)
    results = []
    for event in request["events"]:
        # Anatomical right is image-left in unmirrored camera pixels.
        crops = [cv2.resize(np.asarray(event[side], dtype=np.uint8)[:, :, ::-1], (224, 224))
                 for side in ("right", "left")]
        features = _features_for_bundle(crops, session, bundle)
        pipeline = bundle["pipeline"]
        scores = pipeline.predict_proba(features)[0]
        results.append({
            "eventId": event["eventId"],
            "uncalibratedScores": {str(bundle["classes"][int(k)]): float(v)
                                   for k, v in zip(pipeline.classes_, scores)},
        })
    print(json.dumps({"branch": "hirschberg-v0.5-eye-pair",
                     "status": "RESEARCH_ONLY", "clinicalDiagnosis": None,
                     "limitations": ["UNCOVER_CROP_DOMAIN_UNVALIDATED", "FIXATION_UNVERIFIED",
                                     "NOT_FULL_V0.6_ENSEMBLE", "NO_VALIDATED_TEMPORAL_FUSION"],
                     "events": results}))


if __name__ == "__main__":
    main()
