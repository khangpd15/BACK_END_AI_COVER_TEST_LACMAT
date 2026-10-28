from copy import deepcopy
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from app.services.cover_test.paired_crop_model import (
    assess_timing, timestamp_ms, trajectory_features, crop_features,
    paired_features, validate_source_event, PairedCropPredictor)
from app.services.cover_test.predictor import MultimodalCoverPredictor, _normalize_frame


def samples(fps=30):
    return [{"frameId":i,"timestampMs":10000+i*1000/fps,"t":i*1000/fps,
             "leftX":.45+i*.0001,"leftY":.5,"rightX":.55,"rightY":.5,
             "leftValid":True,"rightValid":True,"coordinateSpace":"CANTHAL_LOCAL_V1",
             "leftEyeLocal":{"x":.45+i*.0001,"y":.5,"valid":True},
             "rightEyeLocal":{"x":.55,"y":.5,"valid":True}} for i in range(90)]


def event(s):
    return {"cycle":1,"frameId":30,"timestampMs":s[30]["timestampMs"],
            "mirrored":False,"sourceWidth":200,"sourceHeight":100,
            "cropGeometryVersion":"source-eye-contour-v1",
            "confirmedAtMs":s[34]["timestampMs"],
            "onsetIntervalMs":[s[29]["timestampMs"],s[30]["timestampMs"]],
            "timingUncertaintyMs":s[30]["timestampMs"]-s[29]["timestampMs"],
            "crops":{side:{"width":64,"height":32,"cropRegion":{"x":x,"y":10,"width":64,"height":32}}
                     for side,x in (("left",100),("right",10))}}


@pytest.mark.parametrize("fps,eligible",[(15,False),(24,False),(29.9573,True),(30,True),(60,True)])
def test_observed_cadence_not_camera_setting(fps,eligible):
    s=samples(fps)
    for item in s: item["leftValid"]=False  # occlusion does not change capture rate
    assert assess_timing(s)["eligible30fps"] is eligible


@pytest.mark.parametrize("field",["timestampMs","frameId"])
def test_duplicate_source_is_rejected(field):
    s=samples();s[30][field]=s[29][field]
    with pytest.raises(ValueError): assess_timing(s)


def test_gaps_cannot_be_hidden_by_fast_median():
    s=samples()
    for item in s[40:]: item["timestampMs"]+=200
    assert assess_timing(s)["eligible30fps"] is False


def test_seconds_and_milliseconds_yield_same_motion():
    s=samples()
    legacy=[{k:v for k,v in row.items() if k!="timestampMs"} for row in s]
    for row in legacy: row["t"]/=1000
    np.testing.assert_allclose(trajectory_features(s),trajectory_features(legacy,unit="seconds"))
    assert timestamp_ms({"t":1},"seconds")==1000
    assert _normalize_frame({**s[30],"t":999999})["t"]==11


def test_hidden_iris_or_nonfinite_is_not_zero_filled():
    s=samples()
    for row in s: row["leftValid"]=False
    with pytest.raises(ValueError,match="INSUFFICIENT_VALID"): trajectory_features(s)
    for row in s: row["leftValid"]=True;row["leftEyeLocal"]["x"]=float("nan")
    with pytest.raises(ValueError,match="INSUFFICIENT_VALID"): trajectory_features(s)


@pytest.mark.parametrize("fault",["frame","rect","orientation","onset"])
def test_source_geometry_and_time_binding(fault):
    s=samples();e=event(s)
    assert validate_source_event(e,s)==11000
    if fault=="frame": e["timestampMs"]+=.001
    elif fault=="rect": e["crops"]["left"]["cropRegion"]["x"]=190
    elif fault=="orientation": e["mirrored"]=True
    else: e["onsetIntervalMs"][0]-=200
    with pytest.raises(ValueError): validate_source_event(e,s)


def test_real_pixel_branch_consumes_both_crops_without_claiming_accuracy():
    rng=np.random.default_rng(42)
    left=rng.integers(30,210,(32,64,3),dtype=np.uint8)
    right=rng.integers(30,210,(32,64,3),dtype=np.uint8)
    altered=np.roll(left,8,axis=1)
    s=samples();e=event(s)
    first=paired_features(left,right,s,11000)
    second=paired_features(altered,right,s,11000)
    assert not np.allclose(first,second)
    model=LogisticRegression(C=1).fit(np.asarray([first,second]),["NORMAL","STRABISMUS"])
    predictor=PairedCropPredictor(model,{"promoted":False})
    payload={"contract":"cover-uncover-multimodal-v1","cycles":{1:{"samples":s,"primaryOcclusionVerified":True,"occlusionVerified":True}},
             "uncoverEvents":{"c1_left_uncover":{"event":e,"images":{"left":left,"right":right}}}}
    a=predictor.predict_multimodal(payload)
    payload["uncoverEvents"]["c1_left_uncover"]["images"]["left"]=altered
    b=predictor.predict_multimodal(payload)
    assert a["classProbabilities"]!=b["classProbabilities"]
    assert a["analyzedEventIds"]==["c1_left_uncover"]
    assert a["prediction"]=="INCONCLUSIVE" and a["accuracy"] is None
    assert a["isReliable"] is False


def test_legacy_model_never_acknowledges_unconsumed_images():
    model=LogisticRegression().fit([[0]*8,[1]*8],[0,1])
    predictor=MultimodalCoverPredictor(model,["unused"])
    result=predictor.predict_multimodal({"contract":"cover-uncover-multimodal-v1",
        "cycles":{1:{"samples":samples()}},"uncoverEvents":{"received":{}}})
    assert result["inputModalities"]==["time_series"]
    assert result["analyzedEventIds"]==[] and result["isReliable"] is False


def test_no_prediction_from_images_using_zero_trajectory():
    predictor=MultimodalCoverPredictor(None,[])
    result=predictor.predict_multimodal({"contract":"cover-uncover-multimodal-v1",
        "cycles":{},"uncoverEvents":{"received":{}}})
    assert result["prediction"]=="INCONCLUSIVE"


def test_deployed_transfer_weights_consume_live_pair(monkeypatch):
    from app.services.cover_test.unified_inference_service import UnifiedCoverInferenceService
    svc=UnifiedCoverInferenceService()
    assert svc._image_transfer is not None
    monkeypatch.setattr(svc,"_predict_legacy",lambda *args:{"prediction":"NORMAL","isReliable":False})
    rng=np.random.default_rng(20)
    images={side:rng.integers(30,210,(32,64,3),dtype=np.uint8) for side in ("left","right")}
    s=samples()
    decoded={"c1_left_uncover":{"event":event(s),"images":images}}
    result=svc.predict({1:{"samples":s}},decoded)
    research=result["imageTransferResearch"]
    assert research["analyzedEventIds"]==["c1_left_uncover"]
    assert abs(sum(research["eventProbabilities"]["c1_left_uncover"].values())-1)<1e-9
    assert research["coverTestAccuracy"] is None and result["isReliable"] is False


def test_unverified_local_coordinates_are_not_mixed_with_image_coordinates():
    s=samples()
    for row in s: row.pop("coordinateSpace")
    with pytest.raises(ValueError,match="INSUFFICIENT_VALID"): trajectory_features(s)
