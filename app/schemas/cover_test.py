"""Pydantic schemas for Cover Test cloud persistence and API contracts."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CoverTestCycleMetaInput(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    cycleNumber: int = Field(1, ge=1, description="Cycle index (1-based)")
    coveredEye: str = Field("LEFT", description="Eye covered: LEFT, RIGHT, ALTERNATING")
    trackedEye: str = Field("RIGHT", description="Eye tracked: RIGHT, LEFT, BOTH")
    sampleCount: Optional[int] = Field(None, ge=0)
    validSampleCount: Optional[int] = Field(None, ge=0)
    validRatio: Optional[float] = Field(None, ge=0.0, le=1.0)
    meanTrackingQuality: Optional[float] = Field(None, ge=0.0, le=1.0)
    durationMs: Optional[int] = Field(None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def resolve_cycle_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "cycleNumber" not in data or data["cycleNumber"] is None:
                for alt in ("cycle_number", "cycle", "cycleIndex", "number"):
                    if alt in data and data[alt] is not None:
                        data["cycleNumber"] = int(data[alt])
                        break
            if "coveredEye" not in data or not data["coveredEye"]:
                for alt in ("covered_eye", "covered", "cover"):
                    if alt in data and data[alt]:
                        data["coveredEye"] = str(data[alt])
                        break
            if "trackedEye" not in data or not data["trackedEye"]:
                for alt in ("tracked_eye", "tracked", "track"):
                    if alt in data and data[alt]:
                        data["trackedEye"] = str(data[alt])
                        break
        return data


class CoverTestSessionMetadataInput(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sessionId: str = Field(..., description="Canonical UUID session ID")
    cycleCount: int = Field(3, ge=1, description="Number of completed cycles")
    samplingRateHz: float = Field(15.0, gt=0, description="Acquisition sampling frequency")
    sourceDevice: str = Field("WEBCAM", description="Device type")
    tracker: str = Field("MEDIAPIPE_IRIS", description="Ocular landmark tracking engine")
    rawSchemaVersion: str = Field("1.0.0", description="Raw dataset schema version")
    clientMetadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    cycles: Optional[List[CoverTestCycleMetaInput]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def resolve_session_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "sessionId" not in data or not data["sessionId"]:
                for alt in ("session_id", "sampleId", "sample_id", "id", "sessionIdStr"):
                    if alt in data and data[alt]:
                        data["sessionId"] = str(data[alt])
                        break
            if "clientMetadata" not in data or data["clientMetadata"] is None:
                if "metadata" in data and isinstance(data["metadata"], dict):
                    data["clientMetadata"] = data["metadata"]
        return data


class CoverTestSessionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    sessionId: str
    saved: bool
    processingStatus: str
    storageRoot: str
    cyclesSaved: int
    imagesSaved: int
    aiResult: Optional[Dict[str, Any]] = None
    message: str = "Cover test session persisted successfully"
