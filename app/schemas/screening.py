"""Pydantic schemas for RemiCare Strabismus AI Backend.

Defines input payload models and output screening response schemas
with strict compatibility for the React Frontend.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PhaseType(str, Enum):
    BASELINE = "BASELINE"
    COVER = "COVER"
    UNCOVER = "UNCOVER"
    TRACKING = "TRACKING"


class EyeSide(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class QualityStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


class ScreeningStatus(str, Enum):
    SCREENING_INCONCLUSIVE = "SCREENING_INCONCLUSIVE"
    SCREENING_NORMAL = "SCREENING_NORMAL"
    SCREENING_ATTENTION = "SCREENING_ATTENTION"


class EyeSample(BaseModel):
    """Raw sample record from webcam eye tracking."""
    model_config = ConfigDict(extra="allow")

    index: int = Field(..., description="Sample index within the cycle")
    t: float = Field(..., description="Timestamp in milliseconds")
    phase: str = Field(..., description="Phase: BASELINE, COVER, UNCOVER, TRACKING")

    leftX: Optional[float] = Field(None, description="Left iris normalized X coordinate [0.0, 1.0]")
    leftY: Optional[float] = Field(None, description="Left iris normalized Y coordinate [0.0, 1.0]")
    leftValid: bool = Field(..., description="Whether left eye detection is valid")

    rightX: Optional[float] = Field(None, description="Right iris normalized X coordinate [0.0, 1.0]")
    rightY: Optional[float] = Field(None, description="Right iris normalized Y coordinate [0.0, 1.0]")
    rightValid: bool = Field(..., description="Whether right eye detection is valid")

    trackingQuality: float = Field(..., description="Detection confidence/quality [0.0, 1.0]")


class CoverCycle(BaseModel):
    """A single cover-uncover cycle."""
    model_config = ConfigDict(extra="allow")

    cycle: int = Field(..., description="Cycle index (1-based)")
    coveredEye: str = Field(..., description="Eye that was covered: LEFT or RIGHT")
    trackedEye: str = Field(..., description="Eye that remained open and tracked: LEFT or RIGHT")
    samples: List[EyeSample] = Field(default_factory=list, description="List of time-series samples")


class ScreeningRequest(BaseModel):
    """Payload sent by React Frontend for screening analysis."""
    model_config = ConfigDict(extra="allow")

    sampleId: str = Field(..., description="Unique sample identifier (e.g. UUID)")
    test: str = Field("COVER_TEST", description="Test type identifier, must be COVER_TEST")
    cycles: List[CoverCycle] = Field(default_factory=list, description="Array of cover test cycles")


class QualitySummary(BaseModel):
    """Summary of data integrity gate checks."""
    model_config = ConfigDict(extra="allow")

    status: str = Field("PASS", description="Gate status: PASS, FAIL, or WARNING")
    reason: Optional[str] = Field(None, description="Reason code if gate failed")
    sampleCount: Optional[int] = Field(None, description="Total sample count")
    validSampleCount: Optional[int] = Field(None, description="Count of valid samples")
    validRatio: Optional[float] = Field(None, description="Ratio of valid samples")
    meanTrackingQuality: Optional[float] = Field(None, description="Average tracking quality score")
    issues: Optional[List[str]] = Field(default_factory=list, description="List of detected anomalies")


class AnalysisSummary(BaseModel):
    """Summary of technical feature extraction across cycles."""
    model_config = ConfigDict(extra="allow")

    cyclesAnalyzed: int = Field(..., description="Number of valid cycles analyzed")
    cycleFeatures: Optional[List[Dict[str, Any]]] = Field(None, description="Cycle-level technical features")
    aggregatedFeatures: Optional[Dict[str, Any]] = Field(None, description="Sample-level aggregated features")
    consistency: Optional[Dict[str, Any]] = Field(None, description="Inter-cycle consistency metrics")


class ScreeningResponse(BaseModel):
    """Screening outcome returned to React Frontend.
    
    CRITICAL: This is a screening outcome only, NOT a clinical diagnosis.
    No clinical angle (prism diopters) or diagnosis is provided.
    """
    model_config = ConfigDict(extra="allow")

    sampleId: str = Field(..., description="Sample identifier echoing the request")
    status: str = Field(
        default=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
        description="Screening outcome: SCREENING_INCONCLUSIVE, SCREENING_NORMAL, or SCREENING_ATTENTION"
    )
    modelVersion: Optional[str] = Field(None, description="Model identifier if inference ran, otherwise None")
    quality: QualitySummary = Field(..., description="Data integrity gate summary")
    analysis: AnalysisSummary = Field(..., description="Cycle and feature extraction summary")
    reason: Optional[str] = Field(None, description="Reason if status is INCONCLUSIVE")
    notice: str = Field(
        default="Screening result only — not a diagnosis.",
        description="Clinical disclaimer notice"
    )


# =============================================================================
# LEGACY TRANSFER EXPERIMENT SCHEMAS
# =============================================================================

class TransferModelMetadata(BaseModel):
    name: str = Field("Retired Transfer Research Model", description="Model identifier")
    version: str = Field("retired", description="Contract/model version")


class TransferDomainShiftInfo(BaseModel):
    source: str = Field("KOREAN_INFRARED_EYE_TRACKER", description="Source training domain")
    target: str = Field("REMICARE_WEBCAM_MEDIAPIPE", description="Target inference domain")
    warning: bool = Field(True, description="Domain shift flag")
    potentialShiftFeatures: Optional[List[str]] = Field(
        default_factory=lambda: ["meanLeftX", "meanLeftY", "meanRightX", "meanRightY"],
        description="Features susceptible to camera positioning and framing"
    )


class TransferComparisonModelResult(BaseModel):
    """One research-only model result in the side-by-side comparison."""
    key: str
    label: str
    prediction: str
    classProbability: Dict[str, float]
    model: TransferModelMetadata
    samplingProfile: str
    domainShiftWarning: bool = True
    clinicalMeaning: Optional[Any] = None
    notice: str = "Research-only model output — not a medical diagnosis."


class TransferExperimentResponse(BaseModel):
    """Legacy output for retired research transfer experiments."""
    model_config = ConfigDict(extra="allow")

    sampleId: str
    status: str = Field("TRANSFER_EXPERIMENT", description="Status code")
    inputCompatible: bool = Field(True, description="True if input met feature contract")
    prediction: str = Field(..., description="Model class output: NORMAL or STRABISMUS")
    classProbability: Dict[str, float] = Field(..., description="Random Forest class probabilities")
    domainShiftWarning: bool = Field(True, description="Explicit domain shift flag")
    clinicalMeaning: Optional[Any] = Field(None, description="Always None; not a medical diagnosis")
    model: TransferModelMetadata = Field(default_factory=TransferModelMetadata)
    domainShift: TransferDomainShiftInfo = Field(default_factory=TransferDomainShiftInfo)
    features: Optional[Dict[str, Optional[float]]] = Field(None, description="30 shared technical features for debug")
    comparisonModels: List[TransferComparisonModelResult] = Field(
        default_factory=list,
        description="Legacy side-by-side research outputs",
    )
    notice: str = Field(
        "Research transfer experiment only — not a diagnosis.",
        description="Mandatory scientific disclaimer"
    )


class TransferInconclusiveResponse(BaseModel):
    """Response returned when time-series validation fails."""
    model_config = ConfigDict(extra="allow")

    sampleId: Optional[str] = None
    status: str = Field("INCONCLUSIVE", description="Inconclusive status")
    inputCompatible: bool = Field(False, description="Incompatible flag")
    reason: str = Field("INVALID_TIME_SERIES", description="Failure reason")
    details: Optional[List[str]] = Field(default_factory=list, description="Detailed validation errors")
    notice: str = Field(
        "Research transfer experiment only — not a diagnosis.",
        description="Mandatory scientific disclaimer"
    )
