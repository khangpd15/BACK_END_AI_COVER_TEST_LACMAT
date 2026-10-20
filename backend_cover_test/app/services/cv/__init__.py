"""Computer Vision services for RemiCare Strabismus AI.

Includes:
- Quality Gate data structures
- Temporal Aggregation & Majority Voting (multi-frame consensus for stable inference)
"""

from app.services.cv.quality_gate import AdvancedQualityGate, QualityGateResult
from app.services.cv.temporal_aggregator import (
    FrameInferenceCandidate,
    TemporalAggregationResult,
    TemporalConsensusAggregator,
)

__all__ = [
    "AdvancedQualityGate",
    "QualityGateResult",
    "TemporalConsensusAggregator",
    "FrameInferenceCandidate",
    "TemporalAggregationResult",
]
