"""SQLAlchemy model for Cover Test Cycles."""

import uuid
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from app.db.models.cover_test_session import CoverTestSessionModel

from sqlalchemy import Float, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base
from app.db.models.cover_test_session import GUID

JSONType = JSON().with_variant(JSONB, "postgresql")


class CoverTestCycleModel(Base):
    """Database model for cover_test_cycles."""
    __tablename__ = "cover_test_cycles"

    cycle_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("cover_test_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    covered_eye: Mapped[str] = mapped_column(String(20), nullable=False)
    tracked_eye: Mapped[str] = mapped_column(String(20), nullable=False)
    raw_trajectory_url: Mapped[str] = mapped_column(String(512), nullable=False)
    sample_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    valid_sample_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    valid_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_tracking_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cycle_metadata: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)

    # Relationships
    session: Mapped["CoverTestSessionModel"] = relationship("CoverTestSessionModel", back_populates="cycles")
