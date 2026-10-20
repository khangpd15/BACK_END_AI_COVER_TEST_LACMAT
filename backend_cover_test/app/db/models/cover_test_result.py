"""SQLAlchemy model for Cover Test Results."""

import uuid
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from app.db.models.cover_test_session import CoverTestSessionModel

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base
from app.db.models.cover_test_session import GUID

JSONType = JSON().with_variant(JSONB, "postgresql")


class CoverTestResultModel(Base):
    """Database model for cover_test_results.
    Customer Privacy: strictly stores NO image binary, NO image URL, NO base64.
    """
    __tablename__ = "cover_test_results"

    result_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("cover_test_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    evaluated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_source: Mapped[str] = mapped_column(String(100), nullable=False, default="fps_10_15_model")
    feature_schema_version: Mapped[str] = mapped_column(String(20), default="shared-v1.0.0", nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    input_compatible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    prediction: Mapped[str] = mapped_column(String(50), nullable=False)
    class_probabilities: Mapped[dict] = mapped_column(JSONType, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    domain_shift_warning: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    features_snapshot: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    comparison_models: Mapped[Optional[list]] = mapped_column(JSONType, nullable=True)
    notice: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    session: Mapped["CoverTestSessionModel"] = relationship("CoverTestSessionModel", back_populates="result")
