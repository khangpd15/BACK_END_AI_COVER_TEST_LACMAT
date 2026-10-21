"""Production container entrypoint for RemiCare Cover Test."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_environment():
    origins = os.getenv("REMICARE_FRONTEND_ORIGINS", "").strip()
    if not origins or "CHANGE_ME" in origins:
        raise RuntimeError("Configure required environment variables: REMICARE_FRONTEND_ORIGINS")
    if "*" in origins:
        raise RuntimeError("Set explicit frontend origins.")

    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url and not db_url.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://", "sqlite://", "sqlite+aiosqlite://")):
        raise RuntimeError("DATABASE_URL must be a valid PostgreSQL or SQLite connection string.")

    from app.services.cover_test.storage_service import get_storage_service
    if not get_storage_service().is_live:
        raise RuntimeError("Storage service initialization failed.")


if __name__ == "__main__":
    validate_environment()
    port = int(os.getenv("PORT", "8000"))
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "app.main:app",
                             "--host", "0.0.0.0", "--port", str(port),
                             "--workers", "1", "--limit-concurrency", "16",
                             "--timeout-keep-alive", "5"])
