"""Production container entrypoint: refuse ephemeral storage and invalid setup."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_environment():
    required = ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
                "REMICARE_FRONTEND_ORIGINS")
    missing = [key for key in required if not os.getenv(key, "").strip()
               or "CHANGE_ME" in os.getenv(key, "")]
    if missing:
        raise RuntimeError("Configure required environment variables: " + ", ".join(missing))
    if not os.environ["DATABASE_URL"].startswith(("postgres://", "postgresql://", "postgresql+asyncpg://")):
        raise RuntimeError("Production Cover Test requires PostgreSQL.")
    if not os.environ["SUPABASE_URL"].startswith("https://"):
        raise RuntimeError("SUPABASE_URL must use HTTPS.")
    if "*" in os.environ["REMICARE_FRONTEND_ORIGINS"]:
        raise RuntimeError("Set explicit frontend origins.")
    from app.services.cover_test.storage_service import get_storage_service
    if not get_storage_service().is_live:
        raise RuntimeError("Supabase storage initialization failed; refusing memory-only mode.")


if __name__ == "__main__":
    validate_environment()
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "app.main:app",
                             "--host", "0.0.0.0", "--port", "8000",
                             "--workers", "1", "--limit-concurrency", "16",
                             "--timeout-keep-alive", "5"])
