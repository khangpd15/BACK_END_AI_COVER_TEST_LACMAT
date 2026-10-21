"""Application Configuration for RemiCare Cover Test AI Backend (Pure Stateless Mode)."""

import os
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment Variables
APP_ENV = os.getenv("APP_ENV", "development").lower()
APP_NAME = "remicare-cover-test-ai"
API_VERSION = "v1"

# Model Configuration
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "app" / "models" / "cover_test_remicare_candidate.joblib")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
FEATURE_CONTRACT_VERSION = "shared-v1.0.0"

# CORS Configuration
DEFAULT_DEV_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "https://remicare.id.vn,"
    "https://www.remicare.id.vn,"
    "https://cover-test.remicare.id.vn,"
    "https://hirschberg.remicare.id.vn"
)
FRONTEND_ORIGINS_RAW = os.getenv("REMICARE_FRONTEND_ORIGINS", DEFAULT_DEV_ORIGINS)

# AI Inference Settings (Stateless - In-Memory Only)
ENABLE_AI_INFERENCE_ON_SAVE = os.getenv("ENABLE_AI_INFERENCE_ON_SAVE", "true").lower() in ("true", "1", "yes")

# Keep-Alive & Self-Ping Configuration
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()
SELF_PING_URL = os.getenv("SELF_PING_URL", "").strip()
KEEP_ALIVE_ENABLED = os.getenv("KEEP_ALIVE_ENABLED", "false").lower() in ("true", "1", "yes")
KEEP_ALIVE_INTERVAL_SECONDS = int(os.getenv("KEEP_ALIVE_INTERVAL_SECONDS", "600"))


def get_allowed_origins() -> List[str]:
    """Parse comma-separated origin string into a clean list of allowed CORS origins."""
    origins = [o.strip() for o in FRONTEND_ORIGINS_RAW.split(",") if o.strip()]
    return origins

KEEP_ALIVE_PING_DB = False
