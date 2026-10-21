"""RemiCare Cover Test AI Backend - Main Application Entrypoint.

Provides FastAPI application initialization, CORS configuration,
health check endpoints, and Cover Test session/persistence routers.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import math
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from app import __version__
from app.api.cover_test import router as cover_test_v1_router
from app.api.cover_test_session import router as cover_test_session_router
from app.config import get_allowed_origins
from app.db.database import check_db_connection, init_db
from app.services.fps_model_service import get_fps_model_service
from app.services.keep_alive import (
    get_keep_alive_service,
    start_keep_alive,
    stop_keep_alive,
)

# Configure structured audit logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("remicare.cover_test.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: Preload Cover Test FPS model, init DB, and start keep-alive worker."""
    logger.info("Initializing RemiCare Cover Test AI Backend...")
    try:
        await asyncio.wait_for(init_db(), timeout=10)
    except Exception as e:
        logger.warning("Database init check note: %s", e)

    try:
        fps_svc = get_fps_model_service()
        logger.info(
            "Cover Test FPS model state: model=%s, version=%s, loaded=%s, features=%d",
            fps_svc.model_name,
            fps_svc.model_version,
            fps_svc.model is not None,
            len(fps_svc.feature_names),
        )
    except Exception as e:
        logger.error("Failed to initialize Cover Test FPS service at startup: %s", e, exc_info=True)

    # Start automated keep-alive self-ping and DB pool warming worker
    start_keep_alive()

    try:
        yield
    finally:
        logger.info("Shutting down RemiCare Cover Test AI Backend...")
        await stop_keep_alive()


# Initialize FastAPI application
app = FastAPI(
    title="RemiCare Cover Test AI Screening Backend",
    description=(
        "Specialized standalone AI backend for webcam-based Cover Test screening. "
        "Provides time-series validation, multi-cycle trajectory storage, "
        "and numeric trajectory consensus inference."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration
allowed_origins = get_allowed_origins()
logger.info("Configured CORS Allowed Origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Enable HTTP response compression for payloads >= 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def performance_measurement_middleware(request: Request, call_next):
    """Performance measurement middleware."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    response.headers["Server-Timing"] = f"total;dur={duration_ms:.1f}"

    content_length = response.headers.get("content-length")
    size_str = f"{int(content_length)}B" if content_length else "chunked"

    if not request.url.path.startswith("/docs") and not request.url.path.startswith("/redoc"):
        logger.info(
            "[CoverTest API Perf] %s %s | status=%d | time=%.1fms | size=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            size_str,
        )
    return response


def _sanitize_validation_errors(obj: Any) -> Any:
    """Recursively replace non-finite float numbers with strings for JSON response safety."""
    if isinstance(obj, list):
        return [_sanitize_validation_errors(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_validation_errors(v) for k, v in obj.items()}
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors safely ensuring non-finite values don't break JSON encoding."""
    sanitized = _sanitize_validation_errors(exc.errors())
    return JSONResponse(status_code=422, content={"detail": sanitized})


# Register API routes
# Legacy endpoint maintained for compatibility
app.include_router(cover_test_session_router, prefix="/api/cover-test")
# Persistent cloud storage & inference router
app.include_router(cover_test_v1_router, prefix="/api/v1/cover-test")


@app.get(
    "/ping",
    tags=["System"],
    summary="Fast keep-alive ping endpoint",
    description="Lightweight endpoint for keep-alive services, Render self-ping, and uptime monitoring.",
)
async def ping() -> Dict[str, Any]:
    """Fast keep-alive ping responding with alive status and timestamp."""
    return {
        "status": "alive",
        "service": "remicare-cover-test-ai",
        "timestamp": time.time(),
    }


@app.get(
    "/health/db",
    tags=["System"],
    summary="Database connectivity diagnostics",
    description="Tests live connection to PostgreSQL / Supabase with SELECT 1 and returns connection diagnostics.",
)
async def db_health() -> Dict[str, Any]:
    """Diagnostics endpoint testing database connection."""
    return await check_db_connection()


@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
    description="Returns Cover Test backend health status, model info, and keep-alive metrics.",
)
async def health_check(check_db: bool = False) -> Dict[str, Any]:
    """Health check endpoint responding with operational status, version, and active Cover Test model info."""
    try:
        svc = get_fps_model_service()
        model_info = {
            "name": svc.model_name,
            "version": svc.model_version,
            "featureCount": len(svc.feature_names),
            "loaded": svc.model is not None,
            "source": "fps_10_15_model",
        }
    except Exception:
        model_info = {"status": "not_loaded"}

    keep_alive_info = get_keep_alive_service().get_status()

    result: Dict[str, Any] = {
        "status": "ok",
        "service": "remicare-cover-test-ai",
        "version": __version__,
        "model": model_info,
        "deploymentCommit": os.getenv("RENDER_GIT_COMMIT"),
        "keepAlive": keep_alive_info,
    }

    if check_db:
        db_res = await check_db_connection()
        result["database"] = db_res
        if not db_res.get("connected", False):
            result["status"] = "degraded"

    return result
