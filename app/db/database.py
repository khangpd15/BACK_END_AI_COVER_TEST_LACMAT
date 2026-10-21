"""Database connection and session management for RemiCare Cover Test."""

from contextlib import asynccontextmanager
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, Optional
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.config import (
    DATABASE_URL,
    DB_MAX_OVERFLOW,
    DB_POOL_RECYCLE,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
)

logger = logging.getLogger("remicare.db")


class Base(DeclarativeBase):
    """Base declarative class for all database models."""
    pass


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def format_async_db_url(url: str) -> str:
    """Ensures DATABASE_URL uses the asyncpg driver and formats SSL parameters properly."""
    if not url:
        return "sqlite+aiosqlite:///:memory:"
    clean = url.strip()
    if clean.startswith("postgres://"):
        clean = clean.replace("postgres://", "postgresql+asyncpg://", 1)
    elif clean.startswith("postgresql://") and not clean.startswith("postgresql+asyncpg://"):
        clean = clean.replace("postgresql://", "postgresql+asyncpg://", 1)

    if "sslmode=require" in clean:
        clean = clean.replace("sslmode=require", "ssl=require")
    elif "sslmode=disable" in clean:
        clean = clean.replace("sslmode=disable", "ssl=disable")
    elif "sslmode=" in clean:
        clean = re.sub(r"sslmode=[^&]+", "ssl=require", clean)
    return clean


def get_db_connect_args(url: str) -> Dict[str, Any]:
    """Inspects connection URL and returns connect_args for asyncpg."""
    connect_args: Dict[str, Any] = {}
    if not url:
        return connect_args

    clean = url.strip().lower()
    
    if "pooler" in clean or ":6543" in clean:
        logger.info(
            "[ConnectionPooler] Transaction pooler detected. "
            "Disabling asyncpg prepared statement cache for transaction pooling."
        )
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_cache_size"] = 0

    return connect_args


def get_engine() -> AsyncEngine:
    """Returns singleton async engine with lazy initialization."""
    global _engine
    if _engine is None:
        db_url = format_async_db_url(DATABASE_URL)
        connect_args = get_db_connect_args(db_url)
        is_sqlite = db_url.startswith("sqlite")

        if is_sqlite:
            engine_kwargs: Dict[str, Any] = {"echo": False, "future": True}
            if ":memory:" in db_url:
                from sqlalchemy.pool import StaticPool
                engine_kwargs["poolclass"] = StaticPool
                engine_kwargs["connect_args"] = {"check_same_thread": False}
            _engine = create_async_engine(
                db_url,
                **engine_kwargs,
            )
        else:
            _engine = create_async_engine(
                db_url,
                echo=False,
                future=True,
                pool_size=DB_POOL_SIZE,
                max_overflow=DB_MAX_OVERFLOW,
                pool_recycle=DB_POOL_RECYCLE,
                pool_timeout=DB_POOL_TIMEOUT,
                connect_args=connect_args,
            )
        logger.info("Database engine initialized: driver=%s", _engine.name)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Returns singleton sessionmaker."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing an async database session per request."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_db_session_ctx() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for standalone async sessions outside request lifecycle."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def auto_migrate_schema(conn) -> None:
    """Non-destructive DDL migrations for Cover Test tables."""
    try:
        dialect_name = conn.dialect.name
        if dialect_name == "postgresql":
            ddl_statements = [
                "CREATE INDEX IF NOT EXISTS idx_cover_test_results_model_source ON cover_test_results (model_source);",
                "CREATE INDEX IF NOT EXISTS idx_cover_test_results_prediction ON cover_test_results (prediction);",
            ]
            # Historical cleanup is destructive; never run it merely on deployment.
            if os.getenv("ENABLE_LEGACY_SCHEMA_CLEANUP", "false").lower() == "true":
                ddl_statements += [
                    "ALTER TABLE cover_test_results DROP COLUMN IF EXISTS image_url;",
                    "ALTER TABLE cover_test_results DROP COLUMN IF EXISTS cloudinary_public_id;",
                    "DROP TABLE IF EXISTS cover_test_images CASCADE;",
                ]
            for stmt in ddl_statements:
                try:
                    await conn.execute(text(stmt))
                except Exception as stmt_err:
                    logger.warning("[AutoMigrateWarning] Non-blocking DDL notice: %s (%s)", stmt, stmt_err)
            logger.info("[AutoMigrate] PostgreSQL schema auto-migration completed successfully.")
    except Exception as e:
        logger.warning("[AutoMigrateError] Error executing auto-migration: %s", e)


async def init_db() -> None:
    """Initializes tables in database if they do not exist and applies non-destructive auto-migrations."""
    # Ensure models are imported so Base.metadata knows about them
    import app.db.models.cover_test_session
    import app.db.models.cover_test_cycle
    import app.db.models.cover_test_result

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await auto_migrate_schema(conn)
    logger.info("Database tables initialized successfully.")


async def check_db_connection() -> Dict[str, Any]:
    """Checks database connectivity with a lightweight SELECT 1 query."""
    engine = get_engine()
    start_time = time.perf_counter()
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            val = result.scalar()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "connected": True,
            "result": val,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        error_msg = str(e)
        logger.warning("[DBHealthCheckFailed] %s", error_msg)
        return {
            "status": "unhealthy",
            "latency_ms": round(latency_ms, 2),
            "connected": False,
            "error": error_msg,
        }
