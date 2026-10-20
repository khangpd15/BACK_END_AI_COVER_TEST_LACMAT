"""Keep-Alive and Self-Ping background service for RemiCare backend on Render.

Prevents Render Free Tier services from spinning down after 15 minutes of inactivity,
and periodically exercises the Supabase database connection pool to avoid idle drops
and inactive project pausing.
"""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, Optional
import httpx

from app.config import (
    KEEP_ALIVE_ENABLED,
    KEEP_ALIVE_INTERVAL_SECONDS,
    KEEP_ALIVE_PING_DB,
    RENDER_EXTERNAL_URL,
    SELF_PING_URL,
)
from app.db.database import check_db_connection

logger = logging.getLogger("remicare.keep_alive")


class KeepAliveService:
    """Orchestrates periodic background self-pings and database connection warming."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        interval_seconds: Optional[int] = None,
        ping_db: Optional[bool] = None,
        custom_url: Optional[str] = None,
    ):
        self.enabled = KEEP_ALIVE_ENABLED if enabled is None else enabled
        self.interval_seconds = (
            KEEP_ALIVE_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
        )
        self.ping_db = KEEP_ALIVE_PING_DB if ping_db is None else ping_db
        self.custom_url = custom_url or SELF_PING_URL or RENDER_EXTERNAL_URL or ""

        self._task: Optional[asyncio.Task] = None
        self._is_running = False
        self._start_time: Optional[float] = None

        # Statistics
        self.total_http_pings = 0
        self.successful_http_pings = 0
        self.failed_http_pings = 0
        self.last_http_ping_at: Optional[str] = None
        self.last_http_status_code: Optional[int] = None
        self.last_http_latency_ms: Optional[float] = None
        self.last_http_error: Optional[str] = None

        self.total_db_pings = 0
        self.successful_db_pings = 0
        self.failed_db_pings = 0
        self.last_db_ping_at: Optional[str] = None
        self.last_db_ping_status: str = "none"
        self.last_db_ping_latency_ms: Optional[float] = None
        self.last_db_ping_error: Optional[str] = None

    def resolve_target_url(self) -> Optional[str]:
        """Resolves target base URL from custom URL, SELF_PING_URL, or RENDER_EXTERNAL_URL."""
        candidate = (self.custom_url or SELF_PING_URL or RENDER_EXTERNAL_URL or "").strip()
        if candidate:
            return candidate.rstrip("/")
        return None

    async def ping_http(self, base_url: str) -> Dict[str, Any]:
        """Performs a lightweight HTTP GET ping against /ping endpoint."""
        ping_endpoint = f"{base_url.rstrip('/')}/ping"
        t0 = time.perf_counter()
        self.total_http_pings += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        self.last_http_ping_at = now_iso

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(ping_endpoint)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.last_http_latency_ms = round(latency_ms, 2)
                self.last_http_status_code = resp.status_code

                if resp.status_code in (200, 204):
                    self.successful_http_pings += 1
                    self.last_http_error = None
                    logger.info(
                        "[KeepAlive] Self-ping SUCCESS: %s | status=%d | latency=%.1fms",
                        ping_endpoint,
                        resp.status_code,
                        latency_ms,
                    )
                    return {
                        "success": True,
                        "url": ping_endpoint,
                        "status_code": resp.status_code,
                        "latency_ms": self.last_http_latency_ms,
                    }
                else:
                    self.failed_http_pings += 1
                    self.last_http_error = f"Unexpected status {resp.status_code}"
                    logger.warning(
                        "[KeepAlive] Self-ping returned non-200: %s | status=%d",
                        ping_endpoint,
                        resp.status_code,
                    )
                    return {
                        "success": False,
                        "url": ping_endpoint,
                        "status_code": resp.status_code,
                        "error": self.last_http_error,
                    }

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.failed_http_pings += 1
            self.last_http_latency_ms = round(latency_ms, 2)
            self.last_http_error = str(e)
            logger.warning(
                "[KeepAlive] Self-ping exception for %s: %s (after %.1fms)",
                ping_endpoint,
                e,
                latency_ms,
            )
            return {
                "success": False,
                "url": ping_endpoint,
                "error": str(e),
                "latency_ms": self.last_http_latency_ms,
            }

    async def ping_database(self) -> Dict[str, Any]:
        """Performs a lightweight SELECT 1 DB ping to warm pool and prevent Supabase timeout."""
        self.total_db_pings += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        self.last_db_ping_at = now_iso

        res = await check_db_connection()
        self.last_db_ping_latency_ms = res.get("latency_ms")

        if res.get("connected"):
            self.successful_db_pings += 1
            self.last_db_ping_status = "healthy"
            self.last_db_ping_error = None
            logger.debug("[KeepAlive] Database ping SUCCESS in %.1fms", res.get("latency_ms", 0))
        else:
            self.failed_db_pings += 1
            self.last_db_ping_status = "unhealthy"
            self.last_db_ping_error = res.get("error")
            logger.warning("[KeepAlive] Database ping WARNING: %s", res.get("error"))

        return res

    async def ping_cycle(self) -> Dict[str, Any]:
        """Executes one complete cycle: HTTP self-ping and optional DB pool warm."""
        target_url = self.resolve_target_url()
        http_result = None
        if target_url:
            http_result = await self.ping_http(target_url)

        db_result = None
        if self.ping_db:
            db_result = await self.ping_database()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "http": http_result,
            "database": db_result,
        }

    async def _run_loop(self) -> None:
        """Internal background loop running every interval_seconds."""
        logger.info(
            "[KeepAlive] Worker started. Interval=%ds, target_url=%s, ping_db=%s",
            self.interval_seconds,
            self.resolve_target_url() or "(none)",
            self.ping_db,
        )
        # Initial boot delay (15 seconds) so application startup completes cleanly
        try:
            await asyncio.sleep(15)
        except asyncio.CancelledError:
            return

        while self._is_running:
            try:
                await self.ping_cycle()
            except Exception as e:
                logger.error("[KeepAlive] Unexpected error during keep-alive cycle: %s", e)

            try:
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break

        logger.info("[KeepAlive] Worker loop stopped cleanly.")

    def start(self) -> Optional[asyncio.Task]:
        """Starts the keep-alive background worker task."""
        if not self.enabled:
            logger.info("[KeepAlive] Disabled via KEEP_ALIVE_ENABLED=false. Skipping worker startup.")
            return None

        if self._is_running and self._task and not self._task.done():
            logger.debug("[KeepAlive] Worker already running.")
            return self._task

        self._is_running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._run_loop())
        return self._task

    async def stop(self) -> None:
        """Stops the keep-alive background worker task cleanly."""
        self._is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic metrics for keep-alive worker."""
        uptime_sec = round(time.time() - self._start_time, 1) if self._start_time else 0.0
        target = self.resolve_target_url()
        return {
            "enabled": self.enabled,
            "isRunning": self._is_running and self._task is not None and not self._task.done(),
            "intervalSeconds": self.interval_seconds,
            "targetUrl": target,
            "uptimeSeconds": uptime_sec,
            "httpPings": {
                "total": self.total_http_pings,
                "successful": self.successful_http_pings,
                "failed": self.failed_http_pings,
                "lastAt": self.last_http_ping_at,
                "lastStatusCode": self.last_http_status_code,
                "lastLatencyMs": self.last_http_latency_ms,
                "lastError": self.last_http_error,
            },
            "databasePings": {
                "total": self.total_db_pings,
                "successful": self.successful_db_pings,
                "failed": self.failed_db_pings,
                "lastAt": self.last_db_ping_at,
                "lastStatus": self.last_db_ping_status,
                "lastLatencyMs": self.last_db_ping_latency_ms,
                "lastError": self.last_db_ping_error,
            },
        }


# Global Singleton
_keep_alive_service: Optional[KeepAliveService] = None


def get_keep_alive_service() -> KeepAliveService:
    """Returns singleton KeepAliveService instance."""
    global _keep_alive_service
    if _keep_alive_service is None:
        _keep_alive_service = KeepAliveService()
    return _keep_alive_service


def start_keep_alive() -> Optional[asyncio.Task]:
    """Helper to start the singleton keep-alive worker."""
    svc = get_keep_alive_service()
    return svc.start()


async def stop_keep_alive() -> None:
    """Helper to stop the singleton keep-alive worker."""
    svc = get_keep_alive_service()
    await svc.stop()
