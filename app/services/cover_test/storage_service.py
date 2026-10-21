"""Local and In-Memory Storage service for Cover Test raw JSON trajectories and eye crops."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.config import LOCAL_STORAGE_DIR

logger = logging.getLogger("remicare.storage")


class StorageService:
    """Encapsulates local file storage and in-memory caching for Cover Test data.
    
    Persists data to LOCAL_STORAGE_DIR without requiring external cloud storage (Supabase).
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        bucket_name: Optional[str] = None,
        **kwargs: Any,
    ):
        self.base_dir = Path(base_dir or LOCAL_STORAGE_DIR).resolve()
        self.bucket = (bucket_name or "cover-test-raw").strip()
        self._in_memory_store: Dict[str, bytes] = {}

        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Local storage service initialized at: %s", self.base_dir)
        except Exception as e:
            logger.warning("Could not create local storage directory '%s': %s", self.base_dir, e)

    @property
    def is_live(self) -> bool:
        """Local file storage is always operational."""
        return True

    def _resolve_path(self, clean_path: str) -> Path:
        return self.base_dir / clean_path

    async def upload_raw_json(self, storage_path: str, data: Union[Dict[str, Any], str, bytes]) -> str:
        """Persists JSON trajectory or manifest data to local filesystem and in-memory cache."""
        if isinstance(data, dict):
            payload_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, str):
            payload_bytes = data.encode("utf-8")
        else:
            payload_bytes = data

        clean_path = storage_path.lstrip("/").replace("\\", "/")
        self._in_memory_store[clean_path] = payload_bytes

        try:
            target_file = self._resolve_path(clean_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(target_file.write_bytes, payload_bytes)
            logger.debug("Persisted JSON to local storage: %s", target_file)
        except Exception as e:
            logger.warning("Failed to persist JSON to disk for '%s' (%s). Retaining in-memory copy.", clean_path, e)

        return clean_path

    async def object_exists(self, storage_path: str) -> bool:
        """Checks if an object exists on disk or in memory."""
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        if clean_path in self._in_memory_store:
            return True
        target_file = self._resolve_path(clean_path)
        return await asyncio.to_thread(target_file.exists)

    async def upload_eye_image(self, storage_path: str, content: bytes) -> str:
        """Store validated event-linked JPEGs in local storage and in-memory cache."""
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        self._in_memory_store[clean_path] = content

        try:
            target_file = self._resolve_path(clean_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(target_file.write_bytes, content)
            logger.debug("Persisted eye crop image to local storage: %s", target_file)
        except Exception as e:
            logger.warning("Failed to persist eye image to disk for '%s' (%s). Retaining in-memory copy.", clean_path, e)

        return clean_path

    async def delete_object(self, storage_path: str) -> bool:
        """Deletes an object from local storage and in-memory cache."""
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        deleted = False

        if clean_path in self._in_memory_store:
            del self._in_memory_store[clean_path]
            deleted = True

        target_file = self._resolve_path(clean_path)
        try:
            if target_file.exists():
                await asyncio.to_thread(target_file.unlink)
                deleted = True
        except Exception as e:
            logger.warning("Failed to delete local object '%s': %s", clean_path, e)

        return deleted

    async def download_object(self, storage_path: str) -> Optional[bytes]:
        """Downloads object bytes from memory or disk."""
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        if clean_path in self._in_memory_store:
            return self._in_memory_store[clean_path]

        target_file = self._resolve_path(clean_path)
        try:
            if await asyncio.to_thread(target_file.exists):
                content = await asyncio.to_thread(target_file.read_bytes)
                self._in_memory_store[clean_path] = content
                return content
        except Exception as e:
            logger.warning("Failed to read local object '%s': %s", clean_path, e)

        return None


# Backward-compatible alias
SupabaseStorageService = StorageService

_default_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    global _default_storage_service
    if _default_storage_service is None:
        _default_storage_service = StorageService()
    return _default_storage_service
