import asyncio
"""Supabase Storage service for Cover Test raw JSON trajectories and eye crops."""

import json
import logging
from typing import Any, Dict, Optional, Union
from app.config import (
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_STORAGE_BUCKET_RAW,
    SUPABASE_URL,
)

logger = logging.getLogger("remicare.storage")


class SupabaseStorageService:
    """Encapsulates all Supabase Object Storage operations for cover-test-raw bucket.
    
    Supports offline/mock mode when cloud credentials are not configured.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        service_role_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        self.url = (supabase_url or SUPABASE_URL or "").strip()
        self.key = (service_role_key or SUPABASE_SERVICE_ROLE_KEY or "").strip()
        self.bucket = (bucket_name or SUPABASE_STORAGE_BUCKET_RAW or "cover-test-raw").strip()
        self._client = None
        self._in_memory_store: Dict[str, bytes] = {}

        if self.url and self.key:
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
                logger.info("Supabase storage client initialized for bucket '%s'.", self.bucket)
            except Exception as e:
                logger.warning(
                    "Failed to initialize live Supabase client (%s). Using offline memory store.", e
                )
                self._client = None
        else:
            logger.info("Supabase credentials not configured. Using offline in-memory storage.")

    @property
    def is_live(self) -> bool:
        return self._client is not None

    async def upload_raw_json(self, storage_path: str, data: Union[Dict[str, Any], str, bytes]) -> str:
        """Uploads JSON trajectory or manifest data to storage."""
        if isinstance(data, dict):
            payload_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, str):
            payload_bytes = data.encode("utf-8")
        else:
            payload_bytes = data

        clean_path = storage_path.lstrip("/")
        # Always retain a local cache / fallback copy
        self._in_memory_store[clean_path] = payload_bytes

        if self._client:
            try:
                await asyncio.to_thread(
                    self._client.storage.from_(self.bucket).upload,
                    path=clean_path,
                    file=payload_bytes,
                    file_options={"content-type": "application/json", "upsert": "true"},
                )
                logger.debug("Uploaded JSON to Supabase Storage: %s", clean_path)
            except Exception as e:
                logger.warning(
                    "Supabase Storage JSON upload failed for '%s' (%s). Retaining in fallback buffer.",
                    clean_path,
                    e,
                )
                raise

        return clean_path

    async def object_exists(self, storage_path: str) -> bool:
        """Checks if an object exists in storage."""
        clean_path = storage_path.lstrip("/")
        if self._client:
            try:
                folder = "/".join(clean_path.split("/")[:-1])
                filename = clean_path.split("/")[-1]
                items = self._client.storage.from_(self.bucket).list(folder)
                return any(item.get("name") == filename for item in items)
            except Exception as e:
                logger.warning("Error checking object existence for '%s': %s", clean_path, e)
                return False
        return clean_path in self._in_memory_store

    async def upload_eye_image(self, storage_path: str, content: bytes) -> str:
        """Store validated event-linked JPEGs in the existing private bucket."""
        clean_path = storage_path.lstrip("/")
        if self._client:
            # Propagate failures: never report an image saved when upload failed.
            await asyncio.to_thread(
                self._client.storage.from_(self.bucket).upload,
                path=clean_path, file=content,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )
        else:
            self._in_memory_store[clean_path] = content
        return clean_path

    async def delete_object(self, storage_path: str) -> bool:
        """Deletes an object from storage."""
        clean_path = storage_path.lstrip("/")
        if self._client:
            try:
                self._client.storage.from_(self.bucket).remove([clean_path])
                return True
            except Exception as e:
                logger.warning("Failed to delete object from Supabase Storage '%s': %s", clean_path, e)
                return False
        if clean_path in self._in_memory_store:
            del self._in_memory_store[clean_path]
            return True
        return False

    async def download_object(self, storage_path: str) -> Optional[bytes]:
        """Downloads object bytes from storage."""
        clean_path = storage_path.lstrip("/")
        if self._client:
            try:
                return self._client.storage.from_(self.bucket).download(clean_path)
            except Exception as e:
                logger.warning("Failed to download object '%s': %s", clean_path, e)
                return None
        return self._in_memory_store.get(clean_path)


_default_storage_service: Optional[SupabaseStorageService] = None


def get_storage_service() -> SupabaseStorageService:
    global _default_storage_service
    if _default_storage_service is None:
        _default_storage_service = SupabaseStorageService()
    return _default_storage_service
