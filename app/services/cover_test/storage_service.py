"""In-memory ephemeral storage service for Cover Test (Pure Stateless Mode).

Zero disk persistence: data is held only in memory during processing and discarded.
"""

import json
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("remicare.storage")


class EphemeralStorageService:
    """Stateless ephemeral in-memory storage handler for Cover Test requests."""

    def __init__(self, **kwargs: Any):
        self._store: Dict[str, bytes] = {}

    @property
    def is_live(self) -> bool:
        return True

    async def upload_raw_json(self, storage_path: str, data: Union[Dict[str, Any], str, bytes]) -> str:
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        if isinstance(data, dict):
            payload_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, str):
            payload_bytes = data.encode("utf-8")
        else:
            payload_bytes = data

        self._store[clean_path] = payload_bytes
        return clean_path

    async def object_exists(self, storage_path: str) -> bool:
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        return clean_path in self._store

    async def upload_eye_image(self, storage_path: str, content: bytes) -> str:
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        self._store[clean_path] = content
        return clean_path

    async def delete_object(self, storage_path: str) -> bool:
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        if clean_path in self._store:
            del self._store[clean_path]
            return True
        return False

    async def download_object(self, storage_path: str) -> Optional[bytes]:
        clean_path = storage_path.lstrip("/").replace("\\", "/")
        return self._store.get(clean_path)


# Backward-compatible aliases
StorageService = EphemeralStorageService
SupabaseStorageService = EphemeralStorageService

_default_storage_service: Optional[EphemeralStorageService] = None


def get_storage_service() -> EphemeralStorageService:
    global _default_storage_service
    if _default_storage_service is None:
        _default_storage_service = EphemeralStorageService()
    return _default_storage_service
