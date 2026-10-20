import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_container_refuses_missing_credentials(monkeypatch):
    for key in ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
                "REMICARE_FRONTEND_ORIGINS"):
        monkeypatch.delenv(key, raising=False)
    entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docker/start.py"))
    with pytest.raises(RuntimeError, match="required environment"):
        entry["validate_environment"]()


def test_default_startup_does_not_delete_legacy_tables(monkeypatch):
    from app.db.database import auto_migrate_schema
    monkeypatch.delenv("ENABLE_LEGACY_SCHEMA_CLEANUP", raising=False)
    conn = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"), execute=AsyncMock())
    asyncio.run(auto_migrate_schema(conn))
    statements = [str(call.args[0]) for call in conn.execute.call_args_list]
    assert len(statements) == 2
    assert all("DROP" not in statement for statement in statements)
