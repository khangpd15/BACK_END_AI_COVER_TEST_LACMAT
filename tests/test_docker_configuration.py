import runpy
from pathlib import Path
import pytest


def test_container_refuses_missing_origins(monkeypatch):
    monkeypatch.delenv("REMICARE_FRONTEND_ORIGINS", raising=False)
    entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docker/start.py"))
    with pytest.raises(RuntimeError, match="required environment"):
        entry["validate_environment"]()


def test_container_accepts_valid_stateless_env(monkeypatch):
    monkeypatch.setenv("REMICARE_FRONTEND_ORIGINS", "https://remicare.id.vn")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docker/start.py"))
    entry["validate_environment"]()
