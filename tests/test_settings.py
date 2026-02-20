from pathlib import Path

import pytest
from pydantic import ValidationError

from rustworkx_framework.config.settings import FrameworkSettings


def test_api_key_required(monkeypatch):
    monkeypatch.delenv("RWXF_API_KEY", raising=False)
    monkeypatch.delenv("RWXF_API_KEY_FILE", raising=False)

    with pytest.raises(ValidationError):
        FrameworkSettings()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("RWXF_API_KEY", "env-secret")
    settings = FrameworkSettings()

    assert settings.resolved_api_key == "env-secret"


def test_api_key_from_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RWXF_API_KEY", raising=False)
    key_path = tmp_path / "api.key"
    key_path.write_text("file-secret", encoding="utf-8")

    settings = FrameworkSettings(api_key_file=key_path)

    assert settings.resolved_api_key == "file-secret"
    assert settings.api_key is not None
    assert key_path.exists()


def test_empty_api_key_file(tmp_path: Path):
    key_path = tmp_path / "api.key"
    key_path.write_text("   ", encoding="utf-8")

    with pytest.raises(ValidationError):
        FrameworkSettings(api_key_file=key_path)
