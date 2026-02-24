from __future__ import annotations

from pathlib import Path

from app.core.config import load_settings


def test_load_settings_uses_toml_values(tmp_path: Path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        "[tool.ai_corner]\napi_debug = false\nimage_base_dir = 'images'\nlog_level = 'warning'\n",
        encoding="utf-8",
    )

    settings = load_settings(toml_file)

    assert settings.api_debug is False
    assert settings.image_base_dir == "images"
    assert settings.log_level == "WARNING"
    assert settings.sources["API_DEBUG"] == "toml"
    assert settings.sources["IMAGE_BASE_DIR"] == "toml"
    assert settings.sources["LOG_LEVEL"] == "toml"


def test_load_settings_env_overrides_toml(monkeypatch, tmp_path: Path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        "[tool.ai_corner]\napi_debug = false\nimage_base_dir = 'images'\nlog_level = 'warning'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("API_DEBUG", "1")
    monkeypatch.setenv("IMAGE_BASE_DIR", "env-images")
    monkeypatch.setenv("LOG_LEVEL", "error")

    settings = load_settings(toml_file)

    assert settings.api_debug is True
    assert settings.image_base_dir == "env-images"
    assert settings.log_level == "ERROR"
    assert settings.sources["API_DEBUG"] == "env"
    assert settings.sources["IMAGE_BASE_DIR"] == "env"
    assert settings.sources["LOG_LEVEL"] == "env"
