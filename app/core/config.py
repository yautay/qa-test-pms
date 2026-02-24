from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


@dataclass(frozen=True)
class AppSettings:
    api_debug: bool
    image_base_dir: str
    log_level: str
    sources: dict[str, str]
    config_toml_path: str


def _parse_bool(raw: Any, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Invalid boolean value for {field_name}: {raw!r}")


def _resolve_toml_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    env_path = os.getenv("AI_CORNER_CONFIG_TOML")
    if env_path:
        return Path(env_path)

    return Path.cwd() / "pyproject.toml"


def load_settings(toml_path: Path | None = None) -> AppSettings:
    resolved_toml = _resolve_toml_path(toml_path)

    values: dict[str, Any] = {
        "API_DEBUG": True,
        "IMAGE_BASE_DIR": ".",
        "LOG_LEVEL": "INFO",
    }
    sources = {key: "default" for key in values}

    config_toml_source = "missing"
    if resolved_toml.exists():
        config_toml_source = "toml"
        with resolved_toml.open("rb") as f:
            data = tomllib.load(f)
        ai_corner_cfg = data.get("tool", {}).get("ai_corner", {})
        if isinstance(ai_corner_cfg, dict):
            toml_mapping = {
                "api_debug": "API_DEBUG",
                "image_base_dir": "IMAGE_BASE_DIR",
                "log_level": "LOG_LEVEL",
            }
            for toml_key, setting_key in toml_mapping.items():
                if toml_key in ai_corner_cfg:
                    values[setting_key] = ai_corner_cfg[toml_key]
                    sources[setting_key] = "toml"

    env_keys = ("API_DEBUG", "IMAGE_BASE_DIR", "LOG_LEVEL")
    for key in env_keys:
        env_value = os.getenv(key)
        if env_value is not None:
            values[key] = env_value
            sources[key] = "env"

    return AppSettings(
        api_debug=_parse_bool(values["API_DEBUG"], "API_DEBUG"),
        image_base_dir=str(values["IMAGE_BASE_DIR"]),
        log_level=str(values["LOG_LEVEL"]).upper(),
        sources={**sources, "CONFIG_TOML": config_toml_source},
        config_toml_path=str(resolved_toml),
    )
