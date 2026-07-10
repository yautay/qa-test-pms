from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core import build_info
from app.core import config as app_config
from app.core import job_store as app_job_store
from app.core.celery_app import create_celery_app
from app.main import create_app


@pytest.fixture(autouse=True)
def _clear_cached_config() -> Iterator[None]:
    app_config._clear_config_cache()
    app_job_store._clear_job_store_cache()
    build_info._clear_git_metadata_cache()
    yield
    app_config._clear_config_cache()
    app_job_store._clear_job_store_cache()
    build_info._clear_git_metadata_cache()


def test_config_toml_values_are_used_when_env_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[env]
API_DEBUG = "0"
LOG_API_ENABLED = true
LOG_API_TIMEOUT_MS = 3500
JOB_STORE_BACKEND = "memory"
JOB_RETENTION_SEC = 120
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg))

    assert app_config.get_bool("API_DEBUG", default=True) is False
    assert app_config.get_bool("LOG_API_ENABLED", default=False) is True
    assert app_config.get_int("LOG_API_TIMEOUT_MS", 2000) == 3500
    assert app_config.get_str("JOB_STORE_BACKEND", "redis") == "memory"
    assert app_config.get_int("JOB_RETENTION_SEC", 86400) == 120


def test_create_job_store_uses_unified_job_retention_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JOB_STORE_BACKEND", "memory")
    monkeypatch.setenv("JOB_RETENTION_SEC", "7")
    monkeypatch.setenv("JOB_TTL_SEC", "1")
    monkeypatch.setenv("HEATMAP_TTL_SEC", "1")

    store = app_job_store.create_job_store()

    assert isinstance(store, app_job_store.MemoryJobStore)
    assert store._retention_sec == 7


def test_system_env_has_priority_over_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[env]
REDIS_PREFIX = "cfg-prefix"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("REDIS_PREFIX", "env-prefix")

    assert app_config.get_str("REDIS_PREFIX", "pms") == "env-prefix"


def test_configure_torch_model_cache_uses_default(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("APP_CONFIG_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.delenv("MODEL_CACHE_DIR", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)

    app_config.configure_torch_model_cache()

    assert os.environ["TORCH_HOME"] == "models"


def test_configure_torch_model_cache_uses_model_cache_dir_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_CACHE_DIR", "/opt/pms-models")
    monkeypatch.delenv("TORCH_HOME", raising=False)

    app_config.configure_torch_model_cache()

    assert os.environ["TORCH_HOME"] == "/opt/pms-models"


def test_configure_torch_model_cache_preserves_explicit_torch_home(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_CACHE_DIR", "/opt/pms-models")
    monkeypatch.setenv("TORCH_HOME", "/explicit/torch-home")

    app_config.configure_torch_model_cache()

    assert os.environ["TORCH_HOME"] == "/explicit/torch-home"


def test_redis_connection_settings_are_built_from_split_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_HOST", "split-host")
    monkeypatch.setenv("REDIS_PORT", "6390")
    monkeypatch.setenv("REDIS_DB", "6")
    monkeypatch.setenv("REDIS_USERNAME", "split-user")
    monkeypatch.setenv("REDIS_PASSWORD", "split-pass")
    monkeypatch.setenv("REDIS_TLS", "true")

    settings = app_config.get_redis_connection_settings()

    assert settings.url == "rediss://split-user:split-pass@split-host:6390/6"
    assert settings.source == "split_vars"
    assert settings.tls_enabled is True
    assert settings.username_configured is True
    assert settings.password_configured is True


def test_redis_connection_settings_fall_back_to_split_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "2")
    monkeypatch.setenv("REDIS_USERNAME", "svc-user")
    monkeypatch.setenv("REDIS_PASSWORD", "super-secret")
    monkeypatch.setenv("REDIS_TLS", "true")

    settings = app_config.get_redis_connection_settings()

    assert settings.url == "rediss://svc-user:super-secret@redis.internal:6380/2"
    assert settings.source == "split_vars"
    assert settings.tls_enabled is True
    assert settings.username_configured is True
    assert settings.password_configured is True


@pytest.mark.parametrize(
    ("env_name", "env_value", "error_message"),
    [
        ("REDIS_PORT", "abc", "REDIS_PORT must be an integer"),
        ("REDIS_PORT", "", "REDIS_PORT must not be empty"),
        ("REDIS_DB", "abc", "REDIS_DB must be an integer"),
        ("REDIS_DB", "", "REDIS_DB must not be empty"),
    ],
)
def test_redis_connection_settings_reject_invalid_split_ints(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    error_message: str,
):
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(RuntimeError, match=error_message):
        app_config.get_redis_connection_settings()


def test_redis_connection_settings_reject_username_without_password(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_USERNAME", "svc-user")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="REDIS_PASSWORD") as exc_info:
        app_config.get_redis_connection_settings()

    assert "svc-user" not in str(exc_info.value)


def test_build_redis_url_encodes_password_with_special_chars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_DB", "0")
    monkeypatch.setenv("REDIS_PASSWORD", "pa?ss&wo:rd`")

    redis_url = app_config.build_redis_url_from_split_vars()

    assert redis_url == "redis://:pa%3Fss%26wo%3Ard%60@redis.internal:6379/0"


@pytest.mark.parametrize(
    ("raw_value", "expected_scheme"),
    [("true", "rediss"), ("false", "redis")],
)
def test_build_redis_url_uses_expected_scheme_for_tls_flag(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected_scheme: str,
):
    monkeypatch.setenv("REDIS_TLS", raw_value)

    redis_url = app_config.build_redis_url_from_split_vars()

    assert redis_url.startswith(f"{expected_scheme}://")


def test_mask_redis_url_hides_password(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_HOST", "redis.example")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_DB", "0")
    monkeypatch.setenv("REDIS_USERNAME", "svc-user")
    monkeypatch.setenv("REDIS_PASSWORD", "super-secret")
    monkeypatch.setenv("REDIS_TLS", "true")

    settings = app_config.get_redis_connection_settings()
    masked = app_config.mask_redis_url(settings.url)

    assert masked == "rediss://svc-user:***@redis.example:6379/0"
    assert "super-secret" not in masked


def test_create_celery_app_uses_shared_redis_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "3")
    monkeypatch.setenv("REDIS_PASSWORD", "shared-secret")
    monkeypatch.setenv("REDIS_TLS", "true")
    monkeypatch.setenv("REDIS_PREFIX", "pms")

    celery_app = create_celery_app()

    assert celery_app.conf.broker_url == "rediss://:shared-secret@redis.internal:6380/3"
    assert celery_app.conf.result_backend == "rediss://:shared-secret@redis.internal:6380/3"
    assert celery_app.conf.broker_transport_options == {"global_keyprefix": "pms:"}
    assert celery_app.conf.result_backend_transport_options == {"global_keyprefix": "pms:"}


def test_create_celery_app_ignores_legacy_celery_broker_and_backend_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "5")
    monkeypatch.setenv("REDIS_PASSWORD", "from-shared-config")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://ignored-broker:6379/9")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://ignored-backend:6379/9")

    celery_app = create_celery_app()

    expected_url = "redis://:from-shared-config@redis.internal:6380/5"
    assert celery_app.conf.broker_url == expected_url
    assert celery_app.conf.result_backend == expected_url
    assert celery_app.conf.broker_url == celery_app.conf.result_backend


def test_create_app_reads_job_settings_from_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[env]
JOB_STORE_BACKEND = "memory"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg))

    app = create_app()
    assert app.state.job_store is None

    with TestClient(app):
        assert app.state.job_store is not None


def test_create_app_with_memory_job_store_backend(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[env]
JOB_STORE_BACKEND = "memory"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg))

    app = create_app()
    assert app.state.job_store is None

    with TestClient(app):
        assert app.state.job_store is not None


def test_create_app_with_redis_job_store_backend(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[env]
JOB_STORE_BACKEND = "redis"
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg))

    app = create_app()
    assert app.state.job_store is None


def test_git_metadata_uses_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_GIT_BRANCH", "release/1.2")
    monkeypatch.setenv("APP_GIT_TAG", "v1.2.3")
    monkeypatch.setenv("APP_GIT_LAST_COMMIT", "abc123")
    monkeypatch.setenv("APP_GIT_COMMITTER", "Jane Doe")
    monkeypatch.setenv("APP_GIT_COMMIT_DATE", "2026-02-24T10:00:00+00:00")

    metadata = build_info.get_git_metadata()

    assert metadata.branch == "release/1.2"
    assert metadata.tag == "v1.2.3"
    assert metadata.last_commit == "abc123"
    assert metadata.committer == "Jane Doe"
    assert metadata.date == "2026-02-24T10:00:00+00:00"


def test_create_app_fails_when_hmac_enabled_without_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HMAC_ENABLED", "true")
    monkeypatch.delenv("HMAC_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="HMAC_SECRET"):
        create_app()


def test_heatmap_device_override_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LPIPS_HEATMAP_DEVICE", raising=False)

    assert app_config.heatmap_device_override() == "cpu"


def test_heatmap_device_override_respects_auto(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LPIPS_HEATMAP_DEVICE", "auto")

    assert app_config.heatmap_device_override() == "auto"


def test_heatmap_device_override_falls_back_to_cpu_for_invalid_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LPIPS_HEATMAP_DEVICE", "cuda")

    assert app_config.heatmap_device_override() == "cpu"


def test_create_celery_app_uses_worker_max_tasks_per_child(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "50")

    celery = create_celery_app()

    assert celery.conf.worker_max_tasks_per_child == 50


def test_create_celery_app_disables_worker_max_tasks_per_child_when_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "0")

    celery = create_celery_app()

    assert celery.conf.worker_max_tasks_per_child is None


def test_create_celery_app_uses_worker_max_memory_per_child(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CELERY_WORKER_MAX_MEMORY_PER_CHILD_KB", "2000000")

    celery = create_celery_app()

    assert celery.conf.worker_max_memory_per_child == 2_000_000


def test_create_celery_app_disables_worker_max_memory_when_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CELERY_WORKER_MAX_MEMORY_PER_CHILD_KB", "0")

    celery = create_celery_app()

    assert celery.conf.worker_max_memory_per_child is None
