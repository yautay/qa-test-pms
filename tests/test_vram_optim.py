from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import config as app_config
from app.core import job_store as job_store_module
from app.metrics.base import Metric, MetricResult


def _asset_bytes(filename: str) -> bytes:
    return (Path(__file__).parent / "assets" / filename).read_bytes()


@pytest.fixture(autouse=True)
def _base_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("JOB_STORE_BACKEND", "memory")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("CELERY_TASK_EAGER_PROPAGATES", "false")
    monkeypatch.setenv("HMAC_ENABLED", "false")
    app_config._clear_config_cache()
    job_store_module._clear_job_store_cache()
    yield
    app_config._clear_config_cache()
    job_store_module._clear_job_store_cache()


@pytest.fixture()
def client() -> Iterator[TestClient]:
    from app.core.celery_app import celery_app
    from app.main import create_app

    old_eager = celery_app.conf.task_always_eager
    old_propagate = celery_app.conf.task_eager_propagates
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        celery_app.conf.task_always_eager = old_eager
        celery_app.conf.task_eager_propagates = old_propagate


def _make_lpips_capturing(captured_heatmap_configs: list) -> type:
    class CapturingLpips(Metric):
        name = "lpips"

        def distance(self, ref_path: str, test_path: str, config) -> MetricResult:
            return MetricResult(value=0.1, meta={"metric": "lpips"})

        def heatmap_png(self, ref_path: str, test_path: str, config) -> bytes:
            captured_heatmap_configs.append(config)
            return b"\x89PNG\r\n\x1a\nFAKE"

    return CapturingLpips


def _post_lpips_job(client: TestClient, job_id: str) -> None:
    r = client.post(
        "/v1/compare/jobs",
        data={
            "job_id": job_id,
            "pair_id": "pair_vram",
            "metric": "lpips",
            "model": "alex",
            "normalize": "true",
        },
        files={
            "img_a": ("a.png", _asset_bytes("ref_1.png"), "image/png"),
            "img_b": ("b.png", _asset_bytes("test_1.png"), "image/png"),
        },
    )
    assert r.status_code == 202


def test_heatmap_force_device_is_cpu_by_default(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    import app.tasks.compare_tasks as tasks_module

    captured: list = []
    cls = _make_lpips_capturing(captured)
    monkeypatch.setattr(tasks_module.registry, "get", lambda _name: cls())

    _post_lpips_job(client, str(uuid.uuid4()))

    assert len(captured) == 1
    assert captured[0].force_device == "cpu"


def test_heatmap_force_device_is_none_when_auto(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    import app.tasks.compare_tasks as tasks_module

    monkeypatch.setenv("LPIPS_HEATMAP_DEVICE", "auto")
    app_config._clear_config_cache()

    captured: list = []
    cls = _make_lpips_capturing(captured)
    monkeypatch.setattr(tasks_module.registry, "get", lambda _name: cls())

    _post_lpips_job(client, str(uuid.uuid4()))

    assert len(captured) == 1
    assert captured[0].force_device is None


def test_heatmap_invalid_device_env_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    import app.tasks.compare_tasks as tasks_module

    monkeypatch.setenv("LPIPS_HEATMAP_DEVICE", "badvalue")
    app_config._clear_config_cache()

    captured: list = []
    cls = _make_lpips_capturing(captured)
    monkeypatch.setattr(tasks_module.registry, "get", lambda _name: cls())

    _post_lpips_job(client, str(uuid.uuid4()))

    assert len(captured) == 1
    assert captured[0].force_device == "cpu"


def test_empty_cache_called_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    import unittest.mock as mock

    import app.tasks.compare_tasks as tasks_module

    empty_cache_calls: list[int] = []

    mock_cuda = mock.MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.memory_allocated.return_value = 1024
    mock_cuda.memory_reserved.return_value = 2048
    mock_cuda.empty_cache.side_effect = lambda: empty_cache_calls.append(1)

    mock_torch = mock.MagicMock()
    mock_torch.cuda = mock_cuda

    monkeypatch.setattr(tasks_module, "torch", mock_torch)

    class DummyLpips(Metric):
        name = "lpips"

        def distance(self, ref_path: str, test_path: str, config) -> MetricResult:
            return MetricResult(value=0.1, meta={"metric": "lpips"})

        def heatmap_png(self, ref_path: str, test_path: str, config) -> bytes:
            return b"\x89PNG\r\n\x1a\nFAKE"

    monkeypatch.setattr(tasks_module.registry, "get", lambda _name: DummyLpips())

    _post_lpips_job(client, str(uuid.uuid4()))

    assert len(empty_cache_calls) >= 1
