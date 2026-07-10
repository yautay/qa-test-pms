from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from app.core.worker_bootstrap import _apply, _detect_role, _extract_queues

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove all env vars that worker_bootstrap may read or write."""
    vars_to_clean = [
        "COMPARE_QUEUE_GPU",
        "COMPARE_QUEUE_CPU",
        "PYTORCH_CUDA_ALLOC_CONF",
        "LPIPS_HEATMAP_DEVICE",
        "ENABLE_GPU_QUEUE",
        "COMPARE_EXECUTION_DEVICE",
        "CUDA_VISIBLE_DEVICES",
    ]
    for var in vars_to_clean:
        monkeypatch.delenv(var, raising=False)
    yield


def _set_argv(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", argv)


# ---------------------------------------------------------------------------
# _extract_queues
# ---------------------------------------------------------------------------


def test_extract_queues_short_flag() -> None:
    assert _extract_queues(["-Q", "compare-gpu"]) == ["compare-gpu"]


def test_extract_queues_short_flag_csv() -> None:
    assert _extract_queues(["-Q", "compare-cpu,compare-gpu"]) == ["compare-cpu", "compare-gpu"]


def test_extract_queues_short_flag_equals() -> None:
    assert _extract_queues(["-Q=compare-gpu"]) == ["compare-gpu"]


def test_extract_queues_long_flag() -> None:
    assert _extract_queues(["--queues", "compare-gpu"]) == ["compare-gpu"]


def test_extract_queues_long_flag_equals() -> None:
    assert _extract_queues(["--queues=compare-gpu"]) == ["compare-gpu"]


def test_extract_queues_long_flag_csv() -> None:
    result = _extract_queues(["--queues=compare-cpu,compare-gpu"])
    assert result == ["compare-cpu", "compare-gpu"]


def test_extract_queues_empty_when_no_flag() -> None:
    assert _extract_queues(["celery", "worker", "--loglevel=info"]) == []


def test_extract_queues_short_flag_at_end_of_argv() -> None:
    # -Q with no following token — should not crash
    assert _extract_queues(["-Q"]) == []


# ---------------------------------------------------------------------------
# _detect_role — argv-based detection
# ---------------------------------------------------------------------------


def test_detect_role_gpu_short_q(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_argv(monkeypatch, ["celery", "-A", "app.core.celery_app:celery_app", "worker", "-Q", "compare-gpu"])
    assert _detect_role() == "gpu"


def test_detect_role_cpu_short_q(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_argv(monkeypatch, ["celery", "worker", "-Q", "compare-cpu"])
    assert _detect_role() == "cpu"


def test_detect_role_gpu_when_csv_contains_gpu_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_argv(monkeypatch, ["celery", "worker", "-Q", "compare-cpu,compare-gpu"])
    assert _detect_role() == "gpu"


def test_detect_role_gpu_long_flag_equals(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_argv(monkeypatch, ["celery", "worker", "--queues=compare-gpu"])
    assert _detect_role() == "gpu"


def test_detect_role_gpu_long_flag_space(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_argv(monkeypatch, ["celery", "worker", "--queues", "compare-gpu"])
    assert _detect_role() == "gpu"


def test_detect_role_cpu_no_queue_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_argv(monkeypatch, ["celery", "worker", "--loglevel=info"])
    assert _detect_role() == "cpu"


def test_detect_role_api_when_no_worker_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_argv(monkeypatch, ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])
    assert _detect_role() == "api"


def test_detect_role_api_for_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    # pytest argv contains no "worker"
    _set_argv(monkeypatch, ["/usr/local/bin/pytest", "-q"])
    assert _detect_role() == "api"


def test_detect_role_uses_custom_gpu_queue_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPARE_QUEUE_GPU", "my-custom-gpu")
    _set_argv(monkeypatch, ["celery", "worker", "-Q", "my-custom-gpu"])
    assert _detect_role() == "gpu"


def test_detect_role_cpu_when_custom_gpu_queue_not_in_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPARE_QUEUE_GPU", "my-custom-gpu")
    _set_argv(monkeypatch, ["celery", "worker", "-Q", "compare-gpu"])
    assert _detect_role() == "cpu"


# ---------------------------------------------------------------------------
# _apply — env configuration
# ---------------------------------------------------------------------------


def test_apply_gpu_sets_all_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply("gpu")
    assert "expandable_segments:True" in (__import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
    assert __import__("os").environ.get("LPIPS_HEATMAP_DEVICE") == "cpu"
    assert __import__("os").environ.get("ENABLE_GPU_QUEUE") == "true"
    assert __import__("os").environ.get("COMPARE_EXECUTION_DEVICE") == "gpu"


def test_apply_cpu_sets_cuda_invisible_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    _apply("cpu")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("COMPARE_EXECUTION_DEVICE") == "cpu"


def test_apply_api_sets_cuda_invisible(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    _apply("api")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_apply_gpu_does_not_set_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    _apply("gpu")
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_apply_does_not_overwrite_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    monkeypatch.setenv("ENABLE_GPU_QUEUE", "false")
    _apply("gpu")
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"
    assert os.environ["ENABLE_GPU_QUEUE"] == "false"


def test_apply_cpu_does_not_overwrite_existing_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _apply("cpu")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_apply_idempotent_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    _apply("gpu")
    first = dict(os.environ)
    _apply("gpu")
    for key in ["PYTORCH_CUDA_ALLOC_CONF", "LPIPS_HEATMAP_DEVICE", "ENABLE_GPU_QUEUE", "COMPARE_EXECUTION_DEVICE"]:
        assert os.environ.get(key) == first.get(key)


def test_apply_idempotent_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    _apply("cpu")
    first_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    first_device = os.environ.get("COMPARE_EXECUTION_DEVICE")
    _apply("cpu")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == first_cuda
    assert os.environ.get("COMPARE_EXECUTION_DEVICE") == first_device
