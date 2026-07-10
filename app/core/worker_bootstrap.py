from __future__ import annotations

import os
import sys
from typing import Literal


def _extract_queues(argv: list[str]) -> list[str]:
    queues: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        value: str | None = None

        if token in ("-Q", "--queues"):
            if i + 1 < len(argv):
                value = argv[i + 1]
                i += 2
            else:
                i += 1
        elif token.startswith("-Q="):
            value = token[3:]
            i += 1
        elif token.startswith("--queues="):
            value = token[len("--queues=") :]
            i += 1
        else:
            i += 1

        if value:
            queues.extend(q.strip() for q in value.split(",") if q.strip())

    return queues


def _detect_role() -> Literal["gpu", "cpu", "api"]:
    argv = sys.argv
    if "worker" not in argv:
        return "api"

    queues = _extract_queues(argv)
    if not queues:
        return "cpu"

    gpu_queue = os.environ.get("COMPARE_QUEUE_GPU", "compare-gpu").strip() or "compare-gpu"
    return "gpu" if gpu_queue in queues else "cpu"


def _apply(role: Literal["gpu", "cpu", "api"]) -> None:
    if role == "gpu":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("LPIPS_HEATMAP_DEVICE", "cpu")
        os.environ.setdefault("ENABLE_GPU_QUEUE", "true")
        os.environ.setdefault("COMPARE_EXECUTION_DEVICE", "gpu")
    elif role == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("COMPARE_EXECUTION_DEVICE", "cpu")
    elif role == "api":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


_apply(_detect_role())
