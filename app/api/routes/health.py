from fastapi import APIRouter, Request

from app.core.device import resolve_device
from app.core.registry import registry

router = APIRouter()


@router.get("/health")
def health(request: Request):
    build = getattr(
        request.app.state,
        "build_info",
        {
            "version": "unknown",
            "tag": "unknown",
            "commit_hash": "unknown",
            "commit_date": "unknown",
            "branch": "unknown",
        },
    )
    return {
        "status": "ok",
        "message": "Service is healthy",
        "device": resolve_device(None),
        "metrics": registry.list(),
        "build": build,
    }
