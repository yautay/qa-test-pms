import base64

from fastapi import APIRouter, HTTPException

from app.core.image_io import cleanup_temp_paths, write_temp_image_from_base64
from app.core.registry import registry
from app.schemas.compare import (
    CompareAllRequest,
    CompareAllResponse,
    DistsCompareRequest,
    DistsCompareResponse,
    DistsDistanceConfig,
    LpipsCompareRequest,
    LpipsCompareResponse,
    LpipsDistanceConfig,
    LpipsHeatmapConfig,
)

router = APIRouter()


def _encode_png_base64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def _materialize_inputs(
    req: CompareAllRequest | LpipsCompareRequest | DistsCompareRequest,
) -> tuple[str, str, list[str]]:
    temp_paths: list[str] = []

    if req.ref_path is not None:
        ref_path = req.ref_path
    else:
        ref_path = write_temp_image_from_base64(req.ref_image_base64 or "", "ref")
        temp_paths.append(ref_path)

    if req.test_path is not None:
        test_path = req.test_path
    else:
        test_path = write_temp_image_from_base64(req.test_image_base64 or "", "test")
        temp_paths.append(test_path)

    return ref_path, test_path, temp_paths


@router.post("/compare", response_model=CompareAllResponse)
def compare(req: CompareAllRequest):
    temp_paths: list[str] = []
    try:
        ref_path, test_path, temp_paths = _materialize_inputs(req)
        lpips_metric = registry.get("lpips")
        dists_metric = registry.get("dists")

        lpips_dist_cfg = LpipsDistanceConfig(
            net=req.config.lpips_net, force_device=req.config.force_device
        )
        lpips_heat_cfg = LpipsHeatmapConfig(
            net=req.config.lpips_net,
            force_device=req.config.force_device,
            max_side=req.config.max_side,
            overlay_on=req.config.overlay_on,
            alpha=req.config.alpha,
        )
        dists_dist_cfg = DistsDistanceConfig(force_device=req.config.force_device)

        lpips_result = lpips_metric.distance(ref_path, test_path, lpips_dist_cfg)
        dists_result = dists_metric.distance(ref_path, test_path, dists_dist_cfg)
        lpips_heatmap = lpips_metric.heatmap_png(ref_path, test_path, lpips_heat_cfg)

        return {
            "lpips": {"value": lpips_result.value, "meta": lpips_result.meta},
            "dists": {"value": dists_result.value, "meta": dists_result.meta},
            "lpips_heatmap_png_base64": _encode_png_base64(lpips_heatmap),
            "integration_mode": req.integration_mode,
            "analyzed_models": [
                {"metric": "lpips", "model": req.config.lpips_net},
                {"metric": "dists", "model": "default"},
            ],
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}") from e
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Metric not registered: {e}") from e
    finally:
        cleanup_temp_paths(temp_paths)


@router.post("/compare/lpips", response_model=LpipsCompareResponse)
def compare_lpips(req: LpipsCompareRequest):
    temp_paths: list[str] = []
    try:
        ref_path, test_path, temp_paths = _materialize_inputs(req)
        lpips_metric = registry.get("lpips")

        lpips_dist_cfg = LpipsDistanceConfig(
            net=req.config.net, force_device=req.config.force_device
        )
        lpips_heat_cfg = LpipsHeatmapConfig(
            net=req.config.net,
            force_device=req.config.force_device,
            max_side=req.config.max_side,
            overlay_on=req.config.overlay_on,
            alpha=req.config.alpha,
        )

        lpips_result = lpips_metric.distance(ref_path, test_path, lpips_dist_cfg)
        lpips_heatmap = lpips_metric.heatmap_png(ref_path, test_path, lpips_heat_cfg)

        return {
            "lpips": {"value": lpips_result.value, "meta": lpips_result.meta},
            "lpips_heatmap_png_base64": _encode_png_base64(lpips_heatmap),
            "integration_mode": req.integration_mode,
            "analyzed_models": [{"metric": "lpips", "model": req.config.net}],
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}") from e
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Metric not registered: {e}") from e
    finally:
        cleanup_temp_paths(temp_paths)


@router.post("/compare/dists", response_model=DistsCompareResponse)
def compare_dists(req: DistsCompareRequest):
    temp_paths: list[str] = []
    try:
        ref_path, test_path, temp_paths = _materialize_inputs(req)
        dists_metric = registry.get("dists")
        dists_dist_cfg = DistsDistanceConfig(force_device=req.config.force_device)
        dists_result = dists_metric.distance(ref_path, test_path, dists_dist_cfg)

        return {
            "dists": {"value": dists_result.value, "meta": dists_result.meta},
            "integration_mode": req.integration_mode,
            "analyzed_models": [{"metric": "dists", "model": "default"}],
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}") from e
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Metric not registered: {e}") from e
    finally:
        cleanup_temp_paths(temp_paths)
