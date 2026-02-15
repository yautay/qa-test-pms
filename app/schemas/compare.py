from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

MetricName = Literal["lpips", "dists"]
IntegrationMode = Literal["paths", "mixed", "api_upload"]


def _present(value: str | None) -> bool:
    return value is not None and value.strip() != ""


class CompareInputBase(BaseModel):
    ref_path: str | None = None
    test_path: str | None = None
    ref_image_base64: str | None = None
    test_image_base64: str | None = None

    @property
    def integration_mode(self) -> IntegrationMode:
        if _present(self.ref_path) and _present(self.test_path):
            return "paths"
        if _present(self.ref_path) and _present(self.test_image_base64):
            return "mixed"
        return "api_upload"

    @model_validator(mode="after")
    def _validate_image_sources(self) -> "CompareInputBase":
        ref_sources = int(_present(self.ref_path)) + int(_present(self.ref_image_base64))
        test_sources = int(_present(self.test_path)) + int(_present(self.test_image_base64))

        if ref_sources != 1:
            raise ValueError("Provide exactly one of ref_path or ref_image_base64")
        if test_sources != 1:
            raise ValueError("Provide exactly one of test_path or test_image_base64")

        if _present(self.ref_image_base64) and _present(self.test_path):
            raise ValueError("Unsupported combination: use paths, mixed, or api_upload mode")

        return self


class HeatmapBaseConfig(BaseModel):
    force_device: Literal["cpu", "cuda"] | None = None
    max_side: int = Field(default=1024, ge=128, le=4096)
    overlay_on: Literal["test", "ref"] = "test"
    alpha: float = Field(default=0.45, ge=0.0, le=1.0)


class DistanceBaseConfig(BaseModel):
    force_device: Literal["cpu", "cuda"] | None = None  # None => auto


class LpipsDistanceConfig(DistanceBaseConfig):
    metric: Literal["lpips"] = "lpips"
    net: Literal["vgg", "alex", "squeeze"] = "vgg"


class LpipsHeatmapConfig(HeatmapBaseConfig):
    metric: Literal["lpips"] = "lpips"
    net: Literal["vgg", "alex", "squeeze"] = "vgg"


class DistsDistanceConfig(DistanceBaseConfig):
    metric: Literal["dists"] = "dists"


class DistsHeatmapConfig(HeatmapBaseConfig):
    metric: Literal["dists"] = "dists"


class LpipsCompareConfig(HeatmapBaseConfig):
    net: Literal["vgg", "alex", "squeeze"] = "vgg"


class DistsCompareConfig(DistanceBaseConfig):
    pass


class CompareAllConfig(HeatmapBaseConfig):
    lpips_net: Literal["vgg", "alex", "squeeze"] = "vgg"


class CompareAllRequest(CompareInputBase):
    config: CompareAllConfig = Field(default_factory=CompareAllConfig)


class LpipsCompareRequest(CompareInputBase):
    config: LpipsCompareConfig = Field(default_factory=LpipsCompareConfig)


class DistsCompareRequest(CompareInputBase):
    config: DistsCompareConfig = Field(default_factory=DistsCompareConfig)


class MetricScoreResponse(BaseModel):
    value: float
    meta: dict[str, Any]


class AnalyzedModel(BaseModel):
    metric: MetricName
    model: str


class LpipsCompareResponse(BaseModel):
    lpips: MetricScoreResponse
    lpips_heatmap_png_base64: str
    integration_mode: IntegrationMode
    analyzed_models: list[AnalyzedModel]


class CompareAllResponse(BaseModel):
    lpips: MetricScoreResponse
    dists: MetricScoreResponse
    lpips_heatmap_png_base64: str
    integration_mode: IntegrationMode
    analyzed_models: list[AnalyzedModel]


class DistsCompareResponse(BaseModel):
    dists: MetricScoreResponse
    integration_mode: IntegrationMode
    analyzed_models: list[AnalyzedModel]
