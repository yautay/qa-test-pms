from app.core.config import configure_torch_model_cache

# Configure Torch's model cache before importing metric modules. Their imports load
# torch/lpips/piq, which may initialize model download/cache paths during import.
# noqa: E402 on the imports below suppresses the "imports must be at top of file"
# lint rule because this ordering is intentional.
configure_torch_model_cache()

from app.metrics.dists_metric import DistsMetric  # noqa: E402
from app.metrics.lpips_metric import LpipsMetric  # noqa: E402


class MetricRegistry:
    def __init__(self):
        self._m = {}

    def register(self, metric):
        self._m[metric.name] = metric

    def get(self, name: str):
        return self._m[name]

    def list(self):
        return sorted(self._m.keys())


registry = MetricRegistry()
registry.register(LpipsMetric())
registry.register(DistsMetric())
