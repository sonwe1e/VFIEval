from __future__ import annotations

from vfieval.config import WorkspaceConfig
from vfieval.metrics.base import MetricAdapter
from vfieval.metrics.names import METRIC_NAMES
from vfieval.metrics.native import ManifestCommandMetric
from vfieval.metrics.vmaf import VmafMetric


def create_metric(name: str, workspace: WorkspaceConfig | None = None) -> MetricAdapter:
    if name == "lpips_vit_patch":
        return ManifestCommandMetric(name, workspace)
    if name == "lpips_convnext":
        return ManifestCommandMetric(name, workspace)
    if name == "vmaf":
        return VmafMetric(workspace)
    if name == "cgvqm":
        return ManifestCommandMetric(name, workspace)
    raise ValueError(f"unsupported metric '{name}'. Supported metrics: {', '.join(METRIC_NAMES)}")
