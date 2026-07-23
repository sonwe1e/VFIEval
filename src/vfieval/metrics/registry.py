from __future__ import annotations

from vfieval.config import WorkspaceConfig
from vfieval.metrics.base import MetricAdapter
from vfieval.metrics.cgvqm import CgvqmMetric
from vfieval.metrics.feature import ConvNextFeatureMetric, DinoPatchMetric
from vfieval.metrics.names import METRIC_NAMES
from vfieval.metrics.vmaf import VmafMetric
from vfieval.process_control import CancelCheck


def create_metric(
    name: str,
    workspace: WorkspaceConfig | None = None,
    device: str | None = None,
    cancel_check: CancelCheck | None = None,
) -> MetricAdapter:
    if name == "lpips_vit_patch":
        return DinoPatchMetric(workspace, device)
    if name == "lpips_convnext":
        return ConvNextFeatureMetric(workspace, device)
    if name == "vmaf":
        return VmafMetric(workspace, cancel_check=cancel_check)
    if name == "cgvqm":
        return CgvqmMetric(workspace, device, cancel_check=cancel_check)
    raise ValueError(f"unsupported metric '{name}'. Supported metrics: {', '.join(METRIC_NAMES)}")
