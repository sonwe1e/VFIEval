from __future__ import annotations

from pathlib import Path

from vfieval.config import WorkspaceConfig
from vfieval.metrics.base import MetricResult, MetricUnavailable
from vfieval.metrics.health import metric_health


class NativeMetric:
    def __init__(self, name: str, workspace: WorkspaceConfig | None = None):
        self.name = name
        self.workspace = workspace or WorkspaceConfig.from_root()

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        health = metric_health(self.workspace, self.name)
        if health["status"] != "ready":
            raise MetricUnavailable(f"{self.name} native adapter is {health['status']}: {health['reason']}")
        raise MetricUnavailable(
            f"{self.name} native evaluator assets are present, but the official evaluator binding "
            "has not been installed in this build."
        )
