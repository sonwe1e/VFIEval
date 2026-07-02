from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from vfieval.config import WorkspaceConfig
from vfieval.metrics.base import MetricResult, MetricUnavailable
from vfieval.metrics.health import metric_health


class ManifestCommandMetric:
    def __init__(self, name: str, workspace: WorkspaceConfig | None = None):
        self.name = name
        self.workspace = workspace or WorkspaceConfig.from_root()

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        health = metric_health(self.workspace, self.name)
        if not health.get("available"):
            raise MetricUnavailable(f"{self.name} metric driver is {health['status']}: {health['reason']}")

        command = list(health.get("driver_command") or [])
        if not command:
            raise MetricUnavailable(f"{self.name} metric driver is missing_evaluator: driver_command is empty")

        work_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "metric_name": self.name,
            "reference": str(reference.resolve()),
            "distorted": str(distorted.resolve()),
            "work_dir": str(work_dir.resolve()),
            "manifest_path": health.get("manifest_path"),
        }
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (health.get("env") or {}).items()})
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
                env=env,
                cwd=str(work_dir),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{self.name} metric driver timed out after 600 seconds")

        data = _parse_driver_output(self.name, completed.stdout, completed.stderr)
        status = data["status"]
        details = data.get("details") if isinstance(data.get("details"), dict) else {}
        details = {
            **details,
            "driver_command": command,
            "manifest_path": health.get("manifest_path"),
            "resolved_executable": health.get("resolved_executable"),
        }
        if status == "completed":
            value = data.get("value")
            if value is None:
                raise RuntimeError(f"{self.name} metric driver returned completed without a numeric value")
            return MetricResult(status="completed", value=float(value), details=details)
        reason = details.get("reason") or data.get("reason") or completed.stderr.strip() or completed.stdout.strip()
        if status == "unavailable":
            raise MetricUnavailable(str(reason or f"{self.name} metric driver reported unavailable"))
        raise RuntimeError(str(reason or f"{self.name} metric driver reported failed"))


NativeMetric = ManifestCommandMetric


def _parse_driver_output(metric_name: str, stdout: str, stderr: str) -> dict:
    raw = (stdout or "").strip()
    if not raw:
        raise RuntimeError(f"{metric_name} metric driver did not write JSON to stdout: {stderr.strip() or 'no stderr'}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{metric_name} metric driver returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{metric_name} metric driver returned a non-object JSON payload")
    status = data.get("status")
    if status not in {"completed", "unavailable", "failed"}:
        raise RuntimeError(f"{metric_name} metric driver returned invalid status {status!r}")
    return data
