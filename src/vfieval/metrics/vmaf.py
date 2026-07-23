from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vfieval.config import WorkspaceConfig
from vfieval.metrics.base import MetricResult, MetricUnavailable
from vfieval.metrics.health import metric_health
from vfieval.process_control import CancelCheck, run_cancellable


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".y4m"}


class VmafMetric:
    name = "vmaf"

    def __init__(
        self,
        workspace: WorkspaceConfig | None = None,
        *,
        cancel_check: CancelCheck | None = None,
    ):
        self.workspace = workspace or WorkspaceConfig.from_root()
        self.cancel_check = cancel_check

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        if reference.suffix.lower() not in VIDEO_SUFFIXES or distorted.suffix.lower() not in VIDEO_SUFFIXES:
            raise MetricUnavailable("VMAF requires reference and distorted video files, not per-frame images.")

        health = metric_health(self.workspace, self.name)
        if not health.get("available"):
            raise MetricUnavailable(f"vmaf evaluator is {health['status']}: {health['reason']}")
        ffmpeg = health.get("resolved_executable")
        if not ffmpeg:
            raise MetricUnavailable("VMAF requires a resolved ffmpeg executable with libvmaf support.")

        work_dir.mkdir(parents=True, exist_ok=True)
        log_path = work_dir / "vmaf.json"
        filter_log_path = _ffmpeg_filter_path(log_path)
        command = [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(distorted),
            "-i",
            str(reference),
            "-lavfi",
            f"libvmaf=log_fmt=json:log_path={filter_log_path}",
            "-f",
            "null",
            "-",
        ]
        try:
            if self.cancel_check is None:
                completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
            else:
                completed = run_cancellable(
                    command,
                    timeout=600,
                    cancel_check=self.cancel_check,
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg vmaf timed out after 600 seconds")
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            if "No such filter" in stderr or "libvmaf" in stderr:
                raise MetricUnavailable("ffmpeg is present but libvmaf is not available.")
            raise RuntimeError(f"ffmpeg vmaf failed: {stderr or completed.stdout.strip()}")

        data = json.loads(log_path.read_text(encoding="utf-8"))
        pooled = data.get("pooled_metrics", {}).get("vmaf", {})
        score = pooled.get("mean")
        if score is None:
            raise RuntimeError("VMAF output did not contain pooled_metrics.vmaf.mean")
        return MetricResult(
            status="completed",
            value=float(score),
            details={
                "log_path": str(log_path),
                "resolved_executable": ffmpeg,
                "manifest_path": health.get("manifest_path"),
                "implementation_mode": health.get("implementation_mode"),
            },
        )


def _ffmpeg_filter_path(path: Path) -> str:
    escaped = path.resolve().as_posix().replace(":", r"\:")
    return f"'{escaped}'"
