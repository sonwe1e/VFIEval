from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from vfieval.metrics.base import MetricResult, MetricUnavailable


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".y4m"}


class VmafMetric:
    name = "vmaf"

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        if reference.suffix.lower() not in VIDEO_SUFFIXES or distorted.suffix.lower() not in VIDEO_SUFFIXES:
            raise MetricUnavailable("VMAF requires reference and distorted video files, not per-frame images.")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise MetricUnavailable("VMAF requires ffmpeg with libvmaf support on PATH.")

        work_dir.mkdir(parents=True, exist_ok=True)
        log_path = work_dir / "vmaf.json"
        command = [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(distorted),
            "-i",
            str(reference),
            "-lavfi",
            f"libvmaf=log_fmt=json:log_path={log_path}",
            "-f",
            "null",
            "-",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
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
        return MetricResult(status="completed", value=float(score), details={"log_path": str(log_path)})
