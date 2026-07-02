from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from vfieval.config import WorkspaceConfig
from vfieval.metrics.base import MetricResult, MetricUnavailable
from vfieval.metrics.health import metric_health
from vfieval.metrics.vmaf import VIDEO_SUFFIXES


class CgvqmMetric:
    name = "cgvqm"

    def __init__(self, workspace: WorkspaceConfig | None = None, device: str | None = None):
        self.workspace = workspace or WorkspaceConfig.from_root()
        self.device_name = device or "cpu"

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        if reference.suffix.lower() not in VIDEO_SUFFIXES or distorted.suffix.lower() not in VIDEO_SUFFIXES:
            raise MetricUnavailable("CGVQM requires reference and distorted video files.")
        health = metric_health(self.workspace, self.name)
        if not health.get("available"):
            raise MetricUnavailable(f"cgvqm evaluator is {health['status']}: {health['reason']}")

        command = list(health.get("driver_command") or [])
        if not command:
            raise MetricUnavailable("cgvqm evaluator is missing_evaluator: driver_command is empty")

        work_dir.mkdir(parents=True, exist_ok=True)
        eval_reference, eval_distorted, resize_details = _prepare_eval_videos(
            reference,
            distorted,
            work_dir,
            int(health.get("video_eval_long_edge") or 720),
        )
        payload = {
            "metric_name": self.name,
            "reference": str(eval_reference.resolve()),
            "distorted": str(eval_distorted.resolve()),
            "source_reference": str(reference.resolve()),
            "source_distorted": str(distorted.resolve()),
            "work_dir": str(work_dir.resolve()),
            "manifest_path": health.get("manifest_path"),
            "device": self.device_name,
            "repo_dir": health.get("repo_dir"),
            "weights_path": health.get("weights_path"),
            "video_eval_long_edge": int(health.get("video_eval_long_edge") or 720),
        }
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (health.get("env") or {}).items()})
        env["VFIEVAL_METRIC_DEVICE"] = self.device_name
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
            raise RuntimeError("cgvqm metric driver timed out after 600 seconds")

        data = _parse_driver_output(completed.stdout, completed.stderr)
        status = data["status"]
        details = data.get("details") if isinstance(data.get("details"), dict) else {}
        details = {
            **details,
            "metric_name": self.name,
            "device": self.device_name,
            "driver_command": command,
            "manifest_path": health.get("manifest_path"),
            "resolved_executable": health.get("resolved_executable"),
            "implementation_mode": health.get("implementation_mode"),
            "video_eval_long_edge": int(health.get("video_eval_long_edge") or 720),
            "eval_resolution": health.get("eval_resolution"),
            **resize_details,
        }
        if status == "completed":
            value = data.get("value")
            if value is None:
                raise RuntimeError("cgvqm metric driver returned completed without a numeric value")
            return MetricResult(status="completed", value=float(value), details=details)
        reason = details.get("reason") or data.get("reason") or completed.stderr.strip() or completed.stdout.strip()
        if status == "unavailable":
            raise MetricUnavailable(str(reason or "cgvqm metric driver reported unavailable"))
        raise RuntimeError(str(reason or "cgvqm metric driver reported failed"))


def _parse_driver_output(stdout: str, stderr: str) -> dict:
    raw = (stdout or "").strip()
    if not raw:
        raise RuntimeError(f"cgvqm metric driver did not write JSON to stdout: {stderr.strip() or 'no stderr'}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"cgvqm metric driver returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("cgvqm metric driver returned a non-object JSON payload")
    status = data.get("status")
    if status not in {"completed", "unavailable", "failed"}:
        raise RuntimeError(f"cgvqm metric driver returned invalid status {status!r}")
    return data


def _prepare_eval_videos(reference: Path, distorted: Path, work_dir: Path, long_edge: int) -> tuple[Path, Path, dict]:
    ref_info = _video_info(reference)
    dist_info = _video_info(distorted)
    if ref_info is None or dist_info is None:
        return reference, distorted, {"resize_status": "skipped_unreadable_video_metadata"}
    width, height, fps = ref_info
    target_width, target_height = _bounded_even_size(width, height, long_edge)
    if (target_width, target_height) == (width, height) and dist_info[:2] == (width, height):
        return reference, distorted, {
            "resize_status": "not_needed",
            "eval_width": width,
            "eval_height": height,
        }
    eval_reference = work_dir / "cgvqm_ref_eval.mp4"
    eval_distorted = work_dir / "cgvqm_dist_eval.mp4"
    _resize_video(reference, eval_reference, target_width, target_height, fps)
    _resize_video(distorted, eval_distorted, target_width, target_height, fps)
    return eval_reference, eval_distorted, {
        "resize_status": "resized",
        "eval_width": target_width,
        "eval_height": target_height,
        "source_width": width,
        "source_height": height,
    }


def _video_info(path: Path) -> tuple[int, int, float] | None:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        capture.release()
        if width <= 0 or height <= 0:
            return None
        return width, height, fps
    except Exception:
        return None


def _bounded_even_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    if max(width, height) <= long_edge:
        return _even(width), _even(height)
    scale = float(long_edge) / float(max(width, height))
    return _even(max(2, int(round(width * scale)))), _even(max(2, int(round(height * scale))))


def _even(value: int) -> int:
    return max(2, int(value) - (int(value) % 2))


def _resize_video(source: Path, target: Path, width: int, height: int, fps: float) -> None:
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise MetricUnavailable(f"CGVQM could not read video for resize: {source}")
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps or 24.0, (width, height))
    if not writer.isOpened():
        capture.release()
        raise MetricUnavailable(f"CGVQM could not create resized evaluation video: {target}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        capture.release()
        writer.release()
