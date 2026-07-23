from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from vfieval.config import WorkspaceConfig
from vfieval.metrics.base import MetricResult, MetricUnavailable
from vfieval.metrics.health import metric_health
from vfieval.metrics.vmaf import VIDEO_SUFFIXES
from vfieval.process_control import CancelCheck, run_cancellable


CGVQM_PATCH_SCALE = 4
DRIVER_LOG_LIMIT = 16 * 1024


class CgvqmMetricUnavailable(MetricUnavailable):
    def __init__(self, message: str, details: dict[str, object]):
        super().__init__(message)
        self.details = details


class CgvqmMetricFailed(RuntimeError):
    def __init__(self, message: str, details: dict[str, object]):
        super().__init__(message)
        self.details = details


class CgvqmMetric:
    name = "cgvqm"

    def __init__(
        self,
        workspace: WorkspaceConfig | None = None,
        device: str | None = None,
        *,
        cancel_check: CancelCheck | None = None,
    ):
        self.workspace = workspace or WorkspaceConfig.from_root()
        self.device_name = device or "cpu"
        self.cancel_check = cancel_check

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
            alignment=CGVQM_PATCH_SCALE,
            cancel_check=self.cancel_check,
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
            "patch_scale": CGVQM_PATCH_SCALE,
        }
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (health.get("env") or {}).items()})
        env["VFIEVAL_METRIC_DEVICE"] = self.device_name
        base_details = {
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
        try:
            if self.cancel_check is None:
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
            else:
                completed = run_cancellable(
                    command,
                    input_text=json.dumps(payload),
                    timeout=600,
                    cancel_check=self.cancel_check,
                    env=env,
                    cwd=str(work_dir),
                )
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_process_output(exc.stdout)
            stderr = _coerce_process_output(exc.stderr)
            stdout_tail, stdout_truncated = _bounded_log(stdout)
            stderr_tail, stderr_truncated = _bounded_log(stderr)
            raise CgvqmMetricFailed(
                "cgvqm metric driver timed out after 600 seconds",
                {
                    **base_details,
                    "reason": "cgvqm metric driver timed out after 600 seconds",
                    "driver_returncode": None,
                    "driver_timed_out": True,
                    "driver_stdout": stdout_tail,
                    "driver_stdout_truncated": stdout_truncated,
                    "driver_stderr": stderr_tail,
                    "driver_stderr_truncated": stderr_truncated,
                },
            ) from exc

        try:
            data = _parse_driver_output(
                completed.stdout,
                completed.stderr,
                returncode=int(completed.returncode),
            )
        except RuntimeError as exc:
            stdout_tail, stdout_truncated = _bounded_log(completed.stdout)
            stderr_tail, stderr_truncated = _bounded_log(completed.stderr)
            raise CgvqmMetricFailed(
                str(exc),
                {
                    **base_details,
                    "reason": str(exc),
                    "driver_returncode": int(completed.returncode),
                    "driver_stdout": stdout_tail,
                    "driver_stdout_truncated": stdout_truncated,
                    "driver_stderr": stderr_tail,
                    "driver_stderr_truncated": stderr_truncated,
                },
            ) from exc
        status = data["status"]
        details = data.get("details") if isinstance(data.get("details"), dict) else {}
        details = {
            **details,
            **base_details,
        }
        if status == "completed":
            value = data.get("value")
            if value is None:
                raise CgvqmMetricFailed(
                    "cgvqm metric driver returned completed without a numeric value",
                    {
                        **details,
                        "reason": "cgvqm metric driver returned completed without a numeric value",
                    },
                )
            return MetricResult(status="completed", value=float(value), details=details)
        reason = details.get("reason") or data.get("reason") or completed.stderr.strip() or completed.stdout.strip()
        if status == "unavailable":
            message = str(reason or "cgvqm metric driver reported unavailable")
            raise CgvqmMetricUnavailable(
                message,
                {**details, "reason": message},
            )
        message = str(reason or "cgvqm metric driver reported failed")
        raise CgvqmMetricFailed(
            message,
            {**details, "reason": message},
        )


def _parse_driver_output(stdout: str, stderr: str, *, returncode: int = 0) -> dict:
    raw = str(stdout or "")
    protocol: list[tuple[int, dict]] = []
    nonprotocol: list[str] = []
    for index, line in enumerate(raw.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            nonprotocol.append(line)
            continue
        if not isinstance(candidate, dict) or candidate.get("status") not in {
            "completed",
            "unavailable",
            "failed",
        }:
            nonprotocol.append(line)
            continue
        protocol.append((index, candidate))

    if not protocol:
        if not raw.strip():
            message = "cgvqm metric driver produced no JSON protocol output"
        else:
            message = "cgvqm metric driver produced no valid JSON protocol object"
        raise RuntimeError(message + _driver_diagnostic(raw, stderr, returncode=returncode))

    _index, data = protocol[-1]
    details = data.get("details") if isinstance(data.get("details"), dict) else {}
    nonprotocol_text = "\n".join(nonprotocol)
    bounded_stdout, stdout_truncated = _bounded_log(nonprotocol_text)
    full_stdout, full_stdout_truncated = _bounded_log(raw)
    bounded_stderr, stderr_truncated = _bounded_log(stderr)
    data["details"] = {
        **details,
        "driver_returncode": int(returncode),
        "driver_stdout": full_stdout,
        "driver_stdout_truncated": full_stdout_truncated,
        "driver_nonprotocol_stdout": bounded_stdout,
        "driver_nonprotocol_stdout_truncated": stdout_truncated,
        "driver_stderr": bounded_stderr,
        "driver_stderr_truncated": stderr_truncated,
    }
    if int(returncode) != 0:
        reason = details.get("reason") or data.get("reason") or "driver process failed"
        raise RuntimeError(
            f"cgvqm metric driver exited with code {int(returncode)}: {reason}"
            + _driver_diagnostic(nonprotocol_text, stderr, returncode=returncode)
        )
    return data


def _bounded_log(value: object, limit: int = DRIVER_LOG_LIMIT) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= int(limit):
        return text, False
    return text[-int(limit):], True


def _driver_diagnostic(stdout: object, stderr: object, *, returncode: int | None) -> str:
    bounded_stdout, stdout_truncated = _bounded_log(stdout)
    bounded_stderr, stderr_truncated = _bounded_log(stderr)
    parts = [f" returncode={returncode if returncode is not None else 'timeout'}"]
    if bounded_stdout:
        parts.append(
            f" stdout_tail={bounded_stdout!r}{' (truncated)' if stdout_truncated else ''}"
        )
    if bounded_stderr:
        parts.append(
            f" stderr_tail={bounded_stderr!r}{' (truncated)' if stderr_truncated else ''}"
        )
    return ";".join(parts)


def _coerce_process_output(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _prepare_eval_videos(
    reference: Path,
    distorted: Path,
    work_dir: Path,
    long_edge: int,
    *,
    alignment: int = CGVQM_PATCH_SCALE,
    cancel_check: CancelCheck | None = None,
) -> tuple[Path, Path, dict]:
    if cancel_check is not None:
        cancel_check()
    ref_info = _video_info(reference)
    dist_info = _video_info(distorted)
    if ref_info is None or dist_info is None:
        reference_frames = _count_decodable_frames(reference, cancel_check=cancel_check)
        distorted_frames = _count_decodable_frames(distorted, cancel_check=cancel_check)
        _validate_eval_frame_counts(reference_frames, distorted_frames)
        return reference, distorted, {
            "resize_status": "skipped_unreadable_video_metadata",
            "eval_frame_count": reference_frames,
            "eval_reference_frame_count": reference_frames,
            "eval_distorted_frame_count": distorted_frames,
        }
    width, height, fps = ref_info
    target_width, target_height = _bounded_aligned_size(width, height, long_edge, alignment)
    if (target_width, target_height) == (width, height) and dist_info[:2] == (width, height):
        reference_frames = _count_decodable_frames(reference, cancel_check=cancel_check)
        distorted_frames = _count_decodable_frames(distorted, cancel_check=cancel_check)
        _validate_eval_frame_counts(reference_frames, distorted_frames)
        return reference, distorted, {
            "resize_status": "not_needed",
            "eval_width": width,
            "eval_height": height,
            "eval_alignment": alignment,
            "eval_frame_count": reference_frames,
            "eval_reference_frame_count": reference_frames,
            "eval_distorted_frame_count": distorted_frames,
        }
    eval_reference = work_dir / "cgvqm_ref_eval.mp4"
    eval_distorted = work_dir / "cgvqm_dist_eval.mp4"
    _resize_video(
        reference,
        eval_reference,
        target_width,
        target_height,
        fps,
        cancel_check=cancel_check,
    )
    _resize_video(
        distorted,
        eval_distorted,
        target_width,
        target_height,
        fps,
        cancel_check=cancel_check,
    )
    reference_frames = _count_decodable_frames(eval_reference, cancel_check=cancel_check)
    distorted_frames = _count_decodable_frames(eval_distorted, cancel_check=cancel_check)
    _validate_eval_frame_counts(reference_frames, distorted_frames)
    return eval_reference, eval_distorted, {
        "resize_status": "resized",
        "eval_width": target_width,
        "eval_height": target_height,
        "eval_alignment": alignment,
        "source_width": width,
        "source_height": height,
        "eval_frame_count": reference_frames,
        "eval_reference_frame_count": reference_frames,
        "eval_distorted_frame_count": distorted_frames,
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


def _bounded_aligned_size(width: int, height: int, long_edge: int, alignment: int) -> tuple[int, int]:
    alignment = max(1, int(alignment))
    if max(width, height) <= long_edge:
        return _aligned_floor(width, alignment), _aligned_floor(height, alignment)
    scale = float(long_edge) / float(max(width, height))
    return (
        _aligned_floor(max(alignment, int(round(width * scale))), alignment),
        _aligned_floor(max(alignment, int(round(height * scale))), alignment),
    )


def _aligned_floor(value: int, alignment: int) -> int:
    return max(alignment, int(value) - (int(value) % alignment))


def _resize_video(
    source: Path,
    target: Path,
    width: int,
    height: int,
    fps: float,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
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
            if cancel_check is not None:
                cancel_check()
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        capture.release()
        writer.release()


def _count_decodable_frames(
    path: Path,
    *,
    cancel_check: CancelCheck | None = None,
) -> int:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return 0
        count = 0
        try:
            while True:
                if cancel_check is not None:
                    cancel_check()
                ok, _frame = capture.read()
                if not ok:
                    break
                count += 1
        finally:
            capture.release()
        return count
    except Exception:
        return 0


def _validate_eval_frame_counts(reference_frames: int, distorted_frames: int) -> None:
    if int(reference_frames) <= 0 or int(distorted_frames) <= 0:
        raise CgvqmMetricUnavailable(
            "CGVQM evaluation videos contain no decodable frames "
            f"(reference={int(reference_frames)}, distorted={int(distorted_frames)})",
            {
                "eval_reference_frame_count": int(reference_frames),
                "eval_distorted_frame_count": int(distorted_frames),
            },
        )
    if int(reference_frames) != int(distorted_frames):
        raise CgvqmMetricUnavailable(
            "CGVQM evaluation video frame counts do not match "
            f"(reference={int(reference_frames)}, distorted={int(distorted_frames)})",
            {
                "eval_reference_frame_count": int(reference_frames),
                "eval_distorted_frame_count": int(distorted_frames),
            },
        )
