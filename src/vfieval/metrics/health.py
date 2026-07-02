from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import hashlib
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.metrics.names import METRIC_NAMES


METRIC_ASSET_VERSION = "v2"
METRIC_ADAPTER_VERSION = "portable-metrics-v2"
STATUS_AVAILABLE = "available"
STATUS_MISSING_WEIGHTS = "missing_weights"
STATUS_MISSING_EVALUATOR = "missing_evaluator"
STATUS_MISSING_DEPENDENCY = "missing_dependency"
PLACEHOLDER_STATUS = "placeholder"

METRIC_REQUIREMENTS = {
    "lpips_vit_patch": {
        "packages": (),
        "granularity": "sample",
        "supports_timeline": True,
        "evaluator": "manifest_command",
        "implementation_mode": "manifest_command",
        "setup_summary": (
            "Provide a project-local manifest, driver command, and metric assets under "
            "set/metrics/lpips_vit_patch/. VFIEval runs the declared driver without auto-downloading "
            "weights or bindings."
        ),
    },
    "lpips_convnext": {
        "packages": (),
        "granularity": "sample",
        "supports_timeline": True,
        "evaluator": "manifest_command",
        "implementation_mode": "manifest_command",
        "setup_summary": (
            "Provide a project-local manifest, driver command, and metric assets under "
            "set/metrics/lpips_convnext/. VFIEval runs the declared driver without auto-downloading "
            "weights or bindings."
        ),
    },
    "vmaf": {
        "packages": (),
        "granularity": "video",
        "supports_timeline": False,
        "requires_video_input": True,
        "evaluator": "ffmpeg_libvmaf",
        "implementation_mode": "ffmpeg_libvmaf",
        "setup_summary": (
            "Install ffmpeg with the libvmaf filter available. You may optionally set "
            "set/metrics/vmaf/manifest.json -> ffmpeg_path to point at a project-local portable "
            "ffmpeg binary; otherwise VFIEval falls back to PATH."
        ),
    },
    "cgvqm": {
        "packages": (),
        "granularity": "video",
        "supports_timeline": False,
        "requires_video_input": True,
        "evaluator": "manifest_command",
        "implementation_mode": "manifest_command",
        "setup_summary": (
            "Provide a project-local manifest, driver command, and metric assets under "
            "set/metrics/cgvqm/. VFIEval runs the declared driver against GT/Pred video artifacts "
            "without auto-downloading assets."
        ),
    },
}


def metric_assets_dir(workspace: WorkspaceConfig) -> Path:
    configured = os.getenv("VFIEVAL_METRIC_ASSETS_DIR")
    if configured:
        return Path(configured).resolve()
    project_root = Path(os.getenv("VFIEVAL_PROJECT_ROOT") or workspace.root.parent).resolve()
    return project_root / "set" / "metrics"


def metric_manifest_path(workspace: WorkspaceConfig, metric_name: str) -> Path:
    return metric_assets_dir(workspace) / metric_name / "manifest.json"


def metrics_health(workspace: WorkspaceConfig) -> dict[str, Any]:
    return {
        "asset_root": str(metric_assets_dir(workspace)),
        "metrics": {
            name: metric_health(workspace, name)
            for name in METRIC_NAMES
        },
    }


def metric_health(workspace: WorkspaceConfig, metric_name: str) -> dict[str, Any]:
    requirement = METRIC_REQUIREMENTS[metric_name]
    manifest_path = metric_manifest_path(workspace, metric_name)
    manifest, manifest_error = _load_manifest(manifest_path)
    if metric_name == "vmaf":
        return _vmaf_health(metric_name, requirement, manifest_path, manifest, manifest_error)
    return _manifest_command_health(metric_name, requirement, manifest_path, manifest, manifest_error)


def metric_requires_video_input(metric_name: str) -> bool:
    requirement = METRIC_REQUIREMENTS[metric_name]
    return bool(requirement.get("requires_video_input"))


def metric_cache_config(workspace: WorkspaceConfig, metric_name: str) -> dict[str, Any]:
    health = metric_health(workspace, metric_name)
    driver_command = list(health.get("driver_command") or [])
    return {
        "metric_name": metric_name,
        "adapter_version": health.get("adapter_version"),
        "asset_version": health.get("asset_version"),
        "status": health.get("status"),
        "reason": health.get("reason"),
        "evaluator": health.get("evaluator"),
        "implementation_mode": health.get("implementation_mode"),
        "granularity": health.get("granularity"),
        "input_mode": health.get("input_mode"),
        "requires_video_input": bool(health.get("requires_video_input")),
        "manifest_path": _path_fingerprint(health.get("manifest_path")) if health.get("manifest_path") else None,
        "expected_paths": _path_fingerprints(health.get("expected_paths") or []),
        "weights_path": _path_fingerprint(health.get("weights_path")) if health.get("weights_path") else None,
        "driver_command": driver_command,
        "driver_files": _path_fingerprints(_command_file_paths(driver_command)),
        "driver_env": dict(health.get("env") or {}),
        "resolved_executable": _path_fingerprint(health.get("resolved_executable")) if health.get("resolved_executable") else None,
        "executable": _executable_fingerprint(health.get("executable")),
        "executable_source": health.get("executable_source"),
    }


def prepare_metric_asset_manifest(workspace: WorkspaceConfig) -> dict[str, Any]:
    root = metric_assets_dir(workspace)
    root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for name in METRIC_NAMES:
        metric_dir = root / name
        metric_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = metric_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(_placeholder_manifest(name), indent=2),
                encoding="utf-8",
            )
        prepared.append(str(manifest_path))
    return {"asset_root": str(root), "prepared": prepared, "health": metrics_health(workspace)}


def _manifest_command_health(
    metric_name: str,
    requirement: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    manifest_error: str | None,
) -> dict[str, Any]:
    expected_paths = [str(manifest_path.resolve())]
    missing_packages = [
        package
        for package in requirement.get("packages", ())
        if importlib.util.find_spec(package) is None
    ]
    if missing_packages:
        return _status(
            metric_name,
            STATUS_MISSING_DEPENDENCY,
            f"missing Python package: {', '.join(missing_packages)}",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
        )
    if manifest is None:
        if manifest_error:
            return _status(
                metric_name,
                STATUS_MISSING_EVALUATOR,
                manifest_error,
                requirement,
                manifest_path=manifest_path,
                expected_paths=expected_paths,
            )
        rel_path = f"{metric_name}/manifest.json"
        return _status(
            metric_name,
            STATUS_MISSING_WEIGHTS,
            f"missing metric weights: {rel_path}",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
        )
    if _manifest_is_placeholder(manifest):
        return _status(
            metric_name,
            STATUS_MISSING_WEIGHTS,
            "metric manifest is still a placeholder",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
        )

    validation_error = _validate_metric_manifest(metric_name, manifest, requirement)
    if validation_error:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            validation_error,
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
        )

    required_files = [
        _resolve_manifest_file_path(manifest_path.parent, rel_path)
        for rel_path in manifest.get("required_files", [])
    ]
    expected_paths.extend(str(path.resolve()) for path in required_files)
    missing_required_files = [str(path.resolve()) for path in required_files if not path.exists()]
    if missing_required_files:
        return _status(
            metric_name,
            STATUS_MISSING_WEIGHTS,
            f"missing metric weights: {', '.join(missing_required_files)}",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            input_mode=str(manifest.get("input_mode") or _input_mode(requirement)),
        )

    driver_command = _resolve_driver_command(manifest_path.parent, manifest["driver"]["command"])
    resolved_executable, executable_source, executable_error = _resolve_command_executable(
        manifest_path.parent,
        driver_command[0],
    )
    if executable_error:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            executable_error,
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            input_mode=str(manifest.get("input_mode") or _input_mode(requirement)),
            driver_command=driver_command,
            env=dict(manifest.get("env") or {}),
        )
    driver_command[0] = resolved_executable
    return _status(
        metric_name,
        STATUS_AVAILABLE,
        None,
        requirement,
        manifest_path=manifest_path,
        expected_paths=expected_paths,
        asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
        input_mode=str(manifest.get("input_mode") or _input_mode(requirement)),
        driver_command=driver_command,
        env=dict(manifest.get("env") or {}),
        resolved_executable=resolved_executable,
        executable_source=executable_source,
    )


def _vmaf_health(
    metric_name: str,
    requirement: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    manifest_error: str | None,
) -> dict[str, Any]:
    expected_paths = [str(manifest_path.resolve())]
    ffmpeg_override: str | None = None
    asset_version = METRIC_ASSET_VERSION

    if manifest_error:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            manifest_error,
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
        )
    if manifest is not None:
        validation_error = _validate_vmaf_manifest(metric_name, manifest)
        if validation_error:
            return _status(
                metric_name,
                STATUS_MISSING_EVALUATOR,
                validation_error,
                requirement,
                manifest_path=manifest_path,
                expected_paths=expected_paths,
                asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            )
        asset_version = str(manifest.get("asset_version") or METRIC_ASSET_VERSION)
        value = str(manifest.get("ffmpeg_path") or "").strip()
        ffmpeg_override = value or None

    resolved_executable = None
    executable_source = "path"
    if ffmpeg_override:
        override_path = _resolve_manifest_file_path(manifest_path.parent, ffmpeg_override)
        if override_path.exists():
            resolved_executable = str(override_path.resolve())
            executable_source = "manifest"

    if resolved_executable is None:
        resolved_executable = shutil.which("ffmpeg")
        executable_source = "path"

    if not resolved_executable:
        reason = "ffmpeg is not on PATH"
        if ffmpeg_override:
            override_path = _resolve_manifest_file_path(manifest_path.parent, ffmpeg_override)
            reason = f"manifest ffmpeg_path does not exist and ffmpeg is not on PATH: {override_path.resolve()}"
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            reason,
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=asset_version,
            resolved_executable=str(_resolve_manifest_file_path(manifest_path.parent, ffmpeg_override).resolve())
            if ffmpeg_override
            else None,
            executable_source=executable_source,
            driver_command=[str(_resolve_manifest_file_path(manifest_path.parent, ffmpeg_override).resolve())]
            if ffmpeg_override
            else ["ffmpeg"],
        )

    filter_check = _inspect_ffmpeg_filters(resolved_executable)
    if not filter_check["available"]:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            str(filter_check["reason"]),
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=asset_version,
            resolved_executable=resolved_executable,
            executable_source=executable_source,
            driver_command=[resolved_executable],
            libvmaf_filter_available=False,
            libvmaf_filter_reason=str(filter_check["reason"]),
        )
    return _status(
        metric_name,
        STATUS_AVAILABLE,
        None,
        requirement,
        manifest_path=manifest_path,
        expected_paths=expected_paths,
        asset_version=asset_version,
        resolved_executable=resolved_executable,
        executable_source=executable_source,
        driver_command=[resolved_executable],
        libvmaf_filter_available=True,
        libvmaf_filter_reason=None,
    )


def _status(
    metric_name: str,
    status: str,
    reason: str | None,
    requirement: dict[str, Any],
    *,
    manifest_path: Path,
    expected_paths: list[str],
    asset_version: str | None = None,
    input_mode: str | None = None,
    driver_command: list[str] | None = None,
    env: dict[str, str] | None = None,
    resolved_executable: str | None = None,
    executable_source: str | None = None,
    libvmaf_filter_available: bool | None = None,
    libvmaf_filter_reason: str | None = None,
) -> dict[str, Any]:
    implementation_mode = str(requirement.get("implementation_mode"))
    effective_input_mode = input_mode or _input_mode(requirement)
    row = {
        "metric_name": metric_name,
        "status": status,
        "available": status == STATUS_AVAILABLE,
        "adapter_version": METRIC_ADAPTER_VERSION,
        "asset_version": asset_version or METRIC_ASSET_VERSION,
        "reason": reason,
        "weights_path": str(manifest_path.resolve()),
        "expected_paths": expected_paths,
        "packages": list(requirement.get("packages", ())),
        "executable": (driver_command or [None])[0],
        "resolved_executable": resolved_executable,
        "executable_source": executable_source,
        "granularity": requirement.get("granularity"),
        "supports_timeline": bool(requirement.get("supports_timeline")),
        "requires_video_input": bool(requirement.get("requires_video_input")),
        "input_mode": effective_input_mode,
        "evaluator": requirement.get("evaluator"),
        "implementation_mode": implementation_mode,
        "manifest_path": str(manifest_path.resolve()),
        "driver_command": driver_command,
        "env": dict(env or {}),
        "libvmaf_filter_available": libvmaf_filter_available,
        "libvmaf_filter_reason": libvmaf_filter_reason,
        "setup_summary": requirement.get("setup_summary"),
        "setup_requirements": _setup_requirements(
            metric_name,
            requirement,
            manifest_path,
            expected_paths,
            driver_command,
            resolved_executable,
        ),
        "auto_download": False,
    }
    return row


def _input_mode(requirement: dict[str, Any]) -> str:
    if bool(requirement.get("requires_video_input")):
        return "video_only"
    if requirement.get("granularity") == "sample":
        return "sample_pair"
    return "artifact_or_video"


def _setup_requirements(
    metric_name: str,
    requirement: dict[str, Any],
    manifest_path: Path,
    expected_paths: list[str],
    driver_command: list[str] | None,
    resolved_executable: str | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "kind": "manifest",
            "target": str(manifest_path.resolve()),
            "description": "Keep the metric manifest here so VFIEval can inspect or run the metric.",
        }
    ]
    if requirement.get("implementation_mode") == "manifest_command":
        for path in expected_paths[1:]:
            rows.append(
                {
                    "kind": "asset",
                    "target": path,
                    "description": "Provide the referenced weights or config file here.",
                }
            )
        rows.append(
            {
                "kind": "driver",
                "target": (driver_command or ["manifest.driver.command[0]"])[0],
                "description": "The manifest driver command must resolve to a runnable evaluator.",
            }
        )
    if metric_name == "vmaf":
        rows.append(
            {
                "kind": "ffmpeg_path",
                "target": str(manifest_path.resolve()),
                "description": "Optional ffmpeg_path override lives in this manifest. Leave it blank to use PATH.",
            }
        )
        rows.append(
            {
                "kind": "executable",
                "target": resolved_executable or "ffmpeg",
                "description": "This ffmpeg binary must be runnable for VMAF execution.",
            }
        )
        rows.append(
            {
                "kind": "ffmpeg_filter",
                "target": "libvmaf",
                "description": "The selected ffmpeg build must expose the libvmaf filter.",
            }
        )
    return rows


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid metric manifest at {path.resolve()}: {exc}"
    if not isinstance(data, dict):
        return None, f"invalid metric manifest at {path.resolve()}: root must be a JSON object"
    return data, None


def _manifest_is_placeholder(manifest: dict[str, Any]) -> bool:
    return str(manifest.get("status") or "").strip().lower() == PLACEHOLDER_STATUS


def _validate_metric_manifest(
    metric_name: str,
    manifest: dict[str, Any],
    requirement: dict[str, Any],
) -> str | None:
    manifest_metric_name = manifest.get("metric_name")
    if manifest_metric_name != metric_name:
        return f"metric manifest metric_name mismatch: expected {metric_name}, got {manifest_metric_name!r}"
    expected_input_mode = _input_mode(requirement)
    manifest_input_mode = manifest.get("input_mode")
    if manifest_input_mode != expected_input_mode:
        return f"metric manifest input_mode mismatch: expected {expected_input_mode}, got {manifest_input_mode!r}"
    required_files = manifest.get("required_files")
    if not isinstance(required_files, list) or not all(isinstance(value, str) and value.strip() for value in required_files):
        return "metric manifest required_files must be a list of relative file paths"
    driver = manifest.get("driver")
    if not isinstance(driver, dict):
        return "metric manifest driver must be an object"
    command = driver.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and value.strip() for value in command):
        return "metric manifest driver.command must be a non-empty command array"
    env = manifest.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        return "metric manifest env must be an object of string values"
    return None


def _validate_vmaf_manifest(metric_name: str, manifest: dict[str, Any]) -> str | None:
    manifest_metric_name = manifest.get("metric_name")
    if manifest_metric_name != metric_name:
        return f"metric manifest metric_name mismatch: expected {metric_name}, got {manifest_metric_name!r}"
    ffmpeg_path = manifest.get("ffmpeg_path", "")
    if ffmpeg_path is not None and not isinstance(ffmpeg_path, str):
        return "vmaf manifest ffmpeg_path must be a string when provided"
    return None


def _resolve_manifest_file_path(base_dir: Path, value: str | None) -> Path:
    if not value:
        return base_dir
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _resolve_driver_command(base_dir: Path, command: list[str]) -> list[str]:
    resolved: list[str] = []
    for index, value in enumerate(command):
        token = value.strip()
        candidate = Path(token)
        should_resolve = index == 0 or "\\" in token or "/" in token or bool(candidate.suffix)
        if should_resolve and not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
            if candidate.exists():
                resolved.append(str(candidate))
                continue
        resolved.append(token)
    return resolved


def _resolve_command_executable(base_dir: Path, token: str) -> tuple[str | None, str | None, str | None]:
    candidate = Path(token)
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate.resolve()), "manifest", None
        return None, None, f"metric driver executable does not exist: {candidate.resolve()}"
    relative = (base_dir / candidate).resolve()
    if relative.exists():
        return str(relative), "manifest", None
    resolved = shutil.which(token)
    if resolved:
        return resolved, "path", None
    return None, None, f"metric driver executable is not available: {token}"


def _inspect_ffmpeg_filters(ffmpeg_path: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "reason": f"failed to inspect ffmpeg filters: {exc}"}
    output = f"{completed.stdout}\n{completed.stderr}"
    if "libvmaf" not in output:
        return {"available": False, "reason": "ffmpeg is present but libvmaf filter is not available"}
    return {"available": True, "reason": None}


def _placeholder_manifest(metric_name: str) -> dict[str, Any]:
    if metric_name == "vmaf":
        return {
            "metric_name": metric_name,
            "asset_version": METRIC_ASSET_VERSION,
            "implementation_mode": "ffmpeg_libvmaf",
            "ffmpeg_path": "",
            "notes": [
                "Optional: set ffmpeg_path to a project-local ffmpeg binary with libvmaf support.",
                "Leave ffmpeg_path empty to fall back to ffmpeg on PATH.",
            ],
        }
    input_mode = "video_only" if metric_requires_video_input(metric_name) else "sample_pair"
    return {
        "metric_name": metric_name,
        "asset_version": METRIC_ASSET_VERSION,
        "status": PLACEHOLDER_STATUS,
        "input_mode": input_mode,
        "driver": {
            "command": ["python", "driver.py"],
        },
        "required_files": ["weights.bin"],
        "env": {},
        "notes": "Replace this placeholder with a real manifest, driver, and metric assets before production evaluation.",
    }


def _command_file_paths(command: list[str]) -> list[str]:
    paths: list[str] = []
    for value in command:
        candidate = Path(value)
        if candidate.is_absolute() and candidate.exists():
            paths.append(str(candidate.resolve()))
    return paths


def _path_fingerprints(paths: list[str]) -> list[dict[str, Any]]:
    return [_path_fingerprint(path) for path in paths]


def _path_fingerprint(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(resolved),
    }


def _executable_fingerprint(executable: str | None) -> dict[str, Any] | None:
    if not executable:
        return None
    candidate = Path(executable)
    if candidate.is_absolute() and candidate.exists():
        resolved = str(candidate.resolve())
    else:
        resolved = shutil.which(str(executable))
    if not resolved:
        return {"name": str(executable), "found": False}
    result = {
        "name": str(executable),
        "found": True,
        "path": resolved,
    }
    try:
        completed = subprocess.run(
            [resolved, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first_line = (completed.stdout or completed.stderr).splitlines()
        result["version"] = first_line[0] if first_line else ""
    except Exception as exc:
        result["version_error"] = str(exc)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
