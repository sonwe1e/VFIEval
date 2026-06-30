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


METRIC_ASSET_VERSION = "v1"
METRIC_ADAPTER_VERSION = "native-v1"
STATUS_AVAILABLE = "available"
STATUS_MISSING_WEIGHTS = "missing_weights"
STATUS_MISSING_EVALUATOR = "missing_evaluator"
STATUS_MISSING_DEPENDENCY = "missing_dependency"

METRIC_REQUIREMENTS = {
    "lpips_vit_patch": {
        "packages": ("torch",),
        "assets": ("lpips_vit_patch/manifest.json",),
        "granularity": "sample",
        "supports_timeline": True,
        "evaluator": "native_binding",
        "binding_available": False,
        "setup_summary": "Place the LPIPS ViT Patch manifest at the expected project-local manifest path and install the official native evaluator binding in this Python environment.",
    },
    "lpips_convnext": {
        "packages": ("torch",),
        "assets": ("lpips_convnext/manifest.json",),
        "granularity": "sample",
        "supports_timeline": True,
        "evaluator": "native_binding",
        "binding_available": False,
        "setup_summary": "Place the LPIPS ConvNeXt manifest at the expected project-local manifest path and install the official native evaluator binding in this Python environment.",
    },
    "vmaf": {
        "packages": (),
        "assets": (),
        "executable": "ffmpeg",
        "granularity": "video",
        "supports_timeline": False,
        "requires_video_input": True,
        "evaluator": "ffmpeg_libvmaf",
        "setup_summary": "Install ffmpeg on PATH with the libvmaf filter enabled. The current build does not use project-local VMAF weights.",
    },
    "cgvqm": {
        "packages": ("torch",),
        "assets": ("cgvqm/manifest.json",),
        "granularity": "video",
        "supports_timeline": False,
        "requires_video_input": True,
        "evaluator": "native_binding",
        "binding_available": False,
        "setup_summary": "Place the CGVQM manifest at the expected project-local manifest path, install the official native evaluator binding in this Python environment, and run it against GT/Pred video artifacts.",
    },
}


def metric_assets_dir(workspace: WorkspaceConfig) -> Path:
    configured = os.getenv("VFIEVAL_METRIC_ASSETS_DIR")
    if configured:
        return Path(configured).resolve()
    project_root = Path(os.getenv("VFIEVAL_PROJECT_ROOT") or workspace.root.parent).resolve()
    return project_root / "set" / "metrics"


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
    asset_root = metric_assets_dir(workspace)
    expected_paths = [
        str((asset_root / rel_path).resolve())
        for rel_path in requirement.get("assets", ())
    ]
    missing_packages = [
        package
        for package in requirement.get("packages", ())
        if importlib.util.find_spec(package) is None
    ]
    missing_assets = [
        rel_path
        for rel_path in requirement.get("assets", ())
        if _asset_missing(metric_assets_dir(workspace) / rel_path)
    ]
    executable = requirement.get("executable")
    if executable and shutil.which(str(executable)) is None:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            f"{executable} is not on PATH",
            requirement,
            expected_paths,
        )
    if metric_name == "vmaf":
        vmaf_reason = _vmaf_reason()
        if vmaf_reason:
            return _status(metric_name, STATUS_MISSING_EVALUATOR, vmaf_reason, requirement, expected_paths)
    if missing_packages:
        return _status(
            metric_name,
            STATUS_MISSING_DEPENDENCY,
            f"missing Python package: {', '.join(missing_packages)}",
            requirement,
            expected_paths,
        )
    if missing_assets:
        return _status(
            metric_name,
            STATUS_MISSING_WEIGHTS,
            f"missing metric weights: {', '.join(missing_assets)}",
            requirement,
            expected_paths,
        )
    native_binding_reason = _native_binding_reason(requirement)
    if native_binding_reason:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            native_binding_reason,
            requirement,
            expected_paths,
        )
    return _status(metric_name, STATUS_AVAILABLE, None, requirement, expected_paths)


def metric_requires_video_input(metric_name: str) -> bool:
    requirement = METRIC_REQUIREMENTS[metric_name]
    return bool(requirement.get("requires_video_input"))


def metric_cache_config(workspace: WorkspaceConfig, metric_name: str) -> dict[str, Any]:
    health = metric_health(workspace, metric_name)
    return {
        "metric_name": metric_name,
        "adapter_version": health.get("adapter_version"),
        "asset_version": health.get("asset_version"),
        "status": health.get("status"),
        "reason": health.get("reason"),
        "evaluator": health.get("evaluator"),
        "granularity": health.get("granularity"),
        "input_mode": health.get("input_mode"),
        "requires_video_input": bool(health.get("requires_video_input")),
        "expected_paths": _path_fingerprints(health.get("expected_paths") or []),
        "weights_path": _path_fingerprint(health.get("weights_path")) if health.get("weights_path") else None,
        "executable": _executable_fingerprint(health.get("executable")),
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
                json.dumps(
                    {
                        "metric_name": name,
                        "asset_version": METRIC_ASSET_VERSION,
                        "status": "placeholder",
                        "note": "Replace with official metric assets before production evaluation.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        prepared.append(str(manifest_path))
    return {"asset_root": str(root), "prepared": prepared, "health": metrics_health(workspace)}


def _status(
    metric_name: str,
    status: str,
    reason: str | None,
    requirement: dict[str, Any],
    expected_paths: list[str],
) -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "status": status,
        "available": status == STATUS_AVAILABLE,
        "adapter_version": METRIC_ADAPTER_VERSION,
        "asset_version": METRIC_ASSET_VERSION,
        "reason": reason,
        "weights_path": expected_paths[0] if expected_paths else None,
        "expected_paths": expected_paths,
        "packages": list(requirement.get("packages", ())),
        "executable": requirement.get("executable"),
        "granularity": requirement.get("granularity"),
        "supports_timeline": bool(requirement.get("supports_timeline")),
        "requires_video_input": bool(requirement.get("requires_video_input")),
        "input_mode": _input_mode(requirement),
        "evaluator": requirement.get("evaluator"),
        "setup_summary": requirement.get("setup_summary"),
        "setup_requirements": _setup_requirements(metric_name, requirement, expected_paths),
        "auto_download": False,
    }


def _input_mode(requirement: dict[str, Any]) -> str:
    if bool(requirement.get("requires_video_input")):
        return "video_only"
    if requirement.get("granularity") == "sample":
        return "sample_pair"
    return "artifact_or_video"


def _setup_requirements(metric_name: str, requirement: dict[str, Any], expected_paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in expected_paths:
        rows.append(
            {
                "kind": "asset",
                "target": path,
                "description": "Place the metric manifest and any referenced weights here.",
            }
        )
    executable = requirement.get("executable")
    if executable:
        rows.append(
            {
                "kind": "executable",
                "target": str(executable),
                "description": "Install this executable on PATH for metric execution.",
            }
        )
    if metric_name == "vmaf":
        rows.append(
            {
                "kind": "ffmpeg_filter",
                "target": "libvmaf",
                "description": "The ffmpeg build on PATH must expose the libvmaf filter.",
            }
        )
    if requirement.get("evaluator") == "native_binding":
        rows.append(
            {
                "kind": "binding",
                "target": "official_native_binding",
                "description": "Install the official native evaluator binding in the active Python environment.",
            }
        )
    return rows


def _vmaf_reason() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "ffmpeg is not on PATH"
    try:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return f"failed to inspect ffmpeg filters: {exc}"
    output = f"{completed.stdout}\n{completed.stderr}"
    if "libvmaf" not in output:
        return "ffmpeg is present but libvmaf filter is not available"
    return None


def _native_binding_reason(requirement: dict[str, Any]) -> str | None:
    if requirement.get("evaluator") != "native_binding":
        return None
    if bool(requirement.get("binding_available", False)):
        return None
    return "official native evaluator binding is not installed in this build"


def _asset_missing(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "placeholder"


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
