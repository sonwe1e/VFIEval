from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.metrics.names import METRIC_NAMES


METRIC_ASSET_VERSION = "v1"
METRIC_ADAPTER_VERSION = "native-v1"

METRIC_REQUIREMENTS = {
    "lpips_vit_patch": {
        "packages": ("torch",),
        "assets": ("lpips_vit_patch/manifest.json",),
    },
    "lpips_convnext": {
        "packages": ("torch",),
        "assets": ("lpips_convnext/manifest.json",),
    },
    "vmaf": {
        "packages": (),
        "assets": (),
        "executable": "ffmpeg",
    },
    "cgvqm": {
        "packages": ("torch",),
        "assets": ("cgvqm/manifest.json",),
    },
}


def metric_assets_dir(workspace: WorkspaceConfig) -> Path:
    return workspace.root / "assets" / "metrics"


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
        return _status(metric_name, "missing_dependency", f"{executable} is not on PATH")
    if metric_name == "vmaf":
        vmaf_reason = _vmaf_reason()
        if vmaf_reason:
            return _status(metric_name, "missing_dependency", vmaf_reason)
    if missing_packages:
        return _status(metric_name, "missing_dependency", f"missing Python package: {', '.join(missing_packages)}")
    if missing_assets:
        return _status(metric_name, "missing_assets", f"missing metric assets: {', '.join(missing_assets)}")
    return _status(metric_name, "ready", None)


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


def _status(metric_name: str, status: str, reason: str | None) -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "status": status,
        "adapter_version": METRIC_ADAPTER_VERSION,
        "asset_version": METRIC_ASSET_VERSION,
        "reason": reason,
    }


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


def _asset_missing(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "placeholder"
