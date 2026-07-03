from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import hashlib
import tempfile
import urllib.request
import zipfile
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

DINO_REPO_URL = "https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip"
DINO_VITS14_REG_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
CONVNEXT_TINY_WEIGHTS_URL = (
    "https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k/resolve/main/model.safetensors"
)
CGVQM_REPO_URL = "https://github.com/IntelLabs/CGVQM/archive/refs/heads/main.zip"

METRIC_REQUIREMENTS = {
    "lpips_vit_patch": {
        "packages": (),
        "granularity": "sample",
        "supports_timeline": True,
        "evaluator": "dinov2_feature_distance",
        "implementation_mode": "dinov2_feature_distance",
        "backbone": "dinov2_vits14_reg",
        "device_policy": "require_run_device",
        "input_size": 518,
        "setup_summary": (
            "Provide a project-local DINOv2 checkout plus ViT-S/14 register backbone weights under "
            "set/metrics/lpips_vit_patch/. prepare-metrics downloads the default assets when missing."
        ),
    },
    "lpips_convnext": {
        "packages": ("timm",),
        "granularity": "sample",
        "supports_timeline": True,
        "evaluator": "convnext_feature_distance",
        "implementation_mode": "convnext_feature_distance",
        "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "device_policy": "require_run_device",
        "input_size": 288,
        "setup_summary": (
            "Provide a project-local timm ConvNeXt V2 tiny checkpoint under set/metrics/lpips_convnext/. "
            "prepare-metrics downloads the default checkpoint when missing."
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
        "packages": ("numpy", "scipy", "av"),
        "granularity": "video",
        "supports_timeline": False,
        "requires_video_input": True,
        "evaluator": "cgvqm_wrapper",
        "implementation_mode": "cgvqm_wrapper",
        "device_policy": "require_run_device",
        "setup_summary": (
            "Provide a project-local IntelLabs CGVQM checkout, model weights, and wrapper command under "
            "set/metrics/cgvqm/. VFIEval passes GT/Pred videos to that command without vendoring or "
            "substituting scores."
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
    if metric_name in {"lpips_vit_patch", "lpips_convnext"}:
        return _feature_metric_health(metric_name, requirement, manifest_path, manifest, manifest_error)
    if metric_name == "vmaf":
        return _vmaf_health(metric_name, requirement, manifest_path, manifest, manifest_error)
    if metric_name == "cgvqm":
        return _cgvqm_health(metric_name, requirement, manifest_path, manifest, manifest_error)
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
        "backbone": health.get("backbone"),
        "input_size": health.get("input_size"),
        "eval_resolution": health.get("eval_resolution"),
        "pad_multiple": health.get("pad_multiple"),
        "normalize": health.get("normalize"),
        "source_url": health.get("source_url"),
        "video_eval_long_edge": health.get("video_eval_long_edge"),
        "device_policy": health.get("device_policy"),
        "repo_dir": _path_fingerprint(health.get("repo_dir")) if health.get("repo_dir") else None,
    }


def prepare_metric_asset_manifest(
    workspace: WorkspaceConfig,
    *,
    force: bool = False,
    downloader: Any | None = None,
) -> dict[str, Any]:
    root = metric_assets_dir(workspace)
    root.mkdir(parents=True, exist_ok=True)
    prepared = []
    downloads = []
    errors = []
    for name in METRIC_NAMES:
        metric_dir = root / name
        metric_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = metric_dir / "manifest.json"
        if name in {"lpips_vit_patch", "lpips_convnext", "cgvqm"}:
            try:
                result = _prepare_downloaded_metric(name, metric_dir, force=force, downloader=downloader)
                prepared.append(str(result["manifest_path"]))
                downloads.extend(result.get("downloads", []))
            except Exception as exc:
                errors.append({"metric_name": name, "status": "failed", "reason": str(exc)})
            continue
        if not manifest_path.exists() or force:
            manifest_path.write_text(
                json.dumps(_placeholder_manifest(name), indent=2),
                encoding="utf-8",
            )
        prepared.append(str(manifest_path))
    return {"asset_root": str(root), "prepared": prepared, "downloads": downloads, "errors": errors, "health": metrics_health(workspace)}


def _prepare_downloaded_metric(
    metric_name: str,
    metric_dir: Path,
    *,
    force: bool,
    downloader: Any | None,
) -> dict[str, Any]:
    manifest_path = metric_dir / "manifest.json"
    if not force and manifest_path.exists() and metric_name in {"lpips_vit_patch", "lpips_convnext", "cgvqm"}:
        manifest, manifest_error = _load_manifest(manifest_path)
        if manifest_error is None and manifest is not None and not _manifest_is_placeholder(manifest):
            if _downloaded_manifest_assets_exist(metric_name, metric_dir, manifest):
                return {"manifest_path": manifest_path, "downloads": [{"metric_name": metric_name, "status": "skipped"}]}

    if force:
        _remove_declared_metric_assets(metric_name, metric_dir)

    downloads: list[dict[str, Any]] = []
    if metric_name == "lpips_vit_patch":
        downloads.append(_download_and_extract(DINO_REPO_URL, metric_dir / "dinov2", downloader))
        downloads.append(_download_file(DINO_VITS14_REG_WEIGHTS_URL, metric_dir / "dinov2_vits14_reg.pth", downloader))
    elif metric_name == "lpips_convnext":
        downloads.append(_download_file(CONVNEXT_TINY_WEIGHTS_URL, metric_dir / "model.safetensors", downloader))
    elif metric_name == "cgvqm":
        downloads.append(_download_and_extract(CGVQM_REPO_URL, metric_dir / "CGVQM", downloader))
        wrapper_path = metric_dir / "run_cgvqm_vfieval.py"
        _atomic_write_text(wrapper_path, _cgvqm_wrapper_source())
        downloads.append({"url": "vfieval-generated-wrapper", "target": str(wrapper_path.resolve()), "status": "written"})
    else:
        raise ValueError(f"unsupported downloadable metric: {metric_name}")

    manifest_path.write_text(
        json.dumps(_downloaded_manifest(metric_name), indent=2),
        encoding="utf-8",
    )
    return {"manifest_path": manifest_path, "downloads": [{"metric_name": metric_name, **row} for row in downloads]}


def _remove_declared_metric_assets(metric_name: str, metric_dir: Path) -> None:
    for relative in _declared_download_targets(metric_name):
        target = (metric_dir / relative).resolve()
        if not _is_relative_to(target, metric_dir.resolve()):
            continue
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _declared_download_targets(metric_name: str) -> tuple[str, ...]:
    if metric_name == "lpips_vit_patch":
        return ("dinov2", "dinov2_vits14_reg.pth", "manifest.json")
    if metric_name == "lpips_convnext":
        return ("model.safetensors", "manifest.json")
    if metric_name == "cgvqm":
        return ("CGVQM", "run_cgvqm_vfieval.py", "manifest.json")
    return ("manifest.json",)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _downloaded_manifest_assets_exist(metric_name: str, metric_dir: Path, manifest: dict[str, Any]) -> bool:
    if metric_name in {"lpips_vit_patch", "lpips_convnext"}:
        weights_path = _resolve_manifest_file_path(metric_dir, str(manifest.get("weights_path") or ""))
        repo_value = str(manifest.get("repo_dir") or "").strip()
        repo_ok = True if not repo_value else _resolve_manifest_file_path(metric_dir, repo_value).exists()
        return weights_path.exists() and repo_ok
    if metric_name == "cgvqm":
        repo_path = _resolve_manifest_file_path(metric_dir, str(manifest.get("repo_dir") or ""))
        driver = manifest.get("driver") if isinstance(manifest.get("driver"), dict) else {}
        command = driver.get("command") if isinstance(driver.get("command"), list) else []
        wrapper_path = _resolve_manifest_file_path(metric_dir, str(command[1] if len(command) > 1 else "run_cgvqm_vfieval.py"))
        return repo_path.exists() and wrapper_path.exists()
    return True


def _download_file(url: str, target: Path, downloader: Any | None) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(target.parent), suffix=".tmp") as handle:
        tmp_path = Path(handle.name)
    try:
        _run_downloader(url, tmp_path, downloader)
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return {"url": url, "target": str(target.resolve()), "status": "downloaded"}


def _download_and_extract(url: str, target_dir: Path, downloader: Any | None) -> dict[str, Any]:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(target_dir.parent), suffix=".zip") as handle:
        archive_path = Path(handle.name)
    extract_root = Path(tempfile.mkdtemp(dir=str(target_dir.parent)))
    staged_dir = target_dir.with_name(f"{target_dir.name}.download")
    try:
        _run_downloader(url, archive_path, downloader)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_root)
        children = [child for child in extract_root.iterdir()]
        source_dir = children[0] if len(children) == 1 and children[0].is_dir() else extract_root
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        shutil.copytree(source_dir, staged_dir)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        staged_dir.replace(target_dir)
    except Exception:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        raise
    finally:
        if archive_path.exists():
            archive_path.unlink()
        if extract_root.exists():
            shutil.rmtree(extract_root)
    return {"url": url, "target": str(target_dir.resolve()), "status": "downloaded"}


def _run_downloader(url: str, target: Path, downloader: Any | None) -> None:
    if downloader is not None:
        downloader(url, target)
        return
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", suffix=".tmp") as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _downloaded_manifest(metric_name: str) -> dict[str, Any]:
    if metric_name == "lpips_vit_patch":
        return {
            "metric_name": metric_name,
            "asset_version": METRIC_ASSET_VERSION,
            "implementation_mode": "dinov2_feature_distance",
            "backbone": "dinov2_vits14_reg",
            "repo_dir": "dinov2",
            "weights_path": "dinov2_vits14_reg.pth",
            "device_policy": "require_run_device",
            "input_size": 518,
            "pad_multiple": 14,
            "normalize": "imagenet",
            "source_url": {
                "repo": DINO_REPO_URL,
                "weights": DINO_VITS14_REG_WEIGHTS_URL,
            },
        }
    if metric_name == "lpips_convnext":
        return {
            "metric_name": metric_name,
            "asset_version": METRIC_ASSET_VERSION,
            "implementation_mode": "convnext_feature_distance",
            "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "weights_path": "model.safetensors",
            "device_policy": "require_run_device",
            "input_size": 288,
            "pad_multiple": 32,
            "normalize": "imagenet",
            "source_url": {
                "weights": CONVNEXT_TINY_WEIGHTS_URL,
            },
        }
    if metric_name == "cgvqm":
        return {
            "metric_name": metric_name,
            "asset_version": METRIC_ASSET_VERSION,
            "implementation_mode": "cgvqm_wrapper",
            "repo_dir": "CGVQM",
            "weights_path": "run_cgvqm_vfieval.py",
            "device_policy": "require_run_device",
            "video_eval_long_edge": 720,
            "source_url": {
                "repo": CGVQM_REPO_URL,
                "wrapper": "vfieval-generated-wrapper",
            },
            "driver": {
                "command": ["python", "run_cgvqm_vfieval.py"],
            },
            "env": {},
        }
    raise ValueError(f"unsupported downloadable metric: {metric_name}")


def _cgvqm_wrapper_source() -> str:
    return '''from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


def main() -> int:
    payload = json.load(sys.stdin)
    repo_dir = Path(payload["repo_dir"]).resolve()
    sys.path.insert(0, str(repo_dir))
    os.environ.setdefault("VFIEVAL_METRIC_DEVICE", str(payload.get("device") or "cpu"))
    try:
        module = importlib.import_module("cgvqm")
        value, details = _call_metric(module, payload)
        print(json.dumps({"status": "completed", "value": float(value), "details": details}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "unavailable", "value": None, "details": {"reason": str(exc), "type": type(exc).__name__}}))
        return 0


def _call_metric(module, payload: dict) -> tuple[float, dict]:
    reference = payload["reference"]
    distorted = payload["distorted"]
    device = payload.get("device") or "cpu"
    run_cgvqm = getattr(module, "run_cgvqm", None)
    if callable(run_cgvqm):
        cgvqm_type = getattr(getattr(module, "CGVQM_TYPE", object), "CGVQM_2", None)
        kwargs = {"device": device, "patch_pool": "max", "patch_scale": 4}
        if cgvqm_type is not None:
            kwargs["cgvqm_type"] = cgvqm_type
        result = run_cgvqm(distorted, reference, **kwargs)
        value = result[0] if isinstance(result, tuple) else result
        return _to_float(value), {
            "backend": "IntelLabs/CGVQM",
            "entrypoint": "run_cgvqm",
            "cgvqm_type": "CGVQM_2",
            "patch_pool": "max",
            "patch_scale": 4,
        }
    for name in ("compute_cgvqm", "calculate_cgvqm", "evaluate", "cgvqm"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            try:
                return _to_float(candidate(reference, distorted, device=device)), {"backend": "custom", "entrypoint": name}
            except TypeError:
                return _to_float(candidate(reference, distorted)), {"backend": "custom", "entrypoint": name}
    demo = getattr(module, "demo_cgvqm", None)
    if callable(demo):
        raise RuntimeError("CGVQM checkout exposes demo_cgvqm but no run_cgvqm entrypoint.")
    raise RuntimeError("CGVQM checkout does not expose a supported callable metric entrypoint.")


def _to_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
'''


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


def _feature_metric_health(
    metric_name: str,
    requirement: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    manifest_error: str | None,
) -> dict[str, Any]:
    expected_paths = [str(manifest_path.resolve())]
    if manifest_error:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            manifest_error,
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
        )
    if manifest is None or _manifest_is_placeholder(manifest):
        return _status(
            metric_name,
            STATUS_MISSING_WEIGHTS,
            f"missing metric weights: {metric_name}/manifest.json",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            extra=_feature_status_extra(requirement, manifest_path, manifest),
        )
    validation_error = _validate_feature_metric_manifest(metric_name, manifest, requirement)
    if validation_error:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            validation_error,
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            extra=_feature_status_extra(requirement, manifest_path, manifest),
        )

    weights_path = _resolve_manifest_file_path(manifest_path.parent, str(manifest["weights_path"]))
    expected_paths.append(str(weights_path.resolve()))
    if not weights_path.exists():
        return _status(
            metric_name,
            STATUS_MISSING_WEIGHTS,
            f"missing metric weights: {weights_path.resolve()}",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            extra=_feature_status_extra(requirement, manifest_path, manifest, weights_path=weights_path),
        )

    repo_dir = _feature_repo_dir(metric_name, manifest_path, manifest)
    if repo_dir is not None:
        expected_paths.append(str(repo_dir.resolve()))
        if not repo_dir.exists():
            return _status(
                metric_name,
                STATUS_MISSING_EVALUATOR,
                f"metric repo_dir does not exist: {repo_dir.resolve()}",
                requirement,
                manifest_path=manifest_path,
                expected_paths=expected_paths,
                asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
                extra=_feature_status_extra(requirement, manifest_path, manifest, weights_path=weights_path, repo_dir=repo_dir),
            )

    missing_packages = [
        package
        for package in requirement.get("packages", ())
        if importlib.util.find_spec(package) is None
    ]
    if weights_path.suffix.lower() == ".safetensors" and importlib.util.find_spec("safetensors") is None:
        missing_packages.append("safetensors")
    if missing_packages:
        return _status(
            metric_name,
            STATUS_MISSING_DEPENDENCY,
            f"missing Python package: {', '.join(sorted(set(missing_packages)))}",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            extra=_feature_status_extra(requirement, manifest_path, manifest, weights_path=weights_path, repo_dir=repo_dir),
        )

    return _status(
        metric_name,
        STATUS_AVAILABLE,
        None,
        requirement,
        manifest_path=manifest_path,
        expected_paths=expected_paths,
        asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
        extra=_feature_status_extra(requirement, manifest_path, manifest, weights_path=weights_path, repo_dir=repo_dir),
    )


def _cgvqm_health(
    metric_name: str,
    requirement: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    manifest_error: str | None,
) -> dict[str, Any]:
    expected_paths = [str(manifest_path.resolve())]
    if manifest_error:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            manifest_error,
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
        )
    if manifest is None or _manifest_is_placeholder(manifest):
        return _status(
            metric_name,
            STATUS_MISSING_WEIGHTS,
            f"missing metric weights: {metric_name}/manifest.json",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            extra=_cgvqm_status_extra(requirement, manifest_path, manifest),
        )
    validation_error = _validate_cgvqm_manifest(metric_name, manifest)
    if validation_error:
        return _status(
            metric_name,
            STATUS_MISSING_EVALUATOR,
            validation_error,
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            extra=_cgvqm_status_extra(requirement, manifest_path, manifest),
        )

    repo_dir = _resolve_manifest_file_path(manifest_path.parent, str(manifest["repo_dir"]))
    weights_path = _resolve_manifest_file_path(manifest_path.parent, str(manifest["weights_path"]))
    expected_paths.extend([str(repo_dir.resolve()), str(weights_path.resolve())])
    missing_assets = [path for path in (repo_dir, weights_path) if not path.exists()]
    if missing_assets:
        return _status(
            metric_name,
            STATUS_MISSING_WEIGHTS,
            f"missing metric weights: {', '.join(str(path.resolve()) for path in missing_assets)}",
            requirement,
            manifest_path=manifest_path,
            expected_paths=expected_paths,
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            extra=_cgvqm_status_extra(requirement, manifest_path, manifest, repo_dir=repo_dir, weights_path=weights_path),
        )

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
            asset_version=str(manifest.get("asset_version") or METRIC_ASSET_VERSION),
            extra=_cgvqm_status_extra(requirement, manifest_path, manifest, repo_dir=repo_dir, weights_path=weights_path),
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
            driver_command=driver_command,
            env=dict(manifest.get("env") or {}),
            extra=_cgvqm_status_extra(requirement, manifest_path, manifest, repo_dir=repo_dir, weights_path=weights_path),
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
        driver_command=driver_command,
        env=dict(manifest.get("env") or {}),
        resolved_executable=resolved_executable,
        executable_source=executable_source,
        extra=_cgvqm_status_extra(requirement, manifest_path, manifest, repo_dir=repo_dir, weights_path=weights_path),
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
    extra: dict[str, Any] | None = None,
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
        "auto_download": metric_name in {"lpips_vit_patch", "lpips_convnext", "cgvqm"},
    }
    row.update(extra or {})
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


def _validate_feature_metric_manifest(
    metric_name: str,
    manifest: dict[str, Any],
    requirement: dict[str, Any],
) -> str | None:
    if manifest.get("metric_name") != metric_name:
        return f"metric manifest metric_name mismatch: expected {metric_name}, got {manifest.get('metric_name')!r}"
    if manifest.get("implementation_mode") != requirement.get("implementation_mode"):
        return (
            "metric manifest implementation_mode mismatch: "
            f"expected {requirement.get('implementation_mode')}, got {manifest.get('implementation_mode')!r}"
        )
    if manifest.get("backbone") != requirement.get("backbone"):
        return f"metric manifest backbone mismatch: expected {requirement.get('backbone')}, got {manifest.get('backbone')!r}"
    if manifest.get("device_policy") != "require_run_device":
        return "metric manifest device_policy must be require_run_device"
    weights_path = manifest.get("weights_path")
    if not isinstance(weights_path, str) or not weights_path.strip():
        return "metric manifest weights_path must be a non-empty string"
    input_size = manifest.get("input_size")
    if input_size is not None:
        try:
            if int(input_size) <= 0:
                return "metric manifest input_size must be positive"
        except Exception:
            return "metric manifest input_size must be an integer"
    pad_multiple = manifest.get("pad_multiple")
    if pad_multiple is not None:
        try:
            if int(pad_multiple) <= 0:
                return "metric manifest pad_multiple must be positive"
        except Exception:
            return "metric manifest pad_multiple must be an integer"
    repo_dir = manifest.get("repo_dir")
    if metric_name == "lpips_vit_patch" and (not isinstance(repo_dir, str) or not repo_dir.strip()):
        return "lpips_vit_patch manifest repo_dir must point at a local DINOv2 checkout"
    if repo_dir is not None and not isinstance(repo_dir, str):
        return "metric manifest repo_dir must be a string when provided"
    return None


def _validate_cgvqm_manifest(metric_name: str, manifest: dict[str, Any]) -> str | None:
    if manifest.get("metric_name") != metric_name:
        return f"metric manifest metric_name mismatch: expected {metric_name}, got {manifest.get('metric_name')!r}"
    if manifest.get("implementation_mode") != "cgvqm_wrapper":
        return "cgvqm manifest implementation_mode must be cgvqm_wrapper"
    for key in ("repo_dir", "weights_path"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"cgvqm manifest {key} must be a non-empty string"
    if manifest.get("device_policy") != "require_run_device":
        return "cgvqm manifest device_policy must be require_run_device"
    long_edge = manifest.get("video_eval_long_edge")
    if long_edge is not None:
        try:
            if int(long_edge) <= 0:
                return "cgvqm manifest video_eval_long_edge must be positive"
        except Exception:
            return "cgvqm manifest video_eval_long_edge must be an integer"
    driver = manifest.get("driver")
    if not isinstance(driver, dict):
        return "cgvqm manifest driver must be an object"
    command = driver.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and value.strip() for value in command):
        return "cgvqm manifest driver.command must be a non-empty command array"
    env = manifest.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        return "cgvqm manifest env must be an object of string values"
    return None


def _feature_repo_dir(metric_name: str, manifest_path: Path, manifest: dict[str, Any]) -> Path | None:
    value = str(manifest.get("repo_dir") or "").strip()
    if not value:
        return None
    return _resolve_manifest_file_path(manifest_path.parent, value)


def _feature_status_extra(
    requirement: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    *,
    weights_path: Path | None = None,
    repo_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = manifest or {}
    default_weights = "dinov2_vits14_reg.pth" if requirement.get("backbone") == "dinov2_vits14_reg" else "model.safetensors"
    return {
        "backbone": manifest.get("backbone") or requirement.get("backbone"),
        "weights_path": str((weights_path or _resolve_manifest_file_path(manifest_path.parent, str(manifest.get("weights_path") or default_weights))).resolve()),
        "repo_dir": str(repo_dir.resolve()) if repo_dir is not None else (str(_resolve_manifest_file_path(manifest_path.parent, str(manifest.get("repo_dir"))).resolve()) if manifest.get("repo_dir") else None),
        "input_size": int(manifest.get("input_size") or requirement.get("input_size") or 0),
        "eval_resolution": {
            "mode": "max_edge",
            "value": int(manifest.get("input_size") or requirement.get("input_size") or 0),
        },
        "pad_multiple": int(manifest.get("pad_multiple") or (14 if requirement.get("backbone") == "dinov2_vits14_reg" else 32)),
        "normalize": manifest.get("normalize") or "imagenet",
        "source_url": manifest.get("source_url"),
        "device_policy": manifest.get("device_policy") or requirement.get("device_policy"),
    }


def _cgvqm_status_extra(
    requirement: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    *,
    repo_dir: Path | None = None,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    manifest = manifest or {}
    return {
        "repo_dir": str(repo_dir.resolve()) if repo_dir is not None else (str(_resolve_manifest_file_path(manifest_path.parent, str(manifest.get("repo_dir"))).resolve()) if manifest.get("repo_dir") else None),
        "weights_path": str(weights_path.resolve()) if weights_path is not None else (str(_resolve_manifest_file_path(manifest_path.parent, str(manifest.get("weights_path") or "run_cgvqm_vfieval.py")).resolve())),
        "device_policy": manifest.get("device_policy") or requirement.get("device_policy"),
        "video_eval_long_edge": int(manifest.get("video_eval_long_edge") or 720),
        "eval_resolution": {
            "mode": "video_long_edge",
            "value": int(manifest.get("video_eval_long_edge") or 720),
        },
        "source_url": manifest.get("source_url"),
    }


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


def _run_ffmpeg_probe(args: list[str]) -> str:
    """Run an ffmpeg introspection command and return combined stdout+stderr.

    ffmpeg writes most of its help/filter listings to stderr, and on some builds
    the text stream is not valid UTF-8 (localized banners, etc.), so decode
    defensively rather than letting a UnicodeDecodeError bubble up.
    """
    completed = subprocess.run(
        args,
        capture_output=True,
        timeout=15,
        check=False,
    )
    parts = []
    for raw in (completed.stdout, completed.stderr):
        if not raw:
            continue
        parts.append(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw))
    return "\n".join(parts)


def _inspect_ffmpeg_filters(ffmpeg_path: str) -> dict[str, Any]:
    # Different ffmpeg builds surface libvmaf in different places. `-filters`
    # lists it for most builds, but some minimal/static builds only reveal it
    # through the per-filter help. Probe both and treat a hit from either as
    # available so a working libvmaf is not reported as missing.
    probes = (
        [ffmpeg_path, "-hide_banner", "-filters"],
        [ffmpeg_path, "-hide_banner", "-h", "filter=libvmaf"],
    )
    last_error: str | None = None
    for args in probes:
        try:
            output = _run_ffmpeg_probe(args)
        except Exception as exc:
            last_error = f"failed to inspect ffmpeg filters: {exc}"
            continue
        lowered = output.lower()
        if "libvmaf" in lowered and "unknown filter" not in lowered:
            return {"available": True, "reason": None}
    if last_error:
        return {"available": False, "reason": last_error}
    return {"available": False, "reason": "ffmpeg is present but libvmaf filter is not available"}


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
    if metric_name == "lpips_vit_patch":
        return {
            "metric_name": metric_name,
            "asset_version": METRIC_ASSET_VERSION,
            "status": PLACEHOLDER_STATUS,
            "implementation_mode": "dinov2_feature_distance",
            "backbone": "dinov2_vits14_reg",
            "repo_dir": "dinov2",
            "weights_path": "dinov2_vits14_reg.pth",
            "device_policy": "require_run_device",
            "input_size": 518,
            "notes": [
                "prepare-metrics downloads the DINOv2 checkout and ViT-S/14 registers weights here by default.",
                "Use --force to replace only VFIEval-declared metric assets.",
            ],
        }
    if metric_name == "lpips_convnext":
        return {
            "metric_name": metric_name,
            "asset_version": METRIC_ASSET_VERSION,
            "status": PLACEHOLDER_STATUS,
            "implementation_mode": "convnext_feature_distance",
            "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "weights_path": "convnextv2_tiny.fcmae_ft_in22k_in1k.pth",
            "device_policy": "require_run_device",
            "input_size": 288,
            "notes": [
                "Install timm in the metric environment.",
                "prepare-metrics downloads the timm ConvNeXt V2 tiny checkpoint here by default.",
                "Use --force to replace only VFIEval-declared metric assets.",
            ],
        }
    if metric_name == "cgvqm":
        return {
            "metric_name": metric_name,
            "asset_version": METRIC_ASSET_VERSION,
            "status": PLACEHOLDER_STATUS,
            "implementation_mode": "cgvqm_wrapper",
            "repo_dir": "CGVQM",
            "weights_path": "run_cgvqm_vfieval.py",
            "device_policy": "require_run_device",
            "video_eval_long_edge": 720,
            "driver": {
                "command": ["python", "run_cgvqm_vfieval.py"],
            },
            "env": {},
            "notes": [
                "prepare-metrics downloads the IntelLabs/CGVQM checkout and writes the VFIEval wrapper here by default.",
                "The driver command reads VFIEval JSON from stdin and writes {status,value,details} JSON to stdout.",
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
    if resolved.is_dir():
        return _directory_fingerprint(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(resolved),
    }


def _directory_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    latest_mtime_ns = 0
    for child in sorted(path.rglob("*")):
        if not child.is_file() or any(part in {".git", "__pycache__"} for part in child.parts):
            continue
        rel = child.relative_to(path).as_posix()
        stat = child.stat()
        file_count += 1
        total_size += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
        digest.update(rel.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return {
        "path": str(path.resolve()),
        "exists": True,
        "type": "directory",
        "file_count": file_count,
        "size": total_size,
        "mtime_ns": latest_mtime_ns,
        "sha256": digest.hexdigest(),
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
