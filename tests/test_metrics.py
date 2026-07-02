from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import textwrap
import unittest
import zipfile
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.metrics import METRIC_NAMES, create_metric
from vfieval.metrics.base import MetricResult, MetricUnavailable
from vfieval.metrics.health import metric_cache_config, metric_health, prepare_metric_asset_manifest
from vfieval.pipeline.metrics_runner import _evaluate_with_cache, metric_cache_key


class MetricTests(unittest.TestCase):
    def test_registry_excludes_psnr(self) -> None:
        self.assertNotIn("psnr", METRIC_NAMES)
        with self.assertRaisesRegex(ValueError, "unsupported metric"):
            create_metric("psnr")

    def test_vmaf_rejects_image_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "ref.png"
            distorted = tmp_path / "dist.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
            Image.new("RGB", (4, 4), (1, 1, 1)).save(distorted)
            metric = create_metric("vmaf")

            with self.assertRaises(MetricUnavailable):
                metric.evaluate(reference, distorted, tmp_path / "work")

    def test_metric_cache_key_uses_file_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "ref.png"
            distorted = tmp_path / "dist.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
            Image.new("RGB", (4, 4), (1, 1, 1)).save(distorted)

            key1 = metric_cache_key("cgvqm", reference, distorted, {})
            key1_config = metric_cache_key("cgvqm", reference, distorted, {"adapter_version": "x"})
            Image.new("RGB", (5, 5), (1, 1, 1)).save(distorted)
            key2 = metric_cache_key("cgvqm", reference, distorted, {})

            self.assertNotEqual(key1, key2)
            self.assertNotEqual(key1, key1_config)

    def test_metric_cache_config_changes_when_manifest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "lpips_vit_patch")
            _write_fake_driver(metric_dir)
            _write_manifest(metric_dir, "lpips_vit_patch")

            config1 = metric_cache_config(workspace, "lpips_vit_patch")
            manifest_path = metric_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["asset_version"] = "v3"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            config2 = metric_cache_config(workspace, "lpips_vit_patch")

            self.assertNotEqual(config1["manifest_path"], config2["manifest_path"])

    def test_metric_cache_config_changes_when_feature_weights_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "lpips_convnext")
            weights_path = metric_dir / "weights.pth"
            weights_path.write_bytes(b"weights-v1")
            _write_feature_manifest(metric_dir, "lpips_convnext", weights_path.name)

            config1 = metric_cache_config(workspace, "lpips_convnext")
            weights_path.write_bytes(b"weights-v2")
            config2 = metric_cache_config(workspace, "lpips_convnext")

            self.assertNotEqual(config1["weights_path"], config2["weights_path"])

    def test_prepare_metric_asset_manifest_includes_vmaf_override_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            result = prepare_metric_asset_manifest(workspace, downloader=_fake_metric_downloader)
            vmaf_manifest = workspace.root.parent / "set" / "metrics" / "vmaf" / "manifest.json"
            data = json.loads(vmaf_manifest.read_text(encoding="utf-8"))

            self.assertIn(str(vmaf_manifest), result["prepared"])
            self.assertEqual(data["metric_name"], "vmaf")
            self.assertIn("ffmpeg_path", data)
            self.assertNotIn("status", data)

    def test_feature_metric_health_requires_local_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            vit = metric_health(workspace, "lpips_vit_patch")
            convnext = metric_health(workspace, "lpips_convnext")

            self.assertEqual(vit["status"], "missing_weights")
            self.assertEqual(vit["implementation_mode"], "dinov2_feature_distance")
            self.assertEqual(vit["backbone"], "dinov2_vits14_reg")
            self.assertEqual(convnext["status"], "missing_weights")
            self.assertEqual(convnext["implementation_mode"], "convnext_feature_distance")
            self.assertEqual(convnext["backbone"], "convnextv2_tiny.fcmae_ft_in22k_in1k")

    def test_prepare_metric_asset_manifest_writes_feature_metric_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            result = prepare_metric_asset_manifest(workspace, downloader=_fake_metric_downloader)
            vit_manifest = workspace.root.parent / "set" / "metrics" / "lpips_vit_patch" / "manifest.json"
            convnext_manifest = workspace.root.parent / "set" / "metrics" / "lpips_convnext" / "manifest.json"
            cgvqm_manifest = workspace.root.parent / "set" / "metrics" / "cgvqm" / "manifest.json"

            vit = json.loads(vit_manifest.read_text(encoding="utf-8"))
            convnext = json.loads(convnext_manifest.read_text(encoding="utf-8"))
            cgvqm = json.loads(cgvqm_manifest.read_text(encoding="utf-8"))
            self.assertFalse(result["errors"])
            self.assertEqual(vit["backbone"], "dinov2_vits14_reg")
            self.assertEqual(vit["device_policy"], "require_run_device")
            self.assertEqual(vit["input_size"], 518)
            self.assertEqual(vit["pad_multiple"], 14)
            self.assertTrue((vit_manifest.parent / "dinov2").is_dir())
            self.assertTrue((vit_manifest.parent / "dinov2_vits14_reg.pth").is_file())
            self.assertEqual(convnext["backbone"], "convnextv2_tiny.fcmae_ft_in22k_in1k")
            self.assertEqual(convnext["weights_path"], "model.safetensors")
            self.assertEqual(convnext["input_size"], 288)
            self.assertEqual(convnext["pad_multiple"], 32)
            self.assertTrue((convnext_manifest.parent / "model.safetensors").is_file())
            self.assertEqual(cgvqm["implementation_mode"], "cgvqm_wrapper")
            self.assertEqual(cgvqm["video_eval_long_edge"], 720)
            self.assertEqual(cgvqm["weights_path"], "run_cgvqm_vfieval.py")
            self.assertIn("driver", cgvqm)
            self.assertTrue((cgvqm_manifest.parent / "CGVQM").is_dir())
            self.assertTrue((cgvqm_manifest.parent / "run_cgvqm_vfieval.py").is_file())

    def test_generated_cgvqm_wrapper_calls_intellabs_run_cgvqm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            result = prepare_metric_asset_manifest(workspace, downloader=_fake_metric_downloader)
            self.assertFalse(result["errors"])
            metric_dir = workspace.root.parent / "set" / "metrics" / "cgvqm"
            wrapper = metric_dir / "run_cgvqm_vfieval.py"
            payload = {
                "metric_name": "cgvqm",
                "reference": str((Path(tmp) / "ref.mp4").resolve()),
                "distorted": str((Path(tmp) / "dist.mp4").resolve()),
                "repo_dir": str((metric_dir / "CGVQM").resolve()),
                "device": "cpu",
            }

            completed = subprocess.run(
                [sys.executable, str(wrapper)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["status"], "completed")
            self.assertEqual(output["value"], 0.5)
            self.assertEqual(output["details"]["entrypoint"], "run_cgvqm")
            self.assertEqual(output["details"]["patch_pool"], "max")
            self.assertEqual(output["details"]["patch_scale"], 4)

    def test_prepare_metrics_check_only_does_not_create_metric_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            from vfieval.metrics.health import metrics_health

            health = metrics_health(workspace)

            self.assertIn("lpips_vit_patch", health["metrics"])
            self.assertFalse((workspace.root.parent / "set").exists())

    def test_prepare_metric_download_failure_leaves_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            def failing_downloader(_url: str, _target: Path) -> None:
                raise RuntimeError("network blocked")

            result = prepare_metric_asset_manifest(workspace, downloader=failing_downloader)
            vit_manifest = workspace.root.parent / "set" / "metrics" / "lpips_vit_patch" / "manifest.json"

            self.assertFalse(vit_manifest.exists())
            self.assertTrue(result["errors"])
            self.assertEqual(result["health"]["metrics"]["lpips_vit_patch"]["status"], "missing_weights")

    def test_cgvqm_wrapper_metric_completes_from_driver_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_fake_driver(metric_dir, value=0.42)
            _write_cgvqm_manifest(metric_dir)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            reference.write_bytes(b"ref")
            distorted.write_bytes(b"dist")

            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                metric = create_metric("cgvqm", workspace, device="cpu")
                result = metric.evaluate(reference, distorted, tmp_path / "work")

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.value, 0.42)
            self.assertEqual(result.details["manifest_path"], str((metric_dir / "manifest.json").resolve()))
            self.assertEqual(result.details["metric_name"], "cgvqm")

    def test_cgvqm_wrapper_metric_maps_unavailable_stdout_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_fake_driver(metric_dir, mode="unavailable")
            _write_cgvqm_manifest(metric_dir)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            reference.write_bytes(b"ref")
            distorted.write_bytes(b"dist")

            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                metric = create_metric("cgvqm", workspace, device="cpu")
                with self.assertRaisesRegex(MetricUnavailable, "driver reported unavailable"):
                    metric.evaluate(reference, distorted, tmp_path / "work")

    def test_cgvqm_wrapper_metric_maps_failed_stdout_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_fake_driver(metric_dir, mode="failed")
            _write_cgvqm_manifest(metric_dir)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            reference.write_bytes(b"ref")
            distorted.write_bytes(b"dist")

            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                metric = create_metric("cgvqm", workspace, device="cpu")
                with self.assertRaisesRegex(RuntimeError, "driver reported failed"):
                    metric.evaluate(reference, distorted, tmp_path / "work")

    def test_feature_and_cgvqm_health_maps_missing_assets_and_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            missing_manifest = metric_health(workspace, "lpips_vit_patch")
            self.assertEqual(missing_manifest["status"], "missing_weights")

            metric_dir = _metric_dir(workspace, "lpips_convnext")
            _write_feature_manifest(metric_dir, "lpips_convnext", "missing.bin")
            missing_weight = metric_health(workspace, "lpips_convnext")
            self.assertEqual(missing_weight["status"], "missing_weights")
            self.assertIn("missing.bin", missing_weight["reason"])

            cgvqm_dir = _metric_dir(workspace, "cgvqm")
            _write_cgvqm_manifest(cgvqm_dir, command=["missing-driver.exe"])
            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                missing_driver = metric_health(workspace, "cgvqm")
            self.assertEqual(missing_driver["status"], "missing_evaluator")
            self.assertIn("driver executable", missing_driver["reason"])

    def test_vmaf_manifest_overrides_ffmpeg_path_for_health_and_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "vmaf")
            ffmpeg_cmd = _write_fake_ffmpeg(metric_dir, value=93.5)
            _write_vmaf_manifest(metric_dir, ffmpeg_cmd.name)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            reference.write_bytes(b"ref")
            distorted.write_bytes(b"dist")

            health = metric_health(workspace, "vmaf")
            self.assertEqual(health["status"], "available")
            self.assertEqual(health["implementation_mode"], "ffmpeg_libvmaf")
            self.assertEqual(health["executable_source"], "manifest")
            self.assertEqual(health["resolved_executable"], str(ffmpeg_cmd.resolve()))

            metric = create_metric("vmaf", workspace)
            result = metric.evaluate(reference, distorted, tmp_path / "work")

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.value, 93.5)
            self.assertEqual(result.details["resolved_executable"], str(ffmpeg_cmd.resolve()))

    def test_unavailable_metric_cache_invalidates_when_metric_setup_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference = tmp_path / "ref.png"
            distorted = tmp_path / "dist.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
            Image.new("RGB", (4, 4), (1, 1, 1)).save(distorted)

            calls: list[str] = []

            class FakeMetric:
                def evaluate(self, _reference: Path, _distorted: Path, _work_dir: Path):
                    calls.append("evaluate")
                    if len(calls) == 1:
                        raise MetricUnavailable("missing weights")
                    return type("Result", (), {"status": "completed", "value": 0.5, "details": {"source": "fake"}})()

            unavailable_config = {"status": "missing_weights", "asset_version": "v1"}
            available_config = {"status": "available", "asset_version": "v2"}

            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=FakeMetric()):
                first = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config=unavailable_config,
                )
                second = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config=available_config,
                )
                third = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config=available_config,
                )

            self.assertEqual(first[0], "unavailable")
            self.assertEqual(second[0], "completed")
            self.assertEqual(third[0], "completed")
            self.assertEqual(third[2]["cached"], True)
            self.assertEqual(calls, ["evaluate", "evaluate"])

    def test_evaluate_with_cache_records_metric_device_and_unavailable_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference, distorted = _write_image_pair(tmp_path)

            class DeviceFailMetric:
                def evaluate(self, _reference: Path, _distorted: Path, _work_dir: Path):
                    raise MetricUnavailable("metric device cuda:7 failed warmup: out of memory")

            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=DeviceFailMetric()) as factory:
                status, value, details = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_convnext",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available"},
                    metric_device="cuda:7",
                )

            self.assertEqual(status, "unavailable")
            self.assertIsNone(value)
            self.assertIn("cuda:7", details["reason"])
            factory.assert_called_once_with("lpips_convnext", workspace, device="cuda:7")

    def test_feature_metric_completed_result_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference, distorted = _write_image_pair(tmp_path)
            calls: list[str] = []

            class FakeMetric:
                def evaluate(self, _reference: Path, _distorted: Path, _work_dir: Path):
                    calls.append("evaluate")
                    return MetricResult("completed", 0.125, {"source": "fake"})

            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=FakeMetric()):
                first = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available", "backbone": "fake"},
                    metric_device="cpu",
                )
                second = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available", "backbone": "fake"},
                    metric_device="cpu",
                )

            self.assertEqual(first[0], "completed")
            self.assertEqual(first[1], 0.125)
            self.assertEqual(second[2]["cached"], True)
            self.assertEqual(calls, ["evaluate"])

    def test_cli_smoke_metric_runs_manifest_driver_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_fake_driver(metric_dir, value=0.77)
            _write_cgvqm_manifest(metric_dir)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            reference.write_bytes(b"ref")
            distorted.write_bytes(b"dist")
            (tmp_path / "av.py").write_text("", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vfieval.cli",
                    "--workspace",
                    str(workspace.root),
                    "smoke-metric",
                    "--metric",
                    "cgvqm",
                    "--reference",
                    str(reference),
                    "--distorted",
                    str(distorted),
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["value"], 0.77)
            self.assertEqual(payload["details"]["metric_name"], "cgvqm")

    def test_vmaf_real_execution_is_conditional_on_local_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            health = metric_health(workspace, "vmaf")
            if not health["available"]:
                self.assertEqual(health["status"], "missing_evaluator")
                return

            video = next((ROOT / "videos" / "test_style").glob("*.*"))
            metric = create_metric("vmaf", workspace)
            result = metric.evaluate(video, video, workspace.tmp_dir / "real-vmaf")

            self.assertEqual(result.status, "completed")
            self.assertIsInstance(result.value, float)


def _metric_dir(workspace: WorkspaceConfig, metric_name: str) -> Path:
    metric_dir = workspace.root.parent / "set" / "metrics" / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    return metric_dir


def _write_image_pair(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "ref.png"
    distorted = tmp_path / "dist.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
    Image.new("RGB", (4, 4), (1, 1, 1)).save(distorted)
    return reference, distorted


def _write_manifest(
    metric_dir: Path,
    metric_name: str,
    *,
    input_mode: str = "sample_pair",
    command: list[str] | None = None,
    required_files: list[str] | None = None,
    env: dict[str, str] | None = None,
    create_required_files: bool = True,
) -> Path:
    manifest_path = metric_dir / "manifest.json"
    driver_command = command or [sys.executable, "driver.py"]
    required = required_files or ["weights.bin"]
    payload = {
        "metric_name": metric_name,
        "asset_version": "v2",
        "input_mode": input_mode,
        "driver": {"command": driver_command},
        "required_files": required,
        "env": env or {},
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if create_required_files:
        for rel_path in required:
            asset_path = metric_dir / rel_path
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            if not asset_path.exists():
                asset_path.write_bytes(b"asset")
    return manifest_path


def _write_feature_manifest(metric_dir: Path, metric_name: str, weights_path: str) -> Path:
    if metric_name == "lpips_vit_patch":
        implementation_mode = "dinov2_feature_distance"
        backbone = "dinov2_vits14_reg"
        input_size = 518
        repo_dir = "dinov2"
    else:
        implementation_mode = "convnext_feature_distance"
        backbone = "convnextv2_tiny.fcmae_ft_in22k_in1k"
        input_size = 288
        repo_dir = None
    payload = {
        "metric_name": metric_name,
        "asset_version": "v2",
        "implementation_mode": implementation_mode,
        "backbone": backbone,
        "weights_path": weights_path,
        "device_policy": "require_run_device",
        "input_size": input_size,
    }
    if repo_dir is not None:
        payload["repo_dir"] = repo_dir
    manifest_path = metric_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _write_cgvqm_manifest(metric_dir: Path, command: list[str] | None = None) -> Path:
    (metric_dir / "CGVQM").mkdir(parents=True, exist_ok=True)
    (metric_dir / "weights").mkdir(parents=True, exist_ok=True)
    manifest_path = metric_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metric_name": "cgvqm",
                "asset_version": "v2",
                "implementation_mode": "cgvqm_wrapper",
                "repo_dir": "CGVQM",
                "weights_path": "weights",
                "device_policy": "require_run_device",
                "driver": {"command": command or [sys.executable, "driver.py"]},
                "env": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_vmaf_manifest(metric_dir: Path, ffmpeg_path: str) -> Path:
    manifest_path = metric_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metric_name": "vmaf",
                "asset_version": "v2",
                "ffmpeg_path": ffmpeg_path,
                "notes": "test override",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_fake_driver(metric_dir: Path, *, mode: str = "completed", value: float = 0.25) -> Path:
    driver_path = metric_dir / "driver.py"
    driver_path.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            payload = json.load(sys.stdin)
            result = {{
                "status": {mode!r},
                "value": {value!r} if {mode!r} == "completed" else None,
                "details": {{
                    "metric_name": payload["metric_name"],
                    "manifest_path": payload["manifest_path"],
                    "reason": "driver reported {mode}" if {mode!r} != "completed" else "",
                }},
            }}
            print(json.dumps(result))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return driver_path


def _write_fake_ffmpeg(metric_dir: Path, *, value: float) -> Path:
    script_path = metric_dir / "fake_ffmpeg.py"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            args = sys.argv[1:]
            if "-version" in args:
                print("ffmpeg version fake-vmaf")
                raise SystemExit(0)
            if "-filters" in args:
                print(" ... libvmaf ... ")
                raise SystemExit(0)
            if "-lavfi" in args:
                spec = args[args.index("-lavfi") + 1]
                log_path = spec.split("log_path=", 1)[1].strip("'").replace("\\:", ":")
                with open(log_path, "w", encoding="utf-8") as handle:
                    json.dump({{"pooled_metrics": {{"vmaf": {{"mean": {value!r}}}}}}}, handle)
                print("fake ffmpeg vmaf run")
                raise SystemExit(0)
            print("unexpected args: " + " ".join(args), file=sys.stderr)
            raise SystemExit(1)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    wrapper_path = metric_dir / "ffmpeg.cmd"
    wrapper_path.write_text(
        f'@echo off\r\n"{sys.executable}" "%~dp0fake_ffmpeg.py" %*\r\n',
        encoding="utf-8",
    )
    return wrapper_path


def _fake_metric_downloader(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if url.endswith(".zip"):
        top = "repo-main"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr(f"{top}/cgvqm.py", _fake_cgvqm_module())
            archive.writestr(f"{top}/hubconf.py", "def dinov2_vits14_reg(*args, **kwargs):\n    return None\n")
        return
    target.write_bytes(f"downloaded from {url}".encode("utf-8"))


def _fake_cgvqm_module() -> str:
    return textwrap.dedent(
        """
        class CGVQM_TYPE:
            CGVQM_2 = "cgvqm-2"

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def item(self):
                return 0.5

        def run_cgvqm(test_vid_path, ref_vid_path, cgvqm_type=None, device='cpu', patch_pool='max', patch_scale=4):
            if patch_pool != 'max' or patch_scale != 4:
                raise RuntimeError('bad wrapper defaults')
            if cgvqm_type != CGVQM_TYPE.CGVQM_2:
                raise RuntimeError('bad cgvqm type')
            if not str(test_vid_path).endswith('dist.mp4') or not str(ref_vid_path).endswith('ref.mp4'):
                raise RuntimeError('wrong argument order')
            return FakeTensor(), 'emap'

        def compute_cgvqm(reference, distorted, device='cpu'):
            return 0.25
        """
    ).strip() + "\n"


if __name__ == "__main__":
    unittest.main()
