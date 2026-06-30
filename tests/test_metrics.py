from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.metrics import METRIC_NAMES, create_metric
from vfieval.metrics.health import metric_cache_config, prepare_metric_asset_manifest
from vfieval.metrics.base import MetricUnavailable
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
            prepare_metric_asset_manifest(workspace)
            manifest_path = workspace.root.parent / "set" / "metrics" / "lpips_vit_patch" / "manifest.json"

            config1 = metric_cache_config(workspace, "lpips_vit_patch")
            manifest_path.write_text(
                '{"metric_name":"lpips_vit_patch","asset_version":"v2","status":"ready","weights":["a.bin"]}',
                encoding="utf-8",
            )
            config2 = metric_cache_config(workspace, "lpips_vit_patch")

            self.assertNotEqual(config1["expected_paths"], config2["expected_paths"])

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


if __name__ == "__main__":
    unittest.main()
