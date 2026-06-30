from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.metrics import METRIC_NAMES, create_metric
from vfieval.metrics.base import MetricUnavailable
from vfieval.pipeline.metrics_runner import metric_cache_key


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


if __name__ == "__main__":
    unittest.main()
