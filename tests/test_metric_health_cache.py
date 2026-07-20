from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.metrics.health import metric_health


class MetricHealthCacheTests(unittest.TestCase):
    def test_vmaf_probe_is_cached_refreshed_and_invalidated_by_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            metric_dir = workspace.root.parent / "set" / "metrics" / "vmaf"
            metric_dir.mkdir(parents=True)
            ffmpeg = metric_dir / "ffmpeg.exe"
            ffmpeg.write_bytes(b"test ffmpeg")
            driver_config = metric_dir / "driver.json"
            driver_config.write_text('{"version": 1}', encoding="utf-8")
            manifest = metric_dir / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "metric_name": "vmaf",
                        "asset_version": "v2",
                        "ffmpeg_path": "ffmpeg.exe",
                        "driver": {"command": ["ffmpeg.exe", "driver.json"]},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "vfieval.metrics.health._inspect_ffmpeg_filters",
                return_value={"available": True, "reason": None},
            ) as probe:
                first = metric_health(workspace, "vmaf")
                second = metric_health(workspace, "vmaf")
                refreshed = metric_health(workspace, "vmaf", refresh=True)
                manifest.write_text(
                    json.dumps(
                        {
                            "metric_name": "vmaf",
                            "asset_version": "v2",
                            "ffmpeg_path": "ffmpeg.exe",
                            "driver": {"command": ["ffmpeg.exe", "driver.json"]},
                            "notes": "fingerprint changed",
                        }
                    ),
                    encoding="utf-8",
                )
                invalidated = metric_health(workspace, "vmaf")
                driver_config.write_text('{"version": 2}', encoding="utf-8")
                driver_invalidated = metric_health(workspace, "vmaf")

            self.assertEqual(first, second)
            self.assertEqual(refreshed["status"], "available")
            self.assertEqual(invalidated["status"], "available")
            self.assertEqual(driver_invalidated["status"], "available")
            self.assertEqual(probe.call_count, 4)


if __name__ == "__main__":
    unittest.main()
