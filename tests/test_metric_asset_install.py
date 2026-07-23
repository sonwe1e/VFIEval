from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from vfieval.metrics.health import (
    METRIC_REQUIREMENTS,
    _download_and_extract,
    _download_file,
    _feature_status_extra,
    _prepare_downloaded_metric,
)


class MetricAssetInstallTests(unittest.TestCase):
    def test_download_hash_mismatch_preserves_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights.bin"
            target.write_bytes(b"known-good")

            def downloader(_url: str, path: Path) -> None:
                path.write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _download_file(
                    "https://example.invalid/weights.bin",
                    target,
                    downloader,
                    expected_sha256="0" * 64,
                )

            self.assertEqual(target.read_bytes(), b"known-good")

    def test_force_prepare_failure_keeps_previous_metric_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metric_dir = Path(directory) / "lpips_vit_patch"
            metric_dir.mkdir()
            sentinel = metric_dir / "sentinel.txt"
            sentinel.write_text("previous-version", encoding="utf-8")
            (metric_dir / "manifest.json").write_text(
                json.dumps({"metric_name": "lpips_vit_patch", "status": "placeholder"}),
                encoding="utf-8",
            )
            calls = 0

            def downloader(url: str, target: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("second asset failed")
                with zipfile.ZipFile(target, "w") as archive:
                    archive.writestr("repo/hubconf.py", "def model():\n    return None\n")

            with self.assertRaisesRegex(RuntimeError, "second asset failed"):
                _prepare_downloaded_metric(
                    "lpips_vit_patch",
                    metric_dir,
                    force=True,
                    downloader=downloader,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "previous-version")
            self.assertFalse(any(metric_dir.parent.glob(".lpips_vit_patch.staging-*")))

    def test_zip_extraction_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def downloader(_url: str, target: Path) -> None:
                with zipfile.ZipFile(target, "w") as archive:
                    archive.writestr("../escape.txt", "unsafe")

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                _download_and_extract(
                    "https://example.invalid/archive.zip",
                    root / "repo",
                    downloader,
                )

            self.assertFalse((root / "escape.txt").exists())

    def test_two_clean_installs_have_the_same_implementation_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def downloader(url: str, target: Path) -> None:
                if url.endswith(".zip"):
                    with zipfile.ZipFile(target, "w") as archive:
                        archive.writestr(
                            "repo/hubconf.py",
                            "def dinov2_vits14_reg(*_args, **_kwargs):\n"
                            "    return None\n",
                        )
                    return
                target.write_bytes(b"stable-test-weights")

            fingerprints: list[str] = []
            manifests: list[bytes] = []
            for install_name in ("first", "second"):
                metric_dir = root / install_name / "lpips_vit_patch"
                metric_dir.parent.mkdir(parents=True)
                _prepare_downloaded_metric(
                    "lpips_vit_patch",
                    metric_dir,
                    force=False,
                    downloader=downloader,
                )
                manifest_path = metric_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                status = _feature_status_extra(
                    METRIC_REQUIREMENTS["lpips_vit_patch"],
                    manifest_path,
                    manifest,
                    weights_path=metric_dir / "dinov2_vits14_reg.pth",
                    repo_dir=metric_dir / "dinov2",
                )
                fingerprints.append(str(status["implementation_fingerprint"]))
                manifests.append(manifest_path.read_bytes())

            self.assertEqual(fingerprints[0], fingerprints[1])
            self.assertEqual(manifests[0], manifests[1])


if __name__ == "__main__":
    unittest.main()
