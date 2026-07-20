from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.file_inputs import preflight_run


class TwoLevelPreflightTests(unittest.TestCase):
    def test_quick_uses_container_metadata_and_skips_model_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            workspace = WorkspaceConfig.from_root(project / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            (project / "models").mkdir()
            (project / "models" / "model.py").write_text("# fixture\n", encoding="utf-8")
            video_dir = project / "videos" / "set-a"
            video_dir.mkdir(parents=True)
            (video_dir / "clip.mp4").write_bytes(b"video-fixture")

            exact_modes: list[bool] = []

            def inspect_video(path, _workspace=None, exact=True):
                exact_modes.append(bool(exact))
                return {
                    "name": Path(path).name,
                    "path": str(path),
                    "decodable": True,
                    "error": None,
                    "frame_count": 12,
                    "frame_count_source": "exact" if exact else "container",
                    "duration_seconds": 0.5,
                    "fps": 24.0,
                    "width": 640,
                    "height": 360,
                    "metadata_source": "opencv",
                }

            request = {
                "model_file": "model.py",
                "video_group": "set-a",
                "selected_videos": ["clip.mp4"],
                "device": "cpu",
                "precision": "fp32",
            }
            with (
                patch("vfieval.file_inputs.inspect_video", side_effect=inspect_video),
                patch(
                    "vfieval.file_inputs._cached_dry_run_model_file",
                    return_value={"output_health": {"warnings": [], "stats": {}}},
                ) as dry_run,
                patch("vfieval.file_inputs.decode_cache_key", return_value="cache-key") as cache_key,
                patch("vfieval.file_inputs.decode_cache_status", return_value="未解码"),
                patch("vfieval.file_inputs.metrics_health", return_value={"metrics": {}}),
            ):
                quick = preflight_run(
                    db,
                    workspace,
                    {**request, "preflight_level": "quick"},
                )
                self.assertTrue(quick["ok"], quick)
                self.assertEqual(quick["preflight_level"], "quick")
                self.assertIsNone(quick["model"]["interface_ok"])
                self.assertFalse(quick["model"]["interface_checked"])
                self.assertEqual(quick["video_group"]["videos"][0]["frame_count_source"], "container")
                self.assertEqual(quick["video_group"]["videos"][0]["cache_status"], "not_checked")
                dry_run.assert_not_called()
                cache_key.assert_not_called()

                deep = preflight_run(db, workspace, request)
                self.assertTrue(deep["ok"], deep)
                self.assertEqual(deep["preflight_level"], "deep")
                self.assertTrue(deep["model"]["interface_ok"])
                self.assertTrue(deep["model"]["interface_checked"])
                self.assertTrue(
                    any(
                        warning.get("type") == "FrameCountNotDecoded"
                        for warning in deep["warnings"]
                    )
                )
                dry_run.assert_called_once()
                cache_key.assert_called_once()

            self.assertEqual(exact_modes, [False, False])

    def test_deep_uses_completed_decode_manifest_as_exact_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            workspace = WorkspaceConfig.from_root(project / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            (project / "models").mkdir()
            (project / "models" / "model.py").write_text("# fixture\n", encoding="utf-8")
            video_dir = project / "videos" / "set-a"
            video_dir.mkdir(parents=True)
            (video_dir / "clip.mp4").write_bytes(b"video-fixture")
            exact_modes: list[bool] = []

            def inspect_video(path, _workspace=None, exact=True):
                exact_modes.append(bool(exact))
                return {
                    "name": Path(path).name,
                    "path": str(path),
                    "decodable": True,
                    "error": None,
                    "frame_count": 12,
                    "frame_count_source": "container",
                    "duration_seconds": 0.5,
                    "fps": 24.0,
                    "width": 640,
                    "height": 360,
                    "metadata_source": "opencv",
                }

            manifest = {
                "cache_key": "cache-key",
                "decode_status": "completed",
                "frame_count": 9,
                "duration_seconds": 0.375,
                "fps": 24.0,
                "width": 640,
                "height": 360,
            }
            request = {
                "model_file": "model.py",
                "video_group": "set-a",
                "selected_videos": ["clip.mp4"],
                "device": "cpu",
                "precision": "fp32",
            }
            with (
                patch("vfieval.file_inputs.inspect_video", side_effect=inspect_video),
                patch(
                    "vfieval.file_inputs._cached_dry_run_model_file",
                    return_value={"output_health": {"warnings": [], "stats": {}}},
                ),
                patch("vfieval.file_inputs.decode_cache_key", return_value="cache-key"),
                patch("vfieval.file_inputs.decode_cache_status", return_value="cached"),
                patch(
                    "vfieval.file_inputs._read_completed_decode_manifest",
                    return_value=manifest,
                ),
                patch("vfieval.file_inputs.metrics_health", return_value={"metrics": {}}),
            ):
                deep = preflight_run(db, workspace, request)

            self.assertTrue(deep["ok"], deep)
            video = deep["video_group"]["videos"][0]
            self.assertEqual(video["frame_count"], 9)
            self.assertEqual(video["frame_count_source"], "manifest_exact")
            self.assertFalse(
                any(warning.get("type") == "FrameCountNotDecoded" for warning in deep["warnings"])
            )
            self.assertEqual(exact_modes, [False])


if __name__ == "__main__":
    unittest.main()
