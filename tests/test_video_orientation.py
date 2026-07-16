from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval import file_inputs
from vfieval.config import WorkspaceConfig


def _ffprobe_result(stream: dict[str, object]) -> subprocess.CompletedProcess[str]:
    payload = {
        "streams": [
            {
                "width": 1206,
                "height": 2622,
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "24/1",
                "nb_frames": "3",
                "duration": "0.125",
                **stream,
            }
        ],
        "format": {"duration": "0.125"},
    }
    return subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout=json.dumps(payload),
        stderr="",
    )


class VideoOrientationTests(unittest.TestCase):
    def test_ffprobe_rotation_normalizes_display_dimensions(self) -> None:
        cases = [
            ({"tags": {"rotate": "90"}}, 90.0, (2622, 1206)),
            (
                {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]},
                -90.0,
                (2622, 1206),
            ),
            (
                {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 180}]},
                180.0,
                (1206, 2622),
            ),
            ({}, 0.0, (1206, 2622)),
        ]

        with patch("vfieval.file_inputs.shutil.which", return_value="ffprobe"):
            for stream, expected_rotation, expected_size in cases:
                with self.subTest(stream=stream), patch(
                    "vfieval.file_inputs.subprocess.run",
                    return_value=_ffprobe_result(stream),
                ) as run:
                    info = file_inputs._inspect_video_ffprobe(Path("clip.mp4"))

                    self.assertIsNotNone(info)
                    assert info is not None
                    self.assertEqual((info["coded_width"], info["coded_height"]), (1206, 2622))
                    self.assertEqual(info["rotation_degrees"], expected_rotation)
                    self.assertEqual((info["width"], info["height"]), expected_size)
                    show_entries = run.call_args.args[0][run.call_args.args[0].index("-show_entries") + 1]
                    self.assertIn("stream_tags=rotate", show_entries)
                    self.assertIn("stream_side_data=side_data_type,displaymatrix,rotation", show_entries)

    def test_display_matrix_rotation_precedes_legacy_rotate_tag(self) -> None:
        stream = {
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
            "tags": {"rotate": "180"},
        }
        with (
            patch("vfieval.file_inputs.shutil.which", return_value="ffprobe"),
            patch("vfieval.file_inputs.subprocess.run", return_value=_ffprobe_result(stream)),
        ):
            info = file_inputs._inspect_video_ffprobe(Path("clip.mp4"))

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info["rotation_degrees"], -90.0)
        self.assertEqual((info["width"], info["height"]), (2622, 1206))

    def test_video_summary_and_original_resolution_use_display_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "rotated.mp4"
            video.write_bytes(b"video")
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            opencv = {
                "frame_count": 3,
                "container_frame_count": 3,
                "frame_count_source": "container",
                "frame_count_warning": None,
                "duration_seconds": 0.125,
                "fps": 24.0,
                "width": 2622,
                "height": 1206,
                "metadata_source": "opencv",
            }
            with (
                patch("vfieval.file_inputs.shutil.which", return_value="ffprobe"),
                patch(
                    "vfieval.file_inputs.subprocess.run",
                    return_value=_ffprobe_result(
                        {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}
                    ),
                ),
                patch("vfieval.file_inputs._inspect_video_opencv", return_value=opencv),
                patch("vfieval.file_inputs.ensure_video_thumbnail", return_value=None),
            ):
                summary = file_inputs.video_summary(workspace, video, exact=False)

            self.assertEqual((summary["width"], summary["height"]), (2622, 1206))
            self.assertEqual(
                file_inputs.resolve_run_dimensions(
                    {"resolution_mode": "original"},
                    [summary],
                ),
                (1206, 2622),
            )

    def test_item_compare_plan_uses_first_decoded_frame_dimensions(self) -> None:
        class _Db:
            @staticmethod
            def next_run_id() -> int:
                return 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            gt = root / "gt.png"
            pred = root / "pred.png"
            Image.new("RGB", (12, 6)).save(gt)
            Image.new("RGB", (8, 4)).save(pred)
            reference = {"path": str(gt), "item_id": 1, "width": 6, "height": 12}
            prediction = {
                "path": str(pred),
                "item_id": 1,
                "member_role": "model_pred",
                "width": 4,
                "height": 8,
            }
            decoded = [
                ([gt, gt, gt], 10.0, [0.0, 0.1, 0.2]),
                ([pred, pred, pred], 10.0, [0.0, 0.1, 0.2]),
            ]
            with (
                patch("vfieval.file_inputs.metrics_health", return_value={"metrics": {}}),
                patch(
                    "vfieval.compare_inputs.resolve_compare_descriptor",
                    side_effect=[reference, prediction],
                ),
                patch("vfieval.datasets._load_compare_source_frames", side_effect=decoded),
            ):
                result = file_inputs._preflight_media_item_compare(
                    _Db(),  # type: ignore[arg-type]
                    workspace,
                    {
                        "reference": {"kind": "member", "id": 1},
                        "distorted": {"kind": "member", "id": 2},
                        "media_item_id": 1,
                        "metrics": [],
                    },
                )

            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual((result["reference"]["width"], result["reference"]["height"]), (12, 6))
            self.assertEqual((result["distorted"]["width"], result["distorted"]["height"]), (8, 4))
            self.assertEqual(
                result["alignment_plan"]["sources"]["gt"]["original"],
                {"width": 12, "height": 6},
            )

    def test_video_inspect_cache_version_invalidates_v3_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            current = file_inputs._video_inspect_cache_path(video, workspace, exact=True)
            with patch.object(file_inputs, "VIDEO_INSPECT_VERSION", "ffprobe-opencv-v3"):
                previous = file_inputs._video_inspect_cache_path(video, workspace, exact=True)

        self.assertEqual(file_inputs.VIDEO_INSPECT_VERSION, "ffprobe-opencv-v4")
        self.assertNotEqual(current, previous)


if __name__ == "__main__":
    unittest.main()
