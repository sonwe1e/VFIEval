from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from vfieval.pipeline.inference import (
    _write_mp4,
    _write_mp4_cv2,
    _write_mp4_ffmpeg_pipe,
    _write_video_artifacts,
)


class VideoArtifactEncodingTests(unittest.TestCase):
    def test_ffmpeg_pipe_explicitly_requests_browser_compatible_h264(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.png"
            frame.write_bytes(b"png")
            output = root / "video.mp4"
            process = Mock()
            process.stdin = io.BytesIO()
            process.stderr = io.BytesIO()
            process.returncode = 0
            process.communicate.return_value = (None, b"")
            commands: list[list[str]] = []

            def fake_popen(command: list[str], **_kwargs):
                commands.append(command)
                output.write_bytes(b"mp4")
                return process

            with patch("vfieval.pipeline.inference.shutil.which", return_value="ffmpeg"), patch(
                "vfieval.pipeline.inference.subprocess.Popen", side_effect=fake_popen
            ):
                self.assertTrue(_write_mp4_ffmpeg_pipe([frame], output, 24.0))

            command = commands[0]
            self.assertEqual(command[command.index("-c:v") + 1], "libx264")
            self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
            self.assertEqual(command[command.index("-movflags") + 1], "+faststart")
            self.assertFalse(any("pad=" in value for value in command))
            self.assertNotIn("-vf", command)

    def test_ffmpeg_failure_is_actionable_and_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.png"
            frame.write_bytes(b"png")
            output = root / "video.mp4"
            process = Mock()
            process.stdin = io.BytesIO()
            process.stderr = io.BytesIO()
            process.returncode = 1
            process.communicate.return_value = (None, b"Unknown encoder 'libx264'")

            def fake_popen(_command: list[str], **_kwargs):
                output.write_bytes(b"partial")
                return process

            with patch("vfieval.pipeline.inference.shutil.which", return_value="ffmpeg"), patch(
                "vfieval.pipeline.inference.subprocess.Popen", side_effect=fake_popen
            ):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg/libx264.*Unknown encoder"):
                    _write_mp4_ffmpeg_pipe([frame], output, 24.0)

            self.assertFalse(output.exists())

    def test_opencv_does_not_fall_back_to_browser_incompatible_mp4v(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.png"
            output = root / "video.mp4"
            codecs: list[str] = []

            def fourcc(*name: str) -> str:
                codec = "".join(name)
                codecs.append(codec)
                return codec

            writer = SimpleNamespace(isOpened=lambda: False)
            fake_cv2 = SimpleNamespace(
                IMREAD_COLOR=1,
                imread=lambda *_args, **_kwargs: np.zeros((4, 4, 3), dtype=np.uint8),
                VideoWriter_fourcc=fourcc,
                VideoWriter=lambda *_args, **_kwargs: writer,
            )
            with patch.dict(sys.modules, {"cv2": fake_cv2}):
                with self.assertRaisesRegex(RuntimeError, "browser-compatible H.264"):
                    _write_mp4_cv2([frame], output, 24.0)

            self.assertEqual(codecs, ["avc1"])

    def test_empty_video_artifact_has_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "without frames"):
                _write_mp4([], Path(tmp) / "video.mp4", 24.0)

    def test_canonical_video_rejects_odd_dimensions_instead_of_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.png"
            Image.new("RGB", (7, 8)).save(frame)
            with patch("vfieval.pipeline.inference._write_mp4_ffmpeg_pipe") as ffmpeg:
                with self.assertRaisesRegex(ValueError, "will not be padded"):
                    _write_mp4([frame], root / "video.mp4", 24.0)
            ffmpeg.assert_not_called()

    def test_canonical_video_rejects_mixed_frame_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.png"
            second = root / "second.png"
            Image.new("RGB", (8, 8)).save(first)
            Image.new("RGB", (10, 8)).save(second)
            with self.assertRaisesRegex(ValueError, "identical dimensions"):
                _write_mp4([first, second], root / "video.mp4", 24.0)

    def test_colliding_sanitized_video_names_get_distinct_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for index in range(3):
                path = root / f"frame-{index}.png"
                Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
                frames.append(path)
            groups = {
                "identity-a": {
                    "video_name": "clip/a",
                    "fps": 24.0,
                    "frames": [{"order": 0, "sample_id": 1, "pred_path": frames[0], "gt_path": None, "diff_path": None}],
                },
                "identity-b": {
                    "video_name": "clip:a",
                    "fps": 24.0,
                    "frames": [{"order": 0, "sample_id": 2, "pred_path": frames[1], "gt_path": None, "diff_path": None}],
                },
                "safe-identity": {
                    "video_name": "safe",
                    "fps": 24.0,
                    "frames": [{"order": 0, "sample_id": 3, "pred_path": frames[2], "gt_path": None, "diff_path": None}],
                },
            }
            db = Mock()

            def encode(_frames, output_path: Path, _fps: float, **_kwargs) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"video")

            with patch("vfieval.pipeline.inference._write_mp4", side_effect=encode):
                _write_video_artifacts(db, 1, root, groups)

            pred_paths = [Path(call.args[3]) for call in db.add_artifact.call_args_list]
            self.assertEqual(len(pred_paths), 3)
            self.assertEqual(len({path.resolve() for path in pred_paths}), 3)
            colliding_dirs = {
                path.parent.name
                for path in pred_paths
                if path.parent.name.startswith("clip_a")
            }
            self.assertEqual(len(colliding_dirs), 2)
            self.assertIn(root / "videos" / "safe" / "pred.mp4", pred_paths)


if __name__ == "__main__":
    unittest.main()
