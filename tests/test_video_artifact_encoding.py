from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vfieval.pipeline.inference import _write_mp4, _write_mp4_cv2, _write_mp4_ffmpeg_pipe


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
            self.assertIn("pad=ceil(iw/2)*2:ceil(ih/2)*2", command)

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


if __name__ == "__main__":
    unittest.main()
