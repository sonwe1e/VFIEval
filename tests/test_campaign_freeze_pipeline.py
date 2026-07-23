from __future__ import annotations

import errno
import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vfieval.alignment import materialize_aligned_rgb, plan_alignment
from vfieval.pipeline import evaluation_freeze


def _plan(frame_count: int = 2, *, gt_size: tuple[int, int] = (8, 6)) -> dict:
    reference = {
        "slot": "gt",
        "width": gt_size[0],
        "height": gt_size[1],
        "frame_count": frame_count,
        "fps": 24.0,
    }
    predictions = [
        {
            "slot": slot,
            "width": 8,
            "height": 6,
            "frame_count": frame_count,
            "fps": 24.0,
        }
        for slot in ("pred_a", "pred_b")
    ]
    return plan_alignment(reference, predictions)


def _frames(root: Path, slot: str, colors: list[tuple[int, int, int]], size=(8, 6)) -> list[Path]:
    directory = root / slot
    directory.mkdir()
    outputs = []
    for index, color in enumerate(colors):
        path = directory / f"{index:06d}.png"
        Image.new("RGB", size, color).save(path)
        outputs.append(path)
    return outputs


class CampaignFreezePipelineTests(unittest.TestCase):
    def test_encoder_broken_pipe_reports_sink_exit_code_and_stderr_to_siblings(self) -> None:
        class BrokenStdin:
            def write(self, _frame: bytes) -> None:
                raise OSError(errno.EPIPE, "Broken pipe")

            def close(self) -> None:
                return None

        class FailedEncoder:
            def __init__(self, stderr) -> None:
                self.stdin = BrokenStdin()
                self.returncode = None
                self.wait_calls = 0
                stderr.write(b"Error initializing output stream: insufficient resources\n")

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.wait_calls += 1
                self.returncode = 23
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 23

            def kill(self) -> None:
                self.returncode = 23

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure_state = evaluation_freeze._SinkFailureState()
            failed = evaluation_freeze._RawVideoSink(
                root / "method-a.mp4",
                width=8,
                height=6,
                fps=24.0,
                threads=1,
                ffmpeg=sys.executable,
                failure_state=failure_state,
            )
            sibling = evaluation_freeze._RawVideoSink(
                root / "method-b.mp4",
                width=8,
                height=6,
                fps=24.0,
                threads=1,
                ffmpeg=sys.executable,
                failure_state=failure_state,
            )
            process = FailedEncoder(failed._stderr)
            failed._process = process
            failed._queue.put(bytes(8 * 6 * 3))
            try:
                failed._writer()
                with self.assertRaises(RuntimeError) as caught:
                    sibling._raise_error()
            finally:
                failed.abort()
                sibling.abort()

        message = str(caught.exception)
        self.assertIn("Campaign encoder method-a.mp4 failed", message)
        self.assertIn("exit code 23", message)
        self.assertIn("insufficient resources", message)
        self.assertNotIn("[Errno 32]", message)
        self.assertNotIn("Broken pipe", message)
        self.assertGreaterEqual(process.wait_calls, 1)

    def test_v3_encoder_uses_fixed_one_second_closed_gop(self) -> None:
        policy = evaluation_freeze._video_stability_policy(23.976)
        encoding = policy["encoding"]
        self.assertEqual(encoding["gop_frames"], 24)
        self.assertEqual(encoding["keyint_min_frames"], 24)
        self.assertEqual(encoding["scene_cut_threshold"], 0)
        self.assertFalse(encoding["open_gop"])
        self.assertEqual(encoding["b_frames"], 0)
        self.assertEqual(len(policy["fingerprint"]), 64)
        self.assertEqual(
            policy["fingerprint"],
            evaluation_freeze._video_stability_policy(23.976)["fingerprint"],
        )

        with tempfile.TemporaryDirectory() as temporary:
            sink = evaluation_freeze._RawVideoSink(
                Path(temporary) / "frozen.mp4",
                width=8,
                height=6,
                fps=23.976,
                threads=1,
                ffmpeg=sys.executable,
                gop_policy=encoding,
            )
            try:
                command = sink.command()
            finally:
                sink.abort()
        self.assertEqual(command[command.index("-crf") + 1], "18")
        self.assertEqual(command[command.index("-g") + 1], "24")
        self.assertEqual(command[command.index("-keyint_min") + 1], "24")
        self.assertEqual(command[command.index("-sc_threshold") + 1], "0")
        self.assertEqual(command[command.index("-bf") + 1], "0")
        self.assertIn("keyint=24:min-keyint=24:scenecut=0:open-gop=0:bframes=0", command)
        self.assertEqual(command[command.index("-pix_fmt") + 1], "rgb24")
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)

    def test_ffprobe_extracts_keyframe_times_and_bounded_summary(self) -> None:
        metadata = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 8,
                    "height": 6,
                    "avg_frame_rate": "1/1",
                    "r_frame_rate": "1/1",
                    "nb_frames": "3",
                }
            ],
            "format": {"duration": "3.0"},
        }
        frames = {
            "frames": [
                {
                    "best_effort_timestamp_time": str(index),
                    "pkt_duration_time": "1.0",
                    "key_frame": 1 if index in {0, 2} else 0,
                    "pict_type": "I" if index in {0, 2} else "P",
                }
                for index in range(3)
            ]
        }
        with patch.object(
            evaluation_freeze,
            "_run_json",
            side_effect=[metadata, frames],
        ):
            probe = evaluation_freeze.probe_video_for_freeze(
                Path("fixture.mp4"),
                ffprobe=sys.executable,
                include_timestamps=True,
            )
        self.assertEqual(probe["keyframe_timestamps"], [0.0, 2.0])
        summary = evaluation_freeze._keyframe_probe_summary(probe)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["first_frame_seconds"], 0.0)
        self.assertEqual(summary["first_keyframe_seconds"], 0.0)
        self.assertEqual(summary["first_keyframe_offset_seconds"], 0.0)
        self.assertEqual(summary["first_seconds"], 0.0)
        self.assertEqual(summary["max_interval_seconds"], 2.0)
        self.assertEqual(len(summary["timestamps_sha256"]), 64)
        incomplete = evaluation_freeze._keyframe_probe_summary(
            {**probe, "timestamps": probe["timestamps"][:-1]}
        )
        self.assertFalse(incomplete["complete"])

    def test_keyframe_only_probe_counts_packets_and_skips_non_keyframes(self) -> None:
        metadata = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 8,
                    "height": 6,
                    "avg_frame_rate": "1/1",
                    "r_frame_rate": "1/1",
                    "nb_read_packets": "3",
                    "start_time": "10.0",
                    "duration": "3.0",
                }
            ],
            "format": {"start_time": "10.0", "duration": "3.0"},
        }
        keyframes = {
            "frames": [
                {"best_effort_timestamp_time": "10.0", "key_frame": 1},
                {"best_effort_timestamp_time": "12.0", "key_frame": 1},
            ]
        }
        with patch.object(
            evaluation_freeze,
            "_run_json",
            side_effect=[metadata, keyframes],
        ) as run_json:
            probe = evaluation_freeze.probe_video_for_freeze(
                Path("fixture.mp4"),
                ffprobe=sys.executable,
                include_timestamps=False,
                include_keyframes=True,
            )
        metadata_command = run_json.call_args_list[0].args[0]
        keyframe_command = run_json.call_args_list[1].args[0]
        self.assertIn("-count_packets", metadata_command)
        self.assertEqual(
            keyframe_command[keyframe_command.index("-skip_frame") + 1],
            "nokey",
        )
        self.assertEqual(probe["packet_count"], 3)
        self.assertEqual(probe["timestamps"], [])
        self.assertEqual(probe["first_frame_timestamp"], 10.0)
        self.assertEqual(probe["keyframe_timestamps"], [10.0, 12.0])
        self.assertEqual(probe["keyframe_probe_mode"], "keyframes_only")
        summary = evaluation_freeze._keyframe_probe_summary(probe)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["first_keyframe_offset_seconds"], 0.0)
        self.assertEqual(summary["max_interval_seconds"], 2.0)

    @unittest.skipUnless(
        shutil.which("ffprobe") and evaluation_freeze.streaming_backend_available(),
        "FFmpeg/libx264 and ffprobe are required",
    )
    def test_actual_fixed_gop_handles_single_frame_and_first_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = evaluation_freeze._video_stability_policy(24.0)
            for frame_count, expected_keyframes in ((1, 1), (25, 2)):
                sink = evaluation_freeze._RawVideoSink(
                    root / f"fixed-gop-{frame_count}.mp4",
                    width=8,
                    height=6,
                    fps=24.0,
                    threads=1,
                    ffmpeg=shutil.which("ffmpeg") or "ffmpeg",
                    gop_policy=policy["encoding"],
                )
                try:
                    sink.start()
                    for _index in range(frame_count):
                        sink.write(bytes(8 * 6 * 3))
                    sink.close_input()
                    sink.wait()
                except Exception:
                    sink.abort()
                    raise
                probe = evaluation_freeze.validate_frozen_video(
                    sink.path,
                    width=8,
                    height=6,
                    frame_count=frame_count,
                    fps=24.0,
                    stability_policy=policy,
                )
                keyframes = probe["keyframe_probe"]
                self.assertTrue(keyframes["complete"])
                self.assertEqual(keyframes["probe_mode"], "keyframes_only")
                self.assertEqual(keyframes["count"], expected_keyframes)
                self.assertAlmostEqual(keyframes["first_seconds"], 0.0, places=6)
                self.assertLessEqual(keyframes["max_interval_seconds"], 1.001)

    def test_streaming_backend_capability_probe_is_cached_per_executable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "ffmpeg.exe"
            executable.write_bytes(b"fake ffmpeg")
            evaluation_freeze._clear_streaming_backend_cache()
            with patch.object(
                evaluation_freeze, "_resolve_executable", return_value=str(executable)
            ), patch.object(
                evaluation_freeze, "_probe_streaming_backend", return_value=True
            ) as probe:
                self.assertTrue(evaluation_freeze.streaming_backend_available())
                self.assertTrue(evaluation_freeze.streaming_backend_available())
            self.assertEqual(probe.call_count, 1)

    def test_in_memory_alignment_uses_strict_dimensions_and_lanczos(self) -> None:
        plan = _plan(frame_count=1, gt_size=(16, 12))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _frames(root, "gt", [(10, 20, 30)], size=(16, 12))[0]
            with Image.new("RGB", (16, 12)) as gradient:
                gradient.putdata(
                    [
                        ((x * 17 + y * 3) % 256, (x * 5 + y * 19) % 256, (x * y * 7) % 256)
                        for y in range(12)
                        for x in range(16)
                    ]
                )
                gradient.save(source)
            image = materialize_aligned_rgb(plan, "gt", source)
            try:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (8, 6))
                with Image.open(source).convert("RGB") as original:
                    expected = original.resize((8, 6), Image.Resampling.LANCZOS)
                    try:
                        self.assertEqual(image.tobytes(), expected.tobytes())
                    finally:
                        expected.close()
            finally:
                image.close()
            changed = root / "changed.png"
            Image.new("RGB", (12, 16), "black").save(changed)
            with self.assertRaisesRegex(ValueError, "dimensions changed"):
                materialize_aligned_rgb(plan, "gt", changed)

    def test_frame_sequence_reads_each_source_once_and_hashes_each_output_once(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "gt": _frames(root, "gt", [(10, 20, 30), (20, 30, 40)]),
                "pred_a": _frames(root, "pred_a", [(9, 18, 27), (22, 33, 44)]),
                "pred_b": _frames(root, "pred_b", [(30, 20, 10), (40, 30, 20)]),
            }
            progress = []
            original_materialize = evaluation_freeze.materialize_aligned_rgb
            original_digest = evaluation_freeze._digest_path
            with patch.object(
                evaluation_freeze,
                "materialize_aligned_rgb",
                wraps=original_materialize,
            ) as materialize, patch.object(
                evaluation_freeze,
                "_digest_path",
                wraps=original_digest,
            ) as digest:
                result = evaluation_freeze.freeze_campaign_media(
                    plan,
                    sources,
                    root / "package",
                    media_kind="frame_sequence",
                    fps=24.0,
                    progress_callback=progress.append,
                )
            self.assertEqual(materialize.call_count, 6)
            self.assertEqual(digest.call_count, 3)
            self.assertEqual(set(result["artifacts"]), set(evaluation_freeze.OUTPUT_SLOTS))
            self.assertTrue(all(item["mode"] == "png_sequence" for item in result["artifacts"].values()))
            self.assertEqual(progress[-1]["stage"], "completed")
            self.assertTrue(progress[-1]["force"])
            self.assertFalse(any(path.name.startswith(".") for path in (root / "package").iterdir()))
            self.assertEqual(result["version"], "campaign-freeze-stream-v3")
            self.assertFalse(result["gop_policy"]["applicable"])
            self.assertEqual(len(result["stability_policy_fingerprint"]), 64)

    def test_frame_sequence_source_change_fails_and_cleans_outputs(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "gt": _frames(root, "gt", [(10, 20, 30), (20, 30, 40)]),
                "pred_a": _frames(root, "pred_a", [(9, 18, 27), (22, 33, 44)]),
                "pred_b": _frames(root, "pred_b", [(30, 20, 10), (40, 30, 20)]),
            }
            changed = False

            def mutate_after_first_frame(event):
                nonlocal changed
                if changed or int(event.get("frame_current") or 0) != 1:
                    return
                changed = True
                Image.new("RGB", (8, 6), (200, 10, 20)).save(sources["gt"][0])

            output = root / "package"
            with self.assertRaises(evaluation_freeze.SourceChanged):
                evaluation_freeze.freeze_campaign_media(
                    plan,
                    sources,
                    output,
                    media_kind="frame_sequence",
                    fps=24.0,
                    progress_callback=mutate_after_first_frame,
                )
            self.assertTrue(changed)
            self.assertFalse(any(output.iterdir()))

    def test_hash_failure_cleans_completed_outputs(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "gt": _frames(root, "gt", [(10, 20, 30), (20, 30, 40)]),
                "pred_a": _frames(root, "pred_a", [(9, 18, 27), (22, 33, 44)]),
                "pred_b": _frames(root, "pred_b", [(30, 20, 10), (40, 30, 20)]),
            }
            output = root / "package"
            with patch.object(
                evaluation_freeze,
                "_digest_path",
                side_effect=RuntimeError("injected final hash failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "final hash failure"):
                    evaluation_freeze.freeze_campaign_media(
                        plan,
                        sources,
                        output,
                        media_kind="frame_sequence",
                        fps=24.0,
                    )
            self.assertFalse(any(output.iterdir()))

    def test_video_streams_three_outputs_with_a_single_source_read_per_frame(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "gt": _frames(root, "gt", [(10, 20, 30), (20, 30, 40)]),
                "pred_a": _frames(root, "pred_a", [(9, 18, 27), (22, 33, 44)]),
                "pred_b": _frames(root, "pred_b", [(30, 20, 10), (40, 30, 20)]),
            }

            class FakeSink:
                instances = []

                def __init__(self, path, **kwargs):
                    self.path = Path(path)
                    self.frames = []
                    self.kwargs = kwargs
                    self.__class__.instances.append(self)

                def start(self):
                    return None

                def write(self, frame):
                    self.frames.append(frame)

                def close_input(self):
                    return None

                def wait(self):
                    self.path.write_bytes(b"encoded")

                def abort(self):
                    self.path.unlink(missing_ok=True)

            ineligible = {
                slot: {"eligible": False, "reasons": ["test"], "probe": None}
                for slot in evaluation_freeze.SOURCE_SLOTS
            }
            original_materialize = evaluation_freeze.materialize_aligned_rgb
            with patch.object(evaluation_freeze, "streaming_backend_available", return_value=True), patch.object(
                evaluation_freeze,
                "_collect_remux_eligibility",
                return_value=ineligible,
            ), patch.object(evaluation_freeze, "_RawVideoSink", FakeSink), patch.object(
                evaluation_freeze,
                "validate_frozen_video",
                return_value={},
            ), patch.object(
                evaluation_freeze,
                "materialize_aligned_rgb",
                wraps=original_materialize,
            ) as materialize:
                result = evaluation_freeze.freeze_campaign_media(
                    plan,
                    sources,
                    root / "package",
                    media_kind="video",
                    fps=24.0,
                    ffmpeg=sys.executable,
                    ffprobe=sys.executable,
                )
            self.assertEqual(materialize.call_count, 6)
            self.assertEqual(len(FakeSink.instances), 3)
            self.assertTrue(all(len(sink.frames) == 2 for sink in FakeSink.instances))
            self.assertTrue(all(sink.kwargs["threads"] == result["encoder_threads"] for sink in FakeSink.instances))
            self.assertEqual(result["pipeline"], "streaming")
            self.assertTrue(all(item["mode"] == "stream" for item in result["artifacts"].values()))
            self.assertEqual(result["version"], "campaign-freeze-stream-v3")
            self.assertEqual(result["gop_policy"]["gop_frames"], 24)
            self.assertEqual(set(result["keyframe_probe"]), set(evaluation_freeze.SOURCE_SLOTS))

    def test_encoder_start_failure_after_first_sink_does_not_fallback(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "gt": _frames(root, "gt", [(10, 20, 30), (20, 30, 40)]),
                "pred_a": _frames(root, "pred_a", [(9, 18, 27), (22, 33, 44)]),
                "pred_b": _frames(root, "pred_b", [(30, 20, 10), (40, 30, 20)]),
            }

            class FailSecondSink:
                starts = 0

                def __init__(self, path, **_kwargs):
                    self.path = Path(path)

                def start(self):
                    self.__class__.starts += 1
                    if self.__class__.starts == 2:
                        raise OSError("injected second encoder start failure")

                def abort(self):
                    self.path.unlink(missing_ok=True)

            ineligible = {
                slot: {"eligible": False, "reasons": ["test"], "probe": None}
                for slot in evaluation_freeze.SOURCE_SLOTS
            }
            output = root / "package"
            with patch.object(
                evaluation_freeze,
                "streaming_backend_available",
                return_value=True,
            ), patch.object(
                evaluation_freeze,
                "_collect_remux_eligibility",
                return_value=ineligible,
            ), patch.object(evaluation_freeze, "_RawVideoSink", FailSecondSink):
                with self.assertRaises(evaluation_freeze.FreezeError) as caught:
                    evaluation_freeze.freeze_campaign_media(
                        plan,
                        sources,
                        output,
                        media_kind="video",
                        fps=24.0,
                        ffmpeg=sys.executable,
                        ffprobe=sys.executable,
                    )
            self.assertNotIsInstance(caught.exception, evaluation_freeze.FreezeBackendUnavailable)
            self.assertIn("after Campaign streaming had started", str(caught.exception))
            self.assertFalse(any(output.iterdir()))

    def test_ffprobe_validation_failure_cleans_stream_outputs(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "gt": _frames(root, "gt", [(10, 20, 30), (20, 30, 40)]),
                "pred_a": _frames(root, "pred_a", [(9, 18, 27), (22, 33, 44)]),
                "pred_b": _frames(root, "pred_b", [(30, 20, 10), (40, 30, 20)]),
            }

            class FinishedSink:
                def __init__(self, path, **_kwargs):
                    self.path = Path(path)

                def start(self):
                    return None

                def write(self, _frame):
                    return None

                def close_input(self):
                    return None

                def wait(self):
                    self.path.write_bytes(b"encoded")

                def abort(self):
                    self.path.unlink(missing_ok=True)

            ineligible = {
                slot: {"eligible": False, "reasons": ["test"], "probe": None}
                for slot in evaluation_freeze.SOURCE_SLOTS
            }
            output = root / "package"
            with patch.object(
                evaluation_freeze,
                "streaming_backend_available",
                return_value=True,
            ), patch.object(
                evaluation_freeze,
                "_collect_remux_eligibility",
                return_value=ineligible,
            ), patch.object(
                evaluation_freeze,
                "_RawVideoSink",
                FinishedSink,
            ), patch.object(
                evaluation_freeze,
                "validate_frozen_video",
                side_effect=RuntimeError("injected ffprobe validation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "ffprobe validation failure"):
                    evaluation_freeze.freeze_campaign_media(
                        plan,
                        sources,
                        output,
                        media_kind="video",
                        fps=24.0,
                        ffmpeg=sys.executable,
                        ffprobe=sys.executable,
                    )
            self.assertFalse(any(output.iterdir()))

    @unittest.skipUnless(
        shutil.which("ffprobe") and evaluation_freeze.streaming_backend_available(),
        "FFmpeg/libx264 and ffprobe are required",
    )
    def test_actual_rawvideo_outputs_are_validated_mp4s(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "gt": _frames(root, "gt", [(10, 20, 30), (20, 30, 40)]),
                "pred_a": _frames(root, "pred_a", [(9, 18, 27), (22, 33, 44)]),
                "pred_b": _frames(root, "pred_b", [(30, 20, 10), (40, 30, 20)]),
            }
            result = evaluation_freeze.freeze_campaign_media(
                plan,
                sources,
                root / "package",
                media_kind="video",
                fps=24.0,
            )
            self.assertEqual(result["pipeline"], "streaming")
            for artifact in result["artifacts"].values():
                probe = evaluation_freeze.validate_frozen_video(
                    artifact["path"],
                    width=8,
                    height=6,
                    frame_count=2,
                    fps=24.0,
                    stability_policy=result["stability_policy"],
                )
                self.assertEqual(probe["codec"], "h264")
                self.assertTrue(probe["keyframe_probe"]["policy_compliant"])

            ffmpeg = shutil.which("ffmpeg")
            assert ffmpeg is not None
            decoded = {}
            for slot in evaluation_freeze.SOURCE_SLOTS:
                frame = root / f"decoded-{slot}.png"
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(result["artifacts"][slot]["path"]),
                        "-frames:v",
                        "1",
                        str(frame),
                    ],
                    check=True,
                )
                with Image.open(frame) as image:
                    decoded[slot] = image.convert("RGB")
                    decoded[slot].load()
            self.assertEqual(set(decoded), set(evaluation_freeze.SOURCE_SLOTS))
            for image in decoded.values():
                image.close()

    @unittest.skipUnless(
        shutil.which("ffprobe") and evaluation_freeze.streaming_backend_available(),
        "FFmpeg/libx264 and ffprobe are required",
    )
    def test_actual_remux_uses_gt_independently_and_predictions_as_a_pair(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "gt": _frames(root, "gt", [(10, 20, 30), (20, 30, 40)]),
                "pred_a": _frames(root, "pred_a", [(9, 18, 27), (22, 33, 44)]),
                "pred_b": _frames(root, "pred_b", [(30, 20, 10), (40, 30, 20)]),
            }
            first = evaluation_freeze.freeze_campaign_media(
                plan,
                sources,
                root / "streamed",
                media_kind="video",
                fps=24.0,
            )
            source_media = {
                slot: first["artifacts"][slot]["path"] for slot in evaluation_freeze.SOURCE_SLOTS
            }
            probes = {
                slot: evaluation_freeze.probe_video_for_freeze(path)
                for slot, path in source_media.items()
            }
            expected_sha256 = {
                slot: evaluation_freeze._source_signature(path)["sha256"]
                for slot, path in source_media.items()
            }
            second = evaluation_freeze.freeze_campaign_media(
                plan,
                sources,
                root / "remuxed",
                media_kind="video",
                fps=24.0,
                source_media=source_media,
                source_timestamps={slot: probe["timestamps"] for slot, probe in probes.items()},
                expected_source_sha256=expected_sha256,
            )
            self.assertEqual(second["pipeline"], "remux")
            self.assertEqual(second["artifacts"]["gt"]["mode"], "remux")
            self.assertEqual(second["artifacts"]["pred_a"]["mode"], "remux")
            self.assertEqual(second["artifacts"]["pred_b"]["mode"], "remux")
            self.assertEqual(set(second["artifacts"]), set(evaluation_freeze.SOURCE_SLOTS))

    def test_remux_eligibility_rejects_rotation_and_non_identity(self) -> None:
        plan = _plan()
        timestamps = [0.0, 1.0 / 24.0]
        probe = {
            "codec": "h264",
            "pix_fmt": "yuv420p",
            "rotation_degrees": 90.0,
            "width": 8,
            "height": 6,
            "frame_count": 2,
            "fps": 24.0,
            "cfr": True,
            "timestamps": timestamps,
        }
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "source.mp4"
            video.write_bytes(b"video")
            with patch.object(evaluation_freeze, "probe_video_for_freeze", return_value=probe):
                result = evaluation_freeze.remux_eligibility(
                    plan,
                    "pred_a",
                    video,
                    timestamps=timestamps,
                    fps=24.0,
                    ffprobe=sys.executable,
                )
        self.assertFalse(result["eligible"])
        self.assertIn("rotation metadata is not zero", result["reasons"])

    def test_indexed_alignment_remuxes_complete_prediction_pair_but_not_gt(self) -> None:
        plan = _plan(frame_count=3)
        plan["temporal"].update(
            {
                "mode": "indexed",
                "reference_frame_count": 5,
                "frame_count": 3,
                "prediction_frame_counts": [3, 3],
                "mapping_count": 3,
                "mapping_first": 0,
                "mapping_last": 4,
                "fps": 5.0,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_media = {}
            for slot in evaluation_freeze.SOURCE_SLOTS:
                source_media[slot] = root / f"{slot}.mp4"
                source_media[slot].write_bytes(slot.encode("ascii"))
            source_timestamps = {
                "gt": [0.0, 0.2, 0.4, 0.6, 0.8],
                "pred_a": [10.0, 10.2, 10.4],
                "pred_b": [20.0, 20.2, 20.4],
            }

            def probe(path, **_kwargs):
                slot = Path(path).stem
                return {
                    "codec": "h264",
                    "pix_fmt": "yuv420p",
                    "rotation_degrees": 0.0,
                    "width": 8,
                    "height": 6,
                    "frame_count": 3,
                    "fps": 5.0,
                    "cfr": True,
                    "timestamps": source_timestamps[slot],
                    "frame_durations": [0.2, 0.2, 0.2],
                    "keyframe_timestamps": [source_timestamps[slot][0]],
                }

            with patch.object(
                evaluation_freeze, "probe_video_for_freeze", side_effect=probe
            ) as probe_call:
                result = evaluation_freeze._collect_remux_eligibility(
                    plan,
                    source_media,
                    source_timestamps,
                    expected_source_sha256={slot: slot * 64 for slot in source_media},
                    fps=5.0,
                    ffprobe=sys.executable,
                )

        self.assertFalse(result["gt"]["eligible"])
        self.assertIn(
            "GT requires indexed temporal materialization", result["gt"]["reasons"]
        )
        self.assertTrue(result["pred_a"]["eligible"])
        self.assertTrue(result["pred_b"]["eligible"])
        self.assertTrue(result["pred_a"]["keyframe_probe"]["policy_compliant"])
        self.assertEqual(
            result["pred_a"]["keyframe_probe"]["first_frame_seconds"],
            10.0,
        )
        self.assertEqual(
            result["pred_b"]["keyframe_probe"]["first_keyframe_offset_seconds"],
            0.0,
        )
        self.assertEqual(probe_call.call_count, 2)

    def test_sparse_or_late_prediction_keyframes_disable_the_pair(self) -> None:
        plan = _plan(frame_count=4)
        timestamps = [0.0, 1.0, 2.0, 3.0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_media = {}
            for slot in evaluation_freeze.SOURCE_SLOTS:
                source_media[slot] = root / f"{slot}.mp4"
                source_media[slot].write_bytes(slot.encode("ascii"))

            def probe(path, **_kwargs):
                slot = Path(path).stem
                keyframes = [0.0, 2.0] if slot == "pred_a" else [0.0]
                return {
                    "codec": "h264",
                    "pix_fmt": "yuv420p",
                    "rotation_degrees": 0.0,
                    "width": 8,
                    "height": 6,
                    "frame_count": 4,
                    "fps": 1.0,
                    "cfr": True,
                    "timestamps": timestamps,
                    "frame_durations": [1.0] * 4,
                    "keyframe_timestamps": keyframes,
                }

            with patch.object(
                evaluation_freeze,
                "probe_video_for_freeze",
                side_effect=probe,
            ):
                result = evaluation_freeze._collect_remux_eligibility(
                    plan,
                    source_media,
                    {slot: timestamps for slot in evaluation_freeze.SOURCE_SLOTS},
                    expected_source_sha256={slot: "a" * 64 for slot in source_media},
                    fps=1.0,
                    ffprobe=sys.executable,
                )

        self.assertFalse(result["pred_a"]["eligible"])
        self.assertFalse(result["pred_b"]["eligible"])
        self.assertIn(
            "maximum keyframe interval exceeds 2.000 seconds",
            result["pred_b"]["reasons"],
        )
        self.assertIn(
            "paired prediction is not remux-eligible",
            result["pred_a"]["reasons"],
        )
        late_reasons = evaluation_freeze._prediction_keyframe_reasons(
            {
                "complete": True,
                "first_frame_seconds": 0.0,
                "first_keyframe_seconds": 0.25,
                "first_keyframe_offset_seconds": 0.25,
                "first_seconds": 0.25,
                "max_interval_seconds": 1.0,
            },
            evaluation_freeze._video_stability_policy(24.0),
        )
        self.assertIn("first keyframe is not near the first frame", late_reasons)

    def test_long_gop_gt_is_reencoded_without_disabling_prediction_pair(self) -> None:
        plan = _plan(frame_count=4)
        source_timestamps = {
            "gt": [5.0, 6.0, 7.0, 8.0],
            "pred_a": [10.0, 11.0, 12.0, 13.0],
            "pred_b": [20.0, 21.0, 22.0, 23.0],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_media = {}
            for slot in evaluation_freeze.SOURCE_SLOTS:
                source_media[slot] = root / f"{slot}.mp4"
                source_media[slot].write_bytes(slot.encode("ascii"))

            def probe(path, **_kwargs):
                slot = Path(path).stem
                timestamps = source_timestamps[slot]
                return {
                    "codec": "h264",
                    "pix_fmt": "yuv420p",
                    "rotation_degrees": 0.0,
                    "width": 8,
                    "height": 6,
                    "frame_count": 4,
                    "fps": 1.0,
                    "cfr": True,
                    "timestamps": timestamps,
                    "frame_durations": [1.0] * 4,
                    "keyframe_timestamps": (
                        [timestamps[0]]
                        if slot == "gt"
                        else [timestamps[0], timestamps[2]]
                    ),
                }

            with patch.object(
                evaluation_freeze,
                "probe_video_for_freeze",
                side_effect=probe,
            ):
                result = evaluation_freeze._collect_remux_eligibility(
                    plan,
                    source_media,
                    source_timestamps,
                    expected_source_sha256={slot: "a" * 64 for slot in source_media},
                    fps=1.0,
                    ffprobe=sys.executable,
                )

        self.assertFalse(result["gt"]["eligible"])
        self.assertIn(
            "maximum keyframe interval exceeds 2.000 seconds",
            result["gt"]["reasons"],
        )
        self.assertTrue(result["gt"]["keyframe_probe"]["remux_gate_applied"])
        self.assertTrue(result["pred_a"]["eligible"])
        self.assertTrue(result["pred_b"]["eligible"])

    def test_missing_trusted_digest_disables_remux_without_probing(self) -> None:
        plan = _plan()
        with patch.object(evaluation_freeze, "remux_eligibility") as eligibility:
            result = evaluation_freeze._collect_remux_eligibility(
                plan,
                {slot: Path(f"{slot}.mp4") for slot in evaluation_freeze.SOURCE_SLOTS},
                {slot: [0.0, 1.0 / 24.0] for slot in evaluation_freeze.SOURCE_SLOTS},
                expected_source_sha256={},
                fps=24.0,
                ffprobe=sys.executable,
            )
        eligibility.assert_not_called()
        self.assertTrue(all(not item["eligible"] for item in result.values()))
        self.assertTrue(
            all("digest" in " ".join(item["reasons"]) for item in result.values())
        )

    def test_output_validation_checks_packet_and_declared_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "frozen.mp4"
            video.write_bytes(b"private-package")
            probe = {
                "codec": "h264",
                "pix_fmt": "yuv420p",
                "rotation_degrees": 0.0,
                "width": 8,
                "height": 6,
                "frame_count": 2,
                "packet_count": 2,
                "declared_frame_count": 3,
                "fps": 24.0,
                "audio_stream_count": 0,
            }
            with patch.object(
                evaluation_freeze,
                "probe_video_for_freeze",
                return_value=probe,
            ), patch.object(evaluation_freeze, "_mp4_has_faststart", return_value=True):
                with self.assertRaisesRegex(ValueError, "declared frame count changed"):
                    evaluation_freeze.validate_frozen_video(
                        video,
                        width=8,
                        height=6,
                        frame_count=2,
                        fps=24.0,
                    )


if __name__ == "__main__":
    unittest.main()
