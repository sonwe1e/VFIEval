from __future__ import annotations

import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.datasets import (
    _decode_video_cached,
    _ffmpeg_reported_fps,
    _load_compare_source_frames_with_cache,
    _publish_decode_staging,
)
from vfieval.db import Database
from vfieval.file_inputs import decode_cache_dir, decode_cache_key


def _workspace(tmp: str) -> tuple[WorkspaceConfig, Database]:
    workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
    workspace.ensure()
    db = Database(workspace.db_path)
    db.init()
    return workspace, db


def _write_valid_cache(cache_dir: Path, cache_key: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = cache_dir / "000000.png"
    Image.new("RGB", (4, 4), (12, 34, 56)).save(frame)
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "fps": 5.0,
                "frames": [str(frame.resolve())],
                "timestamps": [0.0],
                "frame_count": 1,
                "decode_backend": "fake",
            }
        ),
        encoding="utf-8",
    )


class FfmpegDecodeMetadataTests(unittest.TestCase):
    def test_source_fps_comes_from_input_stream_not_progress_duration(self) -> None:
        report = """
Input #0, mov, from 'clip.mp4':
  Stream #0:0: Video: h264, yuv420p, 1920x1080, 23.98 fps, 24 tbr
Output #0, image2, to '%06d.png':
  Stream #0:0: Video: png, rgb24, 1920x1080, 120 fps, 120 tbr
"""
        self.assertEqual(_ffmpeg_reported_fps(report), 23.98)


class DecodeCacheCoordinationTests(unittest.TestCase):
    def test_trusted_catalog_identity_skips_hash_and_stat_change_forces_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"source")
            frame = Path(tmp) / "000000.png"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(frame)
            stat = video.stat()
            trusted = {
                "content_sha256": "a" * 64,
                "size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            }
            decoded = ([frame], 5.0, [0.0], {"cache_hit": False})
            with patch("vfieval.datasets._decode_video_cached", return_value=decoded), patch(
                "vfieval.datasets.file_sha256",
                side_effect=AssertionError("trusted identity must not rehash"),
            ):
                _frames, _fps, _timestamps, descriptor = (
                    _load_compare_source_frames_with_cache(
                        db,
                        workspace,
                        video,
                        "trusted-test",
                        trusted_source_signature=trusted,
                    )
                )
            self.assertEqual(descriptor["source_identity"], "trusted_catalog")
            self.assertEqual(descriptor["source_hash_seconds"], 0.0)

            video.write_bytes(b"source changed and stat no longer matches")
            with patch("vfieval.datasets._decode_video_cached", return_value=decoded), patch(
                "vfieval.datasets.file_sha256", return_value="b" * 64
            ) as digest:
                _frames, _fps, _timestamps, changed = (
                    _load_compare_source_frames_with_cache(
                        db,
                        workspace,
                        video,
                        "trusted-test",
                        trusted_source_signature=trusted,
                    )
                )
            digest.assert_called_once_with(video.resolve())
            self.assertEqual(changed["source_identity"], "full_sha256")
            self.assertEqual(changed["source_sha256"], "b" * 64)

    def test_concurrent_same_key_decodes_once_and_leaves_no_private_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            second_db = Database(workspace.db_path)
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"not-read-by-fake-decoder")
            cache_key = decode_cache_key(video, "compare_concurrent", 1, None)
            started = threading.Event()
            release = threading.Event()
            calls: list[Path] = []
            call_lock = threading.Lock()

            def fake_decode(_video: Path, output_dir: Path, _max_frames: int | None, **_kwargs):
                with call_lock:
                    calls.append(output_dir)
                started.set()
                self.assertTrue(release.wait(5.0))
                frame = output_dir / "000000.png"
                Image.new("RGB", (4, 4), (1, 2, 3)).save(frame)
                return [frame], 5.0, [0.0], {"backend": "fake", "fallback_reason": None}

            def load(client: Database):
                return _decode_video_cached(
                    client,
                    workspace,
                    video,
                    cache_key,
                    None,
                    "compare_concurrent",
                    1,
                )

            with patch("vfieval.datasets._decode_video", side_effect=fake_decode):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(load, db)
                    self.assertTrue(started.wait(5.0))
                    second = executor.submit(load, second_db)
                    time.sleep(0.1)
                    release.set()
                    results = [first.result(timeout=10.0), second.result(timeout=10.0)]

            self.assertEqual(len(calls), 1)
            self.assertTrue(all(result[0] and result[0][0].is_file() for result in results))
            self.assertEqual(sum(result[3].get("cache_hit", False) for result in results), 1)
            cache_dir = decode_cache_dir(workspace, cache_key)
            self.assertTrue((cache_dir / "manifest.json").is_file())
            self.assertFalse(any((workspace.tmp_dir / "decode-cache-staging").glob("*.partial")))
            entry = db.get_cache_entry("decode_cache", cache_key)
            assert entry is not None
            self.assertEqual(entry["state"], "ready")
            self.assertGreater(int(entry["size_bytes"]), 0)

    def test_malformed_final_and_legacy_partial_are_rebuilt_by_current_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"source")
            cache_key = decode_cache_key(video, "compare_recovery", 1, None)
            cache_dir = decode_cache_dir(workspace, cache_key)
            cache_dir.mkdir(parents=True)
            (cache_dir / "manifest.json").write_text("{ malformed", encoding="utf-8")
            legacy_partial = cache_dir.with_name(cache_dir.name + ".partial")
            legacy_partial.mkdir()
            (legacy_partial / "stale.png").write_bytes(b"stale")

            def fake_decode(_video: Path, output_dir: Path, _max_frames: int | None, **_kwargs):
                frame = output_dir / "000000.png"
                Image.new("RGB", (4, 4), (2, 3, 4)).save(frame)
                return [frame], 5.0, [0.0], {"backend": "fake", "fallback_reason": None}

            with patch("vfieval.datasets._decode_video", side_effect=fake_decode) as mocked:
                frames, _fps, _timestamps, _info = _decode_video_cached(
                    db, workspace, video, cache_key, None, "compare_recovery", 1
                )

            self.assertEqual(mocked.call_count, 1)
            self.assertTrue(frames[0].is_file())
            self.assertFalse(legacy_partial.exists())
            manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame_count"], 1)

    def test_expired_owner_can_be_taken_over_without_releasing_new_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace_config, db = _workspace(tmp)
            cache_key = "a" * 64
            self.assertTrue(db.claim_decode_cache_build_lock(cache_key, "old", ttl_seconds=0.02))
            time.sleep(0.04)
            self.assertTrue(db.claim_decode_cache_build_lock(cache_key, "new", ttl_seconds=60))
            db.release_decode_cache_build_lock(cache_key, "old")
            self.assertTrue(db.owns_decode_cache_build_lock(cache_key, "new"))
            with self.assertRaisesRegex(RuntimeError, "lost before publish"):
                with db.decode_cache_build_publish_guard(cache_key, "old"):
                    pass

    def test_publish_reuses_a_completed_winner_after_destination_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_key = "b" * 64
            output_dir = root / cache_key
            staging_dir = root / "private.partial"
            _write_valid_cache(output_dir, cache_key)
            _write_valid_cache(staging_dir, cache_key)

            winner = _publish_decode_staging(staging_dir, output_dir, cache_key)

            self.assertIsNotNone(winner)
            assert winner is not None
            self.assertEqual(winner[0][0].parent, output_dir.resolve())
            self.assertTrue(staging_dir.exists())


if __name__ == "__main__":
    unittest.main()
