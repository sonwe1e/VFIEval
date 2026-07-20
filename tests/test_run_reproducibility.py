from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.input_identity import InputIdentityChanged
from vfieval.file_inputs import preflight_run
from vfieval.pipeline.inference import _write_mp4
from vfieval.server import _clone_run, _create_run_from_files, _make_handler, _retry_run


MODEL_SOURCE = """
class Model:
    def __init__(self, checkpoint_path=None, device='cpu', metadata=None):
        pass

    def infer(self, img0, img1, t=0.5):
        batch, _channels, height, width = img0.shape
        flow = img0.new_zeros((batch, 2, height, width))
        mask = img0.new_zeros((batch, 1, height, width))
        return flow, flow, mask, mask
"""


class RunReproducibilityTests(unittest.TestCase):
    def _project(self, root: Path) -> tuple[WorkspaceConfig, Database, Path]:
        models = root / "models"
        videos = root / "videos" / "group"
        checkpoints = root / "checkpoints"
        models.mkdir(parents=True)
        videos.mkdir(parents=True)
        checkpoints.mkdir(parents=True)
        model_path = models / "simple.py"
        model_path.write_text(MODEL_SOURCE, encoding="utf-8")
        checkpoint_dir = checkpoints / "simple"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "latest.pth").write_bytes(b"original checkpoint")
        frame_dir = root / "frames"
        frame_dir.mkdir()
        frames = []
        for index, color in enumerate(((0, 0, 0), (64, 64, 64), (128, 128, 128))):
            frame = frame_dir / f"{index:06d}.png"
            Image.new("RGB", (8, 8), color).save(frame)
            frames.append(frame)
        _write_mp4(frames, videos / "clip.mp4", 5.0)
        workspace = WorkspaceConfig.from_root(root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        return workspace, db, model_path

    def test_retry_is_exact_and_clone_accepts_current_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, db, model_path = self._project(root)
            payload = {
                "model_file": "simple.py",
                "video_group": "group",
                "selected_videos": ["clip.mp4"],
                "device": "cpu",
                "precision": "fp32",
                "checkpoint": "auto",
                "max_frames": 3,
            }
            with patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": str(root)}, clear=False), patch(
                "vfieval.server.start_decode_worker"
            ):
                created = _create_run_from_files(db, workspace, payload)
                original = db.get_run(int(created["run_id"]))
                identity = original["metadata"]["input_identity"]
                self.assertEqual(identity["schema"], "run-input-identity-v1")
                self.assertEqual(identity["checkpoint"]["requested"], "auto")
                self.assertEqual(
                    identity["checkpoint"]["resolved"]["relative_path"],
                    "simple/latest.pth",
                )
                self.assertEqual(identity["sources"][0]["qualified_name"], "group/clip.mp4")

                newer_checkpoint = root / "checkpoints" / "simple" / "newer.pth"
                newer_checkpoint.write_bytes(b"new checkpoint that auto would now select")
                latest_mtime = (root / "checkpoints" / "simple" / "latest.pth").stat().st_mtime_ns
                os.utime(newer_checkpoint, ns=(latest_mtime + 2_000_000_000,) * 2)
                exact_retry = _retry_run(db, workspace, int(created["run_id"]))
                retried = db.get_run(int(exact_retry["run_id"]))
                self.assertEqual(retried["metadata"]["checkpoint"], "simple/latest.pth")
                self.assertEqual(
                    retried["metadata"]["input_identity"]["fingerprint"],
                    identity["fingerprint"],
                )

                stat_result = model_path.stat()
                os.utime(
                    model_path,
                    ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000),
                )
                with self.assertRaises(InputIdentityChanged) as caught:
                    _retry_run(db, workspace, int(created["run_id"]))
                public = caught.exception.public_payload()
                self.assertTrue(public["differences"])
                self.assertTrue(
                    any(row["field"].startswith("model.") for row in public["differences"])
                )

                clone = _clone_run(db, workspace, int(created["run_id"]))
                cloned = db.get_run(int(clone["run_id"]))
                self.assertEqual(
                    int(cloned["metadata"]["clone_of_run_id"]),
                    int(created["run_id"]),
                )
                self.assertNotEqual(
                    cloned["metadata"]["input_identity"]["fingerprint"],
                    identity["fingerprint"],
                )
                self.assertFalse(cloned["metadata"]["clone_identity_comparison"]["matches"])

    def test_retry_reports_missing_model_checkpoint_and_source_before_preflight(self) -> None:
        cases = (
            ("model", Path("models/simple.py"), "model"),
            (
                "checkpoint",
                Path("checkpoints/simple/latest.pth"),
                "checkpoint.resolved",
            ),
            ("source", Path("videos/group/clip.mp4"), "sources[0].content"),
        )
        for label, relative_path, expected_field in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                workspace, db, _model_path = self._project(root)
                payload = {
                    "model_file": "simple.py",
                    "video_group": "group",
                    "selected_videos": ["clip.mp4"],
                    "device": "cpu",
                    "precision": "fp32",
                    "checkpoint": "auto",
                    "max_frames": 3,
                }
                with patch.dict(
                    os.environ,
                    {"VFIEVAL_PROJECT_ROOT": str(root)},
                    clear=False,
                ), patch("vfieval.server.start_decode_worker"):
                    created = _create_run_from_files(db, workspace, payload)
                    (root / relative_path).rename(root / f"{relative_path.name}.missing")
                    with patch("vfieval.server.preflight_run") as preflight_spy:
                        with self.assertRaises(InputIdentityChanged) as caught:
                            _retry_run(db, workspace, int(created["run_id"]))
                    preflight_spy.assert_not_called()

                public = caught.exception.public_payload()
                rows = {row["field"]: row for row in public["differences"]}
                self.assertEqual(rows[expected_field]["kind"], "missing")
                self.assertIsNone(rows[expected_field]["actual"])
                self.assertNotIn(str(root), json.dumps(public))

    def test_retry_missing_input_returns_structured_http_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, db, _model_path = self._project(root)
            payload = {
                "model_file": "simple.py",
                "video_group": "group",
                "selected_videos": ["clip.mp4"],
                "device": "cpu",
                "precision": "fp32",
                "checkpoint": "auto",
                "max_frames": 3,
            }
            with patch.dict(
                os.environ,
                {"VFIEVAL_PROJECT_ROOT": str(root)},
                clear=False,
            ), patch("vfieval.server.start_decode_worker"):
                created = _create_run_from_files(db, workspace, payload)
                server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                try:
                    source = root / "videos" / "group" / "clip.mp4"
                    source.rename(root / "clip.mp4.missing")
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        self._post(
                            base_url,
                            f"/api/runs/{created['run_id']}/retry",
                            {},
                        )
                    self.assertEqual(caught.exception.code, 409)
                    error = json.loads(caught.exception.read().decode("utf-8"))
                    self.assertEqual(error["type"], "InputIdentityChanged")
                    rows = {row["field"]: row for row in error["differences"]}
                    self.assertEqual(rows["sources[0].content"]["kind"], "missing")
                    self.assertNotIn(str(root), json.dumps(error))
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_preflight_token_reuses_deep_result_and_rejects_changed_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, db, _model_path = self._project(root)
            payload = {
                "model_file": "simple.py",
                "video_group": "group",
                "selected_videos": ["clip.mp4"],
                "device": "cpu",
                "precision": "fp32",
                "checkpoint": "auto",
                "max_frames": 3,
            }
            with patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": str(root)}, clear=False), patch(
                "vfieval.server.preflight_run",
                wraps=preflight_run,
            ) as preflight_spy, patch("vfieval.server.start_decode_worker"):
                server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                try:
                    quick = self._post(
                        base_url,
                        "/api/preflight/quick",
                        payload,
                    )
                    self.assertEqual(quick["preflight_level"], "quick")
                    self.assertNotIn("preflight_token", quick)
                    self.assertFalse(quick["model"]["interface_checked"])
                    preflight = self._post(base_url, "/api/preflight", payload)
                    self.assertTrue(preflight["ok"])
                    self.assertTrue(preflight["preflight_token"])
                    self.assertEqual(len(preflight["input_fingerprint"]), 64)
                    created = self._post(
                        base_url,
                        "/api/runs",
                        {**payload, "preflight_token": preflight["preflight_token"]},
                    )
                    self.assertGreater(int(created["run_id"]), 0)
                    self.assertTrue(created["preflight"]["preflight_cache_hit"])
                    self.assertEqual(
                        created["preflight"]["input_fingerprint"],
                        preflight["input_fingerprint"],
                    )
                    self.assertEqual(preflight_spy.call_count, 2)

                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        self._post(
                            base_url,
                            "/api/runs",
                            {
                                **payload,
                                "batch_size_per_device": 2,
                                "preflight_token": preflight["preflight_token"],
                            },
                        )
                    self.assertEqual(caught.exception.code, 400)
                    error = json.loads(caught.exception.read().decode("utf-8"))
                    self.assertIn("configuration changed", error["error"]["message"])
                    self.assertEqual(preflight_spy.call_count, 2)

                    high_risk_payload = {
                        **payload,
                        "max_frames": None,
                        "batch_size_per_device": 250_001,
                    }
                    risky = self._post(base_url, "/api/preflight", high_risk_payload)
                    self.assertEqual(risky["workload"]["risk_level"], "high")
                    with self.assertRaises(urllib.error.HTTPError) as unconfirmed:
                        self._post(
                            base_url,
                            "/api/runs",
                            {**high_risk_payload, "preflight_token": risky["preflight_token"]},
                        )
                    self.assertEqual(unconfirmed.exception.code, 409)
                    conflict = json.loads(unconfirmed.exception.read().decode("utf-8"))
                    self.assertEqual(conflict["type"], "WorkloadRiskConfirmationRequired")
                    accepted = self._post(
                        base_url,
                        "/api/runs",
                        {
                            **high_risk_payload,
                            "preflight_token": risky["preflight_token"],
                            "risk_ack_fingerprint": risky["workload"]["risk_fingerprint"],
                        },
                    )
                    self.assertGreater(int(accepted["run_id"]), int(created["run_id"]))
                    self.assertEqual(preflight_spy.call_count, 3)

                    clone_path = f"/api/runs/{accepted['run_id']}/clone"
                    with patch(
                        "vfieval.server._host_available_memory_bytes",
                        side_effect=[
                            40_000_000_000,
                            39_000_000_000,
                            38_000_000_000,
                        ],
                    ):
                        with self.assertRaises(urllib.error.HTTPError) as clone_unconfirmed:
                            self._post(base_url, clone_path, {})
                        self.assertEqual(clone_unconfirmed.exception.code, 409)
                        clone_conflict = json.loads(
                            clone_unconfirmed.exception.read().decode("utf-8")
                        )
                        self.assertEqual(
                            clone_conflict["type"],
                            "WorkloadRiskConfirmationRequired",
                        )
                        clone_workload = clone_conflict["workload"]
                        self.assertEqual(clone_workload["risk_level"], "high")
                        replacement_dir = root / "replacement_frames"
                        replacement_dir.mkdir()
                        replacement_frames = []
                        for index in range(5):
                            frame = replacement_dir / f"{index:06d}.png"
                            Image.new("RGB", (16, 16), (index * 32,) * 3).save(frame)
                            replacement_frames.append(frame)
                        _write_mp4(
                            replacement_frames,
                            root / "videos" / "group" / "clip.mp4",
                            5.0,
                        )

                        with self.assertRaises(urllib.error.HTTPError) as clone_changed:
                            self._post(
                                base_url,
                                clone_path,
                                {
                                    "risk_ack_fingerprint": clone_workload[
                                        "risk_fingerprint"
                                    ]
                                },
                            )
                        self.assertEqual(clone_changed.exception.code, 409)
                        changed_conflict = json.loads(
                            clone_changed.exception.read().decode("utf-8")
                        )
                        changed_workload = changed_conflict["workload"]
                        self.assertEqual(changed_workload["effective"]["width"], 16)
                        self.assertEqual(changed_workload["effective"]["height"], 16)
                        self.assertGreater(
                            changed_workload["effective"]["sample_count"],
                            clone_workload["effective"]["sample_count"],
                        )
                        cloned = self._post(
                            base_url,
                            clone_path,
                            {
                                "risk_ack_fingerprint": changed_workload[
                                    "risk_fingerprint"
                                ]
                            },
                        )
                    cloned_run = db.get_run(int(cloned["run_id"]))
                    self.assertEqual(
                        int(cloned_run["metadata"]["clone_of_run_id"]),
                        int(accepted["run_id"]),
                    )
                    self.assertEqual(
                        clone_workload["effective"]["host_available_memory_bytes"],
                        40_000_000_000,
                    )
                    self.assertEqual(
                        cloned_run["metadata"]["workload"]["effective"][
                            "host_available_memory_bytes"
                        ],
                        38_000_000_000,
                    )
                    self.assertEqual(preflight_spy.call_count, 6)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_preflight_token_rehashes_physical_inputs_and_creation_reuses_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, db, _model_path = self._project(root)
            payload = {
                "model_file": "simple.py",
                "video_group": "group",
                "selected_videos": ["clip.mp4"],
                "device": "cpu",
                "precision": "fp32",
                "checkpoint": "auto",
                "max_frames": 3,
            }
            with patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": str(root)}, clear=False), patch(
                "vfieval.server.start_decode_worker"
            ):
                server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                try:
                    preflight = self._post(base_url, "/api/preflight", payload)
                    token = preflight["preflight_token"]
                    from vfieval.file_inputs import file_sha256

                    with patch(
                        "vfieval.input_identity.file_sha256",
                        wraps=file_sha256,
                    ) as identity_hash, patch(
                        "vfieval.media_assets._file_sha256",
                    ) as catalog_hash:
                        created = self._post(
                            base_url,
                            "/api/runs",
                            {**payload, "preflight_token": token},
                        )
                    self.assertGreater(int(created["run_id"]), 0)
                    # POST revalidates model + checkpoint + video content once.
                    # Run input identity construction trusts those exact hashes
                    # instead of reading all three files a third time.
                    self.assertEqual(identity_hash.call_count, 3)
                    catalog_hash.assert_not_called()

                    for input_path in (
                        root / "models" / "simple.py",
                        root / "checkpoints" / "simple" / "latest.pth",
                        root / "videos" / "group" / "clip.mp4",
                    ):
                        with self.subTest(input=input_path.relative_to(root).as_posix()):
                            original = input_path.read_bytes()
                            original_stat = input_path.stat()
                            mutated = bytearray(original)
                            mutated[len(mutated) // 2] ^= 1
                            input_path.write_bytes(mutated)
                            os.utime(
                                input_path,
                                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                            )
                            self.assertEqual(input_path.stat().st_size, original_stat.st_size)
                            self.assertEqual(input_path.stat().st_mtime_ns, original_stat.st_mtime_ns)

                            with self.assertRaises(urllib.error.HTTPError) as changed:
                                self._post(
                                    base_url,
                                    "/api/runs",
                                    {**payload, "preflight_token": token},
                                )
                            self.assertEqual(changed.exception.code, 400)
                            error = json.loads(changed.exception.read().decode("utf-8"))
                            self.assertIn(
                                "inputs changed after preflight",
                                error["error"]["message"],
                            )
                            input_path.write_bytes(original)
                            os.utime(
                                input_path,
                                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                            )
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    @staticmethod
    def _post(base_url: str, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
