from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.server import _make_handler
from vfieval.worker import WorkerOptions, run_worker


class V2RunTests(unittest.TestCase):
    def test_decode_handoff_completes_source_and_publishes_inference_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("atomic-decode", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("atomic-decode", tmp, False)
            source = Path(tmp) / "sample.png"
            Image.new("RGB", (4, 4)).save(source)
            db.add_sample(dataset_id, "sample", str(source), str(source), None, {})
            run_id = db.create_run(
                "atomic-decode", model_id, dataset_id, 4, 4, 1, "cpu", "fp32", [],
                create_inference_job=False,
            )
            decode_job_id = db.add_run_job(
                run_id,
                "decode",
                {"run_id": run_id, "dataset_id": dataset_id},
                progress_total=1,
            )
            self.assertEqual(int(db.claim_next_job("atomic-decode", ["decode"])["id"]), decode_job_id)
            result = {"samples": 1, "status": "completed"}

            job_ids = db.publish_inference_jobs(
                run_id,
                [{"payload": {"dataset_id": dataset_id}, "progress_total": 1, "device": "cpu"}],
                source_job_id=decode_job_id,
                source_job_result=result,
            )

            self.assertEqual(len(job_ids), 1)
            self.assertEqual(db.get_job(decode_job_id)["status"], "completed")
            self.assertEqual(db.get_job(decode_job_id)["result"], result)
            self.assertEqual(db.get_run(run_id)["status"], "queued")
            self.assertEqual(db.get_job(job_ids[0])["status"], "queued")

    def test_last_inference_shard_completion_and_finalize_handoff_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("atomic-shards", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("atomic-shards", tmp, False)
            run_id = db.create_run(
                "atomic-shards", model_id, dataset_id, 4, 4, 1, "cpu", "fp32", [],
                create_inference_job=False,
            )
            job_ids = db.publish_inference_jobs(
                run_id,
                [
                    {
                        "payload": {
                            "run_id": run_id,
                            "dataset_id": dataset_id,
                            "shard_count": 2,
                            "defer_video_finalize": True,
                        },
                        "shard_index": index,
                    }
                    for index in range(2)
                ],
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            first = db.claim_next_job("atomic-shard-0", ["inference"])
            second = db.claim_next_job("atomic-shard-1", ["inference"])
            self.assertEqual([int(first["id"]), int(second["id"])], job_ids)

            self.assertFalse(
                db.maybe_complete_multi_run_inference(
                    run_id,
                    source_job_id=job_ids[0],
                    source_job_result={"samples": 1, "performance": {}},
                )
            )
            self.assertEqual(db.get_job(job_ids[0])["status"], "completed")
            self.assertEqual(db.get_run(run_id)["status"], "running")

            self.assertTrue(
                db.maybe_complete_multi_run_inference(
                    run_id,
                    source_job_id=job_ids[1],
                    source_job_result={"samples": 1, "performance": {}},
                )
            )
            self.assertEqual(db.get_job(job_ids[1])["status"], "completed")
            self.assertEqual(db.get_run(run_id)["status"], "finalize_queued")
            finalize_jobs = db.list_run_jobs(run_id, "finalize")
            self.assertEqual(len(finalize_jobs), 1)
            self.assertEqual(finalize_jobs[0]["status"], "queued")

    def test_payload_only_job_callbacks_are_fenced_after_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("payload-cas", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("payload-cas", tmp, False)
            run_id = db.create_run(
                "payload-cas", model_id, dataset_id, 4, 4, 1, "cpu", "fp32", [],
                create_inference_job=False,
            )
            job_id = db.create_job("inference", {"run_id": run_id})
            with db.connection() as conn:
                conn.execute("DELETE FROM run_jobs WHERE job_id = ?", (job_id,))
                conn.execute("UPDATE runs SET inference_job_id = NULL WHERE id = ?", (run_id,))
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("payload-cas", ["inference"])["id"]), job_id)

            self.assertTrue(db.request_run_cancel(run_id))
            self.assertEqual(db.get_run(run_id)["status"], "cancel_requested")
            self.assertFalse(db.update_job_progress(job_id, 1, 1))
            self.assertFalse(db.complete_job(job_id, {"late": True}))
            self.assertFalse(db.fail_job(job_id, {"message": "late"}))
            self.assertTrue(db.converge_run_cancellation(run_id, job_id))
            self.assertEqual(db.get_job(job_id)["status"], "canceled")
            self.assertEqual(db.get_run(run_id)["status"], "canceled")

    def test_legacy_direct_inference_link_is_backfilled_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("direct-publish", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("direct-publish", tmp, False)
            run_id = db.create_run(
                "direct-publish", model_id, dataset_id, 4, 4, 1, "cpu", "fp32", [],
                create_inference_job=False,
            )
            direct_job_id = db.create_job("inference", {"legacy": True})
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET inference_job_id = ? WHERE id = ?",
                    (direct_job_id, run_id),
                )

            published = db.publish_inference_jobs(
                run_id,
                [{"payload": {"dataset_id": dataset_id}, "progress_total": 1}],
            )

            self.assertEqual(published, [direct_job_id])
            self.assertEqual(
                [int(row["job_id"]) for row in db.list_run_jobs(run_id, "inference")],
                [direct_job_id],
            )
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM jobs WHERE kind = 'inference'")["count"]),
                1,
            )

    def test_legacy_direct_metric_link_blocks_duplicate_wave(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("direct-metric", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("direct-metric", tmp, False)
            run_id = db.create_run(
                "direct-metric", model_id, dataset_id, 4, 4, 1, "cpu", "fp32", [],
                create_inference_job=False,
            )
            metric_job_id = db.create_job("metric", {"legacy": True})
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET metric_job_id = ?, status = 'running' WHERE id = ?",
                    (metric_job_id, run_id),
                )

            with self.assertRaisesRegex(ValueError, "active metric"):
                db.publish_metric_wave(
                    run_id,
                    [{"payload": {"metric_names": ["lpips_convnext"]}}],
                    retry=False,
                )

            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM jobs WHERE kind = 'metric'")["count"]),
                1,
            )
            self.assertEqual(db.get_job(metric_job_id)["status"], "queued")

    def test_cancel_converges_only_workers_that_reached_their_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("cancel-boundary", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("cancel-boundary", tmp, False)
            run_id = db.create_run(
                "cancel-boundary",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            job_ids = db.publish_inference_jobs(
                run_id,
                [
                    {
                        "payload": {"run_id": run_id, "dataset_id": dataset_id},
                        "shard_index": index,
                    }
                    for index in range(3)
                ],
            )
            self.assertEqual(len(job_ids), 3)
            self.assertTrue(db.mark_run_started(run_id, "running"))
            first = db.claim_next_job("cancel-boundary-0", ["inference"])
            second = db.claim_next_job("cancel-boundary-1", ["inference"])
            self.assertEqual([int(first["id"]), int(second["id"])], job_ids[:2])

            self.assertTrue(db.request_run_cancel(run_id))
            self.assertEqual(db.get_run(run_id)["status"], "cancel_requested")
            self.assertEqual(db.get_job(job_ids[2])["status"], "canceled")

            self.assertFalse(db.converge_run_cancellation(run_id, job_ids[0]))
            self.assertEqual(db.get_job(job_ids[0])["status"], "canceled")
            self.assertEqual(db.get_job(job_ids[1])["status"], "running")
            self.assertEqual(db.get_run(run_id)["status"], "cancel_requested")

            self.assertTrue(db.converge_run_cancellation(run_id, job_ids[1]))
            self.assertEqual(db.get_job(job_ids[1])["status"], "canceled")
            self.assertEqual(db.get_run(run_id)["status"], "canceled")

    def test_non_device_claim_fences_legacy_links_payload_phase_and_purge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("claim-fence", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("claim-fence", tmp, False)

            phase_run_id = db.create_run(
                "phase-fence",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            phase_job_id = db.create_job("inference", {"legacy": "direct-inference"})

            terminal_run_id = db.create_run(
                "terminal-fence",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            terminal_job_id = db.create_job("metric", {"legacy": "direct-metric"})

            purge_run_id = db.create_run(
                "payload-purge-fence",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            payload_job_id = db.create_job("inference", {"run_id": purge_run_id})

            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET inference_job_id = ?, status = 'metric_queued' WHERE id = ?",
                    (phase_job_id, phase_run_id),
                )
                conn.execute(
                    "UPDATE runs SET metric_job_id = ?, status = 'completed' WHERE id = ?",
                    (terminal_job_id, terminal_run_id),
                )
                # Leave the payload as the only association for this legacy Job.
                conn.execute("DELETE FROM run_jobs WHERE job_id = ?", (payload_job_id,))
                conn.execute(
                    "UPDATE runs SET inference_job_id = NULL WHERE id = ?",
                    (purge_run_id,),
                )
            db.request_run_purge(purge_run_id, "cleanup_artifacts")

            standalone_job_id = db.create_job("inference", {"standalone": True})
            claimed = db.claim_next_job("legacy-fence", ["inference", "metric"])
            self.assertIsNotNone(claimed)
            self.assertEqual(int(claimed["id"]), standalone_job_id)
            self.assertIsNone(db.claim_next_job("legacy-fence-empty", ["inference", "metric"]))
            for job_id in (phase_job_id, terminal_job_id, payload_job_id):
                self.assertEqual(db.get_job(job_id)["status"], "queued")

    def test_publish_inference_jobs_fast_path_only_returns_active_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("publish-fence", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("publish-fence", tmp, False)
            run_id = db.create_run(
                "publish-fence",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            specs = [
                {
                    "payload": {"dataset_id": dataset_id},
                    "progress_total": 1,
                    "shard_index": 0,
                    "device": "cpu",
                }
            ]

            job_ids = db.publish_inference_jobs(run_id, specs)
            self.assertEqual(len(job_ids), 1)
            self.assertEqual(db.publish_inference_jobs(run_id, specs), job_ids)
            self.assertTrue(db.mark_run_started(run_id, "running"))
            claimed = db.claim_next_job("publish-fence-worker", ["inference"])
            self.assertEqual(int(claimed["id"]), job_ids[0])
            self.assertEqual(db.publish_inference_jobs(run_id, specs), job_ids)
            self.assertTrue(db.complete_job(job_ids[0], {}))
            self.assertEqual(db.publish_inference_jobs(run_id, specs), [])

            terminal_run_id = db.create_run(
                "publish-terminal",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            terminal_job_ids = db.publish_inference_jobs(terminal_run_id, specs)
            self.assertTrue(db.mark_run_started(terminal_run_id, "running"))
            self.assertTrue(db.complete_run_inference(terminal_run_id, {}, {}, "completed"))
            self.assertEqual(db.get_job(terminal_job_ids[0])["status"], "queued")
            self.assertEqual(db.publish_inference_jobs(terminal_run_id, specs), [])

            canceled_run_id = db.create_run(
                "publish-canceled",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            canceled_job_ids = db.publish_inference_jobs(canceled_run_id, specs)
            self.assertTrue(db.request_run_cancel(canceled_run_id))
            self.assertEqual(db.get_run(canceled_run_id)["status"], "canceled")
            self.assertEqual(db.get_job(canceled_job_ids[0])["status"], "canceled")
            self.assertEqual(db.publish_inference_jobs(canceled_run_id, specs), [])
            self.assertEqual(
                [row["job_id"] for row in db.list_run_jobs(canceled_run_id, "inference")],
                canceled_job_ids,
            )

    def test_queue_finalize_fast_path_only_accepts_coherent_active_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("finalize-fence", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("finalize-fence", tmp, False)
            run_id = db.create_run(
                "finalize-fence", model_id, dataset_id, 4, 4, 1, "cpu", "fp32", []
            )
            inference_job_id = int(db.get_run(run_id)["inference_job_id"])
            self.assertTrue(db.mark_run_started(run_id, "running"))
            claimed_inference = db.claim_next_job("finalize-source", ["inference"])
            self.assertEqual(int(claimed_inference["id"]), inference_job_id)
            self.assertTrue(db.complete_job(inference_job_id, {"samples": 1}))

            args = (run_id, {"samples": 1}, {}, [inference_job_id])
            self.assertTrue(db.queue_run_finalize(*args))
            finalize_jobs = db.list_run_jobs(run_id, "finalize")
            self.assertEqual(len(finalize_jobs), 1)
            finalize_job_id = int(finalize_jobs[0]["job_id"])
            self.assertTrue(db.queue_run_finalize(*args))

            claimed_finalize = db.claim_next_job("finalize-worker", ["finalize"])
            self.assertEqual(int(claimed_finalize["id"]), finalize_job_id)
            # Claim and Run-phase CAS are separate calls, so this active pair is
            # valid both immediately before and after the phase transition.
            self.assertTrue(db.queue_run_finalize(*args))
            self.assertTrue(db.mark_run_started(run_id, "finalizing"))
            self.assertTrue(db.queue_run_finalize(*args))

            self.assertTrue(db.complete_job(finalize_job_id, {}))
            self.assertFalse(db.queue_run_finalize(*args))
            self.assertTrue(db.cancel_run(run_id))
            self.assertEqual(db.get_run(run_id)["status"], "canceled")
            self.assertFalse(db.queue_run_finalize(*args))

            terminal_run_id = db.create_run(
                "finalize-terminal",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
            )
            terminal_inference_job_id = int(
                db.get_run(terminal_run_id)["inference_job_id"]
            )
            self.assertTrue(db.mark_run_started(terminal_run_id, "running"))
            terminal_source = db.claim_next_job("finalize-terminal-source", ["inference"])
            self.assertEqual(int(terminal_source["id"]), terminal_inference_job_id)
            self.assertTrue(db.complete_job(terminal_inference_job_id, {}))
            terminal_args = (terminal_run_id, {}, {}, [terminal_inference_job_id])
            self.assertTrue(db.queue_run_finalize(*terminal_args))
            terminal_finalize_job_id = int(
                db.list_run_jobs(terminal_run_id, "finalize")[0]["job_id"]
            )
            self.assertTrue(db.cancel_run(terminal_run_id))
            self.assertEqual(db.get_job(terminal_finalize_job_id)["status"], "queued")
            self.assertFalse(db.queue_run_finalize(*terminal_args))

    def test_cancel_first_is_absorbing_for_late_job_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("cas-cancel", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("cas-cancel", tmp, False)
            run_id = db.create_run(
                "cas-cancel", model_id, dataset_id, 4, 4, 1, "cpu", "fp32", []
            )
            job_id = int(db.get_run(run_id)["inference_job_id"])
            self.assertTrue(db.mark_run_started(run_id, "running"))
            claimed = db.claim_next_job("cas-cancel-worker", ["inference"])
            self.assertEqual(int(claimed["id"]), job_id)

            self.assertTrue(db.request_run_cancel(run_id))
            self.assertEqual(db.get_run(run_id)["status"], "cancel_requested")
            self.assertFalse(db.update_job_progress(job_id, 1, 1))
            self.assertFalse(db.complete_job(job_id, {"late": True}))
            self.assertFalse(db.fail_job(job_id, {"message": "late"}))
            self.assertTrue(db.converge_run_cancellation(run_id, job_id))

            self.assertEqual(db.get_job(job_id)["status"], "canceled")
            self.assertEqual(db.get_run(run_id)["status"], "canceled")
            self.assertFalse(db.mark_run_started(run_id, "running"))
            self.assertFalse(db.complete_run_inference(run_id, {}, {}, "completed"))
            self.assertFalse(db.update_run_progress(run_id, 99, 100))

    def test_completion_or_failure_committed_first_blocks_late_cancel(self) -> None:
        for terminal in ("completed", "failed"):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as tmp:
                db = Database(Path(tmp) / "vfieval.sqlite")
                db.init()
                model_id = db.register_model(f"cas-{terminal}", "dummy", None, 4, 4, {})
                dataset_id = db.create_dataset(f"cas-{terminal}", tmp, False)
                run_id = db.create_run(
                    f"cas-{terminal}", model_id, dataset_id, 4, 4, 1, "cpu", "fp32", []
                )
                job_id = int(db.get_run(run_id)["inference_job_id"])
                self.assertTrue(db.mark_run_started(run_id, "running"))
                self.assertEqual(
                    int(db.claim_next_job(f"cas-{terminal}-worker", ["inference"])["id"]),
                    job_id,
                )
                if terminal == "completed":
                    self.assertTrue(db.complete_run_inference(run_id, {}, {}, "completed"))
                    self.assertTrue(db.complete_job(job_id, {}))
                else:
                    error = {"message": "failure committed first", "type": "RuntimeError"}
                    self.assertTrue(db.fail_run(run_id, error))
                    self.assertFalse(db.complete_job(job_id, {"late": True}))
                    self.assertTrue(db.fail_job(job_id, error))

                self.assertFalse(db.request_run_cancel(run_id))
                self.assertEqual(db.get_run(run_id)["status"], terminal)
                self.assertEqual(db.get_job(job_id)["status"], terminal)

    def test_v1_database_init_adds_v2_tables_without_losing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "vfieval.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                conn.executescript(
                    """
                    CREATE TABLE models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        adapter TEXT NOT NULL,
                        checkpoint_path TEXT,
                        input_height INTEGER NOT NULL,
                        input_width INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        root_path TEXT NOT NULL,
                        has_gt INTEGER NOT NULL DEFAULT 1,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        img0_path TEXT NOT NULL,
                        img1_path TEXT NOT NULL,
                        gt_path TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        UNIQUE(dataset_id, name)
                    );
                    CREATE TABLE jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        worker_id TEXT,
                        progress_current INTEGER NOT NULL DEFAULT 0,
                        progress_total INTEGER NOT NULL DEFAULT 0,
                        result_json TEXT NOT NULL DEFAULT '{}',
                        error_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        started_at REAL,
                        finished_at REAL
                    );
                    CREATE TABLE artifacts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id INTEGER NOT NULL,
                        sample_id INTEGER,
                        kind TEXT NOT NULL,
                        path TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE metric_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id INTEGER NOT NULL,
                        inference_job_id INTEGER NOT NULL,
                        sample_id INTEGER,
                        metric_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        value REAL,
                        details_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE metric_cache (
                        cache_key TEXT PRIMARY KEY,
                        metric_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        value REAL,
                        details_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    );
                    INSERT INTO models(name, adapter, input_height, input_width, metadata_json, created_at)
                    VALUES ('dummy', 'dummy', 4, 4, '{}', 1.0);
                    INSERT INTO datasets(name, root_path, has_gt, metadata_json, created_at)
                    VALUES ('demo', '.', 1, '{}', 1.0);
                    """
                )
                conn.commit()
            finally:
                conn.close()

            db = Database(db_path)
            db.init()

            self.assertEqual(db.list_models()[0]["name"], "dummy")
            upgraded_dataset = db.list_datasets()[0]
            self.assertEqual(upgraded_dataset["name"], "demo")
            self.assertEqual(upgraded_dataset["source_type"], "frames")
            self.assertEqual(upgraded_dataset["decode_mode"], "frames")
            self.assertEqual(db.list_experiments()[0]["name"], "Default")
            run_id = db.create_run("upgrade-smoke", 1, 1, 4, 4, 1, "cpu", "fp32", [])
            self.assertEqual(db.get_run(run_id)["status"], "queued")

    def test_v2_api_and_worker_run_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = _make_dataset(root)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                model = _post(
                    base_url,
                    "/api/models",
                    {"name": "dummy", "adapter": "dummy", "input_height": 4, "input_width": 4},
                )
                dataset = _post(
                    base_url,
                    "/api/datasets",
                    {"name": "demo", "root_path": str(dataset_root), "has_gt": True},
                )
                scan = _post(base_url, f"/api/datasets/{dataset['dataset_id']}/scan", {})
                self.assertEqual(scan["samples"], 2)

                created = _post(
                    base_url,
                    "/api/runs",
                    {
                        "name": "api-run",
                        "model_id": model["model_id"],
                        "dataset_id": dataset["dataset_id"],
                        "height": 4,
                        "width": 4,
                        "batch_size": 2,
                        "device": "cpu",
                        "precision": "fp32",
                        "metrics": ["cgvqm"],
                    },
                )
                run_id = created["run_id"]
                self.assertEqual(created["run"]["status"], "queued")

                run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="v2-inference"))
                after_inference = _get(base_url, f"/api/runs/{run_id}")
                self.assertEqual(after_inference["status"], "metric_queued")
                self.assertEqual(after_inference["artifact_summary"]["by_kind"]["pred"], 2)
                self.assertTrue(any(job["role"] == "metric" and job["job_id"] == after_inference["metric_job_id"] for job in after_inference["jobs"]))

                run_worker(db, workspace, WorkerOptions(role="metric", once=True, worker_id="v2-metric"))
                completed = _get(base_url, f"/api/runs/{run_id}")
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["metric_summary"]["cgvqm"]["unavailable"], 1)

                samples = _get(base_url, f"/api/runs/{run_id}/samples")
                self.assertEqual(len(samples), 2)
                self.assertIn("pred", samples[0]["artifacts"])

                compare = _get(base_url, f"/api/compare?run_id={run_id}")
                self.assertEqual(compare["runs"][0]["run"]["id"], run_id)

                dashboard = _get(base_url, "/api/dashboard")
                self.assertEqual(dashboard["completed_runs"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_cancel_metric_queued_run_cancels_metric_job_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = _make_dataset(root)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                model = _post(
                    base_url,
                    "/api/models",
                    {"name": "dummy", "adapter": "dummy", "input_height": 4, "input_width": 4},
                )
                dataset = _post(
                    base_url,
                    "/api/datasets",
                    {"name": "demo", "root_path": str(dataset_root), "has_gt": True},
                )
                _post(base_url, f"/api/datasets/{dataset['dataset_id']}/scan", {})
                created = _post(
                    base_url,
                    "/api/runs",
                    {
                        "name": "cancel-metric-queued",
                        "model_id": model["model_id"],
                        "dataset_id": dataset["dataset_id"],
                        "height": 4,
                        "width": 4,
                        "batch_size": 2,
                        "device": "cpu",
                        "precision": "fp32",
                        "metrics": ["cgvqm"],
                    },
                )
                run_id = created["run_id"]
                run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="v2-inference"))
                after_inference = _get(base_url, f"/api/runs/{run_id}")
                metric_job_id = int(after_inference["metric_job_id"])
                self.assertEqual(after_inference["status"], "metric_queued")

                canceled = _post(base_url, f"/api/runs/{run_id}/cancel", {})
                self.assertEqual(canceled["run"]["status"], "canceled")
                self.assertEqual(db.get_job(metric_job_id)["status"], "canceled")

                run_worker(db, workspace, WorkerOptions(role="metric", once=True, worker_id="v2-metric"))
                final_run = _get(base_url, f"/api/runs/{run_id}")
                self.assertEqual(final_run["status"], "canceled")
                self.assertEqual(db.get_job(metric_job_id)["status"], "canceled")
                self.assertEqual(db.list_metric_results(inference_job_id=int(after_inference["inference_job_id"])), [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_metric_worker_stops_when_run_was_already_cancel_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = _make_dataset(root)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("demo", str(dataset_root), True)
            for idx in range(2):
                name = f"sample{idx:03d}.png"
                db.add_sample(
                    dataset_id,
                    f"sample{idx:03d}",
                    str(dataset_root / "img0" / name),
                    str(dataset_root / "img1" / name),
                    str(dataset_root / "gt" / name),
                    {},
                )
            run_id = db.create_run("metric-cancel-requested", model_id, dataset_id, 4, 4, 2, "cpu", "fp32", ["cgvqm"])
            run = db.get_run(run_id)
            metric_job_id = db.create_job(
                "metric",
                {
                    "run_id": run_id,
                    "inference_job_id": int(run["inference_job_id"]),
                    "dataset_id": dataset_id,
                    "metric_names": ["cgvqm"],
                },
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertTrue(db.set_run_metric_job(run_id, metric_job_id))
            self.assertTrue(db.mark_run_started(run_id, "metric_running"))
            db.request_run_cancel(run_id)

            run_worker(db, workspace, WorkerOptions(role="metric", once=True, worker_id="v2-metric"))

            canceled_run = db.get_run(run_id)
            self.assertEqual(canceled_run["status"], "canceled")
            self.assertEqual(canceled_run["error"]["type"], "RunCanceled")
            self.assertEqual(db.get_job(metric_job_id)["status"], "canceled")


def _make_dataset(root: Path) -> Path:
    dataset_root = root / "dataset"
    for folder in ("img0", "img1", "gt"):
        (dataset_root / folder).mkdir(parents=True)
    for idx in range(2):
        name = f"sample{idx:03d}.png"
        Image.new("RGB", (8, 8), (idx, 0, 0)).save(dataset_root / "img0" / name)
        Image.new("RGB", (8, 8), (0, idx, 0)).save(dataset_root / "img1" / name)
        Image.new("RGB", (8, 8), (0, 0, idx)).save(dataset_root / "gt" / name)
    return dataset_root


def _post(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
