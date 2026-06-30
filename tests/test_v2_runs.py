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

                run_worker(db, workspace, WorkerOptions(role="metric", once=True, worker_id="v2-metric"))
                completed = _get(base_url, f"/api/runs/{run_id}")
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["metric_summary"]["cgvqm"]["unavailable"], 2)

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
