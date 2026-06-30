from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


def utc_ts() -> float:
    return time.time()


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True)


def _loads(text: str | None) -> Any:
    if not text:
        return {}
    return json.loads(text)


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    adapter TEXT NOT NULL,
    checkpoint_path TEXT,
    input_height INTEGER NOT NULL,
    input_width INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    root_path TEXT NOT NULL,
    has_gt INTEGER NOT NULL DEFAULT 1,
    source_type TEXT NOT NULL DEFAULT 'frames',
    decode_mode TEXT NOT NULL DEFAULT 'frames',
    decoded_root_path TEXT,
    video_count INTEGER NOT NULL DEFAULT 0,
    frame_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    img0_path TEXT NOT NULL,
    img1_path TEXT NOT NULL,
    gt_path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    UNIQUE(dataset_id, name)
);

CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    capabilities_json TEXT NOT NULL DEFAULT '{}',
    last_seen_at REAL NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
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

CREATE INDEX IF NOT EXISTS idx_jobs_status_kind ON jobs(status, kind, created_at);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    sample_id INTEGER REFERENCES samples(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_job_kind ON artifacts(job_id, kind);

CREATE TABLE IF NOT EXISTS metric_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    inference_job_id INTEGER NOT NULL,
    sample_id INTEGER REFERENCES samples(id) ON DELETE SET NULL,
    metric_name TEXT NOT NULL,
    status TEXT NOT NULL,
    value REAL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metric_results_inference ON metric_results(inference_job_id, metric_name);

CREATE TABLE IF NOT EXISTS metric_cache (
    cache_key TEXT PRIMARY KEY,
    metric_name TEXT NOT NULL,
    status TEXT NOT NULL,
    value REAL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    height INTEGER NOT NULL,
    width INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    device TEXT NOT NULL,
    precision TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '[]',
    inference_job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    metric_job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    artifact_summary_json TEXT NOT NULL DEFAULT '{}',
    metric_summary_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_status_created ON runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_jobs ON runs(inference_job_id, metric_job_id);
"""


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)
            now = utc_ts()
            conn.execute(
                """
                INSERT OR IGNORE INTO experiments(name, description, metadata_json, created_at)
                VALUES ('Default', 'Default experiment', '{}', ?)
                """,
                (now,),
            )

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)").fetchall()}
        dataset_columns = {
            "source_type": "TEXT NOT NULL DEFAULT 'frames'",
            "decode_mode": "TEXT NOT NULL DEFAULT 'frames'",
            "decoded_root_path": "TEXT",
            "video_count": "INTEGER NOT NULL DEFAULT 0",
            "frame_count": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, definition in dataset_columns.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE datasets ADD COLUMN {name} {definition}")

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def register_model(
        self,
        name: str,
        adapter: str,
        checkpoint_path: str | None,
        input_height: int,
        input_width: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO models(name, adapter, checkpoint_path, input_height, input_width, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, adapter, checkpoint_path, input_height, input_width, _json(metadata), now),
            )
            return int(cur.lastrowid)

    def upsert_model(
        self,
        name: str,
        adapter: str,
        checkpoint_path: str | None,
        input_height: int,
        input_width: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        existing = self.get("SELECT id FROM models WHERE name = ?", (name,))
        now = utc_ts()
        if existing:
            model_id = int(existing["id"])
            with self.connection() as conn:
                conn.execute(
                    """
                    UPDATE models
                    SET adapter = ?,
                        checkpoint_path = ?,
                        input_height = ?,
                        input_width = ?,
                        metadata_json = ?
                    WHERE id = ?
                    """,
                    (adapter, checkpoint_path, input_height, input_width, _json(metadata), model_id),
                )
            return model_id
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO models(name, adapter, checkpoint_path, input_height, input_width, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, adapter, checkpoint_path, input_height, input_width, _json(metadata), now),
            )
            return int(cur.lastrowid)

    def list_models(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM models ORDER BY id")
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def get_model(self, model_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM models WHERE id = ?", (model_id,))
        if row is None:
            raise KeyError(f"model {model_id} not found")
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def create_dataset(
        self,
        name: str,
        root_path: str,
        has_gt: bool,
        source_type: str = "frames",
        decode_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        source_type = source_type or "frames"
        decode_mode = decode_mode or ("frames" if source_type == "frames" else ("video_gt_triplets" if has_gt else "video_pairs"))
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO datasets(
                    name, root_path, has_gt, source_type, decode_mode, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    str(Path(root_path).resolve()),
                    int(has_gt),
                    source_type,
                    decode_mode,
                    _json(metadata),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def upsert_dataset(
        self,
        name: str,
        root_path: str,
        has_gt: bool,
        source_type: str = "frames",
        decode_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        source_type = source_type or "frames"
        decode_mode = decode_mode or ("frames" if source_type == "frames" else ("video_gt_triplets" if has_gt else "video_pairs"))
        existing = self.get("SELECT id FROM datasets WHERE name = ?", (name,))
        if existing:
            dataset_id = int(existing["id"])
            with self.connection() as conn:
                conn.execute(
                    """
                    UPDATE datasets
                    SET root_path = ?,
                        has_gt = ?,
                        source_type = ?,
                        decode_mode = ?,
                        metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        str(Path(root_path).resolve()),
                        int(has_gt),
                        source_type,
                        decode_mode,
                        _json(metadata),
                        dataset_id,
                    ),
                )
            return dataset_id
        return self.create_dataset(name, root_path, has_gt, source_type, decode_mode, metadata)

    def list_datasets(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM datasets ORDER BY id")
        for row in rows:
            row["has_gt"] = bool(row["has_gt"])
            row["metadata"] = _loads(row.pop("metadata_json"))
            row["sample_count"] = self.count_samples(int(row["id"]))
        return rows

    def get_dataset(self, dataset_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if row is None:
            raise KeyError(f"dataset {dataset_id} not found")
        row["has_gt"] = bool(row["has_gt"])
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def update_dataset_scan_info(
        self,
        dataset_id: int,
        decoded_root_path: str | None,
        video_count: int,
        frame_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        dataset = self.get_dataset(dataset_id)
        merged_metadata = {**(dataset.get("metadata") or {}), **(metadata or {})}
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE datasets
                SET decoded_root_path = ?,
                    video_count = ?,
                    frame_count = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    str(Path(decoded_root_path).resolve()) if decoded_root_path else None,
                    int(video_count),
                    int(frame_count),
                    _json(merged_metadata),
                    dataset_id,
                ),
            )

    def add_sample(
        self,
        dataset_id: int,
        name: str,
        img0_path: str,
        img1_path: str,
        gt_path: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT OR REPLACE INTO samples(dataset_id, name, img0_path, img1_path, gt_path, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    name,
                    str(Path(img0_path).resolve()),
                    str(Path(img1_path).resolve()),
                    str(Path(gt_path).resolve()) if gt_path else None,
                    _json(metadata),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def list_samples(self, dataset_id: int) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM samples WHERE dataset_id = ? ORDER BY name", (dataset_id,))
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def clear_samples(self, dataset_id: int) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM samples WHERE dataset_id = ?", (dataset_id,))

    def count_samples(self, dataset_id: int) -> int:
        row = self.get("SELECT COUNT(*) AS count FROM samples WHERE dataset_id = ?", (dataset_id,))
        return int(row["count"] if row else 0)

    def get_sample(self, sample_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM samples WHERE id = ?", (sample_id,))
        if row is None:
            raise KeyError(f"sample {sample_id} not found")
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def create_experiment(
        self,
        name: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments(name, description, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, description, _json(metadata), now),
            )
            return int(cur.lastrowid)

    def get_default_experiment_id(self) -> int:
        row = self.get("SELECT id FROM experiments WHERE name = 'Default'")
        if row is not None:
            return int(row["id"])
        return self.create_experiment("Default", "Default experiment")

    def list_experiments(self) -> list[dict[str, Any]]:
        rows = self.query(
            """
            SELECT e.*, COUNT(r.id) AS run_count
            FROM experiments e
            LEFT JOIN runs r ON r.experiment_id = e.id
            GROUP BY e.id
            ORDER BY e.id
            """
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        if row is None:
            raise KeyError(f"experiment {experiment_id} not found")
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def create_run(
        self,
        name: str,
        model_id: int,
        dataset_id: int,
        height: int,
        width: int,
        batch_size: int,
        device: str,
        precision: str,
        metrics: list[str],
        experiment_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        experiment_id = experiment_id or self.get_default_experiment_id()
        now = utc_ts()
        progress_total = self.count_samples(dataset_id)
        with self.connection() as conn:
            run_cur = conn.execute(
                """
                INSERT INTO runs(
                    experiment_id, name, model_id, dataset_id, height, width, batch_size,
                    device, precision, metrics_json, status, progress_total,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    name,
                    model_id,
                    dataset_id,
                    height,
                    width,
                    batch_size,
                    device,
                    precision,
                    _json(metrics),
                    progress_total,
                    _json(metadata),
                    now,
                    now,
                ),
            )
            run_id = int(run_cur.lastrowid)
            payload = {
                "run_id": run_id,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "height": height,
                "width": width,
                "batch_size": batch_size,
                "device": device,
                "precision": precision,
                "metrics": metrics,
            }
            job_cur = conn.execute(
                """
                INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                VALUES ('inference', 'queued', ?, ?, ?)
                """,
                (_json(payload), progress_total, now),
            )
            inference_job_id = int(job_cur.lastrowid)
            conn.execute(
                """
                UPDATE runs
                SET inference_job_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (inference_job_id, now, run_id),
            )
            return run_id

    def next_run_id(self) -> int:
        row = self.get("SELECT seq FROM sqlite_sequence WHERE name = 'runs'")
        if row is not None:
            return int(row["seq"]) + 1
        row = self.get("SELECT MAX(id) AS max_id FROM runs")
        max_id = int(row["max_id"] or 0) if row else 0
        return max_id + 1

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.query(
            """
            SELECT
                r.*,
                e.name AS experiment_name,
                m.name AS model_name,
                d.name AS dataset_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            self._decode_run(row)
        return rows

    def get_run(self, run_id: int) -> dict[str, Any]:
        row = self.get(
            """
            SELECT
                r.*,
                e.name AS experiment_name,
                m.name AS model_name,
                d.name AS dataset_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            WHERE r.id = ?
            """,
            (run_id,),
        )
        if row is None:
            raise KeyError(f"run {run_id} not found")
        self._decode_run(row)
        return row

    def get_run_by_job(self, job_id: int) -> dict[str, Any] | None:
        row = self.get(
            """
            SELECT
                r.*,
                e.name AS experiment_name,
                m.name AS model_name,
                d.name AS dataset_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            WHERE r.inference_job_id = ? OR r.metric_job_id = ?
            """,
            (job_id, job_id),
        )
        if row is None:
            return None
        self._decode_run(row)
        return row

    def update_run_progress(
        self,
        run_id: int,
        current: int,
        total: int | None = None,
        status: str | None = None,
    ) -> None:
        now = utc_ts()
        with self.connection() as conn:
            if total is None and status is None:
                conn.execute(
                    "UPDATE runs SET progress_current = ?, updated_at = ? WHERE id = ?",
                    (current, now, run_id),
                )
            elif total is None:
                conn.execute(
                    "UPDATE runs SET progress_current = ?, status = ?, updated_at = ? WHERE id = ?",
                    (current, status, now, run_id),
                )
            elif status is None:
                conn.execute(
                    "UPDATE runs SET progress_current = ?, progress_total = ?, updated_at = ? WHERE id = ?",
                    (current, total, now, run_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE runs
                    SET progress_current = ?, progress_total = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (current, total, status, now, run_id),
                )

    def mark_run_started(self, run_id: int, status: str = "running") -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ?
                """,
                (status, now, now, run_id),
            )

    def set_run_metric_job(self, run_id: int, metric_job_id: int) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET metric_job_id = ?, status = 'metric_queued', updated_at = ?
                WHERE id = ?
                """,
                (metric_job_id, now, run_id),
            )

    def complete_run_inference(
        self,
        run_id: int,
        result: dict[str, Any],
        artifact_summary: dict[str, Any],
        status: str,
    ) -> None:
        now = utc_ts()
        finished_at = now if status == "completed" else None
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?,
                    result_json = ?,
                    artifact_summary_json = ?,
                    finished_at = COALESCE(?, finished_at),
                    updated_at = ?
                WHERE id = ?
                """,
                (status, _json(result), _json(artifact_summary), finished_at, now, run_id),
            )

    def complete_run_metrics(self, run_id: int, metric_summary: dict[str, Any]) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = 'completed',
                    metric_summary_json = ?,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (_json(metric_summary), now, now, run_id),
            )

    def fail_run(self, run_id: int, error: dict[str, Any]) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = 'failed',
                    error_json = ?,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (_json(error), now, now, run_id),
            )

    def request_run_cancel(self, run_id: int) -> None:
        now = utc_ts()
        run = self.get_run(run_id)
        inference_job_id = run.get("inference_job_id")
        with self.connection() as conn:
            if run["status"] == "queued" and inference_job_id is not None:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'canceled',
                        error_json = ?,
                        finished_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (_json({"message": "用户取消了排队中的 Run"}), now, int(inference_job_id)),
                )
                conn.execute(
                    """
                    UPDATE runs
                    SET status = 'canceled',
                        error_json = ?,
                        finished_at = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (_json({"message": "用户取消了排队中的 Run"}), now, now, run_id),
                )
            elif run["status"] not in {"completed", "failed", "canceled"}:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = 'cancel_requested',
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, run_id),
                )

    def cancel_run(self, run_id: int, error: dict[str, Any] | None = None) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = 'canceled',
                    error_json = ?,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (_json(error or {"message": "Run 已取消"}), now, now, run_id),
            )

    def cancel_job(self, job_id: int, error: dict[str, Any] | None = None) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    error_json = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (_json(error or {"message": "Job 已取消"}), now, job_id),
            )

    def summarize_artifacts(self, job_id: int) -> dict[str, Any]:
        rows = self.query(
            """
            SELECT kind, COUNT(*) AS count
            FROM artifacts
            WHERE job_id = ?
            GROUP BY kind
            ORDER BY kind
            """,
            (job_id,),
        )
        by_kind = {row["kind"]: int(row["count"]) for row in rows}
        return {"total": sum(by_kind.values()), "by_kind": by_kind}

    def list_run_artifacts(self, run_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        inference_job_id = run.get("inference_job_id")
        if inference_job_id is None:
            return []
        return self.list_artifacts(job_id=int(inference_job_id), kind=kind)

    def list_run_metrics(self, run_id: int) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        inference_job_id = run.get("inference_job_id")
        if inference_job_id is None:
            return []
        return self.list_metric_results(inference_job_id=int(inference_job_id))

    def list_run_samples(self, run_id: int) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        samples = self.list_samples(int(run["dataset_id"]))
        artifacts = self.list_run_artifacts(run_id)
        metrics = self.list_run_metrics(run_id)
        artifacts_by_sample: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for artifact in artifacts:
            sample_id = artifact.get("sample_id")
            if sample_id is None:
                continue
            artifacts_by_sample.setdefault(int(sample_id), {}).setdefault(artifact["kind"], []).append(artifact)
        metrics_by_sample: dict[int, list[dict[str, Any]]] = {}
        for metric in metrics:
            sample_id = metric.get("sample_id")
            if sample_id is None:
                continue
            metrics_by_sample.setdefault(int(sample_id), []).append(metric)
        for sample in samples:
            sample_id = int(sample["id"])
            sample["artifacts"] = artifacts_by_sample.get(sample_id, {})
            sample["metrics"] = metrics_by_sample.get(sample_id, [])
        return samples

    def register_worker(self, worker_id: str, role: str, capabilities: dict[str, Any] | None = None) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO workers(id, role, capabilities_json, last_seen_at, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    role=excluded.role,
                    capabilities_json=excluded.capabilities_json,
                    last_seen_at=excluded.last_seen_at
                """,
                (worker_id, role, _json(capabilities), now, now),
            )

    def list_workers(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM workers ORDER BY last_seen_at DESC")
        for row in rows:
            row["capabilities"] = _loads(row.pop("capabilities_json"))
        return rows

    def get_worker(self, worker_id: str) -> dict[str, Any]:
        row = self.get("SELECT * FROM workers WHERE id = ?", (worker_id,))
        if row is None:
            raise KeyError(f"worker {worker_id} not found")
        row["capabilities"] = _loads(row.pop("capabilities_json"))
        return row

    def touch_worker(self, worker_id: str, capabilities: dict[str, Any] | None = None) -> None:
        worker = self.get_worker(worker_id)
        self.register_worker(worker_id, worker["role"], capabilities or worker.get("capabilities") or {})

    def create_job(self, kind: str, payload: dict[str, Any], progress_total: int = 0) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (kind, _json(payload), progress_total, now),
            )
            return int(cur.lastrowid)

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        for row in rows:
            self._decode_job(row)
        return rows

    def get_job(self, job_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise KeyError(f"job {job_id} not found")
        self._decode_job(row)
        return row

    def claim_next_job(self, worker_id: str, kinds: list[str]) -> dict[str, Any] | None:
        if not kinds:
            return None
        placeholders = ",".join("?" for _ in kinds)
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE status = 'queued' AND kind IN ({placeholders})
                ORDER BY created_at, id
                LIMIT 1
                """,
                tuple(kinds),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running', worker_id = ?, started_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (worker_id, now, int(row["id"])),
            )
            conn.commit()
        claimed = self.get_job(int(row["id"]))
        return claimed

    def update_job_progress(self, job_id: int, current: int, total: int | None = None) -> None:
        with self.connection() as conn:
            if total is None:
                conn.execute("UPDATE jobs SET progress_current = ? WHERE id = ?", (current, job_id))
            else:
                conn.execute(
                    "UPDATE jobs SET progress_current = ?, progress_total = ? WHERE id = ?",
                    (current, total, job_id),
                )

    def complete_job(self, job_id: int, result: dict[str, Any] | None = None) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed', result_json = ?, finished_at = ?
                WHERE id = ?
                """,
                (_json(result), now, job_id),
            )

    def fail_job(self, job_id: int, error: dict[str, Any]) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', error_json = ?, finished_at = ?
                WHERE id = ?
                """,
                (_json(error), now, job_id),
            )

    def touch_job(self, job_id: int) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET started_at = COALESCE(started_at, ?)
                WHERE id = ? AND status = 'running'
                """,
                (now, job_id),
            )

    def add_artifact(
        self,
        job_id: int,
        sample_id: int | None,
        kind: str,
        path: str,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO artifacts(job_id, sample_id, kind, path, mime_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, sample_id, kind, str(Path(path).resolve()), mime_type, _json(metadata), now),
            )
            return int(cur.lastrowid)

    def list_artifacts(self, job_id: int | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(job_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.query(f"SELECT * FROM artifacts{where} ORDER BY id", params)
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def add_metric_result(
        self,
        job_id: int,
        inference_job_id: int,
        sample_id: int | None,
        metric_name: str,
        status: str,
        value: float | None,
        details: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO metric_results(job_id, inference_job_id, sample_id, metric_name, status, value, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, inference_job_id, sample_id, metric_name, status, value, _json(details), now),
            )
            return int(cur.lastrowid)

    def list_metric_results(self, inference_job_id: int | None = None) -> list[dict[str, Any]]:
        if inference_job_id is None:
            rows = self.query("SELECT * FROM metric_results ORDER BY id")
        else:
            rows = self.query(
                "SELECT * FROM metric_results WHERE inference_job_id = ? ORDER BY id",
                (inference_job_id,),
            )
        for row in rows:
            row["details"] = _loads(row.pop("details_json"))
        return rows

    def get_metric_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = self.get("SELECT * FROM metric_cache WHERE cache_key = ?", (cache_key,))
        if row is None:
            return None
        row["details"] = _loads(row.pop("details_json"))
        return row

    def set_metric_cache(
        self,
        cache_key: str,
        metric_name: str,
        status: str,
        value: float | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO metric_cache(cache_key, metric_name, status, value, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    metric_name=excluded.metric_name,
                    status=excluded.status,
                    value=excluded.value,
                    details_json=excluded.details_json,
                    created_at=excluded.created_at
                """,
                (cache_key, metric_name, status, value, _json(details), now),
            )

    @staticmethod
    def _decode_job(row: dict[str, Any]) -> None:
        row["payload"] = _loads(row.pop("payload_json"))
        row["result"] = _loads(row.pop("result_json"))
        row["error"] = _loads(row.pop("error_json"))

    @staticmethod
    def _decode_run(row: dict[str, Any]) -> None:
        row["metrics"] = _loads(row.pop("metrics_json"))
        row["artifact_summary"] = _loads(row.pop("artifact_summary_json"))
        row["metric_summary"] = _loads(row.pop("metric_summary_json"))
        row["result"] = _loads(row.pop("result_json"))
        row["error"] = _loads(row.pop("error_json"))
        row["metadata"] = _loads(row.pop("metadata_json"))
