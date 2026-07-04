from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from vfieval.job_errors import enrich_job_error


def utc_ts() -> float:
    return time.time()


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True)


def _loads(text: str | None) -> Any:
    if not text:
        return {}
    return json.loads(text)


def _rating_key(score: float) -> str:
    """Canonical 0.25-step histogram key, e.g. 3.0 -> "3.00", 3.25 -> "3.25"."""
    return f"{round(float(score) * 4) / 4:.2f}"


def _combine_output_health(reports: Iterable[dict[str, Any] | None]) -> dict[str, Any] | None:
    valid = [report for report in reports if isinstance(report, dict)]
    if not valid:
        return None
    total_samples = sum(int(report.get("samples") or 0) for report in valid)
    weight_total = sum(int(report.get("samples") or 0) or 1 for report in valid)
    stats: dict[str, dict[str, float | int]] = {}
    for name in ("flowt_0", "flowt_1"):
        weighted_abs_mean = 0.0
        abs_max = 0.0
        nan_count = 0
        for report in valid:
            shard_samples = int(report.get("samples") or 0) or 1
            shard_stats = ((report.get("stats") or {}).get(name) or {})
            weighted_abs_mean += float(shard_stats.get("abs_mean") or 0.0) * shard_samples
            abs_max = max(abs_max, float(shard_stats.get("abs_max") or 0.0))
            nan_count += int(shard_stats.get("nan_count") or 0)
        stats[name] = {
            "abs_mean": float(weighted_abs_mean / weight_total),
            "abs_max": float(abs_max),
            "nan_count": nan_count,
        }
    for name in ("mask0", "mask1"):
        weighted_mean = 0.0
        std = 0.0
        nan_count = 0
        for report in valid:
            shard_samples = int(report.get("samples") or 0) or 1
            shard_stats = ((report.get("stats") or {}).get(name) or {})
            weighted_mean += float(shard_stats.get("mean") or 0.0) * shard_samples
            std = max(std, float(shard_stats.get("std") or 0.0))
            nan_count += int(shard_stats.get("nan_count") or 0)
        stats[name] = {
            "mean": float(weighted_mean / weight_total),
            "std": float(std),
            "nan_count": nan_count,
        }
    flow_flat = all(float(stats[name]["abs_max"]) < 1e-4 for name in ("flowt_0", "flowt_1"))
    mask_flat = all(float(stats[name]["std"]) < 1e-3 for name in ("mask0", "mask1"))
    has_nan = any(int(stats[name]["nan_count"]) > 0 for name in stats)
    warnings: list[str] = []
    seen: set[str] = set()
    for report in valid:
        for warning in report.get("warnings") or []:
            warning_text = str(warning)
            if warning_text in seen:
                continue
            seen.add(warning_text)
            warnings.append(warning_text)
    return {
        "stats": stats,
        "warnings": warnings,
        "flow_flat": flow_flat,
        "mask_flat": mask_flat,
        "has_nan": has_nan,
        "samples": total_samples,
        "shards": len(valid),
    }


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
CREATE INDEX IF NOT EXISTS idx_artifacts_sample ON artifacts(sample_id, kind);

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
CREATE INDEX IF NOT EXISTS idx_metric_results_sample ON metric_results(sample_id, metric_name);

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
    deleted_at REAL,
    artifact_cleaned_at REAL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_status_created ON runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_jobs ON runs(inference_job_id, metric_job_id);

CREATE TABLE IF NOT EXISTS run_jobs (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    shard_index INTEGER NOT NULL DEFAULT 0,
    device TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY(run_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_run_jobs_run_role ON run_jobs(run_id, role, shard_index);
CREATE INDEX IF NOT EXISTS idx_run_jobs_job ON run_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_run_jobs_device ON run_jobs(device);

CREATE TABLE IF NOT EXISTS run_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    username TEXT NOT NULL DEFAULT '',
    rating REAL,
    issue TEXT NOT NULL DEFAULT '',
    video TEXT NOT NULL DEFAULT '',
    track_label TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    checkpoint TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL
);

CREATE INDEX IF NOT EXISTS idx_run_feedback_run ON run_feedback(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_run_feedback_video ON run_feedback(video, model_name, checkpoint);
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
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_jobs (
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                shard_index INTEGER NOT NULL DEFAULT 0,
                device TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                PRIMARY KEY(run_id, job_id)
            );
            CREATE INDEX IF NOT EXISTS idx_run_jobs_run_role ON run_jobs(run_id, role, shard_index);
            CREATE INDEX IF NOT EXISTS idx_run_jobs_job ON run_jobs(job_id);
            CREATE INDEX IF NOT EXISTS idx_run_jobs_device ON run_jobs(device);
            CREATE INDEX IF NOT EXISTS idx_artifacts_sample ON artifacts(sample_id, kind);
            CREATE INDEX IF NOT EXISTS idx_metric_results_sample ON metric_results(sample_id, metric_name);
            CREATE TABLE IF NOT EXISTS run_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                username TEXT NOT NULL DEFAULT '',
                rating REAL,
                issue TEXT NOT NULL DEFAULT '',
                video TEXT NOT NULL DEFAULT '',
                track_label TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL DEFAULT '',
                checkpoint TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_run_feedback_run ON run_feedback(run_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_run_feedback_video ON run_feedback(video, model_name, checkpoint);
            """
        )
        feedback_columns = {row["name"] for row in conn.execute("PRAGMA table_info(run_feedback)").fetchall()}
        for name, definition in {
            "video": "TEXT NOT NULL DEFAULT ''",
            "track_label": "TEXT NOT NULL DEFAULT ''",
            "model_name": "TEXT NOT NULL DEFAULT ''",
            "checkpoint": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "REAL",
        }.items():
            if name not in feedback_columns:
                conn.execute(f"ALTER TABLE run_feedback ADD COLUMN {name} {definition}")
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
        run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        for name, definition in {
            "deleted_at": "REAL",
            "artifact_cleaned_at": "REAL",
        }.items():
            if name not in run_columns:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")

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
                    f"""
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

    def list_samples_by_video(self, run_id: int, video_name: str) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        dataset_id = int(run["dataset_id"])
        video_text = str(video_name or "")
        if not video_text:
            return []
        patterns = [
            f'%"video_name": "{video_text}"%',
            f'%"video_file": "{video_text}"%',
            f'%"compare_group": "{video_text}"%',
        ]
        name_patterns = [
            f"{video_text}__%",
            f"%_{video_text}_%",
        ]
        clauses = ["metadata_json LIKE ?" for _ in patterns] + ["name LIKE ?" for _ in name_patterns]
        rows = self.query(
            f"""
            SELECT *
            FROM samples
            WHERE dataset_id = ?
              AND ({' OR '.join(clauses)})
            ORDER BY name
            """,
            [dataset_id, *patterns, *name_patterns],
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def find_sample_by_video_frame(
        self,
        run_id: int,
        video_name: str,
        frame_index: int,
        track_label: str | None = None,
    ) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        dataset_id = int(run["dataset_id"])
        video_text = str(video_name or "")
        params: list[Any] = [dataset_id, video_text, video_text, int(frame_index), int(frame_index)]
        track_clause = ""
        if track_label:
            track_clause = """
              AND (
                json_extract(metadata_json, '$.compare_track_label') IS NULL
                OR json_extract(metadata_json, '$.compare_track_label') = ?
              )
            """
            params.append(str(track_label))
        rows = self.query(
            f"""
            SELECT *
            FROM samples
            WHERE dataset_id = ?
              AND (
                json_extract(metadata_json, '$.video_name') = ?
                OR json_extract(metadata_json, '$.video_file') = ?
              )
              AND (
                CAST(json_extract(metadata_json, '$.frame_index') AS INTEGER) = ?
                OR CAST(json_extract(metadata_json, '$.sample_index') AS INTEGER) = ?
              )
              {track_clause}
            ORDER BY id
            LIMIT 1
            """,
            params,
        )
        if not rows:
            return None
        row = rows[0]
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def list_run_video_summaries(self, run_id: int, query: str = "") -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        dataset_id = int(run["dataset_id"])
        rows = self.query(
            """
            SELECT
                COALESCE(
                    json_extract(metadata_json, '$.video_name'),
                    json_extract(metadata_json, '$.video_file'),
                    'frames'
                ) AS video_name,
                COALESCE(
                    json_extract(metadata_json, '$.video_file'),
                    json_extract(metadata_json, '$.video_name'),
                    'frames'
                ) AS video_file,
                AVG(CAST(COALESCE(json_extract(metadata_json, '$.fps'), 0) AS REAL)) AS fps,
                COUNT(*) AS sample_count
            FROM samples
            WHERE dataset_id = ?
            GROUP BY video_name, video_file
            ORDER BY video_file, video_name
            """,
            (dataset_id,),
        )
        normalized_query = str(query or "").strip().lower()
        result = []
        for row in rows:
            video_name = str(row.get("video_name") or "frames")
            video_file = str(row.get("video_file") or video_name)
            if normalized_query and normalized_query not in video_name.lower() and normalized_query not in video_file.lower():
                continue
            result.append(
                {
                    "video_name": video_name,
                    "video_file": video_file,
                    "fps": float(row.get("fps") or 0.0),
                    "sample_count": int(row.get("sample_count") or 0),
                }
            )
        return result

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
        create_inference_job: bool = True,
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
            if not create_inference_job:
                return run_id
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
            meta = metadata or {}
            if meta.get("visualize_height") is not None:
                payload["visualize_height"] = meta.get("visualize_height")
            if meta.get("visualize_width") is not None:
                payload["visualize_width"] = meta.get("visualize_width")
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
            conn.execute(
                """
                INSERT INTO run_jobs(run_id, job_id, role, shard_index, device, metadata_json, created_at)
                VALUES (?, ?, 'inference', 0, ?, '{}', ?)
                """,
                (run_id, inference_job_id, device, now),
            )
            return run_id

    def add_run_job(
        self,
        run_id: int,
        role: str,
        payload: dict[str, Any],
        progress_total: int = 0,
        shard_index: int = 0,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (role, _json(payload), int(progress_total), now),
            )
            job_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO run_jobs(run_id, job_id, role, shard_index, device, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, job_id, role, int(shard_index), device, _json(metadata), now),
            )
            if role == "inference":
                conn.execute(
                    """
                    UPDATE runs
                    SET inference_job_id = COALESCE(inference_job_id, ?), updated_at = ?
                    WHERE id = ?
                    """,
                    (job_id, now, run_id),
                )
            elif role == "metric":
                conn.execute(
                    """
                    UPDATE runs
                    SET metric_job_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (job_id, now, run_id),
                )
            return job_id

    def list_run_jobs(self, run_id: int, role: str | None = None) -> list[dict[str, Any]]:
        params: list[Any] = [run_id]
        clause = ""
        if role is not None:
            clause = " AND rj.role = ?"
            params.append(role)
        rows = self.query(
            f"""
            SELECT rj.*, j.kind, j.status, j.payload_json, j.worker_id,
                   j.progress_current, j.progress_total, j.result_json, j.error_json,
                   j.started_at, j.finished_at
            FROM run_jobs rj
            JOIN jobs j ON j.id = rj.job_id
            WHERE rj.run_id = ?{clause}
            ORDER BY rj.role, rj.shard_index, rj.job_id
            """,
            params,
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
            row["payload"] = _loads(row.pop("payload_json"))
            row["result"] = _loads(row.pop("result_json"))
            row["error"] = _loads(row.pop("error_json"))
        return rows

    def next_run_id(self) -> int:
        row = self.get("SELECT seq FROM sqlite_sequence WHERE name = 'runs'")
        if row is not None:
            return int(row["seq"]) + 1
        row = self.get("SELECT MAX(id) AS max_id FROM runs")
        max_id = int(row["max_id"] or 0) if row else 0
        return max_id + 1

    def list_runs(self, limit: int = 100, include_deleted: bool = False) -> list[dict[str, Any]]:
        deleted_clause = "" if include_deleted else "WHERE r.deleted_at IS NULL"
        rows = self.query(
            f"""
            SELECT
                r.*,
                e.name AS experiment_name,
                m.name AS model_name,
                d.name AS dataset_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            {deleted_clause}
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
            LEFT JOIN run_jobs rj ON rj.run_id = r.id
            WHERE r.inference_job_id = ? OR r.metric_job_id = ?
               OR rj.job_id = ?
            """,
            (job_id, job_id, job_id),
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
                INSERT OR IGNORE INTO run_jobs(run_id, job_id, role, shard_index, device, metadata_json, created_at)
                VALUES (?, ?, 'metric', 0, NULL, '{}', ?)
                """,
                (run_id, metric_job_id, now),
            )
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
            # Cancel sibling shard jobs that have not started yet so a worker
            # never claims them once the run is already known to have failed.
            # Already-running shards notice via the run-status check each
            # batch (_raise_if_canceled) and stop themselves.
            conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    error_json = ?,
                    finished_at = ?
                WHERE status = 'queued'
                  AND id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                """,
                (_json({"message": "sibling shard failed the run"}), now, run_id),
            )

    def request_run_cancel(self, run_id: int) -> None:
        now = utc_ts()
        run = self.get_run(run_id)
        inference_job_id = run.get("inference_job_id")
        metric_job_id = run.get("metric_job_id")
        run_job_ids = [int(row["job_id"]) for row in self.list_run_jobs(run_id)]
        job_rows = self.list_run_jobs(run_id)
        has_running_job = any(str(row.get("status") or "") == "running" for row in job_rows)
        with self.connection() as conn:
            if (
                run["status"] in {"queued", "metric_queued", "decoding"}
                and not has_running_job
                and (inference_job_id is not None or metric_job_id is not None or run_job_ids)
            ):
                target_ids = list(dict.fromkeys(
                    [
                        *run_job_ids,
                        *([int(inference_job_id)] if inference_job_id is not None else []),
                        *([int(metric_job_id)] if metric_job_id is not None else []),
                    ]
                ))
                placeholders = ",".join("?" for _ in target_ids)
                conn.execute(
                    f"""
                    UPDATE jobs
                    SET status = 'canceled',
                        error_json = ?,
                        finished_at = ?
                    WHERE id IN ({placeholders}) AND status = 'queued'
                    """,
                    (_json({"message": "用户取消了排队中的 Run"}), now, *target_ids),
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

    def soft_delete_run(self, run_id: int) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET deleted_at = COALESCE(deleted_at, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, run_id),
            )

    def rename_run(self, run_id: int, name: str) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET name = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (name, now, run_id),
            )

    def add_run_feedback(
        self,
        run_id: int,
        username: str,
        rating: float | None,
        issue: str,
        video: str = "",
        track_label: str = "",
        model_name: str = "",
        checkpoint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO run_feedback(
                    run_id, username, rating, issue,
                    video, track_label, model_name, checkpoint,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    username,
                    float(rating) if rating is not None else None,
                    issue,
                    video,
                    track_label,
                    model_name,
                    checkpoint,
                    _json(metadata),
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def update_run_feedback(
        self,
        run_id: int,
        feedback_id: int,
        *,
        username: str | None = None,
        rating: float | None = None,
        issue: str | None = None,
        clear_rating: bool = False,
    ) -> bool:
        """Patch an existing feedback row. Only provided fields change.

        ``clear_rating=True`` explicitly nulls the rating (distinct from leaving
        ``rating`` as ``None`` to mean "unchanged").
        """
        sets: list[str] = []
        params: list[Any] = []
        if username is not None:
            sets.append("username = ?")
            params.append(username)
        if issue is not None:
            sets.append("issue = ?")
            params.append(issue)
        if clear_rating:
            sets.append("rating = NULL")
        elif rating is not None:
            sets.append("rating = ?")
            params.append(float(rating))
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(utc_ts())
        params.extend([feedback_id, run_id])
        with self.connection() as conn:
            cur = conn.execute(
                f"UPDATE run_feedback SET {', '.join(sets)} WHERE id = ? AND run_id = ?",
                tuple(params),
            )
            return cur.rowcount > 0

    def list_run_feedback(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM run_feedback WHERE run_id = ? ORDER BY created_at DESC, id DESC",
            (run_id,),
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json", None))
        return rows

    def delete_run_feedback(self, run_id: int, feedback_id: int) -> bool:
        with self.connection() as conn:
            cur = conn.execute(
                "DELETE FROM run_feedback WHERE id = ? AND run_id = ?",
                (feedback_id, run_id),
            )
            return cur.rowcount > 0

    def list_all_feedback(self, limit: int = 1000, include_deleted: bool = False) -> list[dict[str, Any]]:
        """All feedback joined to its run + dataset context, newest first.

        Runs whose records were deleted are excluded by default so the stats tab
        only reflects live evaluations (matches the #26 "delete run → drop its
        feedback from stats" expectation). Cleaning artifacts cascades the delete
        at the DB level, but a soft-deleted run keeps its rows, so filter here.
        """
        deleted_clause = "" if include_deleted else "WHERE r.deleted_at IS NULL"
        rows = self.query(
            f"""
            SELECT
                f.*,
                r.name AS run_name,
                r.status AS run_status,
                r.deleted_at AS run_deleted_at,
                d.name AS dataset_name
            FROM run_feedback f
            LEFT JOIN runs r ON r.id = f.run_id
            LEFT JOIN datasets d ON d.id = r.dataset_id
            {deleted_clause}
            ORDER BY f.created_at DESC, f.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json", None))
        return rows

    def feedback_stats(
        self,
        *,
        dataset: str | None = None,
        model_name: str | None = None,
        checkpoint: str | None = None,
        video: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate feedback overall and grouped by user/run/video/model/checkpoint.

        Optional filters narrow the population before aggregation so the caller
        can ask e.g. "only this dataset" or "only this checkpoint". Ratings use a
        0.25 step, so the distribution histogram spans the 17 quarter-values from
        1.00 to 5.00.
        """
        entries = self.list_all_feedback(limit=100000)

        def _keep(row: dict[str, Any]) -> bool:
            if dataset and str(row.get("dataset_name") or "") != dataset:
                return False
            if model_name and str(row.get("model_name") or "") != model_name:
                return False
            if checkpoint and str(row.get("checkpoint") or "") != checkpoint:
                return False
            if video and str(row.get("video") or "") != video:
                return False
            return True

        entries = [row for row in entries if _keep(row)]
        total = len(entries)
        ratings = [float(row["rating"]) for row in entries if row.get("rating") is not None]
        issue_count = sum(1 for row in entries if str(row.get("issue") or "").strip())
        distribution = {_rating_key(step / 4): 0 for step in range(4, 21)}
        for score in ratings:
            key = _rating_key(score)
            if key in distribution:
                distribution[key] += 1

        groups: dict[str, dict[Any, dict[str, Any]]] = {
            "by_user": {},
            "by_run": {},
            "by_video": {},
            "by_model": {},
            "by_checkpoint": {},
            "by_model_checkpoint": {},
        }

        def _bucket(dimension: str, key: Any, label: dict[str, Any]) -> dict[str, Any]:
            store = groups[dimension]
            if key not in store:
                store[key] = {**label, "count": 0, "ratings": [], "issues": 0}
            return store[key]

        for row in entries:
            rating = float(row["rating"]) if row.get("rating") is not None else None
            has_issue = bool(str(row.get("issue") or "").strip())
            username = str(row.get("username") or "匿名")
            video_name = str(row.get("video") or "").strip()
            model = str(row.get("model_name") or "").strip()
            ckpt = str(row.get("checkpoint") or "").strip() or "-"
            targets = [
                _bucket("by_user", username, {"username": username}),
            ]
            run_id = row.get("run_id")
            if run_id is not None:
                targets.append(
                    _bucket("by_run", int(run_id), {"run_id": int(run_id), "run_name": row.get("run_name")})
                )
            if video_name:
                targets.append(_bucket("by_video", video_name, {"video": video_name}))
            if model:
                targets.append(_bucket("by_model", model, {"model_name": model}))
                targets.append(_bucket("by_checkpoint", (model, ckpt), {"model_name": model, "checkpoint": ckpt}))
                combo_key = (model, ckpt, video_name)
                targets.append(
                    _bucket(
                        "by_model_checkpoint",
                        combo_key,
                        {"model_name": model, "checkpoint": ckpt, "video": video_name},
                    )
                )
            for bucket in targets:
                bucket["count"] += 1
                if rating is not None:
                    bucket["ratings"].append(rating)
                if has_issue:
                    bucket["issues"] += 1

        def _finalize(bucket: dict[str, Any]) -> dict[str, Any]:
            values = bucket.pop("ratings", [])
            bucket["rating_count"] = len(values)
            bucket["average_rating"] = round(sum(values) / len(values), 2) if values else None
            return bucket

        return {
            "total": total,
            "rating_count": len(ratings),
            "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "issue_count": issue_count,
            "rating_distribution": distribution,
            "by_user": sorted(
                (_finalize(b) for b in groups["by_user"].values()),
                key=lambda item: (-int(item["count"]), str(item["username"])),
            ),
            "by_run": sorted(
                (_finalize(b) for b in groups["by_run"].values()),
                key=lambda item: -int(item["run_id"]),
            ),
            "by_video": sorted(
                (_finalize(b) for b in groups["by_video"].values()),
                key=lambda item: (-int(item["count"]), str(item["video"])),
            ),
            "by_model": sorted(
                (_finalize(b) for b in groups["by_model"].values()),
                key=lambda item: (-int(item["count"]), str(item["model_name"])),
            ),
            "by_checkpoint": sorted(
                (_finalize(b) for b in groups["by_checkpoint"].values()),
                key=lambda item: (str(item["model_name"]), str(item["checkpoint"])),
            ),
            "by_model_checkpoint": sorted(
                (_finalize(b) for b in groups["by_model_checkpoint"].values()),
                key=lambda item: (str(item["video"]), str(item["model_name"]), str(item["checkpoint"])),
            ),
        }

    def feedback_filter_options(self) -> dict[str, list[str]]:
        """Distinct datasets/models/checkpoints/videos present in live feedback.

        Powers the stats-tab filter dropdowns without the frontend having to
        derive them from the full entry list.
        """
        entries = self.list_all_feedback(limit=100000)
        datasets: set[str] = set()
        models: set[str] = set()
        checkpoints: set[str] = set()
        videos: set[str] = set()
        for row in entries:
            if str(row.get("dataset_name") or "").strip():
                datasets.add(str(row["dataset_name"]).strip())
            if str(row.get("model_name") or "").strip():
                models.add(str(row["model_name"]).strip())
            if str(row.get("checkpoint") or "").strip():
                checkpoints.add(str(row["checkpoint"]).strip())
            if str(row.get("video") or "").strip():
                videos.add(str(row["video"]).strip())
        return {
            "datasets": sorted(datasets),
            "models": sorted(models),
            "checkpoints": sorted(checkpoints),
            "videos": sorted(videos),
        }

    def mark_run_artifacts_cleaned(self, run_id: int) -> None:
        now = utc_ts()
        job_ids = self.run_inference_job_ids(run_id)
        with self.connection() as conn:
            if job_ids:
                placeholders = ",".join("?" for _ in job_ids)
                conn.execute(f"DELETE FROM artifacts WHERE job_id IN ({placeholders})", tuple(job_ids))
            # Feedback describes the visual results; once those are cleaned the
            # scores are orphaned (the #26 case: results deleted but ratings
            # lingered), so drop them alongside the artifacts.
            conn.execute("DELETE FROM run_feedback WHERE run_id = ?", (run_id,))
            conn.execute(
                """
                UPDATE runs
                SET artifact_cleaned_at = ?,
                    artifact_summary_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, _json({"total": 0, "by_kind": {}}), now, run_id),
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

    def run_inference_job_ids(self, run_id: int) -> list[int]:
        job_ids = [int(row["job_id"]) for row in self.list_run_jobs(run_id, "inference")]
        if job_ids:
            return job_ids
        run = self.get_run(run_id)
        return [int(run["inference_job_id"])] if run.get("inference_job_id") is not None else []

    def update_run_progress_from_jobs(self, run_id: int, status: str | None = None) -> None:
        rows = self.list_run_jobs(run_id, "inference")
        current = sum(int(row.get("progress_current") or 0) for row in rows)
        total = sum(int(row.get("progress_total") or 0) for row in rows)
        self.update_run_progress(run_id, current, total, status)

    def summarize_run_artifacts(self, run_id: int) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for job_id in self.run_inference_job_ids(run_id):
            summary = self.summarize_artifacts(job_id)
            for kind, count in (summary.get("by_kind") or {}).items():
                by_kind[kind] = by_kind.get(kind, 0) + int(count)
        return {"total": sum(by_kind.values()), "by_kind": by_kind}

    def maybe_complete_multi_run_inference(self, run_id: int) -> bool:
        jobs = self.list_run_jobs(run_id, "inference")
        if not jobs:
            return False
        if any(job["status"] == "failed" for job in jobs):
            failed = next(job for job in jobs if job["status"] == "failed")
            self.fail_run(run_id, enrich_job_error(failed, failed.get("error") or {}))
            return True
        if any(job["status"] == "canceled" for job in jobs):
            self.cancel_run(run_id, {"message": "inference shard canceled"})
            return True
        if any(job["status"] != "completed" for job in jobs):
            self.update_run_progress_from_jobs(run_id, "running")
            return False

        run = self.get_run(run_id)
        output_health = _combine_output_health((job.get("result") or {}).get("output_health") for job in jobs)
        result = {
            "samples": sum(int(job.get("result", {}).get("samples") or 0) for job in jobs),
            "output_dir": (run.get("metadata") or {}).get("output_dir"),
            "shards": [
                {
                    "job_id": int(job["job_id"]),
                    "device": job.get("device"),
                    "status": job["status"],
                    "result": job.get("result") or {},
                }
                for job in jobs
            ],
        }
        if output_health is not None:
            result["output_health"] = output_health
        artifact_summary = self.summarize_run_artifacts(run_id)
        metrics = list(run.get("metrics") or [])
        if metrics:
            metric_payload = {
                "run_id": run_id,
                "dataset_id": int(run["dataset_id"]),
                "inference_job_ids": [int(job["job_id"]) for job in jobs],
                "inference_job_id": int(jobs[0]["job_id"]),
                "metric_names": metrics,
                "metric_device": str(run.get("device") or "cpu"),
            }
            metric_job_id = self.add_run_job(
                run_id,
                "metric",
                metric_payload,
                progress_total=0,
                shard_index=0,
                device=None,
                metadata={"source": (run.get("metadata") or {}).get("execution_mode") or "multi"},
            )
            self.complete_run_inference(run_id, result, artifact_summary, "metric_queued")
            self.set_run_metric_job(run_id, metric_job_id)
        else:
            self.complete_run_inference(run_id, result, artifact_summary, "completed")
        return True

    def list_run_artifacts(self, run_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for job_id in self.run_inference_job_ids(run_id):
            artifacts.extend(self.list_artifacts(job_id=int(job_id), kind=kind))
        return artifacts

    def list_run_video_artifacts(
        self,
        run_id: int,
        video_name: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        job_ids = self.run_inference_job_ids(run_id)
        if not job_ids:
            return []
        placeholders = ",".join("?" for _ in job_ids)
        clauses = ["job_id IN (" + placeholders + ")", "sample_id IS NULL"]
        params: list[Any] = list(job_ids)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if video_name is not None:
            clauses.append("json_extract(metadata_json, '$.video_name') = ?")
            params.append(str(video_name))
        rows = self.query(
            f"""
            SELECT *
            FROM artifacts
            WHERE {' AND '.join(clauses)}
            ORDER BY kind, id
            """,
            params,
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def list_run_metrics(self, run_id: int) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        for job_id in self.run_inference_job_ids(run_id):
            metrics.extend(self.list_metric_results(inference_job_id=int(job_id)))
        return metrics

    def list_run_video_metrics(self, run_id: int, video_name: str | None = None) -> list[dict[str, Any]]:
        job_ids = self.run_inference_job_ids(run_id)
        if not job_ids:
            return []
        placeholders = ",".join("?" for _ in job_ids)
        rows = self.query(
            f"""
            SELECT *
            FROM metric_results
            WHERE sample_id IS NULL
              AND inference_job_id IN ({placeholders})
            ORDER BY metric_name, id
            """,
            job_ids,
        )
        result = []
        for row in rows:
            row["details"] = _loads(row.pop("details_json"))
            if video_name is not None and str((row.get("details") or {}).get("video_name") or "") != str(video_name):
                continue
            result.append(row)
        return result

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

    def claim_next_job(self, worker_id: str, kinds: list[str], device_filter: str | None = None) -> dict[str, Any] | None:
        if not kinds:
            return None
        placeholders = ",".join("?" for _ in kinds)
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if device_filter:
                row = conn.execute(
                    f"""
                    SELECT j.*
                    FROM jobs j
                    JOIN run_jobs rj ON rj.job_id = j.id
                    WHERE j.status = 'queued'
                      AND j.kind IN ({placeholders})
                      AND rj.device = ?
                    ORDER BY j.created_at, j.id
                    LIMIT 1
                    """,
                    (*tuple(kinds), device_filter),
                ).fetchone()
            else:
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

    def update_job_progress(
        self,
        job_id: int,
        current: int,
        total: int | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        with self.connection() as conn:
            if total is None and result is None:
                conn.execute("UPDATE jobs SET progress_current = ? WHERE id = ?", (current, job_id))
            elif total is None:
                conn.execute(
                    "UPDATE jobs SET progress_current = ?, result_json = ? WHERE id = ?",
                    (current, _json(result), job_id),
                )
            elif result is None:
                conn.execute(
                    "UPDATE jobs SET progress_current = ?, progress_total = ? WHERE id = ?",
                    (current, total, job_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET progress_current = ?, progress_total = ?, result_json = ? WHERE id = ?",
                    (current, total, _json(result), job_id),
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

    def list_artifacts_by_sample(self, sample_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        clauses = ["sample_id = ?"]
        params: list[Any] = [sample_id]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        rows = self.query(
            f"SELECT * FROM artifacts WHERE {' AND '.join(clauses)} ORDER BY kind, id",
            params,
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def list_artifacts_by_samples(
        self,
        sample_ids: Iterable[int],
        job_ids: Iterable[int] | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        ids = [int(sample_id) for sample_id in sample_ids]
        if not ids:
            return []
        job_filter: list[int] | None = None
        if job_ids is not None:
            job_filter = [int(job_id) for job_id in job_ids]
            if not job_filter:
                return []
        fixed_param_count = (len(job_filter) if job_filter is not None else 0) + (1 if kind is not None else 0)
        chunk_size = max(1, 900 - fixed_param_count)
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(ids), chunk_size):
            chunk = ids[offset : offset + chunk_size]
            clauses = [f"sample_id IN ({','.join('?' for _ in chunk)})"]
            params: list[Any] = list(chunk)
            if job_filter is not None:
                clauses.append(f"job_id IN ({','.join('?' for _ in job_filter)})")
                params.extend(job_filter)
            if kind is not None:
                clauses.append("kind = ?")
                params.append(kind)
            rows.extend(
                self.query(
                    f"SELECT * FROM artifacts WHERE {' AND '.join(clauses)} ORDER BY sample_id, kind, id",
                    params,
                )
            )
        rows.sort(key=lambda row: (int(row["sample_id"]), str(row["kind"]), int(row["id"])))
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

    def list_metrics_by_sample(self, sample_id: int, metric_name: str | None = None) -> list[dict[str, Any]]:
        clauses = ["sample_id = ?"]
        params: list[Any] = [sample_id]
        if metric_name is not None:
            clauses.append("metric_name = ?")
            params.append(metric_name)
        rows = self.query(
            f"SELECT * FROM metric_results WHERE {' AND '.join(clauses)} ORDER BY metric_name, id",
            params,
        )
        for row in rows:
            row["details"] = _loads(row.pop("details_json"))
        return rows

    def list_metrics_by_samples(
        self,
        sample_ids: Iterable[int],
        inference_job_ids: Iterable[int] | None = None,
        metric_name: str | None = None,
    ) -> list[dict[str, Any]]:
        ids = [int(sample_id) for sample_id in sample_ids]
        if not ids:
            return []
        job_filter: list[int] | None = None
        if inference_job_ids is not None:
            job_filter = [int(job_id) for job_id in inference_job_ids]
            if not job_filter:
                return []
        fixed_param_count = (len(job_filter) if job_filter is not None else 0) + (1 if metric_name is not None else 0)
        chunk_size = max(1, 900 - fixed_param_count)
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(ids), chunk_size):
            chunk = ids[offset : offset + chunk_size]
            clauses = [f"sample_id IN ({','.join('?' for _ in chunk)})"]
            params: list[Any] = list(chunk)
            if job_filter is not None:
                clauses.append(f"inference_job_id IN ({','.join('?' for _ in job_filter)})")
                params.extend(job_filter)
            if metric_name is not None:
                clauses.append("metric_name = ?")
                params.append(metric_name)
            rows.extend(
                self.query(
                    f"SELECT * FROM metric_results WHERE {' AND '.join(clauses)} ORDER BY sample_id, metric_name, id",
                    params,
                )
            )
        rows.sort(key=lambda row: (int(row["sample_id"]), str(row["metric_name"]), int(row["id"])))
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
