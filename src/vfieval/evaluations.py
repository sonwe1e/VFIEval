from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import statistics
import uuid
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any
from urllib.parse import quote

from vfieval.compare_inputs import inspect_compare_path, validate_strict_alignment, validate_strict_decoded_alignment
from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts
from vfieval.media_assets import get_asset, resolve_asset_path


QUALITY_REASONS = {
    "sharpness",
    "temporal_stability",
    "ghosting",
    "artifacts",
    "motion_naturalness",
}
CONFIDENCE_VALUES = {"", "low", "medium", "high"}
METRIC_DIRECTIONS = {
    "vmaf": "higher_is_better",
    "lpips_vit_patch": "lower_is_better",
    "lpips_convnext": "lower_is_better",
    "cgvqm": "lower_is_better",
}


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True, ensure_ascii=False)


def _loads(text: str | None) -> Any:
    return json.loads(text) if text else {}


def upsert_evaluator(db: Database, body: dict[str, Any]) -> dict[str, Any]:
    evaluator_id = str(body.get("evaluator_id") or body.get("id") or uuid.uuid4()).strip()
    display_name = str(body.get("display_name") or "").strip()
    if not display_name:
        raise ValueError("evaluator display_name is required")
    if len(evaluator_id) > 128:
        raise ValueError("evaluator id is too long")
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO evaluators(id, display_name, metadata_json, created_at, updated_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at,
                last_seen_at = excluded.last_seen_at
            """,
            (evaluator_id, display_name[:120], _json(body.get("metadata") or {}), now, now, now),
        )
    return get_evaluator(db, evaluator_id)


def get_evaluator(db: Database, evaluator_id: str) -> dict[str, Any]:
    row = db.get("SELECT * FROM evaluators WHERE id = ?", (str(evaluator_id),))
    if row is None:
        raise KeyError(f"evaluator {evaluator_id} not found")
    row["metadata"] = _loads(row.pop("metadata_json", None))
    return row


def create_campaign(db: Database, body: dict[str, Any], campaign_type: str = "campaign") -> dict[str, Any]:
    name = str(body.get("name") or "").strip()
    if not name:
        raise ValueError("campaign name is required")
    if campaign_type not in {"campaign", "adhoc"}:
        raise ValueError("campaign_type must be campaign or adhoc")
    target_votes = int(body.get("target_votes") or 3)
    if target_votes < 1 or target_votes > 1000:
        raise ValueError("target_votes must be between 1 and 1000")
    seed = int(body.get("seed") if body.get("seed") is not None else random.SystemRandom().randrange(2**31))
    now = utc_ts()
    with db.connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO evaluation_campaigns(
                name, campaign_type, status, target_votes, seed, metadata_json, created_at, updated_at
            ) VALUES (?, ?, 'draft', ?, ?, ?, ?, ?)
            """,
            (name[:240], campaign_type, target_votes, seed, _json(body.get("metadata") or {}), now, now),
        )
        campaign_id = int(cur.lastrowid)
    return get_campaign(db, campaign_id)


def get_campaign(db: Database, campaign_id: int) -> dict[str, Any]:
    row = db.get("SELECT * FROM evaluation_campaigns WHERE id = ?", (int(campaign_id),))
    if row is None:
        raise KeyError(f"evaluation campaign {campaign_id} not found")
    row["metadata"] = _loads(row.pop("metadata_json", None))
    counts = db.get(
        """
        SELECT COUNT(DISTINCT c.id) AS candidates,
               COUNT(DISTINCT t.id) AS tasks,
               COUNT(DISTINCT v.id) AS votes
        FROM evaluation_campaigns ec
        LEFT JOIN evaluation_candidates c ON c.campaign_id = ec.id
        LEFT JOIN evaluation_tasks t ON t.campaign_id = ec.id
        LEFT JOIN evaluation_votes v ON v.task_id = t.id
        WHERE ec.id = ?
        """,
        (int(campaign_id),),
    ) or {}
    row.update({key: int(counts.get(key) or 0) for key in ("candidates", "tasks", "votes")})
    return row


def list_campaigns(db: Database) -> list[dict[str, Any]]:
    rows = db.query("SELECT id FROM evaluation_campaigns ORDER BY id DESC")
    return [get_campaign(db, int(row["id"])) for row in rows]


def add_candidate(
    db: Database,
    workspace: WorkspaceConfig,
    campaign_id: int,
    body: dict[str, Any],
) -> dict[str, Any]:
    campaign = get_campaign(db, campaign_id)
    if campaign["status"] != "draft":
        raise ValueError("candidates can only be changed while a campaign is draft")
    reference_asset_id = int(body.get("reference_asset_id") or 0)
    asset_id = int(body.get("asset_id") or body.get("candidate_asset_id") or 0)
    reference, _reference_path = resolve_asset_path(db, workspace, reference_asset_id, role="reference")
    candidate, _candidate_path = resolve_asset_path(db, workspace, asset_id, role="distorted")
    video_name = str(body.get("video_name") or candidate.get("provenance", {}).get("video_name") or candidate["display_name"]).strip()
    if not video_name:
        raise ValueError("candidate video_name is required")
    provenance = candidate.get("provenance") or {}
    label = str(body.get("label") or candidate["display_name"]).strip()
    model = str(body.get("model_name") or provenance.get("model_name") or "").strip()
    checkpoint = str(body.get("checkpoint") or provenance.get("checkpoint") or "").strip()
    now = utc_ts()
    try:
        with db.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO evaluation_candidates(
                    campaign_id, reference_asset_id, asset_id, video_name, label_snapshot,
                    model_snapshot, checkpoint_snapshot, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(campaign_id), reference_asset_id, asset_id, video_name[:240], label[:240],
                    model[:240], checkpoint[:500],
                    _json({"reference_label": reference["display_name"], **(body.get("metadata") or {})}), now,
                ),
            )
            candidate_id = int(cur.lastrowid)
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise ValueError("candidate already exists for this campaign video") from exc
        raise
    return get_candidate(db, candidate_id)


def get_candidate(db: Database, candidate_id: int) -> dict[str, Any]:
    row = db.get("SELECT * FROM evaluation_candidates WHERE id = ?", (int(candidate_id),))
    if row is None:
        raise KeyError(f"evaluation candidate {candidate_id} not found")
    row["metadata"] = _loads(row.pop("metadata_json", None))
    return row


def list_candidates(db: Database, campaign_id: int) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT * FROM evaluation_candidates WHERE campaign_id = ? ORDER BY video_name, id",
        (int(campaign_id),),
    )
    for row in rows:
        row["metadata"] = _loads(row.pop("metadata_json", None))
    return rows


def publish_campaign(db: Database, workspace: WorkspaceConfig, campaign_id: int) -> dict[str, Any]:
    campaign = get_campaign(db, campaign_id)
    if campaign["status"] == "published":
        return campaign
    if campaign["status"] != "draft":
        raise ValueError("only draft campaigns can be published")
    candidates = list_candidates(db, campaign_id)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(int(candidate["reference_asset_id"]), str(candidate["video_name"]))].append(candidate)
    if not grouped or any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("each campaign video requires at least two candidates")
    alignments: dict[tuple[int, int], dict[str, Any]] = {}
    for (reference_asset_id, _video_name), rows in grouped.items():
        reference_asset, reference_path = resolve_asset_path(db, workspace, reference_asset_id, role="reference")
        reference_info = inspect_compare_path(workspace, reference_path)
        if reference_asset.get("fps") is not None:
            reference_info["fps"] = float(reference_asset["fps"])
        from vfieval.datasets import _load_compare_source_frames

        reference_frames, decoded_reference_fps, reference_timestamps = _load_compare_source_frames(
            db, workspace, reference_path, f"campaign_{campaign_id}_reference"
        )
        reference_fps = reference_asset.get("fps") or decoded_reference_fps
        for candidate in rows:
            distorted_asset, distorted_path = resolve_asset_path(db, workspace, int(candidate["asset_id"]), role="distorted")
            distorted_info = inspect_compare_path(workspace, distorted_path)
            if distorted_asset.get("fps") is not None:
                distorted_info["fps"] = float(distorted_asset["fps"])
            alignments[(reference_asset_id, int(candidate["asset_id"]))] = validate_strict_alignment(
                reference_info, distorted_info
            )
            distorted_frames, decoded_distorted_fps, distorted_timestamps = _load_compare_source_frames(
                db,
                workspace,
                distorted_path,
                f"campaign_{campaign_id}_candidate_{int(candidate['id'])}",
            )
            validate_strict_decoded_alignment(
                reference_frames,
                distorted_frames,
                float(reference_fps) if reference_fps is not None else None,
                float(distorted_asset.get("fps") or decoded_distorted_fps)
                if (distorted_asset.get("fps") or decoded_distorted_fps) is not None
                else None,
                reference_timestamps,
                distorted_timestamps,
            )
    now = utc_ts()
    with db.connection() as conn:
        for (reference_asset_id, video_name), rows in grouped.items():
            for left, right in combinations(sorted(rows, key=lambda row: int(row["id"])), 2):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO evaluation_tasks(
                        campaign_id, reference_asset_id, candidate_a_id, candidate_b_id,
                        video_name, state, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'ready', ?, ?)
                    """,
                    (
                        int(campaign_id), reference_asset_id, int(left["id"]), int(right["id"]),
                        video_name,
                        _json({
                            "alignment_a": alignments[(reference_asset_id, int(left["asset_id"]))],
                            "alignment_b": alignments[(reference_asset_id, int(right["asset_id"]))],
                        }),
                        now,
                    ),
                )
        conn.execute(
            "UPDATE evaluation_campaigns SET status = 'published', updated_at = ? WHERE id = ?",
            (now, int(campaign_id)),
        )
    return get_campaign(db, campaign_id)


def close_campaign(db: Database, campaign_id: int) -> dict[str, Any]:
    campaign = get_campaign(db, campaign_id)
    if campaign["status"] == "closed":
        return campaign
    if campaign["status"] != "published":
        raise ValueError("only a published campaign can be closed")
    with db.connection() as conn:
        conn.execute(
            "UPDATE evaluation_campaigns SET status = 'closed', updated_at = ? WHERE id = ?",
            (utc_ts(), int(campaign_id)),
        )
    return get_campaign(db, campaign_id)


def create_adhoc_task(db: Database, workspace: WorkspaceConfig, body: dict[str, Any]) -> dict[str, Any]:
    reference_asset_id = int(body.get("reference_asset_id") or 0)
    raw_candidates = body.get("candidate_asset_ids") or []
    candidate_ids = [int(value) for value in raw_candidates]
    if len(candidate_ids) != 2 or candidate_ids[0] == candidate_ids[1]:
        raise ValueError("adhoc evaluation requires exactly two distinct candidate_asset_ids")
    video_name = str(body.get("video_name") or "adhoc").strip()
    campaign = create_campaign(
        db,
        {"name": body.get("name") or f"Adhoc · {video_name}", "target_votes": body.get("target_votes") or 1},
        campaign_type="adhoc",
    )
    for asset_id in candidate_ids:
        add_candidate(
            db,
            workspace,
            int(campaign["id"]),
            {"reference_asset_id": reference_asset_id, "asset_id": asset_id, "video_name": video_name},
        )
    publish_campaign(db, workspace, int(campaign["id"]))
    task = db.get("SELECT id FROM evaluation_tasks WHERE campaign_id = ?", (int(campaign["id"]),))
    return {"campaign": get_campaign(db, int(campaign["id"])), "task_id": int((task or {})["id"])}


def _task_row(db: Database, task_id: int) -> dict[str, Any]:
    row = db.get(
        """
        SELECT t.*, ec.seed, ec.target_votes, ec.status AS campaign_status,
               ca.asset_id AS asset_a_id, cb.asset_id AS asset_b_id
        FROM evaluation_tasks t
        JOIN evaluation_campaigns ec ON ec.id = t.campaign_id
        JOIN evaluation_candidates ca ON ca.id = t.candidate_a_id
        JOIN evaluation_candidates cb ON cb.id = t.candidate_b_id
        WHERE t.id = ?
        """,
        (int(task_id),),
    )
    if row is None:
        raise KeyError(f"evaluation task {task_id} not found")
    row["metadata"] = _loads(row.pop("metadata_json", None))
    return row


def presentation_for(db: Database, task_id: int, evaluator_id: str) -> dict[str, Any]:
    get_evaluator(db, evaluator_id)
    task = _task_row(db, task_id)
    digest = hashlib.sha256(f"{task_id}:{evaluator_id}:{task['seed']}".encode("utf-8")).digest()
    swap = bool(digest[0] & 1)
    left_asset_id = int(task["asset_b_id"] if swap else task["asset_a_id"])
    right_asset_id = int(task["asset_a_id"] if swap else task["asset_b_id"])
    return {
        "task_id": int(task_id),
        "evaluator_id": str(evaluator_id),
        "reference_asset_id": int(task["reference_asset_id"]),
        "left_asset_id": left_asset_id,
        "right_asset_id": right_asset_id,
        "swapped": swap,
    }


def next_task(db: Database, campaign_id: int, evaluator_id: str) -> dict[str, Any]:
    evaluator = get_evaluator(db, evaluator_id)
    campaign = get_campaign(db, campaign_id)
    if campaign["status"] != "published":
        raise ValueError("campaign is not published")
    row = db.get(
        """
        SELECT t.id,
               COUNT(v.id) AS vote_count
        FROM evaluation_tasks t
        LEFT JOIN evaluation_votes v ON v.task_id = t.id
        WHERE t.campaign_id = ?
          AND t.state = 'ready'
          AND NOT EXISTS (
              SELECT 1 FROM evaluation_votes mine
              WHERE mine.task_id = t.id AND mine.evaluator_id = ?
          )
        GROUP BY t.id
        HAVING COUNT(v.id) < ?
        ORDER BY COUNT(v.id), t.id
        LIMIT 1
        """,
        (int(campaign_id), str(evaluator_id), int(campaign["target_votes"])),
    )
    if row is None:
        return {"campaign_id": int(campaign_id), "complete": True, "task": None}
    task_id = int(row["id"])
    task = _task_row(db, task_id)
    presentation = presentation_for(db, task_id, evaluator_id)
    media_assets = {
        side: get_asset(db, int(presentation[f"{side}_asset_id"]))
        for side in ("reference", "left", "right")
    }
    evaluator_query = quote(str(evaluator_id), safe="")
    return {
        "campaign_id": int(campaign_id),
        "complete": False,
        "task": {
            "id": task_id,
            "video_name": task["video_name"],
            "reference_url": f"/api/evaluation-tasks/{task_id}/media/reference?evaluator_id={evaluator_query}",
            "left_url": f"/api/evaluation-tasks/{task_id}/media/left?evaluator_id={evaluator_query}",
            "right_url": f"/api/evaluation-tasks/{task_id}/media/right?evaluator_id={evaluator_query}",
            "reference_media_kind": media_assets["reference"]["media_kind"],
            "left_media_kind": media_assets["left"]["media_kind"],
            "right_media_kind": media_assets["right"]["media_kind"],
            "frame_count": min(int(asset.get("frame_count") or 0) for asset in media_assets.values()),
            "quality_reasons": sorted(QUALITY_REASONS),
        },
        "evaluator": {"id": evaluator["id"], "display_name": evaluator["display_name"]},
    }


def task_media_asset_id(db: Database, task_id: int, side: str, evaluator_id: str) -> int:
    presentation = presentation_for(db, task_id, evaluator_id)
    mapping = {
        "reference": presentation["reference_asset_id"],
        "left": presentation["left_asset_id"],
        "right": presentation["right_asset_id"],
    }
    if side not in mapping:
        raise ValueError("evaluation media side must be reference, left, or right")
    return int(mapping[side])


def submit_vote(db: Database, task_id: int, evaluator_id: str, body: dict[str, Any]) -> dict[str, Any]:
    task = _task_row(db, task_id)
    if task["campaign_status"] != "published" or task["state"] != "ready":
        raise ValueError("evaluation task is not accepting votes")
    choice = str(body.get("choice") or "").strip()
    if choice not in {"left", "right", "tie"}:
        raise ValueError("vote choice must be left, right, or tie")
    reasons = [str(value) for value in (body.get("reasons") or [])]
    if any(reason not in QUALITY_REASONS for reason in reasons):
        raise ValueError("vote contains an unsupported quality reason")
    reasons = list(dict.fromkeys(reasons))
    confidence = str(body.get("confidence") or "").strip()
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("confidence must be low, medium, high, or blank")
    presentation = presentation_for(db, task_id, evaluator_id)
    preferred_asset_id = None
    if choice == "left":
        preferred_asset_id = int(presentation["left_asset_id"])
    elif choice == "right":
        preferred_asset_id = int(presentation["right_asset_id"])
    duration = body.get("duration_ms")
    duration_ms = max(0, int(duration)) if duration not in {None, ""} else None
    note = str(body.get("note") or "").strip()[:4000]
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO evaluation_votes(
                task_id, evaluator_id, choice, preferred_asset_id, reasons_json,
                confidence, note, duration_ms, presentation_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, evaluator_id) DO UPDATE SET
                choice = excluded.choice,
                preferred_asset_id = excluded.preferred_asset_id,
                reasons_json = excluded.reasons_json,
                confidence = excluded.confidence,
                note = excluded.note,
                duration_ms = excluded.duration_ms,
                presentation_json = excluded.presentation_json,
                updated_at = excluded.updated_at
            """,
            (
                int(task_id), str(evaluator_id), choice, preferred_asset_id, _json(reasons),
                confidence, note, duration_ms, _json(presentation), now, now,
            ),
        )
    row = db.get(
        "SELECT * FROM evaluation_votes WHERE task_id = ? AND evaluator_id = ?",
        (int(task_id), str(evaluator_id)),
    )
    assert row is not None
    row["reasons"] = _loads(row.pop("reasons_json", None))
    row.pop("presentation_json", None)
    return row


def campaign_analysis(
    db: Database,
    campaign_id: int,
    bootstrap_samples: int = 1000,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    campaign = get_campaign(db, campaign_id)
    candidates = list_candidates(db, campaign_id)
    filters = {key: str(value).strip() for key, value in (filters or {}).items() if str(value).strip()}
    task_clauses = ["t.campaign_id = ?"]
    task_params: list[Any] = [int(campaign_id)]
    if filters.get("video"):
        task_clauses.append("t.video_name = ?")
        task_params.append(filters["video"])
    if filters.get("model"):
        task_clauses.append(
            "EXISTS (SELECT 1 FROM evaluation_candidates fc WHERE fc.id IN (t.candidate_a_id, t.candidate_b_id) AND fc.model_snapshot = ?)"
        )
        task_params.append(filters["model"])
    if filters.get("checkpoint"):
        task_clauses.append(
            "EXISTS (SELECT 1 FROM evaluation_candidates fc WHERE fc.id IN (t.candidate_a_id, t.candidate_b_id) AND fc.checkpoint_snapshot = ?)"
        )
        task_params.append(filters["checkpoint"])
    if filters.get("collection_id"):
        task_clauses.append(
            "EXISTS (SELECT 1 FROM evaluation_candidates fc JOIN media_assets ma ON ma.id = fc.asset_id WHERE fc.id IN (t.candidate_a_id, t.candidate_b_id) AND ma.collection_id = ?)"
        )
        task_params.append(int(filters["collection_id"]))
    task_where = " AND ".join(task_clauses)
    tasks = db.query(
        f"""
        SELECT t.*, ca.asset_id AS asset_a_id, cb.asset_id AS asset_b_id
        FROM evaluation_tasks t
        JOIN evaluation_candidates ca ON ca.id = t.candidate_a_id
        JOIN evaluation_candidates cb ON cb.id = t.candidate_b_id
        WHERE {task_where} ORDER BY t.id
        """,
        task_params,
    )
    vote_clauses = list(task_clauses)
    vote_params = list(task_params)
    if filters.get("evaluator_id"):
        vote_clauses.append("v.evaluator_id = ?")
        vote_params.append(filters["evaluator_id"])
    votes = db.query(
        f"""
        SELECT v.*, t.video_name
        FROM evaluation_votes v
        JOIN evaluation_tasks t ON t.id = v.task_id
        WHERE {' AND '.join(vote_clauses)} ORDER BY v.id
        """,
        vote_params,
    )
    task_by_id = {int(task["id"]): task for task in tasks}
    candidate_by_asset = {int(row["asset_id"]): row for row in candidates}
    observations = _vote_observations(votes, task_by_id)
    overall = _rank_observations(
        observations,
        candidate_by_asset,
        seed=int(campaign["seed"]),
        bootstrap_samples=max(0, min(5000, int(bootstrap_samples))),
    )
    videos = sorted({str(task["video_name"]) for task in tasks})
    by_video = {
        video: _rank_observations(
            [row for row in observations if row["video_name"] == video],
            candidate_by_asset,
            seed=int(campaign["seed"]) ^ int(hashlib.sha256(video.encode()).hexdigest()[:8], 16),
            bootstrap_samples=max(0, min(1000, int(bootstrap_samples))),
        )
        for video in videos
    }
    target_votes = int(campaign["target_votes"])
    votes_by_task = Counter(int(row["task_id"]) for row in votes)
    completed_tasks = sum(1 for task in tasks if votes_by_task[int(task["id"])] >= target_votes)
    reasons = Counter()
    evaluators = Counter()
    agreement_values: list[float] = []
    choices_by_task: dict[int, Counter[str]] = defaultdict(Counter)
    for vote in votes:
        for reason in _loads(vote.get("reasons_json")):
            reasons[str(reason)] += 1
        evaluators[str(vote["evaluator_id"])] += 1
        choices_by_task[int(vote["task_id"])][str(vote["choice"])] += 1
    for counts in choices_by_task.values():
        total = sum(counts.values())
        if total:
            agreement_values.append(max(counts.values()) / total)
    evaluator_names: dict[str, str] = {}
    if evaluators:
        placeholders = ",".join("?" for _ in evaluators)
        evaluator_names = {
            str(row["id"]): str(row["display_name"])
            for row in db.query(
                f"SELECT id, display_name FROM evaluators WHERE id IN ({placeholders})",
                tuple(evaluators),
            )
        }
    objective = _objective_analysis(
        db,
        {
            int(asset_id)
            for task in tasks
            for asset_id in (task["asset_a_id"], task["asset_b_id"])
        },
    )
    return {
        "campaign": campaign,
        "coverage": {
            "tasks": len(tasks),
            "completed_tasks": completed_tasks,
            "target_votes_per_task": target_votes,
            "complete": bool(tasks) and completed_tasks == len(tasks),
            "provisional": not tasks or completed_tasks != len(tasks),
        },
        "human": overall,
        "by_video": by_video,
        "quality_reasons": dict(sorted(reasons.items())),
        "evaluator_votes": [
            {"evaluator_id": key, "evaluator_name": evaluator_names.get(key, key), "votes": value}
            for key, value in evaluators.most_common()
        ],
        "agreement_rate": round(statistics.mean(agreement_values), 4) if agreement_values else None,
        "objective": objective,
        "cross_analysis": _cross_analysis(overall, objective),
        "filters": filters,
    }


def _vote_observations(votes: list[dict[str, Any]], tasks: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for vote in votes:
        task = tasks[int(vote["task_id"])]
        a = int(task["asset_a_id"])
        b = int(task["asset_b_id"])
        preferred = vote.get("preferred_asset_id")
        if preferred is None:
            score_a, score_b = 0.5, 0.5
        elif int(preferred) == a:
            score_a, score_b = 1.0, 0.0
        else:
            score_a, score_b = 0.0, 1.0
        result.append({"a": a, "b": b, "score_a": score_a, "score_b": score_b, "video_name": task["video_name"]})
    return result


def _rank_observations(
    observations: list[dict[str, Any]],
    candidates: dict[int, dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    asset_ids = sorted({value for row in observations for value in (int(row["a"]), int(row["b"]))})
    scores = _bradley_terry(asset_ids, observations)
    intervals: dict[int, tuple[float, float]] = {asset_id: (scores.get(asset_id, 0.0), scores.get(asset_id, 0.0)) for asset_id in asset_ids}
    if observations and bootstrap_samples:
        rng = random.Random(seed)
        samples: dict[int, list[float]] = {asset_id: [] for asset_id in asset_ids}
        for _ in range(bootstrap_samples):
            resampled = [observations[rng.randrange(len(observations))] for _row in observations]
            result = _bradley_terry(asset_ids, resampled)
            for asset_id in asset_ids:
                samples[asset_id].append(result.get(asset_id, 0.0))
        for asset_id, values in samples.items():
            values.sort()
            intervals[asset_id] = (_percentile(values, 0.025), _percentile(values, 0.975))
    pair_stats: dict[tuple[int, int], dict[str, Any]] = {}
    for row in observations:
        pair = (min(int(row["a"]), int(row["b"])), max(int(row["a"]), int(row["b"])))
        stats = pair_stats.setdefault(pair, {"votes": 0, "wins_first": 0.0, "wins_second": 0.0, "ties": 0})
        stats["votes"] += 1
        first_score = float(row["score_a"] if int(row["a"]) == pair[0] else row["score_b"])
        second_score = 1.0 - first_score
        stats["wins_first"] += first_score
        stats["wins_second"] += second_score
        if first_score == 0.5:
            stats["ties"] += 1
    ranking = []
    for asset_id in asset_ids:
        candidate = candidates.get(asset_id) or {}
        low, high = intervals[asset_id]
        ranking.append(
            {
                "asset_id": asset_id,
                "label": candidate.get("label_snapshot") or f"asset-{asset_id}",
                "model_name": candidate.get("model_snapshot") or "",
                "checkpoint": candidate.get("checkpoint_snapshot") or "",
                "score": round(scores.get(asset_id, 0.0), 6),
                "ci95": [round(low, 6), round(high, 6)],
            }
        )
    ranking.sort(key=lambda row: (-float(row["score"]), str(row["label"])))
    head_to_head = []
    matrix: dict[str, dict[str, float | None]] = {
        str(asset_id): {str(other): (0.5 if asset_id == other else None) for other in asset_ids}
        for asset_id in asset_ids
    }
    for pair, stats in sorted(pair_stats.items()):
        votes = int(stats["votes"])
        first_rate = float(stats["wins_first"]) / votes if votes else 0.0
        second_rate = float(stats["wins_second"]) / votes if votes else 0.0
        head_to_head.append(
            {
                "asset_a_id": pair[0],
                "asset_b_id": pair[1],
                **stats,
                "win_rate_a": round(first_rate, 6),
                "win_rate_b": round(second_rate, 6),
            }
        )
        matrix[str(pair[0])][str(pair[1])] = round(first_rate, 6)
        matrix[str(pair[1])][str(pair[0])] = round(second_rate, 6)
    return {
        "vote_count": len(observations),
        "ranking": ranking,
        "head_to_head": head_to_head,
        "head_to_head_matrix": {"asset_ids": asset_ids, "values": matrix},
    }


def _bradley_terry(asset_ids: list[int], observations: list[dict[str, Any]]) -> dict[int, float]:
    if not asset_ids:
        return {}
    ability = {asset_id: 1.0 for asset_id in asset_ids}
    wins = {asset_id: 1e-9 for asset_id in asset_ids}
    comparisons: dict[tuple[int, int], int] = Counter()
    for row in observations:
        a, b = int(row["a"]), int(row["b"])
        wins[a] += float(row["score_a"])
        wins[b] += float(row["score_b"])
        comparisons[(min(a, b), max(a, b))] += 1
    for _ in range(200):
        updated: dict[int, float] = {}
        for asset_id in asset_ids:
            denominator = 0.0
            for other in asset_ids:
                if other == asset_id:
                    continue
                count = comparisons.get((min(asset_id, other), max(asset_id, other)), 0)
                if count:
                    denominator += count / max(ability[asset_id] + ability[other], 1e-12)
            updated[asset_id] = wins[asset_id] / denominator if denominator else ability[asset_id]
        geometric = math.exp(sum(math.log(max(value, 1e-12)) for value in updated.values()) / len(updated))
        updated = {key: max(value / geometric, 1e-12) for key, value in updated.items()}
        delta = max(abs(math.log(updated[key]) - math.log(ability[key])) for key in asset_ids)
        ability = updated
        if delta < 1e-10:
            break
    total = sum(ability.values()) or 1.0
    return {asset_id: ability[asset_id] / total for asset_id in asset_ids}


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _objective_analysis(db: Database, candidate_asset_ids: set[int]) -> dict[str, Any]:
    if not candidate_asset_ids:
        return {"metrics": []}
    placeholders = ",".join("?" for _ in candidate_asset_ids)
    rows = db.query(
        f"""
        SELECT mr.metric_name, mr.status, mr.value, mab.distorted_asset_id
        FROM metric_asset_bindings mab
        JOIN metric_results mr ON mr.id = mab.metric_result_id
        WHERE mab.distorted_asset_id IN ({placeholders})
        """,
        tuple(sorted(candidate_asset_ids)),
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    status_counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (str(row["metric_name"]), int(row["distorted_asset_id"]))
        status_counts[key][str(row["status"])] += 1
        if row["status"] == "completed" and row.get("value") is not None:
            grouped[key].append(row)
    metrics = []
    for key in sorted(set(grouped) | set(status_counts)):
        name, asset_id = key
        values = sorted(float(row["value"]) for row in grouped.get(key, []))
        metrics.append(
            {
                "metric_name": name,
                "direction": METRIC_DIRECTIONS.get(name, "lower_is_better"),
                "asset_id": asset_id,
                "status_counts": dict(status_counts[key]),
                "count": len(values),
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "p10": _percentile(values, 0.10) if values else None,
                "p90": _percentile(values, 0.90) if values else None,
            }
        )
    return {"metrics": metrics}


def _cross_analysis(human: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    human_scores = {int(row["asset_id"]): float(row["score"]) for row in human.get("ranking") or []}
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in objective.get("metrics") or []:
        if row.get("mean") is not None:
            grouped[str(row["metric_name"])][int(row["asset_id"])] = float(row["mean"])
    results = []
    for metric_name, values in sorted(grouped.items()):
        common = sorted(set(values) & set(human_scores))
        if len(common) < 5:
            continue
        human_ranks = _ranks([human_scores[key] for key in common])
        direction = 1 if METRIC_DIRECTIONS.get(metric_name) == "higher_is_better" else -1
        metric_ranks = _ranks([values[key] * direction for key in common])
        correlation = _pearson(human_ranks, metric_ranks)
        human_order = sorted(common, key=lambda key: human_scores[key], reverse=True)
        metric_order = sorted(common, key=lambda key: values[key] * direction, reverse=True)
        conflicts = [asset_id for asset_id in common if abs(human_order.index(asset_id) - metric_order.index(asset_id)) >= 2]
        results.append({"metric_name": metric_name, "spearman": correlation, "candidate_count": len(common), "conflict_asset_ids": conflicts})
    return {"metrics": results}


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2 + 1
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)
    )
    return round(numerator / denominator, 6) if denominator else None


def analysis_csv(analysis: dict[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["section", "video", "rank", "asset_id", "label", "model", "checkpoint", "score", "ci95_low", "ci95_high"])
    sections: list[tuple[str, str, dict[str, Any]]] = [("overall", "", analysis.get("human") or {})]
    sections.extend(("video", video, payload) for video, payload in (analysis.get("by_video") or {}).items())
    for section, video, payload in sections:
        for rank, row in enumerate(payload.get("ranking") or [], 1):
            ci = row.get("ci95") or [None, None]
            writer.writerow(
                [section, video, rank, row.get("asset_id"), row.get("label"), row.get("model_name"), row.get("checkpoint"), row.get("score"), ci[0], ci[1]]
            )
    return output.getvalue().encode("utf-8-sig")


def campaign_export(db: Database, campaign_id: int) -> dict[str, Any]:
    campaign = get_campaign(db, campaign_id)
    candidates = list_candidates(db, campaign_id)
    tasks = db.query(
        "SELECT * FROM evaluation_tasks WHERE campaign_id = ? ORDER BY id",
        (int(campaign_id),),
    )
    for task in tasks:
        task["metadata"] = _loads(task.pop("metadata_json", None))
    votes = db.query(
        """
        SELECT v.*, e.display_name AS evaluator_name, t.video_name
        FROM evaluation_votes v
        JOIN evaluation_tasks t ON t.id = v.task_id
        JOIN evaluators e ON e.id = v.evaluator_id
        WHERE t.campaign_id = ?
        ORDER BY v.id
        """,
        (int(campaign_id),),
    )
    for vote in votes:
        vote["reasons"] = _loads(vote.pop("reasons_json", None))
        vote["presentation"] = _loads(vote.pop("presentation_json", None))
    return {
        "campaign": campaign,
        "candidates": candidates,
        "tasks": tasks,
        "votes": votes,
        "analysis": campaign_analysis(db, campaign_id),
    }


def campaign_export_csv(export: dict[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "row_type", "campaign_id", "video", "task_id", "candidate_id", "asset_id",
            "label", "model", "checkpoint", "evaluator_id", "evaluator_name", "choice",
            "reasons", "confidence", "note", "score", "ci95_low", "ci95_high",
            "metric", "direction", "count", "mean", "median", "p10", "p90",
        ]
    )
    campaign_id = int((export.get("campaign") or {}).get("id") or 0)
    for row in export.get("candidates") or []:
        writer.writerow(
            [
                "candidate", campaign_id, row.get("video_name"), "", row.get("id"), row.get("asset_id"),
                row.get("label_snapshot"), row.get("model_snapshot"), row.get("checkpoint_snapshot"),
            ]
        )
    for row in export.get("votes") or []:
        writer.writerow(
            [
                "vote", campaign_id, row.get("video_name"), row.get("task_id"), "", row.get("preferred_asset_id"),
                "", "", "", row.get("evaluator_id"), row.get("evaluator_name"), row.get("choice"),
                "|".join(row.get("reasons") or []), row.get("confidence"), row.get("note"),
            ]
        )
    for row in ((export.get("analysis") or {}).get("human") or {}).get("ranking") or []:
        ci = row.get("ci95") or [None, None]
        writer.writerow(
            [
                "ranking", campaign_id, "", "", "", row.get("asset_id"), row.get("label"),
                row.get("model_name"), row.get("checkpoint"), "", "", "", "", "", "",
                row.get("score"), ci[0], ci[1],
            ]
        )
    for row in ((export.get("analysis") or {}).get("objective") or {}).get("metrics") or []:
        writer.writerow(
            [
                "metric", campaign_id, "", "", "", row.get("asset_id"), "", "", "", "", "", "", "", "", "",
                "", "", "", row.get("metric_name"), row.get("direction"), row.get("count"), row.get("mean"),
                row.get("median"), row.get("p10"), row.get("p90"),
            ]
        )
    return output.getvalue().encode("utf-8-sig")
