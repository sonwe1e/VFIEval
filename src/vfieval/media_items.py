from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts


ITEM_STATES = {"ready", "unavailable", "deleted"}
MEMBER_STATES = ITEM_STATES
MEMBER_ROLES = {
    "canonical_gt",
    "model_pred",
    "external_pred",
    "compare_snapshot",
    "evaluation_gt",
    "evaluation_pred",
}
PRODUCER_KINDS = {
    "source",
    "model_inference",
    "external",
    "video_compare",
    "evaluation_package",
}
BINDING_ROLES = {"source", "pred_output", "compare_gt", "compare_pred"}
MEMBER_ASSET_ROLES = {
    "canonical_gt": "gt",
    "evaluation_gt": "gt",
    "model_pred": "pred",
    "external_pred": "pred",
    "compare_snapshot": "pred",
    "evaluation_pred": "pred",
}
INITIAL_BINDING_MEMBER_ROLES = {
    "source": {"canonical_gt"},
    "pred_output": {"model_pred"},
    "compare_gt": {"canonical_gt"},
    "compare_pred": {"model_pred", "external_pred"},
}
ACTIVE_BINDING_MEMBER_ROLES = {
    **INITIAL_BINDING_MEMBER_ROLES,
    "compare_pred": {"model_pred", "external_pred", "compare_snapshot"},
}


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True, ensure_ascii=False)


def _loads(text: str | None) -> Any:
    return json.loads(text) if text else {}


def _asset_state(asset: dict[str, Any]) -> str:
    if asset.get("deleted_at") is not None or str(asset.get("state") or "") == "deleted":
        return "deleted"
    return "ready" if str(asset.get("state") or "") == "ready" else "unavailable"


def _run_type(run: dict[str, Any]) -> str:
    metadata = run.get("metadata")
    if metadata is None:
        metadata = _loads(run.get("metadata_json"))
    return str((metadata or {}).get("run_type") or "model_inference")


def _require_member_asset_role(member_role: str, asset_role: str) -> None:
    expected = MEMBER_ASSET_ROLES.get(str(member_role))
    if expected is None:
        raise ValueError(f"unsupported media item member role: {member_role}")
    if str(asset_role) != expected:
        raise ValueError(f"{member_role} requires an asset with role {expected}")


def _require_binding_slot(binding_role: str, slot: str) -> None:
    normalized = str(slot or "")
    if binding_role == "pred_output" and normalized:
        raise ValueError("Pred output bindings require an empty slot")
    if binding_role in {"compare_gt", "compare_pred"} and not normalized.strip():
        raise ValueError("Compare input bindings require a non-empty alignment slot")


def _require_binding_member_role(
    binding_role: str,
    member_role: str,
    *,
    active: bool,
) -> None:
    role_map = ACTIVE_BINDING_MEMBER_ROLES if active else INITIAL_BINDING_MEMBER_ROLES
    allowed = role_map.get(str(binding_role))
    if allowed is None:
        raise ValueError(f"unsupported Run media item binding role: {binding_role}")
    if str(member_role) not in allowed:
        position = "active" if active else "original"
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(
            f"{position} {binding_role} binding member must have role {allowed_text}"
        )


def _decode_item(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = _loads(result.pop("metadata_json", None))
    for key in ("frame_count", "width", "height", "size_bytes"):
        if key in result:
            result[key] = int(result.get(key) or 0)
    return result


def _decode_member(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["reusable_as_pred"] = bool(result.get("reusable_as_pred"))
    result["temporal_mapping"] = _loads(result.pop("temporal_mapping_json", None))
    result["spatial_origin"] = _loads(result.pop("spatial_origin_json", None))
    result["metadata"] = _loads(result.pop("metadata_json", None))
    for key in ("frame_count", "width", "height", "size_bytes"):
        if key in result:
            result[key] = int(result.get(key) or 0)
    return result


def get_media_item(db: Database, item_id: int, *, include_deleted: bool = False) -> dict[str, Any]:
    deleted_clause = "" if include_deleted else "AND mi.deleted_at IS NULL"
    row = db.get(
        f"""
        SELECT mi.*, c.name AS collection_name, c.slug AS collection_slug,
               a.source_key AS canonical_source_key,
               a.source_kind AS canonical_source_kind,
               a.display_name AS canonical_display_name,
               a.storage_path AS canonical_storage_path,
               a.state AS canonical_asset_state,
               a.frame_count, a.width, a.height, a.fps, a.size_bytes,
               a.content_sha256 AS canonical_content_sha256
        FROM media_items mi
        JOIN media_collections c ON c.id = mi.collection_id
        JOIN media_assets a ON a.id = mi.canonical_gt_asset_id
        WHERE mi.id = ? {deleted_clause}
        """,
        (int(item_id),),
    )
    if row is None:
        raise KeyError(f"media item {item_id} not found")
    return _decode_item(row)


def get_media_item_member(
    db: Database,
    member_id: int,
    *,
    include_deleted: bool = False,
) -> dict[str, Any]:
    deleted_clause = "" if include_deleted else "AND mim.deleted_at IS NULL"
    row = db.get(
        f"""
        SELECT mim.*, a.source_key, a.source_kind, a.media_kind,
               a.role AS asset_role, a.display_name, a.original_name,
               a.state AS asset_state, a.storage_path, a.mime_type,
               a.frame_count, a.width, a.height, a.fps, a.size_bytes,
               a.content_sha256, r.name AS run_name, r.status AS run_status,
               r.deleted_at AS run_deleted_at,
               r.artifact_cleaned_at AS run_artifact_cleaned_at
        FROM media_item_members mim
        JOIN media_assets a ON a.id = mim.asset_id
        LEFT JOIN runs r ON r.id = mim.producer_run_id
        WHERE mim.id = ? {deleted_clause}
        """,
        (int(member_id),),
    )
    if row is None:
        raise KeyError(f"media item member {member_id} not found")
    return _decode_member(row)


def resolve_item_reference(
    db: Database,
    workspace: WorkspaceConfig,
    item_id: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Resolve an Item's exact canonical GT through the managed asset resolver."""
    from vfieval.media_assets import resolve_asset_path

    item = get_media_item(db, int(item_id))
    member = get_media_item_member(db, _canonical_member_id(db, int(item_id)))
    asset, path = resolve_asset_path(
        db,
        workspace,
        int(item["canonical_gt_asset_id"]),
        role="reference",
    )
    return item, member, asset, path


def resolve_item_member(
    db: Database,
    workspace: WorkspaceConfig,
    member_id: int,
    *,
    require_reusable: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Resolve a member without ever trusting a client-provided filesystem path."""
    from vfieval.media_assets import resolve_asset_path

    member = get_media_item_member(db, int(member_id))
    item = get_media_item(db, int(member["item_id"]))
    if require_reusable and not member["reusable_as_pred"]:
        raise ValueError("media item member is not a reusable prediction")
    role = "distorted" if member["member_role"] in {
        "model_pred",
        "external_pred",
        "compare_snapshot",
        "evaluation_pred",
    } else "reference"
    asset, path = resolve_asset_path(db, workspace, int(member["asset_id"]), role=role)
    return item, member, asset, path


def ensure_canonical_gt_item(db: Database, asset_id: int) -> dict[str, Any]:
    asset = db.get("SELECT * FROM media_assets WHERE id = ?", (int(asset_id),))
    if asset is None:
        raise KeyError(f"media asset {asset_id} not found")
    if str(asset.get("source_kind") or "") not in {"folder", "upload"}:
        raise ValueError("canonical GT items require a folder or upload asset")
    if str(asset.get("role") or "") != "gt":
        raise ValueError("canonical GT items require an asset with role gt")
    if asset.get("collection_id") is None:
        raise ValueError("canonical GT assets require a Collection")

    state = _asset_state(asset)
    now = utc_ts()
    # The asset is the semantic identity.  ``source_key`` is useful as a
    # stable display/debug key, but it must never be used to merge two
    # canonical assets: an imported or manually repaired catalog can contain
    # a stale Item key even though the assets are distinct.  Resolve only by
    # the immutable canonical asset id and treat a key collision as an
    # integrity error instead of silently repointing an existing Item.
    item_key = f"canonical:{asset['source_key']}"
    existing = db.get(
        "SELECT id FROM media_items WHERE canonical_gt_asset_id = ?",
        (int(asset_id),),
    )
    key_owner = db.get(
        "SELECT id, canonical_gt_asset_id FROM media_items WHERE item_key = ?",
        (item_key,),
    )
    if key_owner is not None and int(key_owner["canonical_gt_asset_id"]) != int(asset_id):
        raise ValueError(
            "canonical media item key collision; refusing to merge distinct GT assets"
        )
    with db.connection() as conn:
        if existing is None:
            cur = conn.execute(
                """
                INSERT INTO media_items(
                    collection_id, item_key, canonical_gt_asset_id, display_name,
                    media_kind, state, metadata_json, created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(asset["collection_id"]),
                    item_key,
                    int(asset_id),
                    str(asset["display_name"]),
                    str(asset["media_kind"]),
                    state,
                    _json({"canonical_source_key": asset["source_key"]}),
                    now,
                    now,
                    now if state == "deleted" else None,
                ),
            )
            item_id = int(cur.lastrowid)
        else:
            item_id = int(existing["id"])
            conn.execute(
                """
                UPDATE media_items
                SET collection_id = ?, item_key = ?, canonical_gt_asset_id = ?,
                    display_name = ?, media_kind = ?, state = ?, updated_at = ?,
                    deleted_at = ?
                WHERE id = ?
                """,
                (
                    int(asset["collection_id"]),
                    item_key,
                    int(asset_id),
                    str(asset["display_name"]),
                    str(asset["media_kind"]),
                    state,
                    now,
                    now if state == "deleted" else None,
                    item_id,
                ),
            )
        conn.execute(
            """
            INSERT INTO media_item_members(
                item_id, asset_id, member_role, producer_kind, producer_run_id,
                method_key, reusable_as_pred, temporal_mapping_json,
                spatial_origin_json, state, metadata_json,
                created_at, updated_at, deleted_at
            ) VALUES (?, ?, 'canonical_gt', 'source', NULL, '', 0, '{}', ?, ?, '{}', ?, ?, ?)
            ON CONFLICT(item_id, asset_id, member_role) DO UPDATE SET
                producer_kind = 'source', producer_run_id = NULL,
                method_key = '', reusable_as_pred = 0,
                spatial_origin_json = excluded.spatial_origin_json,
                state = excluded.state, updated_at = excluded.updated_at,
                deleted_at = excluded.deleted_at
            """,
            (
                item_id,
                int(asset_id),
                _json(
                    {
                        "width": int(asset.get("width") or 0),
                        "height": int(asset.get("height") or 0),
                        "fps": asset.get("fps"),
                        "frame_count": int(asset.get("frame_count") or 0),
                    }
                ),
                state,
                now,
                now,
                now if state == "deleted" else None,
            ),
        )
    return get_media_item(db, item_id, include_deleted=True)


def sync_canonical_gt_items(db: Database) -> dict[str, int]:
    rows = db.query(
        """
        SELECT id FROM media_assets
        WHERE source_kind IN ('folder', 'upload') AND role = 'gt'
        ORDER BY id
        """
    )
    existing_ids = {
        int(row["canonical_gt_asset_id"])
        for row in db.query("SELECT canonical_gt_asset_id FROM media_items")
    }
    created = 0
    updated = 0
    for row in rows:
        asset_id = int(row["id"])
        ensure_canonical_gt_item(db, asset_id)
        if asset_id in existing_ids:
            updated += 1
        else:
            created += 1
    return {"created": created, "updated": updated, "total": len(rows)}


def list_item_groups(db: Database) -> dict[str, list[dict[str, Any]]]:
    rows = db.query(
        """
        SELECT c.id, c.name, c.slug, c.metadata_json,
               MIN(a.source_kind) AS source_kind,
               COUNT(mi.id) AS item_count
        FROM media_collections c
        JOIN media_items mi ON mi.collection_id = c.id
        JOIN media_assets a ON a.id = mi.canonical_gt_asset_id
        WHERE mi.state = 'ready' AND mi.deleted_at IS NULL
          AND a.state = 'ready' AND a.deleted_at IS NULL
          AND a.source_kind IN ('folder', 'upload') AND a.role = 'gt'
        GROUP BY c.id
        ORDER BY c.name, c.id
        """
    )
    groups: list[dict[str, Any]] = []
    for row in rows:
        decoded = dict(row)
        decoded["item_count"] = int(decoded.get("item_count") or 0)
        decoded["metadata"] = _loads(decoded.pop("metadata_json", None))
        groups.append(decoded)
    return {"groups": groups}


def list_media_items(
    db: Database,
    group_id: int,
    *,
    query: str = "",
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    group = db.get("SELECT id FROM media_collections WHERE id = ?", (int(group_id),))
    if group is None:
        raise KeyError(f"media collection {group_id} not found")
    clauses = [
        "mi.collection_id = ?",
        "mi.state = 'ready'",
        "mi.deleted_at IS NULL",
        "a.state = 'ready'",
        "a.deleted_at IS NULL",
        "a.source_kind IN ('folder', 'upload')",
        "a.role = 'gt'",
    ]
    params: list[Any] = [int(group_id)]
    normalized_query = str(query or "").strip()
    if normalized_query:
        clauses.append("(mi.display_name LIKE ? OR a.original_name LIKE ?)")
        needle = f"%{normalized_query}%"
        params.extend([needle, needle])
    where = " AND ".join(clauses)
    count = db.get(
        f"SELECT COUNT(*) AS count FROM media_items mi JOIN media_assets a ON a.id = mi.canonical_gt_asset_id WHERE {where}",
        params,
    )
    total = int((count or {}).get("count") or 0)
    page = max(1, int(page))
    page_size = min(200, max(1, int(page_size)))
    rows = db.query(
        f"""
        SELECT mi.*, c.name AS collection_name, c.slug AS collection_slug,
               a.source_key AS canonical_source_key,
               a.source_kind AS canonical_source_kind,
               a.display_name AS canonical_display_name,
               a.storage_path AS canonical_storage_path,
               a.state AS canonical_asset_state,
               a.frame_count, a.width, a.height, a.fps, a.size_bytes,
               a.content_sha256 AS canonical_content_sha256
        FROM media_items mi
        JOIN media_collections c ON c.id = mi.collection_id
        JOIN media_assets a ON a.id = mi.canonical_gt_asset_id
        WHERE {where}
        ORDER BY mi.display_name, mi.id
        LIMIT ? OFFSET ?
        """,
        (*params, page_size, (page - 1) * page_size),
    )
    return {
        "items": [_decode_item(row) for row in rows],
        "page": page,
        "page_size": page_size,
        "total": total,
        "page_count": max(1, (total + page_size - 1) // page_size),
    }


def _canonical_member_id(db: Database, item_id: int) -> int:
    row = db.get(
        """
        SELECT mim.id
        FROM media_item_members mim
        JOIN media_items mi ON mi.id = mim.item_id
        WHERE mim.item_id = ? AND mim.member_role = 'canonical_gt'
          AND mim.asset_id = mi.canonical_gt_asset_id
          AND mim.state = 'ready' AND mim.deleted_at IS NULL
        ORDER BY mim.id LIMIT 1
        """,
        (int(item_id),),
    )
    if row is None:
        raise ValueError(f"media item {item_id} has no ready canonical GT member")
    return int(row["id"])


def _upsert_member(
    db: Database,
    *,
    item_id: int,
    asset_id: int,
    member_role: str,
    producer_kind: str,
    producer_run_id: int | None,
    method_key: str,
    reusable_as_pred: bool,
    temporal_mapping: dict[str, Any] | None,
    spatial_origin: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if member_role not in MEMBER_ROLES:
        raise ValueError(f"unsupported media item member role: {member_role}")
    if producer_kind not in PRODUCER_KINDS:
        raise ValueError(f"unsupported media item producer kind: {producer_kind}")
    item = get_media_item(db, int(item_id))
    if item["state"] != "ready":
        raise ValueError(f"media item {item_id} is not ready")
    asset = db.get("SELECT * FROM media_assets WHERE id = ? AND deleted_at IS NULL", (int(asset_id),))
    if asset is None:
        raise KeyError(f"media asset {asset_id} not found")
    if str(asset.get("state") or "") != "ready":
        raise ValueError(f"media asset {asset_id} is not ready")
    _require_member_asset_role(member_role, str(asset.get("role") or ""))
    if reusable_as_pred and not (
        (member_role == "model_pred" and producer_kind == "model_inference")
        or (member_role == "external_pred" and producer_kind == "external")
    ):
        raise ValueError("only model inference and explicit external predictions may be reusable")

    now = utc_ts()
    existing = db.get(
        """
        SELECT id, item_id FROM media_item_members
        WHERE asset_id = ? AND member_role = ?
        """,
        (int(asset_id), member_role),
    )
    if existing is not None and int(existing["item_id"]) != int(item_id):
        raise ValueError("a media asset cannot represent the same member role for multiple Items")
    with db.connection() as conn:
        if existing is None:
            cur = conn.execute(
                """
                INSERT INTO media_item_members(
                    item_id, asset_id, member_role, producer_kind, producer_run_id,
                    method_key, reusable_as_pred, temporal_mapping_json,
                    spatial_origin_json, state, metadata_json,
                    created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, NULL)
                """,
                (
                    int(item_id),
                    int(asset_id),
                    member_role,
                    producer_kind,
                    int(producer_run_id) if producer_run_id is not None else None,
                    str(method_key or ""),
                    1 if reusable_as_pred else 0,
                    _json(temporal_mapping),
                    _json(spatial_origin),
                    _json(metadata),
                    now,
                    now,
                ),
            )
            member_id = int(cur.lastrowid)
        else:
            member_id = int(existing["id"])
            conn.execute(
                """
                UPDATE media_item_members
                SET producer_kind = ?, producer_run_id = ?, method_key = ?,
                    reusable_as_pred = ?, temporal_mapping_json = ?,
                    spatial_origin_json = ?, state = 'ready', metadata_json = ?,
                    updated_at = ?, deleted_at = NULL
                WHERE id = ?
                """,
                (
                    producer_kind,
                    int(producer_run_id) if producer_run_id is not None else None,
                    str(method_key or ""),
                    1 if reusable_as_pred else 0,
                    _json(temporal_mapping),
                    _json(spatial_origin),
                    _json(metadata),
                    now,
                    member_id,
                ),
            )
    return get_media_item_member(db, member_id)


def _upsert_binding(
    db: Database,
    *,
    run_id: int,
    item_id: int,
    binding_role: str,
    member_id: int,
    slot: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if binding_role not in BINDING_ROLES:
        raise ValueError(f"unsupported Run media item binding role: {binding_role}")
    run = db.get_run(int(run_id))
    item = get_media_item(db, int(item_id))
    member = get_media_item_member(db, int(member_id))
    if int(member["item_id"]) != int(item["id"]):
        raise ValueError("Run binding member must belong to the selected media item")
    _require_binding_member_role(
        binding_role,
        str(member.get("member_role") or ""),
        active=False,
    )
    _require_binding_slot(binding_role, str(slot or ""))
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO run_media_item_bindings(
                run_id, item_id, binding_role, slot,
                original_member_id, active_member_id,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, item_id, binding_role, slot) DO UPDATE SET
                original_member_id = excluded.original_member_id,
                active_member_id = excluded.active_member_id,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                int(run["id"]),
                int(item["id"]),
                binding_role,
                str(slot or ""),
                int(member["id"]),
                int(member["id"]),
                _json(metadata),
                now,
                now,
            ),
        )
    row = db.get(
        """
        SELECT * FROM run_media_item_bindings
        WHERE run_id = ? AND item_id = ? AND binding_role = ? AND slot = ?
        """,
        (int(run_id), int(item_id), binding_role, str(slot or "")),
    )
    assert row is not None
    row["metadata"] = _loads(row.pop("metadata_json", None))
    return row


def bind_run_source(
    db: Database,
    run_id: int,
    item_id: int,
    *,
    video_name: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binding_metadata = dict(metadata or {})
    if video_name:
        binding_metadata["video_name"] = str(video_name)
    return _upsert_binding(
        db,
        run_id=int(run_id),
        item_id=int(item_id),
        binding_role="source",
        member_id=_canonical_member_id(db, int(item_id)),
        slot=str(video_name or ""),
        metadata=binding_metadata,
    )


def find_run_source_item(
    db: Database,
    run_id: int,
    video_name: str,
    *,
    source_video_group: str = "",
    source_video_file: str = "",
) -> dict[str, Any] | None:
    """Find an explicitly bound source Item; never infer identity from a filename stem."""
    rows = db.query(
        """
        SELECT DISTINCT rib.item_id
        FROM run_media_item_bindings rib
        JOIN media_items mi ON mi.id = rib.item_id
        JOIN media_assets canonical ON canonical.id = mi.canonical_gt_asset_id
        LEFT JOIN run_media_assets rma
          ON rma.run_id = rib.run_id
         AND rma.asset_id = mi.canonical_gt_asset_id
         AND rma.role = 'source'
        WHERE rib.run_id = ? AND rib.binding_role = 'source'
          AND (
            rib.slot = ?
            OR json_extract(rib.metadata_json, '$.video_name') = ?
            OR rma.video_name = ?
            OR (
              ? != '' AND ? != ''
              AND json_extract(canonical.provenance_json, '$.video_group') = ?
              AND json_extract(canonical.provenance_json, '$.video') = ?
            )
          )
        ORDER BY rib.item_id
        """,
        (
            int(run_id),
            str(video_name),
            str(video_name),
            str(video_name),
            str(source_video_group),
            str(source_video_file),
            str(source_video_group),
            str(source_video_file),
        ),
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError(f"Run {run_id} has ambiguous source Item bindings for {video_name}")
    return get_media_item(db, int(rows[0]["item_id"]))


def register_model_prediction(
    db: Database,
    run_id: int,
    item_id: int,
    asset_id: int,
    *,
    method_key: str = "",
    temporal_mapping: dict[str, Any] | None = None,
    spatial_origin: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = db.get_run(int(run_id))
    if _run_type(run) != "model_inference":
        raise ValueError("only model_inference Runs may register reusable model predictions")
    source_bindings = db.query(
        """
        SELECT id, item_id, slot, metadata_json
        FROM run_media_item_bindings
        WHERE run_id = ? AND binding_role = 'source'
        ORDER BY id
        """,
        (int(run_id),),
    )
    item_source_bindings = [
        row for row in source_bindings if int(row["item_id"]) == int(item_id)
    ]
    if not item_source_bindings:
        raise ValueError("model prediction requires an explicit source media item binding")
    pred_outputs = db.query(
        """
        SELECT video_name FROM run_media_assets
        WHERE run_id = ? AND asset_id = ? AND role = 'pred'
        ORDER BY video_name
        """,
        (int(run_id), int(asset_id)),
    )
    if not pred_outputs:
        raise ValueError("prediction asset is not a Pred output of this Run")

    # A Run with exactly one source and exactly one Pred output is unambiguous
    # even when an older writer used a filename in one binding and a stem in
    # the other.  Any broader Run needs an exact server-owned timeline key:
    # accepting an Item-level source binding alone would let a Pred for video
    # B be published under video A. Never infer a mapping from stems, hashes,
    # or model labels.
    if len(source_bindings) > 1 or len(pred_outputs) > 1:
        for output in pred_outputs:
            video_name = str(output.get("video_name") or "")
            matches = []
            for source in item_source_bindings:
                source_metadata = _loads(source.get("metadata_json"))
                source_video_name = (
                    str(source_metadata.get("video_name") or "")
                    if isinstance(source_metadata, dict)
                    else ""
                )
                if video_name and (
                    str(source.get("slot") or "") == video_name
                    or source_video_name == video_name
                ):
                    matches.append(source)
            if len(matches) != 1:
                raise ValueError(
                    "prediction asset does not map to exactly one bound source media item"
                )
    member = _upsert_member(
        db,
        item_id=int(item_id),
        asset_id=int(asset_id),
        member_role="model_pred",
        producer_kind="model_inference",
        producer_run_id=int(run_id),
        method_key=str(method_key or f"run:{int(run_id)}"),
        reusable_as_pred=True,
        temporal_mapping=temporal_mapping,
        spatial_origin=spatial_origin,
        metadata=metadata,
    )
    _upsert_binding(
        db,
        run_id=int(run_id),
        item_id=int(item_id),
        binding_role="pred_output",
        member_id=int(member["id"]),
        metadata=metadata,
    )
    return member


def register_external_prediction(
    db: Database,
    item_id: int,
    asset_id: int,
    *,
    method_key: str,
    temporal_mapping: dict[str, Any] | None = None,
    spatial_origin: dict[str, Any] | None = None,
    aspect_stretch_confirmed: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    asset = db.get("SELECT source_kind FROM media_assets WHERE id = ?", (int(asset_id),))
    if asset is None:
        raise KeyError(f"media asset {asset_id} not found")
    if str(asset.get("source_kind") or "") != "upload":
        raise ValueError("explicit external predictions require an upload asset")
    normalized_key = str(method_key or "").strip()
    if not normalized_key:
        raise ValueError("external prediction method_key is required")
    member_metadata = dict(metadata or {})
    member_metadata["aspect_stretch_confirmed"] = bool(aspect_stretch_confirmed)
    return _upsert_member(
        db,
        item_id=int(item_id),
        asset_id=int(asset_id),
        member_role="external_pred",
        producer_kind="external",
        producer_run_id=None,
        method_key=normalized_key,
        reusable_as_pred=True,
        temporal_mapping=temporal_mapping,
        spatial_origin=spatial_origin,
        metadata=member_metadata,
    )


def bind_compare_input(
    db: Database,
    run_id: int,
    item_id: int,
    member_id: int,
    *,
    binding_role: str,
    slot: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if binding_role not in {"compare_gt", "compare_pred"}:
        raise ValueError("Compare input binding role must be compare_gt or compare_pred")
    run = db.get_run(int(run_id))
    if _run_type(run) != "video_compare":
        raise ValueError("Compare input bindings require a video_compare Run")
    member = get_media_item_member(db, int(member_id))
    if binding_role == "compare_gt" and member["member_role"] != "canonical_gt":
        raise ValueError("Compare GT binding requires the canonical GT member")
    if binding_role == "compare_pred" and not member["reusable_as_pred"]:
        raise ValueError("Compare Pred binding requires a reusable prediction member")
    return _upsert_binding(
        db,
        run_id=int(run_id),
        item_id=int(item_id),
        binding_role=binding_role,
        member_id=int(member_id),
        slot=str(slot or ""),
        metadata=metadata,
    )


def register_compare_snapshot(
    db: Database,
    compare_run_id: int,
    item_id: int,
    asset_id: int,
    *,
    source_member_id: int,
    slot: str = "",
    temporal_mapping: dict[str, Any] | None = None,
    spatial_origin: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = db.get_run(int(compare_run_id))
    if _run_type(run) != "video_compare":
        raise ValueError("Compare snapshots require a video_compare Run")
    source = get_media_item_member(db, int(source_member_id))
    if int(source["item_id"]) != int(item_id):
        raise ValueError("Compare snapshot source must belong to the selected media item")
    snapshot_metadata = dict(metadata or {})
    snapshot_metadata["source_member_id"] = int(source_member_id)
    snapshot_metadata["slot"] = str(slot or "")
    member = _upsert_member(
        db,
        item_id=int(item_id),
        asset_id=int(asset_id),
        member_role="compare_snapshot",
        producer_kind="video_compare",
        producer_run_id=int(compare_run_id),
        method_key=str(source.get("method_key") or ""),
        reusable_as_pred=False,
        temporal_mapping=temporal_mapping or source.get("temporal_mapping"),
        spatial_origin=spatial_origin or source.get("spatial_origin"),
        metadata=snapshot_metadata,
    )
    replace_active_binding_member(
        db,
        int(compare_run_id),
        int(item_id),
        "compare_pred",
        str(slot or ""),
        int(member["id"]),
        expected_active_member_id=int(source_member_id),
    )
    return member


def replace_active_binding_member(
    db: Database,
    run_id: int,
    item_id: int,
    binding_role: str,
    slot: str,
    new_member_id: int,
    *,
    expected_active_member_id: int | None = None,
) -> dict[str, Any]:
    member = get_media_item_member(db, int(new_member_id))
    if int(member["item_id"]) != int(item_id):
        raise ValueError("replacement member must belong to the bound media item")
    _require_binding_member_role(
        str(binding_role),
        str(member.get("member_role") or ""),
        active=True,
    )
    _require_binding_slot(str(binding_role), str(slot or ""))
    binding = db.get(
        """
        SELECT original_member_id
        FROM run_media_item_bindings
        WHERE run_id = ? AND item_id = ? AND binding_role = ? AND slot = ?
        """,
        (int(run_id), int(item_id), str(binding_role), str(slot or "")),
    )
    if binding is None:
        raise ValueError("Compare input binding changed or no longer exists")
    if str(member.get("member_role") or "") != "compare_snapshot":
        if int(member["id"]) != int(binding["original_member_id"]):
            raise ValueError("replacement member must preserve the original binding identity")
    else:
        metadata = member.get("metadata") or {}
        if (
            str(member.get("producer_kind") or "") != "video_compare"
            or member.get("producer_run_id") is None
            or int(member["producer_run_id"]) != int(run_id)
            or not isinstance(metadata, dict)
            or int(metadata.get("source_member_id") or 0)
            != int(binding["original_member_id"])
        ):
            raise ValueError("Compare Pred replacement is not a snapshot of the original member")
    clauses = [
        "run_id = ?",
        "item_id = ?",
        "binding_role = ?",
        "slot = ?",
    ]
    params: list[Any] = [int(run_id), int(item_id), str(binding_role), str(slot or "")]
    if expected_active_member_id is not None:
        clauses.append("active_member_id = ?")
        params.append(int(expected_active_member_id))
    with db.connection() as conn:
        cur = conn.execute(
            f"UPDATE run_media_item_bindings SET active_member_id = ?, updated_at = ? WHERE {' AND '.join(clauses)}",
            (int(new_member_id), utc_ts(), *params),
        )
        if int(cur.rowcount or 0) != 1:
            raise ValueError("Compare input binding changed or no longer exists")
    row = db.get(
        """
        SELECT * FROM run_media_item_bindings
        WHERE run_id = ? AND item_id = ? AND binding_role = ? AND slot = ?
        """,
        (int(run_id), int(item_id), str(binding_role), str(slot or "")),
    )
    assert row is not None
    row["metadata"] = _loads(row.pop("metadata_json", None))
    return row


def list_item_predictions(db: Database, item_id: int) -> dict[str, Any]:
    item = get_media_item(db, int(item_id))
    rows = db.query(
        """
        SELECT mim.*, a.source_key, a.source_kind, a.media_kind,
               a.display_name, a.original_name, a.storage_path, a.mime_type,
               a.frame_count, a.width, a.height, a.fps, a.size_bytes,
               a.content_sha256, r.name AS run_name, r.status AS run_status,
               r.deleted_at AS run_deleted_at,
               r.artifact_cleaned_at AS run_artifact_cleaned_at,
               r.metadata_json AS run_metadata_json
        FROM media_item_members mim
        JOIN media_assets a ON a.id = mim.asset_id
        LEFT JOIN runs r ON r.id = mim.producer_run_id
        WHERE mim.item_id = ? AND mim.reusable_as_pred = 1
          AND mim.state = 'ready' AND mim.deleted_at IS NULL
          AND a.state = 'ready' AND a.deleted_at IS NULL
          AND (
            (mim.member_role = 'external_pred' AND mim.producer_kind = 'external'
             AND mim.producer_run_id IS NULL AND a.source_kind = 'upload')
            OR
            (mim.member_role = 'model_pred' AND mim.producer_kind = 'model_inference'
             AND mim.producer_run_id IS NOT NULL AND a.source_kind = 'run_artifact'
             AND r.status IN ('completed', 'metric_queued', 'metric_running')
             AND r.deleted_at IS NULL AND r.artifact_cleaned_at IS NULL
             AND COALESCE(json_extract(r.metadata_json, '$.run_type'), 'model_inference') = 'model_inference')
          )
        ORDER BY r.created_at DESC, mim.id DESC
        """,
        (int(item_id),),
    )
    predictions: list[dict[str, Any]] = []
    for row in rows:
        run_metadata = _loads(row.pop("run_metadata_json", None))
        decoded = _decode_member(row)
        decoded["member_id"] = int(decoded["id"])
        decoded["run_id"] = decoded.get("producer_run_id")
        decoded["run_metadata"] = run_metadata
        predictions.append(decoded)
    return {"item": item, "predictions": predictions}


def _mapping_object(value: Any, field_name: str) -> dict[str, Any]:
    """Return persisted mapping metadata in the only shape Compare accepts.

    The database schema intentionally stores JSON so old deployments can carry
    richer mapping reports.  Compare, however, needs an object it can inspect
    and fingerprint; accepting an arbitrary JSON list/string here would make a
    malformed member look like a valid source.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"media item member {field_name} must be an object")
    return dict(value)


def _prediction_label(member: dict[str, Any], asset: dict[str, Any]) -> str:
    metadata = member.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("track_label", "label", "method_label"):
            label = str(metadata.get(key) or "").strip()
            if label:
                return label
    for value in (
        member.get("method_key"),
        member.get("run_name"),
        asset.get("display_name"),
        asset.get("original_name"),
    ):
        label = str(value or "").strip()
        if label:
            return label
    return f"member-{int(member['id'])}"


def _assert_reusable_compare_prediction(
    db: Database,
    item: dict[str, Any],
    member: dict[str, Any],
    asset: dict[str, Any],
) -> None:
    """Enforce the reusable-Pred contract at the server-side resolution edge."""
    if int(member["item_id"]) != int(item["id"]):
        raise ValueError("compare prediction member does not belong to the selected media item")
    if member.get("state") != "ready" or member.get("deleted_at") is not None:
        raise ValueError("compare prediction member is not ready")
    if not bool(member.get("reusable_as_pred")):
        raise ValueError("media item member is not a reusable prediction")
    if asset.get("state") != "ready" or asset.get("deleted_at") is not None:
        raise ValueError("compare prediction asset is not ready")
    if str(asset.get("role") or "") != "pred":
        raise ValueError("compare prediction asset must have role pred")
    if str(asset.get("media_kind") or "") != str(item.get("media_kind") or ""):
        raise ValueError("compare prediction media kind does not match the selected media item")
    # The member's role/producer columns are not enough on their own: a
    # hand-edited or stale row could point at a Compare-owned artifact while
    # claiming to be a normal model output.  The asset provenance is the
    # authority for this non-reuse boundary.
    from vfieval.media_assets import is_compare_derived_asset

    if is_compare_derived_asset(db, asset):
        raise ValueError("Compare cannot reuse video_compare-derived media")

    member_role = str(member.get("member_role") or "")
    producer_kind = str(member.get("producer_kind") or "")
    if member_role == "model_pred":
        if producer_kind != "model_inference" or member.get("producer_run_id") is None:
            raise ValueError("model prediction member has an invalid producer")
        if str(asset.get("source_kind") or "") != "run_artifact":
            raise ValueError("model prediction member must reference a Run artifact")
        run = db.get_run(int(member["producer_run_id"]))
        if _run_type(run) != "model_inference":
            raise ValueError("Compare cannot reuse a prediction produced by a non-model-inference Run")
        if (
            str(run.get("status") or "") not in {"completed", "metric_queued", "metric_running"}
            or run.get("deleted_at") is not None
            or run.get("artifact_cleaned_at") is not None
        ):
            raise ValueError("model prediction source Run is not available")
        linked = db.get(
            """
            SELECT 1 FROM run_media_assets
            WHERE run_id = ? AND asset_id = ? AND role = 'pred'
            LIMIT 1
            """,
            (int(run["id"]), int(asset["id"])),
        )
        if linked is None:
            raise ValueError("model prediction asset is not a published Pred output of its producer Run")
        return

    if member_role == "external_pred":
        if producer_kind != "external" or member.get("producer_run_id") is not None:
            raise ValueError("external prediction member has an invalid producer")
        if str(asset.get("source_kind") or "") != "upload":
            raise ValueError("external prediction member must reference an uploaded asset")
        if not str(member.get("method_key") or "").strip():
            raise ValueError("external prediction member is missing its method identity")
        return

    # This explicitly rules out Compare snapshots, evaluation packages, and
    # any future derived role even if a malformed row tries to claim reuse.
    raise ValueError("only model inference or explicit external predictions may be Compare sources")


def resolve_media_item_compare(
    db: Database,
    workspace: WorkspaceConfig,
    item_id: int,
    member_ids: Iterable[int],
) -> dict[str, Any]:
    """Resolve a GT-first Compare request exclusively through Item members.

    Returned paths are generated by ``resolve_asset_path`` and are intended for
    server-side preflight/materialization only.  No caller-provided path is
    accepted or consulted.  The order of ``member_ids`` is preserved as the
    requested Compare-track order.
    """
    if isinstance(member_ids, (str, bytes)):
        raise ValueError("pred_member_ids must be a list of one or two member IDs")
    try:
        normalized_ids = [int(value) for value in member_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError("pred_member_ids must contain integer member IDs") from exc
    if not 1 <= len(normalized_ids) <= 2:
        raise ValueError("Compare requires one or two prediction members")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Compare prediction members must be distinct")

    item, canonical_member, reference_asset, reference_path = resolve_item_reference(
        db, workspace, int(item_id)
    )
    if item.get("state") != "ready":
        raise ValueError(f"media item {item_id} is not ready")
    if int(canonical_member["asset_id"]) != int(item["canonical_gt_asset_id"]):
        raise ValueError("media item canonical GT member does not match its canonical asset")
    if (
        canonical_member.get("member_role") != "canonical_gt"
        or canonical_member.get("producer_kind") != "source"
        or bool(canonical_member.get("reusable_as_pred"))
    ):
        raise ValueError("media item canonical GT member is invalid")
    if (
        str(reference_asset.get("source_kind") or "") not in {"folder", "upload"}
        or str(reference_asset.get("role") or "") != "gt"
        or str(reference_asset.get("media_kind") or "") != str(item.get("media_kind") or "")
    ):
        raise ValueError("media item canonical GT asset is not an eligible source asset")

    reference_mapping = _mapping_object(canonical_member.get("temporal_mapping"), "temporal_mapping")
    reference_spatial_origin = _mapping_object(canonical_member.get("spatial_origin"), "spatial_origin")
    reference = {
        "item_id": int(item["id"]),
        "member_id": int(canonical_member["id"]),
        "asset_id": int(reference_asset["id"]),
        "asset": reference_asset,
        "path": str(reference_path),
        "label": str(item.get("display_name") or reference_asset.get("display_name") or "GT"),
        "temporal_mapping": reference_mapping,
        "spatial_origin": reference_spatial_origin,
        "frame_count": int(reference_asset.get("frame_count") or 0),
        "width": int(reference_asset.get("width") or 0),
        "height": int(reference_asset.get("height") or 0),
        "fps": reference_asset.get("fps"),
        "media_kind": reference_asset.get("media_kind"),
    }

    members: list[dict[str, Any]] = []
    for member_id in normalized_ids:
        member_item, member, asset, path = resolve_item_member(
            db,
            workspace,
            member_id,
            require_reusable=True,
        )
        if int(member_item["id"]) != int(item["id"]):
            raise ValueError(
                "all Compare prediction members must belong to the same media item as the selected GT"
            )
        _assert_reusable_compare_prediction(db, item, member, asset)
        temporal_mapping = _mapping_object(member.get("temporal_mapping"), "temporal_mapping")
        spatial_origin = _mapping_object(member.get("spatial_origin"), "spatial_origin")
        members.append(
            {
                **member,
                "member_id": int(member["id"]),
                "asset": asset,
                "path": str(path),
                "label": _prediction_label(member, asset),
                "track_label": _prediction_label(member, asset),
                "temporal_mapping": temporal_mapping,
                "spatial_origin": spatial_origin,
                "frame_count": int(asset.get("frame_count") or 0),
                "width": int(asset.get("width") or 0),
                "height": int(asset.get("height") or 0),
                "fps": asset.get("fps"),
                "media_kind": asset.get("media_kind"),
            }
        )
    return {"item": item, "reference": reference, "members": members}


def list_methods_for_items(db: Database, item_ids: Iterable[int]) -> dict[str, Any]:
    normalized_ids = list(dict.fromkeys(int(item_id) for item_id in item_ids))
    if not normalized_ids:
        return {"item_ids": [], "methods": []}
    for item_id in normalized_ids:
        get_media_item(db, item_id)
    grouped: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    for item_id in normalized_ids:
        for prediction in list_item_predictions(db, item_id)["predictions"]:
            run_id = prediction.get("producer_run_id")
            method_key = str(prediction.get("method_key") or "")
            identity = (str(prediction["producer_kind"]), int(run_id) if run_id else None, method_key)
            method = grouped.setdefault(
                identity,
                {
                    "kind": "run" if run_id else "external",
                    "run_id": int(run_id) if run_id else None,
                    "method_key": method_key,
                    "label": str(prediction.get("run_name") or method_key or prediction.get("display_name") or "Pred"),
                    "bindings": {},
                },
            )
            current = method["bindings"].get(item_id)
            if current is None or int(prediction["id"]) > int(current["member_id"]):
                method["bindings"][item_id] = {
                    "item_id": item_id,
                    "member_id": int(prediction["id"]),
                    "asset_id": int(prediction["asset_id"]),
                    "width": int(prediction.get("width") or 0),
                    "height": int(prediction.get("height") or 0),
                    "fps": prediction.get("fps"),
                    "frame_count": int(prediction.get("frame_count") or 0),
                }
    methods: list[dict[str, Any]] = []
    requested = set(normalized_ids)
    for method in grouped.values():
        binding_map = method.pop("bindings")
        covered = [item_id for item_id in normalized_ids if item_id in binding_map]
        missing = [item_id for item_id in normalized_ids if item_id not in binding_map]
        method.update(
            {
                "bindings": [binding_map[item_id] for item_id in covered],
                "covered_item_ids": covered,
                "missing_item_ids": missing,
                "covered_count": len(covered),
                "total_items": len(requested),
                "complete": not missing,
            }
        )
        methods.append(method)
    methods.sort(key=lambda row: (-int(row["covered_count"]), str(row["label"]), str(row["method_key"])))
    return {"item_ids": normalized_ids, "methods": methods}


# Public names used by the HTTP layer.  Keep the smaller internal names above
# for existing call sites while making the Item API contracts self-describing.
def sync_media_items(db: Database) -> dict[str, int]:
    """Synchronize canonical GT Items for managed folder/upload assets only."""
    return sync_canonical_gt_items(db)


def list_media_item_groups(
    db: Database,
    *,
    role: str = "gt",
) -> dict[str, list[dict[str, Any]]]:
    if str(role or "gt").lower() not in {"gt", "canonical_gt"}:
        raise ValueError("media item groups currently support only role=gt")
    sync_media_items(db)
    return list_item_groups(db)


def list_media_item_predictions(db: Database, item_id: int) -> dict[str, Any]:
    return list_item_predictions(db, int(item_id))


def list_media_methods(db: Database, item_ids: Iterable[int]) -> dict[str, Any]:
    return list_methods_for_items(db, item_ids)


def list_unbound_predictions(
    db: Database,
    *,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    where = """
        a.source_kind = 'run_artifact' AND a.role = 'pred'
        AND NOT EXISTS (
            SELECT 1 FROM media_item_members mim
            WHERE mim.asset_id = a.id AND mim.member_role = 'model_pred'
        )
    """
    count = db.get(f"SELECT COUNT(DISTINCT a.id) AS count FROM media_assets a WHERE {where}")
    total = int((count or {}).get("count") or 0)
    page = max(1, int(page))
    page_size = min(200, max(1, int(page_size)))
    rows = db.query(
        f"""
        SELECT DISTINCT a.*, rma.run_id, rma.video_name, rma.track_label,
               r.name AS run_name, r.status AS run_status,
               r.deleted_at AS run_deleted_at,
               r.artifact_cleaned_at AS run_artifact_cleaned_at,
               r.metadata_json AS run_metadata_json
        FROM media_assets a
        LEFT JOIN run_media_assets rma ON rma.asset_id = a.id AND rma.role = 'pred'
        LEFT JOIN runs r ON r.id = rma.run_id
        WHERE {where}
        ORDER BY a.id DESC
        LIMIT ? OFFSET ?
        """,
        (page_size, (page - 1) * page_size),
    )
    predictions: list[dict[str, Any]] = []
    for row in rows:
        run_metadata = _loads(row.pop("run_metadata_json", None))
        decoded = dict(row)
        decoded["provenance"] = _loads(decoded.pop("provenance_json", None))
        decoded["metadata"] = _loads(decoded.pop("metadata_json", None))
        if str(run_metadata.get("run_type") or "model_inference") == "video_compare":
            reason = "compare_run_not_reusable"
        elif decoded.get("run_deleted_at") is not None:
            reason = "source_run_deleted"
        elif decoded.get("run_artifact_cleaned_at") is not None:
            reason = "source_run_cleaned"
        else:
            reason = "legacy_or_missing_item_binding"
        decoded["reason"] = reason
        decoded["run_metadata"] = run_metadata
        predictions.append(decoded)
    return {
        "predictions": predictions,
        "page": page,
        "page_size": page_size,
        "total": total,
        "page_count": max(1, (total + page_size - 1) // page_size),
    }
