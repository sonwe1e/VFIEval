from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts
from vfieval.file_inputs import resolve_checkpoint, resolve_model_file


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True, ensure_ascii=False)


def _loads(text: str | None) -> Any:
    return json.loads(text) if text else {}


def _sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_device_model(device_kind: str, device: str) -> str:
    try:
        import torch

        index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        if device_kind == "cuda" and torch.cuda.is_available():
            return str(torch.cuda.get_device_name(index))
        if device_kind == "npu" and hasattr(torch, "npu"):
            function = getattr(torch.npu, "get_device_name", None)
            if function is not None:
                return str(function(index))
    except Exception:
        pass
    return "CPU" if device_kind == "cpu" else device_kind.upper()


def execution_profile_identity(workspace: WorkspaceConfig, payload: dict[str, Any]) -> dict[str, Any]:
    model_path = resolve_model_file(workspace, str(payload.get("model_file") or ""))
    checkpoint_path = resolve_checkpoint(workspace, payload.get("checkpoint"), model_path.name)
    devices = [str(value) for value in (payload.get("devices") or [])]
    device = str(payload.get("device") or (devices[0] if devices else "auto"))
    execution_mode = str(payload.get("execution_mode") or "single")
    if execution_mode == "multi_npu" or device.startswith("npu"):
        device_kind = "npu"
    elif execution_mode == "multi_cuda" or device.startswith("cuda"):
        device_kind = "cuda"
    else:
        device_kind = "cpu"
    identity = {
        "model_name": model_path.name,
        "model_sha256": _sha256(model_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "device_kind": device_kind,
        "device_model": str(
            payload.get("device_model")
            or payload.get("device_name")
            or _runtime_device_model(device_kind, devices[0] if devices else device)
        ),
        "device_count": len(devices) if devices else 1,
        "height": int(payload.get("height") or 0),
        "width": int(payload.get("width") or 0),
        "resolution_mode": str(payload.get("resolution_mode") or "original"),
        "precision": str(payload.get("precision") or "fp32"),
        "artifact_profile": str(payload.get("artifact_profile") or "evaluation"),
    }
    identity["fingerprint"] = hashlib.sha256(_json(identity).encode("utf-8")).hexdigest()
    return identity


def record_execution_profile(
    db: Database,
    identity: dict[str, Any],
    settings: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    now = utc_ts()
    fingerprint = str(identity["fingerprint"])
    with db.connection() as conn:
        existing = conn.execute(
            "SELECT performance_json FROM execution_profiles WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        if existing is not None:
            old = _loads(existing["performance_json"])
            old_fps = float(old.get("steady_state_fps") or 0.0)
            new_fps = float(performance.get("steady_state_fps") or 0.0)
            if old_fps > new_fps:
                return get_execution_profile(db, fingerprint)
        conn.execute(
            """
            INSERT INTO execution_profiles(
                fingerprint, model_name, checkpoint, device_kind, device_count,
                device_model, height, width, precision, artifact_profile, settings_json,
                performance_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                settings_json = excluded.settings_json,
                performance_json = excluded.performance_json,
                updated_at = excluded.updated_at
            """,
            (
                fingerprint,
                identity.get("model_name") or "",
                identity.get("checkpoint") or "",
                identity.get("device_kind") or "cpu",
                int(identity.get("device_count") or 1),
                identity.get("device_model") or "",
                int(identity.get("height") or 0),
                int(identity.get("width") or 0),
                identity.get("precision") or "fp32",
                identity.get("artifact_profile") or "evaluation",
                _json(settings),
                _json(performance),
                now,
                now,
            ),
        )
    return get_execution_profile(db, fingerprint)


def get_execution_profile(db: Database, fingerprint: str) -> dict[str, Any]:
    row = db.get("SELECT * FROM execution_profiles WHERE fingerprint = ?", (str(fingerprint),))
    if row is None:
        raise KeyError(f"execution profile {fingerprint} not found")
    row["settings"] = _loads(row.pop("settings_json", None))
    row["performance"] = _loads(row.pop("performance_json", None))
    return row


def recommend_execution_profile(
    db: Database,
    workspace: WorkspaceConfig,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        identity = execution_profile_identity(workspace, payload)
    except Exception:
        return None
    row = db.get("SELECT fingerprint FROM execution_profiles WHERE fingerprint = ?", (identity["fingerprint"],))
    if row is None:
        return None
    return get_execution_profile(db, str(row["fingerprint"]))
