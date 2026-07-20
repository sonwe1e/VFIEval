from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from vfieval.file_inputs import file_sha256


INPUT_IDENTITY_SCHEMA = "run-input-identity-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[/\\]")
_REQUEST_EXCLUDED_KEYS = {
    "clone_of_run_id",
    "input_identity",
    "name",
    "preflight_token",
    "retry_of_run_id",
    "risk_ack_fingerprint",
}


class InputIdentityChanged(ValueError):
    """A public, structured exact-retry identity mismatch."""

    def __init__(self, comparison: Mapping[str, Any]):
        self.comparison = {
            "matches": False,
            "expected_fingerprint": str(comparison.get("expected_fingerprint") or ""),
            "actual_fingerprint": str(comparison.get("actual_fingerprint") or ""),
            "differences": [dict(row) for row in comparison.get("differences") or []],
        }
        super().__init__("Run inputs no longer match the stored input identity")

    def public_payload(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": str(self),
            "expected_fingerprint": self.comparison["expected_fingerprint"],
            "actual_fingerprint": self.comparison["actual_fingerprint"],
            "differences": list(self.comparison["differences"]),
        }


def build_file_identity(
    path: str | Path,
    *,
    trusted_root: str | Path,
    display_path: str | None = None,
    content_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a portable identity for a file below a trusted root.

    Only root-relative and explicitly public display paths are persisted. The
    resolved absolute path is used for validation and hashing but never appears
    in the returned identity.
    """

    root = Path(trusted_root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("identity file is outside its trusted root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"identity file is unavailable: {relative}")

    before = resolved.stat()
    digest = _normalize_sha256(content_sha256) if content_sha256 else file_sha256(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"identity file changed while it was being read: {relative}")

    return build_file_identity_from_values(
        relative_path=relative,
        display_path=display_path or relative,
        size_bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest,
    )


def build_file_identity_from_values(
    *,
    relative_path: str,
    display_path: str | None,
    size_bytes: int,
    mtime_ns: int,
    sha256: str,
) -> dict[str, Any]:
    """Build a file identity from a trusted catalog signature."""

    relative = _normalize_public_path(relative_path, "relative_path")
    display = _normalize_public_path(display_path or relative, "display_path")
    size = _non_negative_int(size_bytes, "size_bytes")
    modified = _non_negative_int(mtime_ns, "mtime_ns")
    return {
        "relative_path": relative,
        "display_path": display,
        "size_bytes": size,
        "mtime_ns": modified,
        "sha256": _normalize_sha256(sha256),
    }


def build_checkpoint_identity(
    requested: Any,
    *,
    resolved: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Preserve checkpoint selection intent and the exact resolved file."""

    raw = str(requested or "none").strip()
    lowered = raw.lower()
    if lowered in {"", "none"}:
        normalized_request = "none"
    elif lowered == "auto":
        normalized_request = "auto"
    else:
        normalized_request = _normalize_public_path(raw, "checkpoint requested path")

    normalized_resolved = _normalize_file_identity(resolved) if resolved is not None else None
    if normalized_request == "none" and normalized_resolved is not None:
        raise ValueError("checkpoint requested as none cannot have a resolved file")
    if normalized_request not in {"none", "auto"} and normalized_resolved is None:
        raise ValueError("an explicit checkpoint request requires a resolved file")
    return {
        "requested": normalized_request,
        "resolved": normalized_resolved,
    }


def build_source_identity(
    *,
    item_id: int,
    asset_id: int,
    qualified_name: str,
    file_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine canonical Item/Asset identity with immutable file content."""

    return {
        "item_id": _positive_int(item_id, "item_id"),
        "asset_id": _positive_int(asset_id, "asset_id"),
        "qualified_name": _normalize_public_path(qualified_name, "qualified_name"),
        "content": _normalize_file_identity(file_identity),
    }


def normalize_request_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize execution-affecting request data for stable JSON hashing.

    Provenance/UI fields that must change on retry are excluded. Keys carrying
    internal absolute paths are excluded at every nesting level; their portable
    identity belongs in the model/checkpoint/source sections instead.
    """

    if not isinstance(request, Mapping):
        raise TypeError("request identity must be an object")
    normalized = _normalize_request_mapping(request, top_level=True)
    return dict(normalized)


def build_run_input_identity(
    *,
    model: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete portable identity persisted with a Run."""

    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise TypeError("sources must be an ordered sequence")
    normalized_sources = [_normalize_source_identity(source) for source in sources]
    if not normalized_sources:
        raise ValueError("input identity requires at least one source")
    payload = {
        "schema": INPUT_IDENTITY_SCHEMA,
        "model": _normalize_file_identity(model),
        "checkpoint": _normalize_checkpoint_identity(checkpoint),
        "sources": normalized_sources,
        "request": normalize_request_identity(request),
    }
    payload["fingerprint"] = input_identity_fingerprint(payload)
    return payload


def input_identity_fingerprint(identity: Mapping[str, Any]) -> str:
    """Return the canonical SHA256 for an identity, ignoring its stored digest."""

    if not isinstance(identity, Mapping):
        raise TypeError("input identity must be an object")
    unsigned = {str(key): value for key, value in identity.items() if str(key) != "fingerprint"}
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_input_identity(identity: Mapping[str, Any]) -> None:
    """Validate schema and the stored fingerprint before trusting an identity."""

    if not isinstance(identity, Mapping):
        raise TypeError("input identity must be an object")
    if str(identity.get("schema") or "") != INPUT_IDENTITY_SCHEMA:
        raise ValueError("unsupported Run input identity schema")
    stored = _normalize_sha256(identity.get("fingerprint"))
    computed = input_identity_fingerprint(identity)
    if not hmac.compare_digest(stored, computed):
        raise ValueError("Run input identity fingerprint is invalid")


def resolved_checkpoint_relative_path(identity: Mapping[str, Any]) -> str | None:
    """Return the frozen checkpoint path used by exact retry."""

    checkpoint = identity.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Run input identity checkpoint is missing")
    resolved = checkpoint.get("resolved")
    if resolved is None:
        return None
    if not isinstance(resolved, Mapping):
        raise ValueError("Run input identity resolved checkpoint is invalid")
    return _normalize_public_path(resolved.get("relative_path"), "resolved checkpoint path")


def diff_input_identities(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return field-level public differences, excluding redundant fingerprints."""

    expected_payload = {key: value for key, value in expected.items() if key != "fingerprint"}
    actual_payload = {key: value for key, value in actual.items() if key != "fingerprint"}
    differences: list[dict[str, Any]] = []
    _diff_values(expected_payload, actual_payload, "", differences)
    return differences


def compare_input_identities(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare identities for exact retry without exposing absolute paths."""

    differences = diff_input_identities(expected, actual)
    expected_fingerprint = input_identity_fingerprint(expected)
    actual_fingerprint = input_identity_fingerprint(actual)
    return {
        "matches": not differences and hmac.compare_digest(expected_fingerprint, actual_fingerprint),
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual_fingerprint,
        "differences": differences,
    }


def assert_input_identity_matches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    """Raise ``InputIdentityChanged`` when an exact retry is no longer exact."""

    comparison = compare_input_identities(expected, actual)
    if not comparison["matches"]:
        raise InputIdentityChanged(comparison)
    return comparison


def assert_input_identity_files_available(
    identity: Mapping[str, Any],
    *,
    models_root: str | Path,
    checkpoints_root: str | Path,
    videos_root: str | Path,
) -> None:
    """Raise a public identity mismatch when an exact-retry input disappeared.

    Exact Retry must report deleted or renamed files as identity drift before
    deep preflight turns the same condition into a generic validation error.
    The comparison is built only from the stored portable identity; resolved
    filesystem paths never enter the exception payload.
    """

    validate_input_identity(identity)
    actual = dict(identity)
    missing = False

    model = _normalize_file_identity(identity.get("model"))
    if not _portable_identity_file_exists(model, models_root):
        actual.pop("model", None)
        missing = True

    checkpoint = _normalize_checkpoint_identity(identity.get("checkpoint"))
    resolved_checkpoint = checkpoint.get("resolved")
    if resolved_checkpoint is not None and not _portable_identity_file_exists(
        resolved_checkpoint,
        checkpoints_root,
    ):
        actual_checkpoint = dict(identity.get("checkpoint") or {})
        actual_checkpoint.pop("resolved", None)
        actual["checkpoint"] = actual_checkpoint
        missing = True

    raw_sources = identity.get("sources")
    if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
        raise TypeError("Run input identity sources must be an ordered sequence")
    actual_sources: list[dict[str, Any]] | None = None
    for index, raw_source in enumerate(raw_sources):
        source = _normalize_source_identity(raw_source)
        if _portable_identity_file_exists(source["content"], videos_root):
            continue
        if actual_sources is None:
            actual_sources = [dict(row) for row in raw_sources]
        actual_source = dict(actual_sources[index])
        actual_source.pop("content", None)
        actual_sources[index] = actual_source
        missing = True
    if actual_sources is not None:
        actual["sources"] = actual_sources

    if missing:
        raise InputIdentityChanged(compare_input_identities(identity, actual))


def _portable_identity_file_exists(
    identity: Mapping[str, Any],
    trusted_root: str | Path,
) -> bool:
    relative = _normalize_public_path(identity.get("relative_path"), "relative_path")
    root = Path(trusted_root).resolve()
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate.is_file()


def _normalize_file_identity(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise TypeError("file identity must be an object")
    return build_file_identity_from_values(
        relative_path=str(identity.get("relative_path") or ""),
        display_path=str(identity.get("display_path") or ""),
        size_bytes=identity.get("size_bytes"),
        mtime_ns=identity.get("mtime_ns"),
        sha256=str(identity.get("sha256") or ""),
    )


def _normalize_checkpoint_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise TypeError("checkpoint identity must be an object")
    return build_checkpoint_identity(
        identity.get("requested"),
        resolved=identity.get("resolved"),
    )


def _normalize_source_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise TypeError("source identity must be an object")
    return build_source_identity(
        item_id=identity.get("item_id"),
        asset_id=identity.get("asset_id"),
        qualified_name=str(identity.get("qualified_name") or ""),
        file_identity=identity.get("content"),
    )


def _normalize_request_mapping(value: Mapping[str, Any], *, top_level: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        if not isinstance(raw_key, str):
            raise TypeError("request identity keys must be strings")
        key = raw_key.strip()
        if not key:
            raise ValueError("request identity keys must not be empty")
        if (top_level and key in _REQUEST_EXCLUDED_KEYS) or _private_request_path_key(key):
            continue
        result[key] = _normalize_request_value(value[raw_key])
    return result


def _normalize_request_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("request identity does not support non-finite numbers")
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        return _normalize_request_mapping(value, top_level=False)
    if isinstance(value, (list, tuple)):
        return [_normalize_request_value(item) for item in value]
    raise TypeError(f"request identity does not support {type(value).__name__}")


def _private_request_path_key(key: str) -> bool:
    lowered = key.lower()
    return lowered == "path" or lowered.endswith("_path") or lowered.endswith("_dir")


def _normalize_public_path(value: Any, field_name: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw or "\x00" in raw:
        raise ValueError(f"{field_name} must be a non-empty relative path")
    if raw.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(raw):
        raise ValueError(f"{field_name} must be relative")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be relative and cannot contain '..'")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{field_name} must identify a file")
    return normalized


def _normalize_sha256(value: Any) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    return digest


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _diff_values(expected: Any, actual: Any, field: str, output: list[dict[str, Any]]) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            public_key = _public_field_key(key)
            child_field = f"{field}.{public_key}" if field else public_key
            if key not in actual:
                output.append(
                    {
                        "field": child_field,
                        "kind": "missing",
                        "expected": _public_value(expected[key]),
                        "actual": None,
                    }
                )
            elif key not in expected:
                output.append(
                    {
                        "field": child_field,
                        "kind": "added",
                        "expected": None,
                        "actual": _public_value(actual[key]),
                    }
                )
            else:
                _diff_values(expected[key], actual[key], child_field, output)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        for index in range(max(len(expected), len(actual))):
            child_field = f"{field}[{index}]"
            if index >= len(actual):
                output.append(
                    {
                        "field": child_field,
                        "kind": "missing",
                        "expected": _public_value(expected[index]),
                        "actual": None,
                    }
                )
            elif index >= len(expected):
                output.append(
                    {
                        "field": child_field,
                        "kind": "added",
                        "expected": None,
                        "actual": _public_value(actual[index]),
                    }
                )
            else:
                _diff_values(expected[index], actual[index], child_field, output)
        return
    if type(expected) is type(actual) and expected == actual:
        return
    output.append(
        {
            "field": field or "$",
            "kind": "changed",
            "expected": _public_value(expected),
            "actual": _public_value(actual),
        }
    )


def _public_value(value: Any) -> Any:
    if isinstance(value, str):
        return "<redacted-path>" if _looks_like_absolute_path(value) else value
    if isinstance(value, Mapping):
        return {_public_field_key(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [_public_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def _looks_like_absolute_path(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.replace("\\", "/")
    return (
        normalized.startswith("/")
        or normalized.lower().startswith("file://")
        or bool(_WINDOWS_ABSOLUTE_RE.match(stripped))
    )


def _public_field_key(value: Any) -> str:
    key = str(value)
    return "<redacted-key>" if _looks_like_absolute_path(key) else key
