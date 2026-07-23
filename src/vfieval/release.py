from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import os
import re
from typing import Any


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_VERSION = "0.1.0"


def package_release_metadata() -> dict[str, Any]:
    """Return release identity embedded in the installed package.

    Release builds replace ``_build_info.json`` only while building the wheel,
    so source checkouts retain the explicit ``development`` identity.
    """

    embedded: dict[str, Any] = {}
    try:
        resource = importlib.resources.files("vfieval").joinpath("_build_info.json")
        parsed = json.loads(resource.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            embedded = parsed
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        embedded = {}

    try:
        version = importlib.metadata.version("vfieval")
    except importlib.metadata.PackageNotFoundError:
        version = str(embedded.get("version") or _DEFAULT_VERSION)

    embedded_commit = str(embedded.get("commit_sha") or "").lower()
    commit_sha = embedded_commit if _COMMIT_RE.fullmatch(embedded_commit) else None
    embedded_build_id = str(embedded.get("build_id") or "").strip()
    build_id = os.getenv("VFIEVAL_BUILD_ID", "").strip() or embedded_build_id or "development"
    return {
        "version": version,
        "build_id": build_id,
        "commit_sha": commit_sha,
        "source_date_epoch": embedded.get("source_date_epoch"),
    }
