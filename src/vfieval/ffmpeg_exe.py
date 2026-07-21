"""Shared helper for locating the ffmpeg binary used for video encode/decode.

Set the ``VFIEVAL_VIDEO_FFMPEG`` environment variable to an absolute path to
override the system PATH lookup.  When the variable is unset or empty the
helper falls back to ``shutil.which("ffmpeg")``.
"""
from __future__ import annotations

import os
import shutil


def resolve_ffmpeg() -> str | None:
    """Return the ffmpeg executable path.

    Priority:
    1. ``VFIEVAL_VIDEO_FFMPEG`` environment variable (if non-empty)
    2. ``shutil.which("ffmpeg")`` (PATH lookup)

    Returns ``None`` when ffmpeg cannot be found by either method.
    """
    override = os.environ.get("VFIEVAL_VIDEO_FFMPEG", "").strip()
    if override:
        return override
    return shutil.which("ffmpeg")
