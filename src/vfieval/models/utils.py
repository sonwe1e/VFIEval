from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint_cpu(checkpoint_path: str | Path) -> Any:
    """Load a checkpoint on CPU so it can be moved to any worker device."""
    return torch.load(str(checkpoint_path), map_location="cpu")


def load_state_dict_portable(module: Any, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint into ``module`` and return a structured load report.

    The report ({checkpoint_path, matched, total_in_checkpoint, missing_keys,
    unexpected_keys}) is also attached to ``module._last_load_report`` so the
    inference pipeline can surface it in the run detail UI. ``strict=False`` is
    used deliberately so partial loads surface as diagnostics rather than a
    hard failure — callers that need strictness must inspect the report.
    """
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    incompatible = module.load_state_dict(state_dict, strict=False)
    move_module_to_device(module, device)
    missing = [str(key) for key in getattr(incompatible, "missing_keys", []) or []]
    unexpected = [str(key) for key in getattr(incompatible, "unexpected_keys", []) or []]
    total = len(state_dict) if hasattr(state_dict, "__len__") else 0
    report = {
        "checkpoint_path": str(checkpoint_path),
        "matched": max(0, total - len(unexpected)),
        "total_in_checkpoint": total,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    try:
        setattr(module, "_last_load_report", report)
    except Exception:
        pass
    return report


def move_module_to_device(module: Any, device: str | torch.device) -> Any:
    to_device = getattr(module, "to", None)
    if callable(to_device):
        to_device(device)
    eval_module = getattr(module, "eval", None)
    if callable(eval_module):
        eval_module()
    return module


def _state_dict_from_checkpoint(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint
