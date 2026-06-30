from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint_cpu(checkpoint_path: str | Path) -> Any:
    """Load a checkpoint on CPU so it can be moved to any worker device."""
    return torch.load(str(checkpoint_path), map_location="cpu")


def load_state_dict_portable(module: Any, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> Any:
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    result = module.load_state_dict(state_dict)
    move_module_to_device(module, device)
    return result


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
