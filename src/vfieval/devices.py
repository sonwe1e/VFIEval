from __future__ import annotations

from contextlib import nullcontext
import re
from typing import Any

import torch


def npu_module():
    try:
        import torch_npu  # type: ignore

        return torch_npu
    except ImportError:
        return None


def npu_is_available() -> bool:
    if npu_module() is None:
        return False
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "is_available"):
        try:
            return bool(npu.is_available())
        except Exception:
            return True
    return True


def npu_unavailable_reason() -> str | None:
    if npu_module() is None:
        return "torch_npu is not installed"
    if not npu_is_available() or not list_npu_devices():
        return "torch_npu is installed but no NPU devices were reported"
    return None


def npu_device_count() -> int:
    if npu_module() is None:
        return 0
    candidates = [
        getattr(getattr(torch, "npu", None), "device_count", None),
        getattr(getattr(npu_module(), "npu", None), "device_count", None),
    ]
    for fn in candidates:
        if callable(fn):
            try:
                return max(0, int(fn()))
            except Exception:
                continue
    return 0


def list_npu_devices() -> list[dict[str, Any]]:
    count = npu_device_count()
    devices = []
    for index in range(count):
        name = f"Ascend NPU {index}"
        get_name = getattr(getattr(torch, "npu", None), "get_device_name", None)
        if callable(get_name):
            try:
                name = str(get_name(index))
            except Exception:
                pass
        devices.append({"id": f"npu:{index}", "name": name, "index": index})
    return devices


def normalize_device_name(device_name: str) -> str:
    requested = str(device_name or "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if list_npu_devices():
            return "npu:0"
        return "cpu"
    if requested == "cuda":
        return "cuda:0"
    if requested == "npu":
        return "npu:0"
    return requested


def device_type_name(device_name: str | torch.device) -> str:
    return str(device_name).split(":", 1)[0]


def npu_device_index(device_name: str | torch.device) -> int:
    normalized = normalize_device_name(str(device_name))
    match = re.fullmatch(r"npu(?::(\d+))?", normalized)
    if not match:
        raise ValueError(f"not an NPU device: {device_name}")
    return int(match.group(1) or 0)


def set_npu_device(device_name: str | torch.device) -> None:
    normalized = normalize_device_name(str(device_name))
    if not normalized.startswith("npu"):
        return
    reason = npu_unavailable_reason()
    if reason is not None:
        raise RuntimeError(f"NPU device requested but {reason}")
    set_device = getattr(getattr(torch, "npu", None), "set_device", None)
    if callable(set_device):
        index = npu_device_index(normalized)
        try:
            set_device(index)
        except TypeError:
            set_device(normalized)


def prepare_worker_device(device_filter: str | None) -> None:
    if device_filter and normalize_device_name(device_filter).startswith("npu"):
        set_npu_device(device_filter)


def resolve_torch_device(device_name: str) -> torch.device:
    normalized = normalize_device_name(device_name)
    if normalized.startswith("npu"):
        set_npu_device(normalized)
    return torch.device(normalized)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if device.type in {"cuda", "npu"}:
        try:
            return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)
        except Exception:
            return nullcontext()
    return nullcontext()


def supported_precisions(kind: str, available: bool | None = None) -> list[str]:
    if kind == "cpu":
        return ["fp32"]
    if kind == "cuda":
        if available is None:
            available = torch.cuda.is_available()
        if not available:
            return []
        supported = ["fp32", "fp16"]
        if _cuda_bf16_supported():
            supported.append("bf16")
        return supported
    if kind == "npu":
        if available is None:
            available = npu_is_available()
        if not available:
            return []
        supported = ["fp32", "fp16"]
        if _npu_bf16_supported():
            supported.append("bf16")
        return supported
    return []


def _cuda_bf16_supported() -> bool:
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def _npu_bf16_supported() -> bool:
    module = npu_module()
    candidates = [
        getattr(getattr(torch, "npu", None), "is_bf16_supported", None),
        getattr(getattr(module, "npu", None), "is_bf16_supported", None),
        getattr(module, "is_bf16_supported", None),
    ]
    for candidate in candidates:
        supported = _probe_npu_bf16(candidate)
        if supported is not None:
            return supported
    return False


def _probe_npu_bf16(candidate: Any) -> bool | None:
    if not callable(candidate):
        return None
    for args in ((), (0,), ("npu:0",)):
        try:
            return bool(candidate(*args))
        except TypeError:
            continue
        except Exception:
            return None
    return None
