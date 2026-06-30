from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from vfieval.models.base import FlowMaskModel
from vfieval.models.dummy import DummyFlowMaskModel


OUTPUT_KEYS = ("flowt_0", "flowt_1", "mask0", "mask1")


class InferModelAdapter:
    def __init__(self, infer_callable):
        self._infer = infer_callable

    def predict(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> dict[str, torch.Tensor]:
        return normalize_infer_output(self._infer(img0, img1))


def normalize_infer_output(output: object) -> dict[str, torch.Tensor]:
    if isinstance(output, dict):
        return dict(output)
    if isinstance(output, (tuple, list)) and len(output) == 4:
        return dict(zip(OUTPUT_KEYS, output))
    raise TypeError("模型 infer 必须返回包含 flowt_0/flowt_1/mask0/mask1 的 dict，或四元组")


def load_flow_mask_model(
    adapter: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    metadata: dict[str, Any] | None = None,
) -> FlowMaskModel:
    """Load a model adapter.

    Supported adapter strings:
    - "dummy"
    - "file:/absolute/path/to/model.py"
    - "module.submodule:factory_or_object"

    Factories may accept keyword arguments `checkpoint_path`, `device`, and `metadata`.
    """
    if adapter == "dummy":
        return DummyFlowMaskModel(device=device)
    if adapter.startswith("file:"):
        return load_model_file(Path(adapter.removeprefix("file:")), checkpoint_path, device, metadata or {})

    if ":" not in adapter:
        raise ValueError(f"unknown adapter '{adapter}'. Use 'dummy' or 'module:factory'.")

    module_name, attr_name = adapter.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    if callable(target):
        try:
            model = target(checkpoint_path=checkpoint_path, device=device, metadata=metadata or {})
        except TypeError:
            model = target(checkpoint_path, device)
    else:
        model = target
    if hasattr(model, "infer") and not hasattr(model, "predict"):
        model = InferModelAdapter(model.infer)
    if callable(model) and not hasattr(model, "predict"):
        model = InferModelAdapter(model)
    if not hasattr(model, "predict"):
        raise TypeError(f"adapter '{adapter}' did not return an object with predict(img0, img1, t)")
    return model


def load_model_file(
    path: Path,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    metadata: dict[str, Any] | None = None,
) -> FlowMaskModel:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    if path.suffix.lower() != ".py":
        raise ValueError(f"模型文件必须是 .py: {path.name}")

    digest = hashlib.sha1(f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}".encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location(f"vfieval_user_model_{digest}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模型文件: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass

    if hasattr(module, "Model"):
        model_class = getattr(module, "Model")
        try:
            model = model_class(checkpoint_path=checkpoint_path, device=device, metadata=metadata or {})
        except TypeError:
            model = model_class()
        if not hasattr(model, "infer"):
            raise TypeError(f"{path.name} 的 Model 类缺少 infer(img0, img1)")
        return InferModelAdapter(model.infer)

    if hasattr(module, "infer"):
        infer = getattr(module, "infer")
        if not callable(infer):
            raise TypeError(f"{path.name} 的 infer 不是可调用对象")
        return InferModelAdapter(infer)

    raise TypeError(f"{path.name} 必须定义 class Model 且包含 infer(img0, img1)，或定义顶层 infer(img0, img1)")
