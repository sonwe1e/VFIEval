from __future__ import annotations

import hashlib
import inspect
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from vfieval.models.base import FlowMaskModel
from vfieval.models.dummy import DummyFlowMaskModel
from vfieval.models.utils import move_module_to_device


OUTPUT_KEYS = ("flowt_0", "flowt_1", "mask0", "mask1")


class InferModelAdapter:
    def __init__(self, infer_callable):
        self._infer = infer_callable

    def predict(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> dict[str, torch.Tensor]:
        return normalize_infer_output(self._infer(img0, img1))


def normalize_infer_output(output: object) -> dict[str, torch.Tensor]:
    if isinstance(output, dict):
        return dict(output)
    if isinstance(output, (tuple, list)):
        if len(output) != 4:
            raise TypeError(
                f"Model infer returned {type(output).__name__} with {len(output)} items; "
                "expected exactly 4 items: flowt_0, flowt_1, mask0, mask1"
            )
        return dict(zip(OUTPUT_KEYS, output))
    raise TypeError(
        f"Model infer returned {type(output).__name__}; expected a dict with "
        "flowt_0/flowt_1/mask0/mask1 or a 4-item tuple/list"
    )


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
        model = _call_model_factory(target, checkpoint_path, device, metadata or {})
    else:
        model = target
    _prepare_user_model(model, device)
    if hasattr(model, "infer") and not hasattr(model, "predict"):
        model = InferModelAdapter(model.infer)
    if callable(model) and not hasattr(model, "predict"):
        model = InferModelAdapter(model)
    if not hasattr(model, "predict"):
        raise TypeError(f"adapter '{adapter}' did not return an object with predict(img0, img1, t)")
    return model


def _prepare_user_model(model: object, device: str) -> None:
    seen: set[int] = set()
    _prepare_model_object(model, device, seen)
    for attr_name in ("model", "net", "network", "module"):
        child = getattr(model, attr_name, None)
        if child is not None:
            _prepare_model_object(child, device, seen)


def _prepare_model_object(model: object, device: str, seen: set[int]) -> None:
    identity = id(model)
    if identity in seen:
        return
    seen.add(identity)
    if callable(getattr(model, "to", None)) or callable(getattr(model, "eval", None)):
        move_module_to_device(model, device)


def load_model_file(
    path: Path,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    metadata: dict[str, Any] | None = None,
) -> FlowMaskModel:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {path}")
    if path.suffix.lower() != ".py":
        raise ValueError(f"model file must be a .py file: {path.name}")

    digest = hashlib.sha1(f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}".encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location(f"vfieval_user_model_{digest}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to import model file: {path}")
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
        model = _call_model_factory(model_class, checkpoint_path, device, metadata or {})
        if not hasattr(model, "infer"):
            raise TypeError(f"{path.name} Model class must define infer(img0, img1)")
        _prepare_user_model(model, device)
        return InferModelAdapter(model.infer)

    if hasattr(module, "infer"):
        infer = getattr(module, "infer")
        if not callable(infer):
            raise TypeError(f"{path.name} infer is not callable")
        return InferModelAdapter(infer)

    raise TypeError(f"{path.name} must define class Model with infer(img0, img1), or top-level infer(img0, img1)")


def _call_model_factory(factory, checkpoint_path: str | None, device: str, metadata: dict[str, Any]) -> object:
    signature = inspect.signature(factory)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return factory(checkpoint_path=checkpoint_path, device=device, metadata=metadata)
    if any(name in parameters for name in ("checkpoint_path", "device", "metadata")):
        kwargs: dict[str, Any] = {}
        if "checkpoint_path" in parameters:
            kwargs["checkpoint_path"] = checkpoint_path
        if "device" in parameters:
            kwargs["device"] = device
        if "metadata" in parameters:
            kwargs["metadata"] = metadata
        return factory(**kwargs)
    positional = [
        param
        for param in parameters.values()
        if param.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if len(positional) >= 2:
        return factory(checkpoint_path, device)
    return factory()
