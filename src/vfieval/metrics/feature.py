from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.devices import resolve_torch_device
from vfieval.metrics.base import MetricBatchOutOfMemory, MetricResult, MetricUnavailable
from vfieval.metrics.health import metric_health


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class DinoPatchMetric:
    name = "lpips_vit_patch"

    def __init__(self, workspace: WorkspaceConfig | None = None, device: str | None = None):
        self.workspace = workspace or WorkspaceConfig.from_root()
        self.device_name = device or "cpu"
        self._health: dict[str, Any] | None = None
        self._device: torch.device | None = None
        self._model = None
        self._warmed_shapes: set[tuple[int, int]] = set()
        self._timings = {"model_load_seconds": 0.0, "warmup_seconds": 0.0, "preprocess_seconds": 0.0, "compute_seconds": 0.0}
        self._model_load_count = 0

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        return self.evaluate_batch([(reference, distorted, work_dir)])[0]

    def evaluate_batch(self, pairs: list[tuple[Path, Path, Path]]) -> list[MetricResult]:
        health, device, model = self._ensure_model()
        prepared = _prepare_image_pairs(pairs, int(health.get("input_size") or 518), 14, self.name, self._timings)
        results: list[MetricResult | None] = [None] * len(pairs)
        try:
            for shape, rows in _group_prepared_pairs(prepared).items():
                refs = torch.cat([row[1] for row in rows], dim=0).to(device)
                dists = torch.cat([row[2] for row in rows], dim=0).to(device)
                self._warmup(model, refs[:1], shape)
                started = time.perf_counter()
                with torch.inference_mode():
                    features = _dino_patch_features(model, torch.cat([refs, dists], dim=0))
                    values = _feature_distances(features[: len(rows)], features[len(rows) :])
                self._timings["compute_seconds"] += time.perf_counter() - started
                for (index, _ref, _dist), value in zip(rows, values):
                    results[index] = MetricResult("completed", float(value), self._details(health))
        except MetricUnavailable:
            raise
        except Exception as exc:
            if _is_out_of_memory(exc):
                _clear_device_cache(self.device_name)
                raise MetricBatchOutOfMemory(f"metric device {self.device_name} ran out of memory") from exc
            raise MetricUnavailable(
                f"metric device {self.device_name} failed warmup or evaluation: {exc}"
            ) from exc
        return [result for result in results if result is not None]

    def _ensure_model(self):
        if self._model is not None:
            return self._health, self._device, self._model
        health = metric_health(self.workspace, self.name)
        if not health.get("available"):
            raise MetricUnavailable(f"{self.name} metric is {health['status']}: {health['reason']}")
        device = _resolve_metric_device(self.device_name)
        started = time.perf_counter()
        try:
            self._model = _load_dino_model(health, device)
        except MetricUnavailable:
            raise
        except Exception as exc:
            raise MetricUnavailable(f"metric device {self.device_name} failed model load: {exc}") from exc
        self._timings["model_load_seconds"] += time.perf_counter() - started
        self._model_load_count += 1
        self._health, self._device = health, device
        return health, device, self._model

    def _warmup(self, model, sample: torch.Tensor, shape: tuple[int, int]) -> None:
        if shape in self._warmed_shapes:
            return
        started = time.perf_counter()
        with torch.inference_mode():
            _dino_patch_features(model, torch.zeros_like(sample))
        self._timings["warmup_seconds"] += time.perf_counter() - started
        self._warmed_shapes.add(shape)

    def _details(self, health: dict[str, Any]) -> dict[str, Any]:
        return _feature_details(self.name, self.device_name, health)

    def performance(self) -> dict[str, Any]:
        return {**self._timings, "model_load_count": self._model_load_count, "warmed_shape_count": len(self._warmed_shapes)}


class ConvNextFeatureMetric:
    name = "lpips_convnext"

    def __init__(self, workspace: WorkspaceConfig | None = None, device: str | None = None):
        self.workspace = workspace or WorkspaceConfig.from_root()
        self.device_name = device or "cpu"
        self._health: dict[str, Any] | None = None
        self._device: torch.device | None = None
        self._model = None
        self._warmed_shapes: set[tuple[int, int]] = set()
        self._timings = {"model_load_seconds": 0.0, "warmup_seconds": 0.0, "preprocess_seconds": 0.0, "compute_seconds": 0.0}
        self._model_load_count = 0

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        return self.evaluate_batch([(reference, distorted, work_dir)])[0]

    def evaluate_batch(self, pairs: list[tuple[Path, Path, Path]]) -> list[MetricResult]:
        health, device, model = self._ensure_model()
        prepared = _prepare_image_pairs(pairs, int(health.get("input_size") or 288), 32, self.name, self._timings)
        results: list[MetricResult | None] = [None] * len(pairs)
        try:
            for shape, rows in _group_prepared_pairs(prepared).items():
                refs = torch.cat([row[1] for row in rows], dim=0).to(device)
                dists = torch.cat([row[2] for row in rows], dim=0).to(device)
                self._warmup(model, refs[:1], shape)
                started = time.perf_counter()
                with torch.inference_mode():
                    features = model(torch.cat([refs, dists], dim=0))
                    ref_features, dist_features = _split_feature_batch(features, len(rows))
                    values = _feature_distances_list(ref_features, dist_features)
                self._timings["compute_seconds"] += time.perf_counter() - started
                for (index, _ref, _dist), value in zip(rows, values):
                    results[index] = MetricResult("completed", float(value), self._details(health))
        except MetricUnavailable:
            raise
        except Exception as exc:
            if _is_out_of_memory(exc):
                _clear_device_cache(self.device_name)
                raise MetricBatchOutOfMemory(f"metric device {self.device_name} ran out of memory") from exc
            raise MetricUnavailable(
                f"metric device {self.device_name} failed warmup or evaluation: {exc}"
            ) from exc
        return [result for result in results if result is not None]

    def _ensure_model(self):
        if self._model is not None:
            return self._health, self._device, self._model
        health = metric_health(self.workspace, self.name)
        if not health.get("available"):
            raise MetricUnavailable(f"{self.name} metric is {health['status']}: {health['reason']}")
        device = _resolve_metric_device(self.device_name)
        started = time.perf_counter()
        try:
            self._model = _load_convnext_model(health, device)
        except MetricUnavailable:
            raise
        except Exception as exc:
            raise MetricUnavailable(f"metric device {self.device_name} failed model load: {exc}") from exc
        self._timings["model_load_seconds"] += time.perf_counter() - started
        self._model_load_count += 1
        self._health, self._device = health, device
        return health, device, self._model

    def _warmup(self, model, sample: torch.Tensor, shape: tuple[int, int]) -> None:
        if shape in self._warmed_shapes:
            return
        started = time.perf_counter()
        with torch.inference_mode():
            model(torch.zeros_like(sample))
        self._timings["warmup_seconds"] += time.perf_counter() - started
        self._warmed_shapes.add(shape)

    def _details(self, health: dict[str, Any]) -> dict[str, Any]:
        return _feature_details(self.name, self.device_name, health)

    def performance(self) -> dict[str, Any]:
        return {**self._timings, "model_load_count": self._model_load_count, "warmed_shape_count": len(self._warmed_shapes)}


def _feature_details(name: str, device_name: str, health: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric_name": name,
        "backbone": health.get("backbone"),
        "input_size": int(health.get("input_size") or (518 if name == "lpips_vit_patch" else 288)),
        "eval_resolution": health.get("eval_resolution"),
        "pad_multiple": health.get("pad_multiple"),
        "normalize": health.get("normalize"),
        "device": device_name,
        "device_policy": health.get("device_policy"),
        "manifest_path": health.get("manifest_path"),
        "implementation_mode": health.get("implementation_mode"),
    }


def _prepare_image_pairs(
    pairs: list[tuple[Path, Path, Path]],
    input_size: int,
    multiple: int,
    metric_name: str,
    timings: dict[str, float],
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    started = time.perf_counter()
    prepared = []
    for index, (reference, distorted, _work_dir) in enumerate(pairs):
        if reference.suffix.lower() not in IMAGE_SUFFIXES or distorted.suffix.lower() not in IMAGE_SUFFIXES:
            raise MetricUnavailable(f"{metric_name} requires reference and distorted image files.")
        ref = _load_image_tensor(reference, input_size, multiple=multiple)
        dist = _load_image_tensor(distorted, input_size, multiple=multiple)
        if ref.shape != dist.shape:
            raise MetricUnavailable(f"{metric_name} requires spatially aligned image pairs.")
        prepared.append((index, ref, dist))
    timings["preprocess_seconds"] += time.perf_counter() - started
    return prepared


def _group_prepared_pairs(
    prepared: list[tuple[int, torch.Tensor, torch.Tensor]],
) -> dict[tuple[int, int], list[tuple[int, torch.Tensor, torch.Tensor]]]:
    groups: dict[tuple[int, int], list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
    for row in prepared:
        groups.setdefault((int(row[1].shape[-2]), int(row[1].shape[-1])), []).append(row)
    return groups


def _resolve_metric_device(device_name: str) -> torch.device:
    text = str(device_name or "cpu")
    try:
        # This resolves and binds NPU devices before any metric model is
        # constructed.  It also keeps metric device handling aligned with the
        # inference/preflight paths instead of relying on the process default.
        return resolve_torch_device(text)
    except Exception as exc:
        raise MetricUnavailable(f"metric device {text} initialization failed: {exc}") from exc


def _load_dino_model(health: dict[str, Any], device: torch.device):
    repo_dir = Path(str(health.get("repo_dir") or "")).resolve()
    weights_path = Path(str(health.get("weights_path") or "")).resolve()
    backbone = str(health.get("backbone") or "dinov2_vits14_reg")
    model = torch.hub.load(str(repo_dir), backbone, source="local", pretrained=False)
    _load_state_dict(model, weights_path)
    return model.to(device).eval()


def _load_convnext_model(health: dict[str, Any], device: torch.device):
    if importlib.util.find_spec("timm") is None:
        raise MetricUnavailable("lpips_convnext requires the timm package")
    import timm

    backbone = str(health.get("backbone") or "convnextv2_tiny.fcmae_ft_in22k_in1k")
    weights_path = Path(str(health.get("weights_path") or "")).resolve()
    model = timm.create_model(backbone, pretrained=False, features_only=True)
    _load_state_dict(model, weights_path)
    return model.to(device).eval()


def _load_state_dict(model, weights_path: Path) -> None:
    if weights_path.suffix.lower() == ".safetensors":
        if importlib.util.find_spec("safetensors") is None:
            raise MetricUnavailable("safetensors weights require the safetensors package")
        from safetensors.torch import load_file

        state = load_file(str(weights_path), device="cpu")
    else:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    model.load_state_dict(state, strict=False)


def _load_image_tensor(path: Path, input_size: int, multiple: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = float(input_size) / max(width, height)
        resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        image = image.resize(resized, Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = _normalize_imagenet(tensor)
    pad_h = (-tensor.shape[-2]) % multiple
    pad_w = (-tensor.shape[-1]) % multiple
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor


def _normalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _warmup_model(model, sample: torch.Tensor) -> None:
    warmup = torch.zeros_like(sample[:, :, : sample.shape[-2], : sample.shape[-1]])
    model(warmup)


def _dino_patch_features(model, tensor: torch.Tensor) -> torch.Tensor:
    output = model.forward_features(tensor) if hasattr(model, "forward_features") else model(tensor)
    if isinstance(output, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, torch.Tensor):
        if output.ndim == 3:
            return output[:, 1:] if output.shape[1] > 1 else output
        if output.ndim == 4:
            return output.flatten(2).transpose(1, 2)
    raise RuntimeError("DINO backbone did not return patch features")


def _feature_distance(reference: torch.Tensor, distorted: torch.Tensor) -> float:
    return float(_feature_distances(reference, distorted)[0])


def _feature_distances(reference: torch.Tensor, distorted: torch.Tensor) -> list[float]:
    count = min(reference.shape[1], distorted.shape[1]) if reference.ndim == 3 else None
    if count is not None:
        reference = reference[:, :count]
        distorted = distorted[:, :count]
    reference = F.normalize(reference.float(), dim=-1)
    distorted = F.normalize(distorted.float(), dim=-1)
    values = (reference - distorted).pow(2).sum(dim=-1).flatten(1).mean(dim=1)
    return [float(value) for value in values.detach().cpu().tolist()]


def _feature_distance_list(reference: Any, distorted: Any) -> float:
    return float(_feature_distances_list(reference, distorted)[0])


def _feature_distances_list(reference: Any, distorted: Any) -> list[float]:
    if not isinstance(reference, (list, tuple)):
        reference = [reference]
    if not isinstance(distorted, (list, tuple)):
        distorted = [distorted]
    values = []
    for ref, dist in zip(reference, distorted):
        if not isinstance(ref, torch.Tensor) or not isinstance(dist, torch.Tensor):
            continue
        ref_norm = F.normalize(ref.float(), dim=1)
        dist_norm = F.normalize(dist.float(), dim=1)
        height = min(ref_norm.shape[-2], dist_norm.shape[-2])
        width = min(ref_norm.shape[-1], dist_norm.shape[-1])
        values.append(
            (ref_norm[..., :height, :width] - dist_norm[..., :height, :width])
            .pow(2)
            .sum(dim=1)
            .flatten(1)
            .mean(dim=1)
        )
    if not values:
        raise RuntimeError("ConvNeXt backbone did not return feature maps")
    combined = torch.stack(values).mean(dim=0)
    return [float(value) for value in combined.detach().cpu().tolist()]


def _split_feature_batch(features: Any, count: int) -> tuple[Any, Any]:
    if isinstance(features, (list, tuple)):
        return [value[:count] for value in features], [value[count:] for value in features]
    return features[:count], features[count:]


def _is_out_of_memory(exc: Exception) -> bool:
    text = str(exc).lower()
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    return (
        (oom_type is not None and isinstance(exc, oom_type))
        or type(exc).__name__.lower().endswith("outofmemoryerror")
        or "out of memory" in text
        or ("alloc" in text and "memory" in text)
    )


def _clear_device_cache(device_name: str) -> None:
    try:
        if str(device_name).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif str(device_name).startswith("npu") and hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except Exception:
        pass


def feature_metric_details_json(health: dict[str, Any]) -> str:
    return json.dumps(
        {
            "backbone": health.get("backbone"),
            "weights_path": health.get("weights_path"),
            "repo_dir": health.get("repo_dir"),
            "input_size": health.get("input_size"),
            "device_policy": health.get("device_policy"),
        },
        sort_keys=True,
    )
