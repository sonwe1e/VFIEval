from __future__ import annotations

import importlib.util
import hashlib
import json
import math
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
from vfieval.metrics.health import metric_health, record_feature_metric_validation


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
WEIGHT_LOAD_CONTRACT = "strict-state-dict-v1"
FEATURE_CONFORMANCE_CONTRACT = "feature-distance-smoke-v1"
CONFORMANCE_IDENTITY_TOLERANCE = 1e-6
CONFORMANCE_MIN_PERTURBED_DISTANCE = 1e-8


class DinoPatchMetric:
    name = "lpips_vit_patch"

    def __init__(self, workspace: WorkspaceConfig | None = None, device: str | None = None):
        self.workspace = workspace or WorkspaceConfig.from_root()
        self.device_name = device or "cpu"
        self._health: dict[str, Any] | None = None
        self._device: torch.device | None = None
        self._model = None
        self._weight_load_report: dict[str, Any] | None = None
        self._conformance_report: dict[str, Any] | None = None
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
            raise MetricUnavailable(
                f"{self.name} metric is {health['status']}: {health['reason']}",
                _feature_details(self.name, self.device_name, health),
            )
        device = _resolve_metric_device(self.device_name)
        started = time.perf_counter()
        try:
            self._model = _load_dino_model(health, device)
        except MetricUnavailable as exc:
            self._record_failed_validation(health, exc)
            raise
        except Exception as exc:
            raise MetricUnavailable(f"metric device {self.device_name} failed model load: {exc}") from exc
        self._timings["model_load_seconds"] += time.perf_counter() - started
        self._model_load_count += 1
        self._weight_load_report = getattr(self._model, "_vfieval_weight_load_report", None)
        self._conformance_report = getattr(self._model, "_vfieval_conformance_report", None)
        self._record_successful_validation(health)
        self._health, self._device = health, device
        return health, device, self._model

    def _record_failed_validation(self, health: dict[str, Any], exc: MetricUnavailable) -> None:
        details = dict(getattr(exc, "details", {}) or {})
        if details.get("validation_scope") != "asset":
            return
        record_feature_metric_validation(
            self.workspace,
            self.name,
            str(health.get("implementation_fingerprint") or ""),
            {"status": "incompatible", **details},
        )

    def _record_successful_validation(self, health: dict[str, Any]) -> None:
        if not self._weight_load_report:
            return
        record_feature_metric_validation(
            self.workspace,
            self.name,
            str(health.get("implementation_fingerprint") or ""),
            {
                "status": "compatible",
                "weight_load": self._weight_load_report,
                "conformance": self._conformance_report,
            },
        )

    def _warmup(self, model, sample: torch.Tensor, shape: tuple[int, int]) -> None:
        if shape in self._warmed_shapes:
            return
        started = time.perf_counter()
        with torch.inference_mode():
            _dino_patch_features(model, torch.zeros_like(sample))
        self._timings["warmup_seconds"] += time.perf_counter() - started
        self._warmed_shapes.add(shape)

    def _details(self, health: dict[str, Any]) -> dict[str, Any]:
        return _feature_details(
            self.name,
            self.device_name,
            health,
            weight_load=self._weight_load_report,
            conformance=self._conformance_report,
        )

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
        self._weight_load_report: dict[str, Any] | None = None
        self._conformance_report: dict[str, Any] | None = None
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
                    features = _convnext_features(model, torch.cat([refs, dists], dim=0))
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
            raise MetricUnavailable(
                f"{self.name} metric is {health['status']}: {health['reason']}",
                _feature_details(self.name, self.device_name, health),
            )
        device = _resolve_metric_device(self.device_name)
        started = time.perf_counter()
        try:
            self._model = _load_convnext_model(health, device)
        except MetricUnavailable as exc:
            self._record_failed_validation(health, exc)
            raise
        except Exception as exc:
            raise MetricUnavailable(f"metric device {self.device_name} failed model load: {exc}") from exc
        self._timings["model_load_seconds"] += time.perf_counter() - started
        self._model_load_count += 1
        self._weight_load_report = getattr(self._model, "_vfieval_weight_load_report", None)
        self._conformance_report = getattr(self._model, "_vfieval_conformance_report", None)
        self._record_successful_validation(health)
        self._health, self._device = health, device
        return health, device, self._model

    def _record_failed_validation(self, health: dict[str, Any], exc: MetricUnavailable) -> None:
        details = dict(getattr(exc, "details", {}) or {})
        if details.get("validation_scope") != "asset":
            return
        record_feature_metric_validation(
            self.workspace,
            self.name,
            str(health.get("implementation_fingerprint") or ""),
            {"status": "incompatible", **details},
        )

    def _record_successful_validation(self, health: dict[str, Any]) -> None:
        if not self._weight_load_report:
            return
        record_feature_metric_validation(
            self.workspace,
            self.name,
            str(health.get("implementation_fingerprint") or ""),
            {
                "status": "compatible",
                "weight_load": self._weight_load_report,
                "conformance": self._conformance_report,
            },
        )

    def _warmup(self, model, sample: torch.Tensor, shape: tuple[int, int]) -> None:
        if shape in self._warmed_shapes:
            return
        started = time.perf_counter()
        with torch.inference_mode():
            _convnext_features(model, torch.zeros_like(sample))
        self._timings["warmup_seconds"] += time.perf_counter() - started
        self._warmed_shapes.add(shape)

    def _details(self, health: dict[str, Any]) -> dict[str, Any]:
        return _feature_details(
            self.name,
            self.device_name,
            health,
            weight_load=self._weight_load_report,
            conformance=self._conformance_report,
        )

    def performance(self) -> dict[str, Any]:
        return {**self._timings, "model_load_count": self._model_load_count, "warmed_shape_count": len(self._warmed_shapes)}


def _feature_details(
    name: str,
    device_name: str,
    health: dict[str, Any],
    *,
    weight_load: dict[str, Any] | None = None,
    conformance: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "manifest_fingerprint": health.get("manifest_fingerprint"),
        "driver_fingerprint": health.get("driver_fingerprint"),
        "weights_fingerprint": health.get("weights_fingerprint"),
        "implementation_fingerprint": health.get("implementation_fingerprint"),
        "weight_load": weight_load or health.get("weight_load_validation"),
        "conformance": conformance,
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
    report = _load_state_dict(model, weights_path)
    model = model.to(device).eval()
    try:
        conformance = _run_dino_conformance(model, device)
    except MetricUnavailable as exc:
        raise MetricUnavailable(
            str(exc),
            {**dict(exc.details), "weight_load": report},
        ) from exc
    setattr(model, "_vfieval_weight_load_report", report)
    setattr(model, "_vfieval_conformance_report", conformance)
    return model


def _load_convnext_model(health: dict[str, Any], device: torch.device):
    if importlib.util.find_spec("timm") is None:
        raise MetricUnavailable("lpips_convnext requires the timm package")
    import timm

    backbone = str(health.get("backbone") or "convnextv2_tiny.fcmae_ft_in22k_in1k")
    weights_path = Path(str(health.get("weights_path") or "")).resolve()
    # Load the original model strictly before exposing intermediate feature
    # maps. timm's ``features_only`` wrapper rewrites root parameter names
    # (for example ``stem.0`` -> ``stem_0``), so loading a classifier
    # checkpoint into that wrapper with strict=False can silently match zero
    # backbone parameters and produce random-feature scores.
    model = timm.create_model(backbone, pretrained=False)
    report = _load_state_dict(model, weights_path)
    model = model.to(device).eval()
    try:
        conformance = _run_convnext_conformance(model, device)
    except MetricUnavailable as exc:
        raise MetricUnavailable(
            str(exc),
            {**dict(exc.details), "weight_load": report},
        ) from exc
    setattr(model, "_vfieval_weight_load_report", report)
    setattr(model, "_vfieval_conformance_report", conformance)
    return model


def _load_state_dict(model, weights_path: Path) -> dict[str, Any]:
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
    if not isinstance(state, dict) or not state:
        report = _weight_load_report(model, {})
        raise MetricUnavailable(
            "metric checkpoint does not contain a non-empty state dictionary",
            {"validation_scope": "asset", "weight_load": report},
        )
    normalized = _normalize_state_dict_keys(state, set(model.state_dict()))
    report = _weight_load_report(model, normalized)
    if (
        not report["matched_key_count"]
        or report["missing_keys"]
        or report["unexpected_keys"]
        or report["shape_mismatches"]
    ):
        raise MetricUnavailable(
            "metric checkpoint is structurally incompatible with the declared backbone",
            {"validation_scope": "asset", "weight_load": report},
        )
    try:
        model.load_state_dict(normalized, strict=True)
    except Exception as exc:
        raise MetricUnavailable(
            f"metric checkpoint strict load failed: {exc}",
            {"validation_scope": "asset", "weight_load": report},
        ) from exc
    return report


def _normalize_state_dict_keys(state: dict[str, Any], model_keys: set[str]) -> dict[str, Any]:
    normalized = {str(key): value for key, value in state.items()}
    candidates = [normalized]
    for prefix in ("module.", "model.", "backbone."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            candidates.append({key[len(prefix) :]: value for key, value in normalized.items()})
    return max(candidates, key=lambda candidate: len(set(candidate) & model_keys))


def _weight_load_report(model, state: dict[str, Any]) -> dict[str, Any]:
    model_state = model.state_dict()
    model_keys = set(model_state)
    state_keys = set(state)
    common = sorted(model_keys & state_keys)
    shape_mismatches = []
    matched = []
    for key in common:
        expected_shape = tuple(getattr(model_state[key], "shape", ()))
        actual_shape = tuple(getattr(state[key], "shape", ()))
        if expected_shape != actual_shape:
            shape_mismatches.append(
                {"key": key, "expected": list(expected_shape), "actual": list(actual_shape)}
            )
        else:
            matched.append(key)
    matched_digest = hashlib.sha256("\n".join(matched).encode("utf-8")).hexdigest()
    return {
        "contract": WEIGHT_LOAD_CONTRACT,
        "status": "compatible"
        if matched and not (model_keys - state_keys) and not (state_keys - model_keys) and not shape_mismatches
        else "incompatible",
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(state_keys),
        "matched_key_count": len(matched),
        "matched_keys_fingerprint": matched_digest,
        "matched_key_examples": matched[:12],
        "missing_keys": sorted(model_keys - state_keys),
        "unexpected_keys": sorted(state_keys - model_keys),
        "shape_mismatches": shape_mismatches,
    }


def _run_dino_conformance(model, device: torch.device) -> dict[str, Any]:
    sample = _conformance_inputs(device, multiple=14, edge=56)
    try:
        with torch.inference_mode():
            features = _dino_patch_features(model, sample)
    except Exception as exc:
        raise MetricUnavailable(
            f"lpips_vit_patch conformance smoke failed: {exc}",
            {
                # A model may load correctly while a requested accelerator
                # lacks one of its forward operators. Do not poison the
                # device-independent asset health for that execution failure.
                "validation_scope": "device",
                "conformance": {
                    "contract": FEATURE_CONFORMANCE_CONTRACT,
                    "status": "failed",
                    "reason": str(exc),
                },
            },
        ) from exc
    try:
        identity = _feature_distance(features[:1], features[1:2])
        perturbed = _feature_distance(features[:1], features[2:3])
    except MetricUnavailable as exc:
        raise MetricUnavailable(
            f"lpips_vit_patch conformance distance validation failed: {exc}",
            {
                **dict(exc.details),
                "validation_scope": "runtime",
                "conformance": {
                    "contract": FEATURE_CONFORMANCE_CONTRACT,
                    "status": "failed",
                    "reason": str(exc),
                },
            },
        ) from exc
    except Exception as exc:
        raise MetricUnavailable(
            f"lpips_vit_patch conformance distance validation failed: {exc}",
            {
                "validation_scope": "runtime",
                "conformance": {
                    "contract": FEATURE_CONFORMANCE_CONTRACT,
                    "status": "failed",
                    "reason": str(exc),
                },
            },
        ) from exc
    return _validate_conformance("lpips_vit_patch", identity, perturbed)


def _run_convnext_conformance(model, device: torch.device) -> dict[str, Any]:
    sample = _conformance_inputs(device, multiple=32, edge=64)
    try:
        with torch.inference_mode():
            features = _convnext_features(model, sample)
    except Exception as exc:
        raise MetricUnavailable(
            f"lpips_convnext conformance smoke failed: {exc}",
            {
                "validation_scope": "device",
                "conformance": {
                    "contract": FEATURE_CONFORMANCE_CONTRACT,
                    "status": "failed",
                    "reason": str(exc),
                },
            },
        ) from exc
    try:
        identity_features, remainder = _split_feature_batch(features, 1)
        same_features, perturbed_features = _split_feature_batch(remainder, 1)
        identity = _feature_distance_list(identity_features, same_features)
        perturbed = _feature_distance_list(identity_features, perturbed_features)
    except MetricUnavailable as exc:
        raise MetricUnavailable(
            f"lpips_convnext conformance distance validation failed: {exc}",
            {
                **dict(exc.details),
                "validation_scope": "runtime",
                "conformance": {
                    "contract": FEATURE_CONFORMANCE_CONTRACT,
                    "status": "failed",
                    "reason": str(exc),
                },
            },
        ) from exc
    except Exception as exc:
        raise MetricUnavailable(
            f"lpips_convnext conformance distance validation failed: {exc}",
            {
                "validation_scope": "runtime",
                "conformance": {
                    "contract": FEATURE_CONFORMANCE_CONTRACT,
                    "status": "failed",
                    "reason": str(exc),
                },
            },
        ) from exc
    return _validate_conformance("lpips_convnext", identity, perturbed)


def _conformance_inputs(device: torch.device, *, multiple: int, edge: int) -> torch.Tensor:
    aligned_edge = max(multiple, (int(edge) // multiple) * multiple)
    base = torch.zeros((1, 3, aligned_edge, aligned_edge), dtype=torch.float32, device=device)
    perturbed = base.clone()
    start = aligned_edge // 4
    stop = max(start + 1, aligned_edge // 2)
    perturbed[:, :, start:stop, start:stop] = 0.1
    return torch.cat([base, base.clone(), perturbed], dim=0)


def _validate_conformance(metric_name: str, identity: float, perturbed: float) -> dict[str, Any]:
    report = {
        "contract": FEATURE_CONFORMANCE_CONTRACT,
        "status": "passed",
        "identity_distance": float(identity),
        "perturbed_distance": float(perturbed),
        "identity_tolerance": CONFORMANCE_IDENTITY_TOLERANCE,
        "minimum_perturbed_distance": CONFORMANCE_MIN_PERTURBED_DISTANCE,
    }
    if not math.isfinite(identity) or not math.isfinite(perturbed):
        report["status"] = "failed"
        raise MetricUnavailable(
            f"{metric_name} conformance produced a non-finite distance",
            {"validation_scope": "runtime", "conformance": report},
        )
    if abs(identity) > CONFORMANCE_IDENTITY_TOLERANCE:
        report["status"] = "failed"
        raise MetricUnavailable(
            f"{metric_name} conformance identity distance is invalid: {identity}",
            {"validation_scope": "asset", "conformance": report},
        )
    if perturbed < 0.0:
        report["status"] = "failed"
        raise MetricUnavailable(
            f"{metric_name} conformance perturbed distance is invalid: {perturbed}",
            {"validation_scope": "asset", "conformance": report},
        )
    minimum = max(
        CONFORMANCE_MIN_PERTURBED_DISTANCE,
        float(identity) + CONFORMANCE_MIN_PERTURBED_DISTANCE,
    )
    report["required_perturbed_distance"] = minimum
    if perturbed <= minimum:
        report["status"] = "failed"
        raise MetricUnavailable(
            (
                f"{metric_name} conformance is semantically degenerate: perturbed distance "
                f"{perturbed} must be greater than {minimum} and the identity baseline"
            ),
            {"validation_scope": "asset", "conformance": report},
        )
    return report


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


def _convnext_features(model, tensor: torch.Tensor) -> Any:
    if hasattr(model, "forward_intermediates"):
        features = model.forward_intermediates(tensor, intermediates_only=True)
    elif hasattr(model, "forward_features"):
        features = [model.forward_features(tensor)]
    else:
        features = model(tensor)
    if isinstance(features, tuple) and len(features) == 2 and isinstance(features[1], (list, tuple)):
        features = features[1]
    if not isinstance(features, (list, tuple)):
        features = [features]
    if not features or not all(isinstance(value, torch.Tensor) for value in features):
        raise RuntimeError("ConvNeXt backbone did not return feature maps")
    return list(features)


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
    return _validate_distance_values(values.detach().cpu().tolist())


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
    return _validate_distance_values(combined.detach().cpu().tolist())


def _validate_distance_values(values: Any) -> list[float]:
    normalized = [float(value) for value in values]
    if not normalized or any(not math.isfinite(value) or value < -1e-8 for value in normalized):
        raise MetricUnavailable(
            "feature metric produced a non-finite or negative distance",
            {
                "validation_scope": "runtime",
                "distance_values": normalized[:8],
            },
        )
    return [max(0.0, value) for value in normalized]


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
            "manifest_fingerprint": health.get("manifest_fingerprint"),
            "driver_fingerprint": health.get("driver_fingerprint"),
            "weights_fingerprint": health.get("weights_fingerprint"),
            "implementation_fingerprint": health.get("implementation_fingerprint"),
            "weight_load_validation": health.get("weight_load_validation"),
        },
        sort_keys=True,
    )
