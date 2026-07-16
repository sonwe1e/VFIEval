from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vfieval.pipeline.io import save_rgb_tensor, tensor_to_uint8_image


def save_mask(mask: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_2d = mask.detach().float().cpu().squeeze().clamp(0.0, 1.0).numpy()
    Image.fromarray((mask_2d * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def flow_to_rgb(flow: torch.Tensor) -> np.ndarray:
    flow_np = flow.detach().float().cpu().numpy()
    if flow_np.shape[0] != 2:
        raise ValueError(f"expected flow CHW with 2 channels, got {flow_np.shape}")
    u = flow_np[0]
    v = flow_np[1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hue = ((ang + math.pi) / (2.0 * math.pi)).astype(np.float32)
    sat = np.ones_like(hue, dtype=np.float32)
    value = mag / (np.percentile(mag, 99) + 1e-6)
    value = np.clip(value, 0.0, 1.0).astype(np.float32)
    return _hsv_to_rgb_uint8(hue, sat, value)


def _hsv_to_rgb_uint8(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    choices = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ]
    for idx, channels in enumerate(choices):
        mask = i_mod == idx
        for c, channel in enumerate(channels):
            rgb[..., c] = np.where(mask, channel, rgb[..., c])
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def save_flow(flow: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(flow_to_rgb(flow)).save(path)


def save_difference(pred: torch.Tensor, reference: torch.Tensor, path: str | Path) -> None:
    diff = (pred.detach().float().cpu() - reference.detach().float().cpu()).abs().clamp(0.0, 1.0)
    save_rgb_tensor(diff, path)


def save_visual_bundle(bundle: dict[str, torch.Tensor], sample_dir: str | Path, index: int = 0) -> dict[str, Path]:
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pred": sample_dir / "pred.png",
        "warp0": sample_dir / "warp0.png",
        "warp1": sample_dir / "warp1.png",
        "blend": sample_dir / "blend.png",
        "mask0": sample_dir / "mask0.png",
        "mask1": sample_dir / "mask1.png",
        "flowt_0": sample_dir / "flowt_0.png",
        "flowt_1": sample_dir / "flowt_1.png",
    }
    save_rgb_tensor(bundle["pred"][index], paths["pred"])
    save_rgb_tensor(bundle["warp0"][index], paths["warp0"])
    save_rgb_tensor(bundle["warp1"][index], paths["warp1"])
    save_rgb_tensor(bundle["blend"][index], paths["blend"])
    save_mask(bundle["mask0"][index], paths["mask0"])
    save_mask(bundle["mask1"][index], paths["mask1"])
    save_flow(bundle["flowt_0"][index], paths["flowt_0"])
    save_flow(bundle["flowt_1"][index], paths["flowt_1"])
    return paths


def save_extra_tensor(tensor: torch.Tensor, path: str | Path, index: int = 0) -> Path:
    path = Path(path)
    if tensor.ndim == 4:
        tensor = tensor[index]
    if tensor.ndim == 2:
        save_mask(tensor, path)
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        save_mask(tensor, path)
    elif tensor.ndim == 3 and tensor.shape[0] == 2:
        save_flow(tensor, path)
    elif tensor.ndim == 3 and tensor.shape[0] >= 3:
        save_rgb_tensor(_normalize_extra_rgb(tensor[:3]), path)
    else:
        raise ValueError(f"unsupported extra visualization tensor shape: {tuple(tensor.shape)}")
    return path


def save_preview_image(
    source_path: str | Path,
    preview_path: str | Path,
    max_edge: int = 512,
    *,
    height: int | None = None,
    width: int | None = None,
) -> Path:
    source_path = Path(source_path)
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path).convert("RGB") as image:
        if height is not None or width is not None:
            if height is None or width is None or int(height) <= 0 or int(width) <= 0:
                raise ValueError("preview height and width must both be positive")
            image = image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        else:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        image.save(preview_path)
    return preview_path


def _normalize_extra_rgb(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float()
    if float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0:
        return tensor
    low = tensor.amin(dim=(-2, -1), keepdim=True)
    high = tensor.amax(dim=(-2, -1), keepdim=True)
    return ((tensor - low) / (high - low + 1e-6)).clamp(0.0, 1.0)
