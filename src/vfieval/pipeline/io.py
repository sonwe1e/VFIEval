from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_rgb_tensor(path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.to(device)


def resize_batch(batch: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if batch.shape[-2:] == (height, width):
        return batch
    return F.interpolate(batch, size=(height, width), mode="bilinear", align_corners=False)


def batch_tensors(items: list[torch.Tensor]) -> torch.Tensor:
    if not items:
        raise ValueError("cannot batch an empty tensor list")
    return torch.stack(items, dim=0)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {tuple(tensor.shape)}")
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return arr


def save_rgb_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)
