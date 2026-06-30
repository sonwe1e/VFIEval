from __future__ import annotations

import torch
import torch.nn.functional as F


REQUIRED_OUTPUTS = ("flowt_0", "flowt_1", "mask0", "mask1")


def validate_model_outputs(outputs: dict[str, torch.Tensor], img0: torch.Tensor) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if name not in outputs]
    if missing:
        raise ValueError(f"model output missing required fields: {', '.join(missing)}")

    batch, _channels, height, width = img0.shape
    flow_shape = (batch, 2, height, width)
    mask_shape = (batch, 1, height, width)
    for name in ("flowt_0", "flowt_1"):
        if not isinstance(outputs[name], torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(outputs[name].shape) != flow_shape:
            raise ValueError(f"{name} must have shape {flow_shape}, got {tuple(outputs[name].shape)}")
    for name in ("mask0", "mask1"):
        if not isinstance(outputs[name], torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(outputs[name].shape) != mask_shape:
            raise ValueError(f"{name} must have shape {mask_shape}, got {tuple(outputs[name].shape)}")
    for name in REQUIRED_OUTPUTS:
        if outputs[name].device != img0.device:
            raise ValueError(f"{name} must stay on device {img0.device}, got {outputs[name].device}")
        if outputs[name].dtype != img0.dtype:
            raise ValueError(f"{name} must keep dtype {img0.dtype}, got {outputs[name].dtype}")


def _pixel_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=-1)
    return grid.unsqueeze(0).expand(batch, height, width, 2)


def _normalize_grid(pixel_grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = pixel_grid[..., 0]
    y = pixel_grid[..., 1]
    if width > 1:
        x = 2.0 * x / float(width - 1) - 1.0
    else:
        x = torch.zeros_like(x)
    if height > 1:
        y = 2.0 * y / float(height - 1) - 1.0
    else:
        y = torch.zeros_like(y)
    return torch.stack((x, y), dim=-1)


def backward_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample `image` at target pixel plus backward `flow` pixel displacement."""
    if image.ndim != 4 or flow.ndim != 4:
        raise ValueError("image and flow must be BCHW tensors")
    batch, _channels, height, width = image.shape
    if tuple(flow.shape) != (batch, 2, height, width):
        raise ValueError(f"flow must have shape {(batch, 2, height, width)}, got {tuple(flow.shape)}")

    flow_hw2 = flow.permute(0, 2, 3, 1)
    grid = _pixel_grid(batch, height, width, image.device, image.dtype) + flow_hw2.to(dtype=image.dtype)
    normalized = _normalize_grid(grid, height, width)
    return F.grid_sample(image, normalized, mode="bilinear", padding_mode="border", align_corners=True)


def compose_interpolated(
    img0: torch.Tensor,
    img1: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    validate_model_outputs(outputs, img0)
    warp0 = backward_warp(img0, outputs["flowt_0"])
    warp1 = backward_warp(img1, outputs["flowt_1"])
    mask0 = torch.sigmoid(outputs["mask0"])
    mask1 = torch.sigmoid(outputs["mask1"])
    blended = mask0 * warp0 + (1.0 - mask0) * warp1
    pred = mask1 * img1 + (1.0 - mask1) * blended
    return {
        "warp0": warp0.clamp(0.0, 1.0),
        "warp1": warp1.clamp(0.0, 1.0),
        "mask0": mask0,
        "mask1": mask1,
        "blend": blended.clamp(0.0, 1.0),
        "pred": pred.clamp(0.0, 1.0),
    }
