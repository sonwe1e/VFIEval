from __future__ import annotations

import torch
import torch.nn.functional as F


REQUIRED_OUTPUTS = ("flowt_0", "flowt_1", "mask0", "mask1")


def validate_model_outputs(outputs: dict[str, torch.Tensor], img0: torch.Tensor) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if name not in outputs]
    if missing:
        raise ValueError(f"model output missing required fields: {', '.join(missing)}")

    batch, _channels, _height, _width = img0.shape
    for name in ("flowt_0", "flowt_1"):
        if not isinstance(outputs[name], torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if outputs[name].ndim != 4:
            raise ValueError(f"{name} must be a BCHW tensor, got {tuple(outputs[name].shape)}")
        if outputs[name].shape[0] != batch or outputs[name].shape[1] != 2:
            raise ValueError(f"{name} must have shape [B,2,h,w] with B={batch}, got {tuple(outputs[name].shape)}")
        if outputs[name].shape[-2] <= 0 or outputs[name].shape[-1] <= 0:
            raise ValueError(f"{name} spatial size must be positive, got {tuple(outputs[name].shape)}")
    for name in ("mask0", "mask1"):
        if not isinstance(outputs[name], torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if outputs[name].ndim != 4:
            raise ValueError(f"{name} must be a BCHW tensor, got {tuple(outputs[name].shape)}")
        if outputs[name].shape[0] != batch or outputs[name].shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,h,w] with B={batch}, got {tuple(outputs[name].shape)}")
        if outputs[name].shape[-2] <= 0 or outputs[name].shape[-1] <= 0:
            raise ValueError(f"{name} spatial size must be positive, got {tuple(outputs[name].shape)}")
    for name in REQUIRED_OUTPUTS:
        if outputs[name].device != img0.device:
            raise ValueError(
                f"Model output field {name} is on {_device_label(outputs[name].device)}, "
                f"expected {_device_label(img0.device)}"
            )
        if outputs[name].dtype != img0.dtype:
            raise ValueError(
                f"Model output field {name} has dtype {outputs[name].dtype}, expected {img0.dtype}"
            )


def _device_label(device: torch.device) -> str:
    text = str(device)
    if text == "cpu":
        return "CPU"
    return text


def normalize_model_outputs(outputs: dict[str, torch.Tensor], img0: torch.Tensor) -> dict[str, torch.Tensor]:
    """Resize core model outputs to the inference resolution.

    Flow is in pixel units, so resizing from h/w to H/W also scales x/y
    displacement magnitudes. Mask tensors remain logits and are only resized.
    """
    validate_model_outputs(outputs, img0)
    _batch, _channels, height, width = img0.shape
    normalized = dict(outputs)
    for name in ("flowt_0", "flowt_1"):
        flow = outputs[name]
        src_h, src_w = int(flow.shape[-2]), int(flow.shape[-1])
        if (src_h, src_w) != (height, width):
            flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=True)
            scale_x = float(width) / float(src_w)
            scale_y = float(height) / float(src_h)
            scale = flow.new_tensor([scale_x, scale_y]).view(1, 2, 1, 1)
            flow = flow * scale
        normalized[name] = flow
    for name in ("mask0", "mask1"):
        mask = outputs[name]
        if tuple(mask.shape[-2:]) != (height, width):
            mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=True)
        normalized[name] = mask
    return normalized


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


def resize_bundle(bundle: dict[str, torch.Tensor], height: int, width: int) -> dict[str, torch.Tensor]:
    """Resize every tensor in a composed bundle to (height, width).

    Flow tensors are in pixel units, so resizing scales their displacement
    magnitudes to match the new resolution. RGB/mask tensors are resized
    directly. Tensors already at the target size are returned unchanged.
    """
    resized: dict[str, torch.Tensor] = {}
    for name, tensor in bundle.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            resized[name] = tensor
            continue
        src_h, src_w = int(tensor.shape[-2]), int(tensor.shape[-1])
        if (src_h, src_w) == (height, width):
            resized[name] = tensor
            continue
        out = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=True)
        if name in ("flowt_0", "flowt_1"):
            scale = out.new_tensor([float(width) / float(src_w), float(height) / float(src_h)]).view(1, 2, 1, 1)
            out = out * scale
        resized[name] = out
    return resized


def compose_interpolated(
    img0: torch.Tensor,
    img1: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized = normalize_model_outputs(outputs, img0)
    warp0 = backward_warp(img0, normalized["flowt_0"])
    warp1 = backward_warp(img1, normalized["flowt_1"])
    mask0 = torch.sigmoid(normalized["mask0"])
    mask1 = torch.sigmoid(normalized["mask1"])
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


def _resize_to(tensor: torch.Tensor, height: int, width: int, *, mode: str = "bilinear") -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    align = None if mode == "nearest" else True
    return F.interpolate(tensor, size=(height, width), mode=mode, align_corners=align)


def compose_interpolated_native(
    img0: torch.Tensor,
    img1: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compose the interpolated frame at the model's native flow resolution.

    The default :func:`compose_interpolated` upsamples flow/mask to the input
    resolution before warping, so warp/blend/compose all run at full size. When
    the model emits flow/mask at a much smaller resolution (e.g. 208x448 for a
    832x1792 input) this is wasteful: the extra detail is synthesized by the
    upsampler, not the model. This variant instead downsamples the RGB inputs to
    the flow resolution, so warp/compose run at native size. Outputs are at the
    native flow resolution and can be resized once for visualization.
    """
    validate_model_outputs(outputs, img0)
    flow0 = outputs["flowt_0"]
    flow1 = outputs["flowt_1"]
    height, width = int(flow0.shape[-2]), int(flow0.shape[-1])
    # flow1 usually matches flow0; rescale (and rescale displacements) if not.
    if tuple(flow1.shape[-2:]) != (height, width):
        src_h, src_w = int(flow1.shape[-2]), int(flow1.shape[-1])
        flow1 = _resize_to(flow1, height, width)
        scale = flow1.new_tensor([float(width) / float(src_w), float(height) / float(src_h)]).view(1, 2, 1, 1)
        flow1 = flow1 * scale
    img0r = _resize_to(img0, height, width)
    img1r = _resize_to(img1, height, width)
    warp0 = backward_warp(img0r, flow0)
    warp1 = backward_warp(img1r, flow1)
    mask0 = torch.sigmoid(_resize_to(outputs["mask0"], height, width))
    mask1 = torch.sigmoid(_resize_to(outputs["mask1"], height, width))
    blended = mask0 * warp0 + (1.0 - mask0) * warp1
    pred = mask1 * img1r + (1.0 - mask1) * blended
    return {
        "warp0": warp0.clamp(0.0, 1.0),
        "warp1": warp1.clamp(0.0, 1.0),
        "mask0": mask0,
        "mask1": mask1,
        "blend": blended.clamp(0.0, 1.0),
        "pred": pred.clamp(0.0, 1.0),
        "flowt_0": flow0,
        "flowt_1": flow1,
    }


_BUNDLE_RGB_KEYS = ("pred", "warp0", "warp1", "blend")
_BUNDLE_MASK_KEYS = ("mask0", "mask1")
_BUNDLE_FLOW_KEYS = ("flowt_0", "flowt_1")


def resize_bundle(bundle: dict[str, torch.Tensor], height: int, width: int) -> dict[str, torch.Tensor]:
    """Resize a composed bundle to a target visualization resolution.

    RGB and mask tensors are bilinearly resized. Flow tensors are resized and
    their pixel displacements rescaled so the flow color visualization keeps a
    consistent magnitude at the new resolution.
    """
    resized: dict[str, torch.Tensor] = {}
    for key, tensor in bundle.items():
        if not isinstance(tensor, torch.Tensor):
            resized[key] = tensor
            continue
        if tuple(tensor.shape[-2:]) == (height, width):
            resized[key] = tensor
            continue
        if key in _BUNDLE_FLOW_KEYS:
            src_h, src_w = int(tensor.shape[-2]), int(tensor.shape[-1])
            flow = _resize_to(tensor, height, width)
            scale = flow.new_tensor([float(width) / float(src_w), float(height) / float(src_h)]).view(1, 2, 1, 1)
            resized[key] = flow * scale
        else:
            resized[key] = _resize_to(tensor, height, width)
    return resized
