from __future__ import annotations

import torch


def zero_flow(img0: torch.Tensor) -> torch.Tensor:
    batch, _channels, height, width = img0.shape
    return img0.new_zeros((batch, 2, height, width))


def logits(img0: torch.Tensor, value: float) -> torch.Tensor:
    batch, _channels, height, width = img0.shape
    return img0.new_full((batch, 1, height, width), value)


def outputs_for_mode(img0: torch.Tensor, mode: str) -> dict[str, torch.Tensor]:
    flow = zero_flow(img0)
    if mode == "average":
        mask0 = logits(img0, 0.0)
        mask1 = logits(img0, -20.0)
    elif mode == "img0":
        mask0 = logits(img0, 20.0)
        mask1 = logits(img0, -20.0)
    elif mode == "img1":
        mask0 = logits(img0, -20.0)
        mask1 = logits(img0, -20.0)
    else:
        raise ValueError(f"unknown test mode: {mode}")
    return {
        "flowt_0": flow,
        "flowt_1": flow.clone(),
        "mask0": mask0,
        "mask1": mask1,
    }
