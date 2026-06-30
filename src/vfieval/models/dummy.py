from __future__ import annotations

import torch


class DummyFlowMaskModel:
    """A zero-flow model for smoke tests and pipeline validation."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def predict(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> dict[str, torch.Tensor]:
        batch, _channels, height, width = img0.shape
        flow = torch.zeros((batch, 2, height, width), dtype=img0.dtype, device=img0.device)
        mask = torch.zeros((batch, 1, height, width), dtype=img0.dtype, device=img0.device)
        return {
            "flowt_0": flow,
            "flowt_1": flow.clone(),
            "mask0": mask,
            "mask1": mask.clone(),
        }
