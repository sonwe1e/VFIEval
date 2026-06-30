from __future__ import annotations

from typing import Any

import torch


class AverageFlowMaskModel:
    """Minimal flow/mask-only adapter that lets platform post-processing average frames."""

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu", metadata: dict[str, Any] | None = None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.metadata = metadata or {}

    def predict(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> dict[str, torch.Tensor]:
        batch, _channels, height, width = img0.shape
        flow = img0.new_zeros((batch, 2, height, width))
        mask0 = img0.new_zeros((batch, 1, height, width))
        mask1 = img0.new_full((batch, 1, height, width), -20.0)
        return {
            "flowt_0": flow,
            "flowt_1": flow.clone(),
            "mask0": mask0,
            "mask1": mask1,
        }


def create_model(
    checkpoint_path: str | None = None,
    device: str = "cpu",
    metadata: dict[str, Any] | None = None,
) -> AverageFlowMaskModel:
    return AverageFlowMaskModel(checkpoint_path=checkpoint_path, device=device, metadata=metadata)
