from __future__ import annotations

from typing import Protocol

import torch


class FlowMaskModel(Protocol):
    def predict(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> dict[str, torch.Tensor]:
        """Return flow and mask logits only."""
