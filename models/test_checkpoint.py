from __future__ import annotations

from pathlib import Path

from _test_helpers import outputs_for_mode


class Model:
    def __init__(self, checkpoint_path=None, device="cpu", metadata=None):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for this test model")
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

    def infer(self, img0, img1):
        return outputs_for_mode(img0, "average")
