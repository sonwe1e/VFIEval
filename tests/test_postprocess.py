from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.pipeline.postprocess import backward_warp, compose_interpolated, validate_model_outputs


class PostprocessTests(unittest.TestCase):
    def test_zero_flow_compose_uses_platform_masks(self) -> None:
        img0 = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        img1 = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        outputs = {
            "flowt_0": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "flowt_1": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "mask0": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            "mask1": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        }

        composed = compose_interpolated(img0, img1, outputs)

        self.assertTrue(torch.allclose(composed["warp0"], img0))
        self.assertTrue(torch.allclose(composed["warp1"], img1))
        self.assertTrue(torch.allclose(composed["blend"], torch.full_like(img0, 0.5)))
        self.assertTrue(torch.allclose(composed["pred"], torch.full_like(img0, 0.75)))

    def test_validate_rejects_missing_fields(self) -> None:
        img0 = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_model_outputs({"flowt_0": torch.zeros((1, 2, 4, 4))}, img0)

    def test_backward_warp_identity_for_zero_flow(self) -> None:
        image = torch.rand((2, 3, 5, 7), dtype=torch.float32)
        flow = torch.zeros((2, 2, 5, 7), dtype=torch.float32)
        warped = backward_warp(image, flow)
        self.assertTrue(torch.allclose(warped, image, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
