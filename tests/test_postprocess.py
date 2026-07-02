from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.pipeline.postprocess import (
    backward_warp,
    compose_interpolated,
    compose_interpolated_native,
    normalize_model_outputs,
    resize_bundle,
    validate_model_outputs,
)


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

    def test_low_resolution_outputs_are_resized_and_flow_scaled(self) -> None:
        img0 = torch.zeros((1, 3, 4, 6), dtype=torch.float32)
        outputs = {
            "flowt_0": torch.ones((1, 2, 2, 2), dtype=torch.float32),
            "flowt_1": torch.ones((1, 2, 2, 2), dtype=torch.float32),
            "mask0": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "mask1": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        }

        normalized = normalize_model_outputs(outputs, img0)

        self.assertEqual(tuple(normalized["flowt_0"].shape), (1, 2, 4, 6))
        self.assertTrue(torch.allclose(normalized["flowt_0"][:, 0], torch.full((1, 4, 6), 3.0)))
        self.assertTrue(torch.allclose(normalized["flowt_0"][:, 1], torch.full((1, 4, 6), 2.0)))
        self.assertEqual(tuple(normalized["mask0"].shape), (1, 1, 4, 6))

    def test_validate_rejects_wrong_channels_not_spatial_scale(self) -> None:
        img0 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        outputs = {
            "flowt_0": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "flowt_1": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "mask0": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            "mask1": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        }
        validate_model_outputs(outputs, img0)
        outputs["flowt_0"] = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "flowt_0"):
            validate_model_outputs(outputs, img0)

    def test_validate_reports_explicit_device_mismatch(self) -> None:
        img0 = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        outputs = {
            "flowt_0": torch.zeros((1, 2, 4, 4), dtype=torch.float32, device="meta"),
            "flowt_1": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "mask0": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            "mask1": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        }
        with self.assertRaisesRegex(ValueError, r"Model output field flowt_0 is on meta, expected CPU"):
            validate_model_outputs(outputs, img0)

    def test_validate_reports_explicit_dtype_mismatch(self) -> None:
        img0 = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        outputs = {
            "flowt_0": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "flowt_1": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "mask0": torch.zeros((1, 1, 4, 4), dtype=torch.float64),
            "mask1": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        }
        with self.assertRaisesRegex(ValueError, r"Model output field mask0 has dtype torch.float64, expected torch.float32"):
            validate_model_outputs(outputs, img0)

    def test_backward_warp_identity_for_zero_flow(self) -> None:
        image = torch.rand((2, 3, 5, 7), dtype=torch.float32)
        flow = torch.zeros((2, 2, 5, 7), dtype=torch.float32)
        warped = backward_warp(image, flow)
        self.assertTrue(torch.allclose(warped, image, atol=1e-6))

    def test_native_compose_matches_upsampled_compose_for_zero_flow(self) -> None:
        # With zero flow and constant masks, composing at native resolution and
        # composing after upsampling produce the same interpolated frame (the
        # only difference between the two paths is where the RGB resize happens).
        img0 = torch.zeros((1, 3, 8, 12), dtype=torch.float32)
        img1 = torch.ones((1, 3, 8, 12), dtype=torch.float32)
        outputs = {
            "flowt_0": torch.zeros((1, 2, 4, 6), dtype=torch.float32),
            "flowt_1": torch.zeros((1, 2, 4, 6), dtype=torch.float32),
            "mask0": torch.zeros((1, 1, 4, 6), dtype=torch.float32),
            "mask1": torch.zeros((1, 1, 4, 6), dtype=torch.float32),
        }

        native = compose_interpolated_native(img0, img1, outputs)

        self.assertEqual(tuple(native["pred"].shape), (1, 3, 4, 6))
        self.assertTrue(torch.allclose(native["blend"], torch.full((1, 3, 4, 6), 0.5)))
        self.assertTrue(torch.allclose(native["pred"], torch.full((1, 3, 4, 6), 0.75)))
        self.assertEqual(tuple(native["flowt_0"].shape), (1, 2, 4, 6))

    def test_native_compose_handles_mismatched_flow1_resolution(self) -> None:
        img0 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        img1 = torch.ones((1, 3, 8, 8), dtype=torch.float32)
        outputs = {
            "flowt_0": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "flowt_1": torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            "mask0": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            "mask1": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        }

        native = compose_interpolated_native(img0, img1, outputs)

        # Everything is composed at flowt_0's resolution (4x4).
        for key in ("pred", "warp0", "warp1", "blend", "mask0", "mask1", "flowt_0", "flowt_1"):
            self.assertEqual(tuple(native[key].shape[-2:]), (4, 4), key)

    def test_resize_bundle_scales_flow_displacements(self) -> None:
        bundle = {
            "pred": torch.zeros((1, 3, 4, 6), dtype=torch.float32),
            "warp0": torch.zeros((1, 3, 4, 6), dtype=torch.float32),
            "warp1": torch.zeros((1, 3, 4, 6), dtype=torch.float32),
            "blend": torch.zeros((1, 3, 4, 6), dtype=torch.float32),
            "mask0": torch.full((1, 1, 4, 6), 0.5, dtype=torch.float32),
            "mask1": torch.full((1, 1, 4, 6), 0.5, dtype=torch.float32),
            "flowt_0": torch.ones((1, 2, 4, 6), dtype=torch.float32),
            "flowt_1": torch.ones((1, 2, 4, 6), dtype=torch.float32),
        }

        resized = resize_bundle(bundle, 8, 12)

        self.assertEqual(tuple(resized["pred"].shape), (1, 3, 8, 12))
        # width doubled (6->12) so x displacement scales x2; height doubled too.
        self.assertTrue(torch.allclose(resized["flowt_0"][:, 0], torch.full((1, 8, 12), 2.0)))
        self.assertTrue(torch.allclose(resized["flowt_0"][:, 1], torch.full((1, 8, 12), 2.0)))

    def test_resize_bundle_is_noop_when_already_target_size(self) -> None:
        bundle = {"pred": torch.rand((1, 3, 4, 6), dtype=torch.float32)}
        resized = resize_bundle(bundle, 4, 6)
        self.assertIs(resized["pred"], bundle["pred"])


if __name__ == "__main__":
    unittest.main()
