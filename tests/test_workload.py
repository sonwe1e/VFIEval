from __future__ import annotations

import json
import unittest

from vfieval.workload import (
    WORKLOAD_ESTIMATE_CONTRACT,
    estimate_workload,
    workload_confirmation_scope,
    workload_confirmation_scope_fingerprint,
)


class WorkloadEstimateTests(unittest.TestCase):
    def test_report_is_json_safe_and_exposes_deterministic_effective_values(self) -> None:
        report = estimate_workload(
            device=" CUDA:0 ",
            precision="FP16",
            device_memory_bytes=8_000_000,
            host_available_memory_bytes=32_000_000,
            batch_size_per_device=2,
            height=10,
            width=20,
            sample_count=7,
            artifact_profile="Evaluation",
            prefetch_depth=3,
        )

        self.assertEqual(report["contract"], WORKLOAD_ESTIMATE_CONTRACT)
        self.assertEqual(
            report["effective"],
            {
                "device": "cuda:0",
                "device_kind": "cuda",
                "precision": "fp16",
                "bytes_per_element": 2,
                "device_memory_bytes": 8_000_000,
                "host_available_memory_bytes": 32_000_000,
                "batch_size_per_device": 2,
                "height": 10,
                "width": 20,
                "sample_count": 7,
                "artifact_profile": "evaluation",
                "prefetch_depth": 3,
            },
        )
        self.assertEqual(report["batch_pixels_per_device"], 400)
        self.assertEqual(report["input_tensor_bytes_lower_bound"], 4_800)
        # Host-side decode tensors are float32 regardless of model precision.
        self.assertEqual(report["prefetch_host_bytes_lower_bound"], 28_800)
        self.assertEqual(report["risk_level"], "normal")
        self.assertEqual(report["risk_reasons"], [])
        self.assertEqual(json.loads(json.dumps(report, sort_keys=True)), report)

    def test_input_pair_at_five_percent_of_device_memory_is_high_risk(self) -> None:
        report = estimate_workload(
            device="cuda:0",
            precision="fp32",
            device_memory_bytes=4_800_000,
            host_available_memory_bytes=1_000_000_000,
            batch_size_per_device=1,
            height=100,
            width=100,
            sample_count=1,
            artifact_profile="benchmark",
            prefetch_depth=1,
        )

        self.assertEqual(report["input_tensor_bytes_lower_bound"], 240_000)
        self.assertEqual(report["risk_level"], "high")
        self.assertEqual(
            [reason["code"] for reason in report["risk_reasons"]],
            ["input_pair_device_memory_ge_5_percent"],
        )
        self.assertEqual(report["device_memory_fraction"], 0.05)

    def test_prefetch_at_twenty_five_percent_of_host_memory_is_high_risk(self) -> None:
        report = estimate_workload(
            device="npu:0",
            precision="bf16",
            device_memory_bytes=1_000_000_000,
            host_available_memory_bytes=19_200,
            batch_size_per_device=1,
            height=10,
            width=10,
            sample_count=1,
            artifact_profile="evaluation",
            prefetch_depth=2,
        )

        self.assertEqual(report["prefetch_host_bytes_lower_bound"], 4_800)
        self.assertEqual(report["risk_level"], "high")
        self.assertEqual(
            [reason["code"] for reason in report["risk_reasons"]],
            ["prefetch_host_memory_ge_25_percent"],
        )
        self.assertEqual(report["host_memory_fraction"], 0.25)

    def test_unknown_device_memory_uses_strict_sixteen_million_pixel_threshold(self) -> None:
        common = {
            "device": "npu:0",
            "precision": "fp16",
            "device_memory_bytes": None,
            "host_available_memory_bytes": None,
            "height": 1_000,
            "width": 1_000,
            "sample_count": 1,
            "artifact_profile": "benchmark",
            "prefetch_depth": 0,
        }

        boundary = estimate_workload(batch_size_per_device=16, **common)
        over = estimate_workload(batch_size_per_device=17, **common)

        self.assertEqual(boundary["batch_pixels_per_device"], 16_000_000)
        self.assertEqual(boundary["risk_level"], "normal")
        self.assertEqual(over["risk_level"], "high")
        self.assertEqual(
            [reason["code"] for reason in over["risk_reasons"]],
            ["unknown_device_memory_batch_pixels_gt_16000000"],
        )

    def test_multiple_reasons_have_stable_order(self) -> None:
        report = estimate_workload(
            device="cuda:0",
            precision="fp32",
            device_memory_bytes=1,
            host_available_memory_bytes=1,
            batch_size_per_device=1,
            height=1,
            width=1,
            sample_count=1,
            artifact_profile="evaluation",
            prefetch_depth=1,
        )

        self.assertEqual(
            [reason["code"] for reason in report["risk_reasons"]],
            [
                "input_pair_device_memory_ge_5_percent",
                "prefetch_host_memory_ge_25_percent",
            ],
        )

    def test_artifact_budget_depends_on_profile_and_sample_count(self) -> None:
        common = {
            "device": "cpu",
            "precision": "fp32",
            "device_memory_bytes": 1_000_000,
            "host_available_memory_bytes": 1_000_000,
            "batch_size_per_device": 1,
            "height": 4,
            "width": 5,
            "sample_count": 2,
            "prefetch_depth": 0,
        }

        benchmark = estimate_workload(artifact_profile="benchmark", **common)
        evaluation = estimate_workload(artifact_profile="evaluation", **common)
        diagnostic = estimate_workload(artifact_profile="diagnostic", **common)

        self.assertEqual(benchmark["artifact_budget_bytes"], 0)
        self.assertEqual(evaluation["artifact_budget_bytes"], 900)
        self.assertEqual(
            evaluation["artifact_budget_breakdown"],
            {
                "canonical_images_bytes": 360,
                "encoded_video_reserve_bytes": 360,
                "overhead_reserve_bytes": 180,
                "model_specific_extras_included": False,
            },
        )
        self.assertEqual(diagnostic["artifact_budget_bytes"], 1_750)
        self.assertGreater(diagnostic["artifact_budget_bytes"], evaluation["artifact_budget_bytes"])

    def test_fingerprint_is_normalized_stable_and_binds_all_effective_inputs(self) -> None:
        inputs = {
            "device": "cuda:0",
            "precision": "fp16",
            "device_memory_bytes": 8_000_000,
            "host_available_memory_bytes": 16_000_000,
            "batch_size_per_device": 4,
            "height": 32,
            "width": 48,
            "sample_count": 10,
            "artifact_profile": "evaluation",
            "prefetch_depth": 3,
        }
        first = estimate_workload(**inputs)
        normalized_equivalent = estimate_workload(
            **{**inputs, "device": " CUDA:0 ", "precision": "FP16"}
        )
        changed = estimate_workload(**{**inputs, "sample_count": 11})

        self.assertEqual(first["risk_fingerprint"], normalized_equivalent["risk_fingerprint"])
        self.assertNotEqual(first["risk_fingerprint"], changed["risk_fingerprint"])
        self.assertEqual(len(first["risk_fingerprint"]), 64)

    def test_confirmation_scope_ignores_volatile_memory_but_binds_execution_shape(self) -> None:
        inputs = {
            "device": "cuda:0",
            "precision": "fp16",
            "device_memory_bytes": 8_000_000,
            "host_available_memory_bytes": 16_000_000,
            "batch_size_per_device": 4,
            "height": 32,
            "width": 48,
            "sample_count": 10,
            "artifact_profile": "evaluation",
            "prefetch_depth": 3,
        }
        first = estimate_workload(**inputs)
        volatile_memory_changed = estimate_workload(
            **{
                **inputs,
                "device_memory_bytes": 7_000_000,
                "host_available_memory_bytes": 15_000_000,
            }
        )
        dimensions_changed = estimate_workload(**{**inputs, "width": 64})
        samples_changed = estimate_workload(**{**inputs, "sample_count": 11})

        self.assertNotEqual(
            first["risk_fingerprint"],
            volatile_memory_changed["risk_fingerprint"],
        )
        self.assertEqual(
            workload_confirmation_scope_fingerprint(first),
            workload_confirmation_scope_fingerprint(volatile_memory_changed),
        )
        self.assertNotEqual(
            workload_confirmation_scope_fingerprint(first),
            workload_confirmation_scope_fingerprint(dimensions_changed),
        )
        self.assertNotEqual(
            workload_confirmation_scope_fingerprint(first),
            workload_confirmation_scope_fingerprint(samples_changed),
        )
        scope = workload_confirmation_scope(first)
        self.assertNotIn("host_available_memory_bytes", scope["effective"])
        self.assertNotIn("device_memory_bytes", scope["effective"])

    def test_invalid_effective_values_are_rejected_without_silent_clamping(self) -> None:
        valid = {
            "device": "cuda:0",
            "precision": "fp16",
            "device_memory_bytes": None,
            "host_available_memory_bytes": None,
            "batch_size_per_device": 1,
            "height": 10,
            "width": 10,
            "sample_count": 1,
            "artifact_profile": "evaluation",
            "prefetch_depth": 1,
        }

        for field, value in (
            ("batch_size_per_device", 0),
            ("height", -1),
            ("sample_count", -1),
            ("prefetch_depth", -1),
            ("device_memory_bytes", 0),
            ("host_available_memory_bytes", -1),
            ("precision", "int8"),
            ("artifact_profile", "everything"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                estimate_workload(**{**valid, field: value})


if __name__ == "__main__":
    unittest.main()
