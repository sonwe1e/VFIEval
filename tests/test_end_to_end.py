from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_triplet_dataset
from vfieval.db import Database
from vfieval.worker import WorkerOptions, run_worker


class EndToEndTests(unittest.TestCase):
    def test_each_required_image_stage_failure_is_fatal(self) -> None:
        cases = (
            ("pred", "evaluation", "vfieval.pipeline.inference._save_visual_bundle_from_cpu"),
            ("gt", "evaluation", "vfieval.pipeline.inference.save_rgb_tensor"),
            ("difference", "evaluation", "vfieval.pipeline.inference.save_difference"),
            ("diagnostic_flow", "diagnostic", "vfieval.pipeline.visualize.save_flow"),
        )
        for failure_kind, profile, target in cases:
            with self.subTest(failure_kind=failure_kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset_root = root / "dataset"
                for folder in ("img0", "img1", "gt"):
                    (dataset_root / folder).mkdir(parents=True)
                    Image.new("RGB", (8, 8)).save(dataset_root / folder / "sample.png")
                workspace = WorkspaceConfig.from_root(root / ".vfieval")
                workspace.ensure()
                db = Database(workspace.db_path)
                db.init()
                model_id = db.register_model("dummy", "dummy", None, 8, 8)
                dataset_id = db.create_dataset(f"failure-{failure_kind}", str(dataset_root), has_gt=True)
                self.assertEqual(scan_triplet_dataset(db, dataset_id), 1)
                run_id = db.create_run(
                    f"failure-{failure_kind}",
                    model_id,
                    dataset_id,
                    8,
                    8,
                    1,
                    "cpu",
                    "fp32",
                    ["lpips_vit_patch"],
                    create_inference_job=False,
                )
                job_id = db.add_run_job(
                    run_id,
                    "inference",
                    {
                        "run_id": run_id,
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "height": 8,
                        "width": 8,
                        "batch_size": 1,
                        "device": "cpu",
                        "precision": "fp32",
                        "metrics": ["lpips_vit_patch"],
                        "artifact_profile": profile,
                    },
                    progress_total=1,
                )

                with patch(target, side_effect=OSError(f"synthetic {failure_kind} failure")):
                    run_worker(
                        db,
                        workspace,
                        WorkerOptions(role="inference", once=True, worker_id=f"failure-{failure_kind}"),
                    )

                self.assertEqual(db.get_job(job_id)["status"], "failed")
                failed_run = db.get_run(run_id)
                self.assertEqual(failed_run["status"], "failed")
                self.assertEqual(db.list_run_jobs(run_id, "metric"), [])
                self.assertEqual(db.query("SELECT * FROM run_media_assets WHERE run_id = ?", (run_id,)), [])

    def test_core_artifact_save_failure_fails_run_before_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
            for folder, color in (("img0", (1, 0, 0)), ("img1", (0, 1, 0)), ("gt", (0, 0, 1))):
                Image.new("RGB", (8, 8), color).save(dataset_root / folder / "sample.png")

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8)
            dataset_id = db.create_dataset("failure", str(dataset_root), has_gt=True)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 1)
            run_id = db.create_run(
                "save-failure",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                ["lpips_vit_patch"],
            )
            inference_job_id = int(db.get_run(run_id)["inference_job_id"])

            with patch(
                "vfieval.pipeline.inference._save_visual_bundle_from_cpu",
                side_effect=OSError("synthetic PNG write failure"),
            ):
                run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="save-failure"))

            self.assertEqual(db.get_job(inference_job_id)["status"], "failed")
            failed_run = db.get_run(run_id)
            self.assertEqual(failed_run["status"], "failed")
            self.assertIn("synthetic PNG write failure", failed_run["error"]["message"])
            self.assertEqual(db.list_run_jobs(run_id, "metric"), [])
            sample_errors = db.list_artifacts(job_id=inference_job_id, kind="sample_error")
            self.assertEqual(len(sample_errors), 1)

    def test_artifact_bulk_insert_failure_is_fatal_and_persists_integrity_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
                Image.new("RGB", (8, 8)).save(dataset_root / folder / "sample.png")
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8)
            dataset_id = db.create_dataset("bulk-failure", str(dataset_root), has_gt=True)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 1)
            run_id = db.create_run(
                "bulk-failure", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", ["lpips_vit_patch"]
            )
            job_id = int(db.get_run(run_id)["inference_job_id"])

            with patch.object(db, "add_artifacts_bulk", side_effect=OSError("synthetic SQLite artifact failure")):
                run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="bulk-failure"))

            self.assertEqual(db.get_job(job_id)["status"], "failed")
            failed_run = db.get_run(run_id)
            self.assertEqual(failed_run["status"], "failed")
            self.assertEqual(db.list_run_jobs(run_id, "metric"), [])
            self.assertFalse(db.list_run_artifacts(run_id))
            integrity = failed_run["result"]["artifact_integrity"]
            self.assertFalse(integrity["valid"])
            self.assertTrue(
                (workspace.runs_dir / str(run_id) / "logs" / "artifact_integrity" / f"{job_id}.json").is_file()
            )

    def test_video_encoding_failure_blocks_metrics_and_media_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            frame_paths = []
            for index, color in enumerate(((10, 0, 0), (0, 10, 0), (0, 0, 10))):
                path = root / f"frame-{index}.png"
                Image.new("RGB", (8, 8), color).save(path)
                frame_paths.append(path)
            model_id = db.register_model("dummy", "dummy", None, 8, 8)
            dataset_id = db.create_dataset("video", str(root), has_gt=True, source_type="video")
            db.add_sample(
                dataset_id,
                "clip-0",
                str(frame_paths[0]),
                str(frame_paths[2]),
                str(frame_paths[1]),
                {"source_type": "video", "video_name": "clip", "frame_index": 0, "fps": 24.0},
            )
            run_id = db.create_run(
                "video-failure", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", ["vmaf"]
            )
            job_id = int(db.get_run(run_id)["inference_job_id"])
            self.assertEqual(db.get_job(job_id)["payload"].get("dataset_id"), dataset_id)

            with patch("vfieval.pipeline.inference._write_mp4", side_effect=OSError("synthetic encoder failure")):
                run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="video-failure"))

            self.assertEqual(db.get_job(job_id)["status"], "failed")
            failed_run = db.get_run(run_id)
            self.assertEqual(failed_run["status"], "failed")
            self.assertEqual(db.list_run_jobs(run_id, "metric"), [])
            self.assertEqual(db.query("SELECT * FROM run_media_assets WHERE run_id = ?", (run_id,)), [])
            self.assertIn(
                "video_encoding_failed",
                {item["code"] for item in failed_run["result"]["artifact_integrity"]["errors"]},
            )

    def test_inference_saves_canonical_artifacts_and_visualization_previews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
            size = (256, 128)
            for idx in range(2):
                name = f"sample{idx:03d}.png"
                Image.new("RGB", size, (idx, 0, 0)).save(dataset_root / "img0" / name)
                Image.new("RGB", size, (0, idx, 0)).save(dataset_root / "img1" / name)
                Image.new("RGB", size, (0, 0, idx)).save(dataset_root / "gt" / name)

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, size[1], size[0])
            dataset_id = db.create_dataset("viz", str(dataset_root), has_gt=True)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 2)

            preview_sizes = [(64, 32), (128, 64)]
            job_ids = []
            for vis_w, vis_h in preview_sizes:
                job_ids.append(db.create_job(
                    "inference",
                    {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "height": size[1],
                    "width": size[0],
                    "batch_size": 2,
                    "device": "cpu",
                    "precision": "fp32",
                    "metrics": [],
                    "visualize_height": vis_h,
                    "visualize_width": vis_w,
                    },
                ))

            for index, job_id in enumerate(job_ids):
                run_worker(
                    db,
                    workspace,
                    WorkerOptions(role="inference", once=True, worker_id=f"viz-inference-{index}"),
                )
                self.assertEqual(db.get_job(job_id)["status"], "completed")

            for kind in ("pred", "gt", "difference"):
                canonical_by_job = []
                for job_id, preview_size in zip(job_ids, preview_sizes):
                    artifacts = db.list_artifacts(job_id=job_id, kind=kind)
                    self.assertTrue(artifacts, kind)
                    canonical_by_job.append(
                        {int(artifact["sample_id"]): Path(artifact["path"]).read_bytes() for artifact in artifacts}
                    )
                    for artifact in artifacts:
                        with Image.open(artifact["path"]) as image:
                            self.assertEqual(image.size, size, kind)
                        self.assertEqual(artifact["metadata"]["artifact_contract"], "canonical-v1")
                        with Image.open(artifact["metadata"]["preview_path"]) as preview:
                            self.assertEqual(preview.size, preview_size, kind)
                self.assertEqual(canonical_by_job[0], canonical_by_job[1], kind)

    def test_dummy_inference_and_unavailable_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
            for idx in range(2):
                name = f"sample{idx:03d}.png"
                Image.new("RGB", (8, 8), (idx, 0, 0)).save(dataset_root / "img0" / name)
                Image.new("RGB", (8, 8), (0, idx, 0)).save(dataset_root / "img1" / name)
                Image.new("RGB", (8, 8), (0, 0, idx)).save(dataset_root / "gt" / name)

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 4, 4)
            dataset_id = db.create_dataset("demo", str(dataset_root), has_gt=True)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 2)

            job_id = db.create_job(
                "inference",
                {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "height": 4,
                    "width": 4,
                    "batch_size": 2,
                    "device": "cpu",
                    "precision": "fp32",
                    "metrics": ["cgvqm"],
                },
            )

            run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="test-inference"))
            inference_job = db.get_job(job_id)
            self.assertEqual(inference_job["status"], "completed")
            self.assertEqual(len(db.list_artifacts(job_id=job_id, kind="pred")), 2)
            self.assertEqual(len(db.list_artifacts(job_id=job_id, kind="difference")), 2)

            run_worker(db, workspace, WorkerOptions(role="metric", once=True, worker_id="test-metric"))
            metric_jobs = [job for job in db.list_jobs() if job["kind"] == "metric"]
            self.assertEqual(len(metric_jobs), 1)
            self.assertEqual(metric_jobs[0]["status"], "completed")
            results = db.list_metric_results(inference_job_id=job_id)
            self.assertEqual(len(results), 1)
            self.assertTrue(all(row["metric_name"] == "cgvqm" for row in results))
            self.assertTrue(all(row["status"] == "unavailable" for row in results))
            self.assertTrue(all(row["sample_id"] is None for row in results))
            self.assertIn("requires video artifacts", results[0]["details"]["reason"])


if __name__ == "__main__":
    unittest.main()
