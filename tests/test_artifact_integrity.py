from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.pipeline.artifact_integrity import (
    ArtifactIntegrityError,
    SHARD_MANIFEST_SCHEMA,
    validate_finalize_inputs,
    validate_finalize_video_artifact_integrity,
    validate_job_artifact_integrity,
    validate_metric_retry_integrity,
    validate_video_artifact_integrity,
)
from vfieval.pipeline.finalize_runner import run_finalize_job
from vfieval.pipeline.inference import RunCanceled


class ArtifactIntegrityTests(unittest.TestCase):
    def _workspace(self, root: Path, *, samples: int = 1, source_type: str = "frames"):
        workspace = WorkspaceConfig.from_root(root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
        dataset_id = db.create_dataset("dataset", str(root / "source"), True, source_type=source_type)
        sample_ids = []
        for index in range(samples):
            source = root / "source" / f"{index}.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"source")
            sample_ids.append(
                db.add_sample(
                    dataset_id,
                    f"sample-{index}",
                    str(source),
                    str(source),
                    str(source),
                    {
                        "source_type": source_type,
                        "video_path": "clip",
                        "video_name": "clip",
                        "frame_index": index,
                        "fps": 24.0,
                    },
                )
            )
        return workspace, db, model_id, dataset_id, sample_ids

    @staticmethod
    def _core_artifacts(db: Database, job_id: int, sample_id: int, run_dir: Path) -> dict[str, Path]:
        paths = {}
        for kind in ("pred", "gt", "difference"):
            path = run_dir / f"sample-{sample_id}" / f"{kind}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(kind.encode("ascii"))
            db.add_artifact(job_id, sample_id, kind, str(path), "image/png", {"sample": f"sample-{sample_id}"})
            paths[kind] = path
        return paths

    @staticmethod
    def _manifest(
        run_dir: Path,
        run_id: int,
        job_id: int,
        sample_id: int,
        paths: dict[str, Path],
        *,
        order: int,
        frame_sample_id: int | None = None,
    ) -> Path:
        manifest_path = run_dir / "logs" / "shards" / f"{job_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "version": SHARD_MANIFEST_SCHEMA,
                    "run_id": run_id,
                    "job_id": job_id,
                    "expected_sample_ids": [sample_id],
                    "successful_sample_ids": [sample_id],
                    "core_artifact_counts": {"difference": 1, "gt": 1, "pred": 1},
                    "video_groups": {
                        "clip": {
                            "video_name": "clip",
                            "fps": 24.0,
                            "source_video_path": "clip",
                            "frames": [
                                {
                                    "sample_id": sample_id if frame_sample_id is None else frame_sample_id,
                                    "order": order,
                                    "pred_path": str(paths["pred"]),
                                    "gt_path": str(paths["gt"]),
                                    "diff_path": str(paths["difference"]),
                                }
                            ],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

    def _finalize_ready_run(self, root: Path):
        workspace, db, model_id, dataset_id, sample_ids = self._workspace(root, source_type="video")
        run_id = db.create_run(
            "finalize",
            model_id,
            dataset_id,
            8,
            8,
            1,
            "multi_npu",
            "fp32",
            [],
            metadata={"execution_mode": "multi_npu"},
            create_inference_job=False,
        )
        run_dir = workspace.runs_dir / str(run_id)
        inference_job_id = db.add_run_job(
            run_id,
            "inference",
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "sample_ids": sample_ids,
                "artifact_profile": "evaluation",
                "defer_video_finalize": True,
            },
            progress_total=1,
        )
        self.assertTrue(db.mark_run_started(run_id, "running"))
        self.assertEqual(int(db.claim_next_job("inference", ["inference"])["id"]), inference_job_id)
        paths = self._core_artifacts(db, inference_job_id, sample_ids[0], run_dir)
        self._manifest(run_dir, run_id, inference_job_id, sample_ids[0], paths, order=0)
        self.assertTrue(db.complete_job(inference_job_id, {"samples": 1, "performance": {}}))
        self.assertTrue(
            db.queue_run_finalize(
                run_id,
                {"samples": 1},
                db.summarize_run_artifacts(run_id),
                [inference_job_id],
            )
        )
        finalize_job_id = int(db.list_run_jobs(run_id, "finalize")[0]["job_id"])
        self.assertEqual(int(db.claim_next_job("finalize", ["finalize"])["id"]), finalize_job_id)
        return workspace, db, run_id, inference_job_id, finalize_job_id, run_dir

    @staticmethod
    def _fake_video_encoder(db: Database, job_id: int, run_dir: Path, video_groups, **_kwargs) -> None:
        for kind in ("pred_video", "gt_video", "diff_video"):
            path = run_dir / "videos" / "clip" / f"{kind}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video")
            db.add_artifact(
                job_id,
                None,
                kind,
                str(path),
                "video/mp4",
                {"video_name": "clip", "frames": 1, "width": 8, "height": 8, "fps": 24.0},
            )

    def test_job_integrity_rejects_missing_empty_duplicate_and_sample_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, model_id, dataset_id, sample_ids = self._workspace(Path(tmp))
            run_id = db.create_run(
                "run", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [], create_inference_job=False
            )
            sample_id = sample_ids[0]
            job_id = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "dataset_id": dataset_id, "sample_ids": [sample_id], "artifact_profile": "evaluation"},
                progress_total=1,
            )
            paths = self._core_artifacts(db, job_id, sample_id, workspace.runs_dir / str(run_id))
            valid = validate_job_artifact_integrity(db, job_id)
            self.assertTrue(valid["valid"], valid)

            paths["gt"].write_bytes(b"")
            db.add_artifact(job_id, sample_id, "pred", str(paths["pred"]), "image/png", {})
            db.add_artifact(
                job_id,
                sample_id,
                "sample_error",
                "",
                "application/json",
                {"error_type": "OSError", "message": "save failed"},
            )
            invalid = validate_job_artifact_integrity(db, job_id)
            codes = {error["code"] for error in invalid["errors"]}
            self.assertFalse(invalid["valid"])
            self.assertTrue({"duplicate_core_artifact", "invalid_core_artifact_file", "sample_error"}.issubset(codes))
            self.assertEqual(invalid["successful_sample_ids"], [])

    def test_canonical_contract_checks_actual_size_and_unique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, model_id, dataset_id, sample_ids = self._workspace(Path(tmp), samples=2)
            run_id = db.create_run(
                "run",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
                metadata={"artifact_contract": "canonical-v1"},
                create_inference_job=False,
            )
            job_id = db.add_run_job(
                run_id,
                "inference",
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "sample_ids": sample_ids,
                    "artifact_profile": "evaluation",
                },
                progress_total=2,
            )
            shared_pred = workspace.runs_dir / str(run_id) / "shared-pred.png"
            shared_pred.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(shared_pred)
            for sample_id in sample_ids:
                for kind in ("pred", "gt", "difference"):
                    path = shared_pred if kind == "pred" else shared_pred.parent / f"{sample_id}-{kind}.png"
                    if path != shared_pred:
                        Image.new("RGB", (8, 8)).save(path)
                    db.add_artifact(
                        job_id,
                        sample_id,
                        kind,
                        str(path),
                        "image/png",
                        {
                            "artifact_contract": "canonical-v1",
                            "canonical_height": 8,
                            "canonical_width": 8,
                        },
                    )
            shared = validate_job_artifact_integrity(db, job_id)
            self.assertIn("shared_core_artifact_path", {item["code"] for item in shared["errors"]})

            Image.new("RGB", (4, 8)).save(shared_pred)
            wrong_size = validate_job_artifact_integrity(db, job_id)
            self.assertIn("canonical_file_size_mismatch", {item["code"] for item in wrong_size["errors"]})

    def test_canonical_video_rejects_shared_paths_and_encoded_size_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, db, model_id, dataset_id, sample_ids = self._workspace(
                root,
                samples=2,
                source_type="video",
            )
            run_id = db.create_run(
                "run",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
                metadata={"artifact_contract": "canonical-v1"},
                create_inference_job=False,
            )
            job_id = db.add_run_job(
                run_id,
                "inference",
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "sample_ids": sample_ids,
                    "artifact_profile": "evaluation",
                    "artifact_contract": "canonical-v1",
                },
                progress_total=2,
            )
            frame_paths = []
            groups = {}
            for index, (group_key, video_name, sample_id) in enumerate(
                zip(("identity-a", "identity-b"), ("clip/a", "clip:a"), sample_ids)
            ):
                frame_path = root / f"frame-{index}.png"
                Image.new("RGB", (8, 8)).save(frame_path)
                frame_paths.append(frame_path)
                groups[group_key] = {
                    "video_name": video_name,
                    "fps": 24.0,
                    "frames": [
                        {
                            "order": 0,
                            "sample_id": sample_id,
                            "pred_path": frame_path,
                            "gt_path": None,
                            "diff_path": None,
                        }
                    ],
                }

            shared_path = workspace.runs_dir / str(run_id) / "videos" / "shared.mp4"
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            shared_path.write_bytes(b"video")
            for video_name in ("clip/a", "clip:a"):
                db.add_artifact(
                    job_id,
                    None,
                    "pred_video",
                    str(shared_path),
                    "video/mp4",
                    {
                        "artifact_contract": "canonical-v1",
                        "video_name": video_name,
                        "frames": 1,
                        "width": 8,
                        "height": 8,
                        "fps": 24.0,
                    },
                )

            with patch(
                "vfieval.file_inputs.inspect_video",
                return_value={
                    "decodable": True,
                    "frame_count": 1,
                    "width": 10,
                    "height": 8,
                },
            ):
                report = validate_video_artifact_integrity(
                    db,
                    job_id,
                    groups,
                    expected_sample_ids=sample_ids,
                )
            codes = {item["code"] for item in report["errors"]}
            self.assertIn("shared_canonical_video_path", codes)
            self.assertIn("encoded_video_size_mismatch", codes)

            missing_reference_frames = validate_video_artifact_integrity(
                db,
                job_id,
                {
                    "clip": {
                        "video_name": "clip",
                        "fps": 24.0,
                        "frames": [
                            {
                                "order": 0,
                                "sample_id": sample_ids[0],
                                "pred_path": frame_paths[0],
                                "gt_path": None,
                                "diff_path": None,
                            }
                        ],
                    }
                },
                expected_sample_ids=[sample_ids[0]],
            )
            self.assertIn(
                "incomplete_video_group_frames",
                {item["code"] for item in missing_reference_frames["errors"]},
            )

    def test_metric_retry_requires_gt_videos_per_video_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("mixed", str(root), True, source_type="video")
            source = root / "source.png"
            source.write_bytes(b"source")
            with_gt = db.add_sample(
                dataset_id,
                "with-gt",
                str(source),
                str(source),
                str(source),
                {"source_type": "video", "video_name": "with-gt", "frame_index": 0},
            )
            without_gt = db.add_sample(
                dataset_id,
                "without-gt",
                str(source),
                str(source),
                None,
                {"source_type": "video", "video_name": "without-gt", "frame_index": 0},
            )
            run_id = db.create_run(
                "mixed",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            job_id = db.add_run_job(
                run_id,
                "inference",
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "sample_ids": [with_gt, without_gt],
                    "artifact_profile": "evaluation",
                },
                progress_total=2,
            )
            for sample_id, kinds in (
                (with_gt, ("pred", "gt", "difference")),
                (without_gt, ("pred",)),
            ):
                for kind in kinds:
                    path = root / f"{sample_id}-{kind}.png"
                    path.write_bytes(kind.encode("ascii"))
                    db.add_artifact(job_id, sample_id, kind, str(path), "image/png", {})

            for video_name, kinds in (
                ("with-gt", ("pred_video", "gt_video", "diff_video")),
                ("without-gt", ("pred_video",)),
            ):
                for kind in kinds:
                    path = root / f"{video_name}-{kind}.mp4"
                    path.write_bytes(b"video")
                    db.add_artifact(
                        job_id,
                        None,
                        kind,
                        str(path),
                        "video/mp4",
                        {"video_name": video_name, "frames": 1, "width": 8, "height": 8, "fps": 24.0},
                    )

            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("mixed-retry", ["inference"])["id"]), job_id)
            self.assertTrue(db.complete_job(job_id, {"samples": 2}))
            with patch(
                "vfieval.file_inputs.inspect_video",
                return_value={"decodable": True, "frame_count": 1, "width": 8, "height": 8, "fps": 24.0},
            ):
                report = validate_metric_retry_integrity(db, run_id)
            self.assertTrue(report["valid"], report)

            stray_path = root / "without-gt-stray-gt.mp4"
            stray_path.write_bytes(b"video")
            db.add_artifact(
                job_id,
                None,
                "gt_video",
                str(stray_path),
                "video/mp4",
                {"video_name": "without-gt", "frames": 1, "width": 8, "height": 8},
            )
            with patch(
                "vfieval.file_inputs.inspect_video",
                return_value={"decodable": True, "frame_count": 1, "width": 8, "height": 8, "fps": 24.0},
            ):
                rejected = validate_metric_retry_integrity(db, run_id)
            self.assertIn(
                "unexpected_video_artifact",
                {item["code"] for item in rejected["errors"]},
            )

    def test_metric_retry_rejects_nonidentical_pred_gt_video_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, db, model_id, dataset_id, sample_ids = self._workspace(
                root,
                source_type="video",
            )
            run_id = db.create_run(
                "legacy-video-pair",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            job_id = db.add_run_job(
                run_id,
                "inference",
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "sample_ids": sample_ids,
                    "artifact_profile": "evaluation",
                },
                progress_total=1,
            )
            self._core_artifacts(db, job_id, sample_ids[0], workspace.runs_dir / str(run_id))
            video_paths = {}
            for kind, frames in (("pred_video", 2), ("gt_video", 1), ("diff_video", 2)):
                path = root / f"{kind}.mp4"
                path.write_bytes(kind.encode("ascii"))
                video_paths[kind] = path
                db.add_artifact(
                    job_id,
                    None,
                    kind,
                    str(path),
                    "video/mp4",
                    {"video_name": "clip", "frames": frames, "width": 8, "height": 8, "fps": 24.0},
                )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("strict-retry", ["inference"])["id"]), job_id)
            self.assertTrue(db.complete_job(job_id, {"samples": 1}))

            def inspect(path, exact=True):
                name = Path(path).stem
                return {
                    "decodable": True,
                    "frame_count": 1 if name == "gt_video" else 2,
                    "width": 8,
                    "height": 8,
                    "fps": 24.0,
                }

            with patch("vfieval.file_inputs.inspect_video", side_effect=inspect):
                report = validate_metric_retry_integrity(db, run_id)

            self.assertIn("video_pair_mismatch", {item["code"] for item in report["errors"]})
            pair_error = next(item for item in report["errors"] if item["code"] == "video_pair_mismatch")
            self.assertEqual(pair_error["mismatches"]["observed_frame_count"], {"pred": 2, "gt": 1})

    def test_finalize_missing_video_manifest_stops_before_encoding_or_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, model_id, dataset_id, sample_ids = self._workspace(Path(tmp), source_type="video")
            run_id = db.create_run(
                "run",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "multi_npu",
                "fp32",
                ["lpips_vit_patch"],
                metadata={"execution_mode": "multi_npu"},
                create_inference_job=False,
            )
            inference_job_id = db.add_run_job(
                run_id,
                "inference",
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "sample_ids": sample_ids,
                    "artifact_profile": "evaluation",
                    "defer_video_finalize": True,
                },
                progress_total=1,
            )
            self._core_artifacts(db, inference_job_id, sample_ids[0], workspace.runs_dir / str(run_id))
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("missing-manifest-inference", ["inference"])["id"]), inference_job_id)
            self.assertTrue(db.complete_job(inference_job_id, {"samples": 1, "performance": {}}))
            self.assertTrue(
                db.queue_run_finalize(
                    run_id,
                    {"samples": 1},
                    db.summarize_run_artifacts(run_id),
                    [inference_job_id],
                )
            )
            finalize_job_id = int(db.list_run_jobs(run_id, "finalize")[0]["job_id"])
            self.assertEqual(int(db.claim_next_job("missing-manifest-finalize", ["finalize"])["id"]), finalize_job_id)

            with (
                patch("vfieval.pipeline.finalize_runner._write_video_artifacts") as encode,
                patch("vfieval.pipeline.finalize_runner.sync_run_assets") as sync,
            ):
                with self.assertRaises(ArtifactIntegrityError) as raised:
                    run_finalize_job(db, workspace, finalize_job_id)
            self.assertIn("missing_shard_manifest", {item["code"] for item in raised.exception.report["errors"]})
            encode.assert_not_called()
            sync.assert_not_called()
            self.assertEqual(db.list_run_jobs(run_id, "metric"), [])
            self.assertTrue((workspace.runs_dir / str(run_id) / "logs" / "artifact_integrity.json").is_file())

    def test_finalize_publishes_canonical_media_before_no_metric_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, run_id, _inference_job_id, finalize_job_id, _run_dir = self._finalize_ready_run(Path(tmp))
            with patch(
                "vfieval.pipeline.finalize_runner._write_video_artifacts",
                side_effect=self._fake_video_encoder,
            ):
                result = run_finalize_job(db, workspace, finalize_job_id)
            self.assertTrue(db.complete_job(finalize_job_id, result))
            self.assertEqual(db.get_run(run_id)["status"], "completed")
            assets = db.query(
                """
                SELECT ma.role, ma.state, ma.storage_path
                FROM run_media_assets rma
                JOIN media_assets ma ON ma.id = rma.asset_id
                WHERE rma.run_id = ? AND ma.source_kind = 'run_artifact'
                ORDER BY ma.role
                """,
                (run_id,),
            )
            self.assertEqual([(row["role"], row["state"]) for row in assets], [("gt", "ready"), ("pred", "ready")])
            self.assertTrue(all("preview" not in str(row["storage_path"]) for row in assets))

    def test_finalize_cancel_during_media_publication_converges_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, run_id, _inference_job_id, finalize_job_id, _run_dir = self._finalize_ready_run(Path(tmp))

            def cancel_during_sync(_db, _workspace, selected_run_id):
                self.assertEqual(selected_run_id, run_id)
                self.assertTrue(db.request_run_cancel(run_id))
                return []

            with (
                patch(
                    "vfieval.pipeline.finalize_runner._write_video_artifacts",
                    side_effect=self._fake_video_encoder,
                ),
                patch("vfieval.pipeline.finalize_runner.sync_run_assets", side_effect=cancel_during_sync),
            ):
                with self.assertRaises(RunCanceled):
                    run_finalize_job(db, workspace, finalize_job_id)

            self.assertEqual(db.get_run(run_id)["status"], "cancel_requested")
            self.assertEqual(db.get_job(finalize_job_id)["status"], "running")
            self.assertTrue(db.converge_run_cancellation(run_id, finalize_job_id))
            self.assertEqual(db.get_run(run_id)["status"], "canceled")
            self.assertEqual(db.get_job(finalize_job_id)["status"], "canceled")
            self.assertEqual(db.list_run_jobs(run_id, "metric"), [])
            self.assertEqual(db.query("SELECT * FROM run_media_assets WHERE run_id = ?", (run_id,)), [])

    def test_finalize_rejects_duplicate_or_missing_manifest_sample_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, model_id, dataset_id, sample_ids = self._workspace(Path(tmp), samples=2, source_type="video")
            run_id = db.create_run(
                "run", model_id, dataset_id, 8, 8, 1, "multi_npu", "fp32", [], create_inference_job=False
            )
            run_dir = workspace.runs_dir / str(run_id)
            jobs = []
            paths_by_sample = {}
            self.assertTrue(db.mark_run_started(run_id, "running"))
            for index, sample_id in enumerate(sample_ids):
                job_id = db.add_run_job(
                    run_id,
                    "inference",
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "sample_ids": [sample_id],
                        "artifact_profile": "evaluation",
                        "defer_video_finalize": True,
                    },
                    progress_total=1,
                    shard_index=index,
                )
                paths_by_sample[sample_id] = self._core_artifacts(db, job_id, sample_id, run_dir)
                self.assertEqual(int(db.claim_next_job(f"coverage-{index}", ["inference"])["id"]), job_id)
                self.assertTrue(db.complete_job(job_id, {"samples": 1, "performance": {}}))
                jobs.append(db.list_run_jobs(run_id, "inference")[-1])
            first_job, second_job = [int(row["job_id"]) for row in db.list_run_jobs(run_id, "inference")]
            self._manifest(run_dir, run_id, first_job, sample_ids[0], paths_by_sample[sample_ids[0]], order=0)
            self._manifest(
                run_dir,
                run_id,
                second_job,
                sample_ids[1],
                paths_by_sample[sample_ids[1]],
                order=1,
                frame_sample_id=sample_ids[0],
            )

            _merged, report = validate_finalize_inputs(db, run_id, db.list_run_jobs(run_id, "inference"), run_dir)
            codes = {item["code"] for item in report["errors"]}
            self.assertFalse(report["valid"])
            self.assertIn("manifest_frame_wrong_shard", codes)
            self.assertIn("manifest_video_sample_coverage_mismatch", codes)
            self.assertIn("manifest_frame_artifact_mismatch", codes)

    def test_finalize_manifest_rejects_swapped_video_identity_order_and_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, db, model_id, dataset_id, sample_ids = self._workspace(
                root,
                samples=2,
                source_type="video",
            )
            video_paths = [root / "source" / "a.mp4", root / "source" / "b.mp4"]
            sample_metadata = []
            for index, (sample_id, video_path) in enumerate(zip(sample_ids, video_paths)):
                metadata = {
                    "source_type": "video",
                    "video_name": video_path.stem,
                    "video_path": str(video_path),
                    "video_group": "source",
                    "video_file": video_path.name,
                    "frame_index": index,
                    "sample_index": index,
                    "gt_index": index,
                    "fps": 24.0,
                    "timestamps": {"gt": index / 24.0},
                }
                sample_metadata.append(metadata)
                with db.connection() as conn:
                    conn.execute(
                        "UPDATE samples SET metadata_json = ? WHERE id = ?",
                        (json.dumps(metadata), sample_id),
                    )

            run_id = db.create_run(
                "swapped-manifest",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "multi_npu",
                "fp32",
                [],
                create_inference_job=False,
            )
            run_dir = workspace.runs_dir / str(run_id)
            job_id = db.add_run_job(
                run_id,
                "inference",
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "sample_ids": sample_ids,
                    "artifact_profile": "evaluation",
                    "defer_video_finalize": True,
                },
                progress_total=2,
            )
            paths_by_sample = {
                sample_id: self._core_artifacts(db, job_id, sample_id, run_dir)
                for sample_id in sample_ids
            }
            groups = {}
            for target_index, source_index in ((0, 1), (1, 0)):
                target = sample_metadata[target_index]
                source = sample_metadata[source_index]
                source_paths = paths_by_sample[sample_ids[source_index]]
                groups[target["video_path"]] = {
                    "video_name": target["video_name"],
                    "fps": 24.0,
                    "source_video_path": target["video_path"],
                    "source_video_group": target["video_group"],
                    "source_video_file": target["video_file"],
                    "frames": [
                        {
                            "sample_id": sample_ids[source_index],
                            "order": 99 if target_index == 0 else int(source["frame_index"]),
                            "pred_path": str(source_paths["pred"]),
                            "gt_path": str(source_paths["gt"]),
                            "diff_path": str(source_paths["difference"]),
                            "source_frame_index": target["gt_index"],
                            "source_timestamp": target["timestamps"]["gt"],
                        }
                    ],
                }
            manifest_path = run_dir / "logs" / "shards" / f"{job_id}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": SHARD_MANIFEST_SCHEMA,
                        "run_id": run_id,
                        "job_id": job_id,
                        "expected_sample_ids": sample_ids,
                        "successful_sample_ids": sample_ids,
                        "core_artifact_counts": {"pred": 2, "gt": 2, "difference": 2},
                        "video_groups": groups,
                    }
                ),
                encoding="utf-8",
            )

            _merged, report = validate_finalize_inputs(
                db,
                run_id,
                db.list_run_jobs(run_id, "inference"),
                run_dir,
            )

            codes = {item["code"] for item in report["errors"]}
            self.assertIn("manifest_frame_video_identity_mismatch", codes)
            self.assertIn("manifest_frame_order_mismatch", codes)
            self.assertIn("manifest_frame_source_mapping_mismatch", codes)

    def test_valid_manifests_merge_and_video_counts_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, model_id, dataset_id, sample_ids = self._workspace(Path(tmp), samples=2, source_type="video")
            run_id = db.create_run(
                "run", model_id, dataset_id, 8, 8, 1, "multi_npu", "fp32", [], create_inference_job=False
            )
            run_dir = workspace.runs_dir / str(run_id)
            self.assertTrue(db.mark_run_started(run_id, "running"))
            for index, sample_id in enumerate(sample_ids):
                job_id = db.add_run_job(
                    run_id,
                    "inference",
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "sample_ids": [sample_id],
                        "artifact_profile": "evaluation",
                        "defer_video_finalize": True,
                    },
                    progress_total=1,
                    shard_index=index,
                )
                paths = self._core_artifacts(db, job_id, sample_id, run_dir)
                self._manifest(run_dir, run_id, job_id, sample_id, paths, order=index)
                self.assertEqual(int(db.claim_next_job(f"valid-manifest-{index}", ["inference"])["id"]), job_id)
                self.assertTrue(db.complete_job(job_id, {"samples": 1, "performance": {}}))

            jobs = db.list_run_jobs(run_id, "inference")
            merged, report = validate_finalize_inputs(db, run_id, jobs, run_dir)
            self.assertTrue(report["valid"], report)
            self.assertEqual([frame["sample_id"] for frame in merged["clip"]["frames"]], sample_ids)

            artifact_job_id = int(jobs[0]["job_id"])
            for kind in ("pred_video", "gt_video", "diff_video"):
                path = run_dir / "videos" / "clip" / f"{kind}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
                db.add_artifact(
                    artifact_job_id,
                    None,
                    kind,
                    str(path),
                    "video/mp4",
                    {"video_name": "clip", "frames": 2},
                )
            video_report = validate_video_artifact_integrity(
                db,
                artifact_job_id,
                merged,
                expected_sample_ids=sample_ids,
            )
            self.assertTrue(video_report["valid"], video_report)

            pred = db.list_artifacts(job_id=artifact_job_id, kind="pred_video")[0]
            with db.connection() as conn:
                conn.execute(
                    "UPDATE artifacts SET metadata_json = ? WHERE id = ?",
                    (json.dumps({"video_name": "clip", "frames": 1}), int(pred["id"])),
                )
            mismatch = validate_video_artifact_integrity(
                db,
                artifact_job_id,
                merged,
                expected_sample_ids=sample_ids,
            )
            self.assertIn("video_frame_count_mismatch", {item["code"] for item in mismatch["errors"]})

            stray_path = run_dir / "videos" / "clip" / "stray.mp4"
            stray_path.write_bytes(b"video")
            second_job_id = int(jobs[1]["job_id"])
            db.add_artifact(
                second_job_id,
                None,
                "pred_video",
                str(stray_path),
                "video/mp4",
                {"video_name": "clip", "frames": 2},
            )
            shard_report = validate_finalize_video_artifact_integrity(
                db,
                artifact_job_id,
                [int(row["job_id"]) for row in jobs],
                merged,
                expected_sample_ids=sample_ids,
            )
            self.assertIn(
                "unexpected_shard_video_artifact",
                {item["code"] for item in shard_report["errors"]},
            )


if __name__ == "__main__":
    unittest.main()
