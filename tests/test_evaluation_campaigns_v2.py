from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
import sys
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.evaluations_v2 import (
    CampaignDependencyError,
    EvaluationConflict,
    SourceChanged,
    _capture_source_tree_guard,
    _objective_by_method,
    _objective_metric_snapshot,
    _validate_final_campaign_sources,
    blind_heartbeat,
    blind_media_asset,
    blind_payload,
    blind_review_task,
    blind_reviews,
    blind_session,
    blind_submit_vote,
    campaign_analysis_v2,
    campaign_export_v2,
    campaign_objective_curve_v2,
    close_campaign_v2,
    create_campaign_v2,
    delete_campaign_v2,
    get_campaign_v2,
    get_preparation_v2,
    list_run_outputs,
    preview_campaign_v2,
    publish_campaign_v2,
    request_publish_campaign_v2,
    run_pending_preparations,
)
from vfieval.evaluations import add_candidate, create_campaign, publish_campaign
from vfieval.media_assets import (
    bind_metric_result,
    create_collection,
    ensure_collection,
    get_asset,
    soft_delete_asset,
    sync_run_assets,
    upsert_asset,
)
from vfieval.media_items import (
    bind_run_source,
    ensure_canonical_gt_item,
    list_item_predictions,
    register_external_prediction,
)
from vfieval.run_cleanup import RunCleanupService
from vfieval.pipeline.evaluation_freeze import FreezeBackendUnavailable

from v13_test_utils import add_completed_pred_run, make_workspace, write_mp4


class _FakeRawVideoSink:
    instances = []

    def __init__(self, path, **kwargs):
        self.path = Path(path)
        self.kwargs = dict(kwargs)
        self.frames = []
        self.aborted = False
        type(self).instances.append(self)

    def start(self):
        return None

    def write(self, frame):
        self.frames.append(bytes(frame))

    def close_input(self):
        return None

    def wait(self):
        self.path.write_bytes(b"fake-campaign-mp4")

    def finish(self):
        self.close_input()
        self.wait()

    def abort(self):
        self.aborted = True
        self.path.unlink(missing_ok=True)


class EvaluationCampaignV2Tests(unittest.TestCase):
    def test_final_campaign_source_validation_catches_earlier_item_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "item-one.mp4"
            source.write_bytes(b"0123456789abcdef")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            stat = source.stat()
            guarded_source = {
                "decode_cache": {
                    "source_path": source,
                    "source_sha256": digest,
                    "source_size_bytes": stat.st_size,
                    "source_mtime_ns": stat.st_mtime_ns,
                }
            }
            guarded_asset = {"content_sha256": digest}
            source.write_bytes(b"fedcba9876543210")
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(SourceChanged, "source changed"):
                _validate_final_campaign_sources(
                    [(guarded_source, guarded_asset, "Item one Method A")]
                )

    def test_final_campaign_source_validation_hashes_frame_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "frames"
            source.mkdir()
            frame = source / "000000.png"
            frame.write_bytes(b"frame-before")
            guarded_source = {"source_tree_guard": _capture_source_tree_guard(source)}
            frame.write_bytes(b"frame-after!")
            with self.assertRaisesRegex(SourceChanged, "frame source changed"):
                _validate_final_campaign_sources(
                    [(guarded_source, {}, "Item one GT")]
                )

    def test_objective_snapshot_merges_shards_and_keeps_latest_real_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_id, _other_run, _body, _paths = self._two_runs(workspace, db)
            run = db.get_run(run_id)
            inference_job_id = int(run["inference_job_id"])
            dataset_id = int(run["dataset_id"])
            first_sample = db.query(
                "SELECT id FROM samples WHERE dataset_id = ? ORDER BY id", (dataset_id,)
            )[0]
            sample_ids = [int(first_sample["id"])]
            for index in (1, 2, 3):
                sample_ids.append(
                    db.add_sample(
                        dataset_id,
                        f"clip_{index:06d}",
                        str(_paths[1]),
                        str(_paths[1]),
                        str(_paths[0]),
                        {"video_name": "clip", "frame_index": index},
                    )
                )
            unrelated_sample = db.add_sample(
                dataset_id,
                "other_000000",
                str(_paths[1]),
                str(_paths[1]),
                str(_paths[0]),
                {"video_name": "other", "frame_index": 0},
            )
            asset_id = int(
                db.get(
                    """
                    SELECT asset_id FROM run_media_assets
                    WHERE run_id = ? AND role = 'pred' AND video_name = 'clip'
                    """,
                    (run_id,),
                )["asset_id"]
            )
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET metrics_json = ? WHERE id = ?",
                    (json.dumps(["lpips_vit_patch"]), run_id),
                )

            old_job = db.add_run_job(
                run_id,
                "metric",
                {"metric_names": ["lpips_vit_patch"], "metric_wave_id": "old"},
                shard_index=0,
                device="npu:0",
            )
            shard_a = db.add_run_job(
                run_id,
                "metric",
                {"metric_names": ["lpips_vit_patch"], "metric_wave_id": "good"},
                shard_index=0,
                device="npu:0",
            )
            shard_b = db.add_run_job(
                run_id,
                "metric",
                {"metric_names": ["lpips_vit_patch"], "metric_wave_id": "good"},
                shard_index=1,
                device="npu:1",
            )
            with db.connection() as conn:
                conn.execute(
                    "UPDATE jobs SET status = 'completed' WHERE id IN (?, ?, ?)",
                    (old_job, shard_a, shard_b),
                )

            def add_result(job_id, sample_id, status, value, details, *, bind):
                result_id = db.add_metric_result(
                    job_id,
                    inference_job_id,
                    sample_id,
                    "lpips_vit_patch",
                    status,
                    value,
                    details,
                )
                if bind:
                    bind_metric_result(
                        db,
                        result_id,
                        None,
                        asset_id,
                        video_name="clip",
                    )
                return result_id

            add_result(
                old_job,
                sample_ids[0],
                "unavailable",
                None,
                {"reason": "stale unavailable"},
                bind=True,
            )
            add_result(shard_a, sample_ids[0], "completed", 0.2, {}, bind=True)
            add_result(shard_b, sample_ids[1], "completed", 0.4, {}, bind=True)
            add_result(
                shard_a,
                sample_ids[2],
                "unavailable",
                None,
                {"reason": "bound frame unavailable"},
                bind=True,
            )
            add_result(
                shard_b,
                sample_ids[3],
                "unavailable",
                None,
                {"reason": "lpips_vit unavailable: frame 3"},
                bind=False,
            )
            add_result(
                shard_b,
                unrelated_sample,
                "unavailable",
                None,
                {"reason": "unrelated video failure"},
                bind=False,
            )
            global_result = add_result(
                shard_b,
                None,
                "unavailable",
                None,
                {"reason": "global startup failure"},
                bind=False,
            )
            bind_metric_result(db, global_result, None, None, video_name="")

            failed_a = db.add_run_job(
                run_id,
                "metric",
                {"metric_names": ["lpips_vit_patch"], "metric_wave_id": "failed"},
                shard_index=0,
                device="npu:0",
            )
            failed_b = db.add_run_job(
                run_id,
                "metric",
                {"metric_names": ["lpips_vit_patch"], "metric_wave_id": "failed"},
                shard_index=1,
                device="npu:1",
            )
            with db.connection() as conn:
                conn.execute(
                    "UPDATE jobs SET status = 'failed', error_json = ? WHERE id = ?",
                    (json.dumps({"message": "root shard failed"}), failed_a),
                )
                conn.execute(
                    "UPDATE jobs SET status = 'canceled', error_json = ? WHERE id = ?",
                    (json.dumps({"message": "sibling canceled"}), failed_b),
                )

            methods = {
                1: {"id": 1, "source_run_id": run_id, "label_snapshot": "Method A"}
            }
            bindings = [
                {
                    "source_asset_id": asset_id,
                    "method_id": 1,
                    "video_name": "clip",
                    "expected_frame_count": 4,
                }
            ]
            before_identity_fill = _objective_metric_snapshot(db, bindings, methods)
            bind_metric_result(db, global_result, None, None, video_name="other")
            snapshot = _objective_metric_snapshot(db, bindings, methods)
            self.assertNotEqual(before_identity_fill["fingerprint"], snapshot["fingerprint"])
            selected = sorted(snapshot["rows"], key=lambda row: int(row["sample_id"]))
            self.assertEqual(
                [row["status"] for row in selected],
                ["completed", "completed", "unavailable"],
            )
            self.assertEqual(
                [float(row["value"]) for row in selected if row["value"] is not None],
                [0.2, 0.4],
            )
            self.assertNotIn(
                (run_id, "lpips_vit_patch", "other"),
                snapshot["result_fallback_reasons"],
            )
            self.assertEqual(
                snapshot["job_fallback_reasons"][(run_id, "lpips_vit_patch", "")],
                Counter({"root shard failed": 1, "sibling canceled": 1}),
            )
            objective = _objective_by_method(
                db, bindings, methods, snapshot=snapshot
            )
            item = objective["items"][0]
            self.assertEqual(item["status"], "unavailable")
            self.assertEqual(item["frame_coverage"], {"expected": 4, "observed": 3, "completed": 2})
            self.assertEqual(item["reason_counts"]["bound frame unavailable"], 1)
            self.assertEqual(item["reason_counts"]["lpips_vit unavailable: frame 3"], 1)
            self.assertNotIn("stale unavailable", item["reason_counts"])
            self.assertNotIn("root shard failed", item["reason_counts"])
            self.assertNotIn("unrelated video failure", item["reason_counts"])

    def test_objective_statistics_weight_items_not_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = make_workspace(tmp)
            methods = {
                1: {"id": 1, "source_run_id": 10, "label_snapshot": "Method A"}
            }
            bindings = [
                {
                    "source_asset_id": 101,
                    "method_id": 1,
                    "video_name": "short",
                    "expected_frame_count": 2,
                },
                {
                    "source_asset_id": 102,
                    "method_id": 1,
                    "video_name": "long",
                    "expected_frame_count": 4,
                },
            ]
            rows = []
            row_id = 0
            for asset_id, values in ((101, [0.0, 0.0]), (102, [1.0] * 4)):
                for sample_id, value in enumerate(values):
                    row_id += 1
                    rows.append(
                        {
                            "id": row_id,
                            "distorted_asset_id": asset_id,
                            "metric_name": "lpips_vit_patch",
                            "sample_id": asset_id * 10 + sample_id,
                            "status": "completed",
                            "value": value,
                            "details": {},
                        }
                    )
            snapshot = {
                "asset_to_binding": {row["source_asset_id"]: row for row in bindings},
                "rows": rows,
                "run_states": {10: {"metrics": ["lpips_vit_patch"]}},
                "fingerprint": "fixture",
                "producer_state": [],
            }
            objective = _objective_by_method(db, bindings, methods, snapshot=snapshot)
            metric = objective["metrics"][0]
            self.assertEqual(metric["item_count"], 2)
            self.assertEqual(metric["count"], 2)
            self.assertEqual(metric["mean"], 0.5)
            self.assertEqual(metric["frame_coverage"]["completed"], 6)

    def test_objective_curve_compares_two_methods_on_semantic_frame_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _media_item, body, paths = self._item_campaign(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            detail = get_campaign_v2(db, int(campaign["id"]))
            evaluation_item = detail["items"][0]
            methods = {int(row["id"]): row for row in detail["methods"]}

            rows_by_slot = {
                "a": [("completed", 0.10, {}), ("unavailable", None, {"reason": "weights missing"})],
                "b": [("completed", 0.15, {}), ("completed", 0.25, {}), ("completed", None, {})],
            }
            for binding in evaluation_item["bindings"]:
                method = methods[int(binding["method_id"])]
                run_id = int(method["source_run_id"])
                run = db.get_run(run_id)
                dataset_id = int(run["dataset_id"])
                inference_job_id = int(run["inference_job_id"])
                samples = db.query(
                    "SELECT id, metadata_json FROM samples WHERE dataset_id = ? ORDER BY id",
                    (dataset_id,),
                )
                sample_ids = [int(samples[0]["id"])]
                for frame_index in (2, 4):
                    sample_ids.append(
                        db.add_sample(
                            dataset_id,
                            f"clip_{frame_index:06d}",
                            str(paths[1]),
                            str(paths[1]),
                            str(paths[0]),
                            {"video_name": "clip", "frame_index": frame_index},
                        )
                    )
                metric_job_id = db.add_run_job(
                    run_id,
                    "metric",
                    {"metric_names": ["lpips_vit_patch"], "metric_wave_id": f"curve-{method['slot']}"},
                )
                with db.connection() as conn:
                    conn.execute(
                        "UPDATE runs SET metrics_json = ? WHERE id = ?",
                        (json.dumps(["lpips_vit_patch"]), run_id),
                    )
                    conn.execute(
                        "UPDATE jobs SET status = 'completed' WHERE id = ?",
                        (metric_job_id,),
                    )
                for sample_id, (status, value, details_json) in zip(
                    sample_ids,
                    rows_by_slot[str(method["slot"])],
                ):
                    result_id = db.add_metric_result(
                        metric_job_id,
                        inference_job_id,
                        sample_id,
                        "lpips_vit_patch",
                        status,
                        value,
                        details_json,
                    )
                    bind_metric_result(
                        db,
                        result_id,
                        None,
                        int(binding["source_asset_id"]),
                        video_name="clip",
                    )

            curve = campaign_objective_curve_v2(
                db,
                int(campaign["id"]),
                int(evaluation_item["id"]),
                "lpips_vit_patch",
            )

            self.assertEqual(curve["frame_count"], 3)
            self.assertEqual(
                [point["frame_index"] for point in curve["series"][0]["points"]],
                [0, 2, 4],
            )
            by_slot = {row["slot"]: row for row in curve["series"]}
            self.assertEqual(
                [point["status"] for point in by_slot["a"]["points"]],
                ["completed", "unavailable", "missing"],
            )
            self.assertEqual(
                [point["status"] for point in by_slot["b"]["points"]],
                ["completed", "completed", "unavailable"],
            )
            self.assertIn("weights missing", by_slot["a"]["reason_counts"])
            self.assertIn(
                "metric completed without a finite value",
                by_slot["b"]["reason_counts"],
            )
            serialized = json.dumps(curve, sort_keys=True)
            for forbidden in ("run_id", "asset_id", "sample_id", "storage_path"):
                self.assertNotIn(forbidden, serialized)
            with self.assertRaisesRegex(ValueError, "metric_name"):
                campaign_objective_curve_v2(
                    db,
                    int(campaign["id"]),
                    int(evaluation_item["id"]),
                    "vmaf",
                )

    def _upload_asset(self, db, collection_id, path, role, name):
        return upsert_asset(
            db,
            collection_id=collection_id,
            source_key=f"upload:v2:{name}",
            source_kind="upload",
            media_kind="video",
            role=role,
            display_name=name,
            original_name=path.name,
            storage_path=path,
            size_bytes=path.stat().st_size,
            frame_count=3,
            width=8,
            height=8,
            fps=5,
        )

    def _two_runs(self, workspace, db, *, size=(8, 8)):
        gt_path = write_mp4(
            workspace.runs_dir / "shared" / "gt.mp4",
            [(0, 0, 0), (20, 0, 0), (40, 0, 0)],
            size=size,
        )
        pred_a = write_mp4(
            workspace.runs_dir / "method-a" / "pred.mp4",
            [(0, 2, 0), (20, 2, 0), (40, 2, 0)],
            size=size,
        )
        pred_b = write_mp4(
            workspace.runs_dir / "method-b" / "pred.mp4",
            [(0, 0, 2), (20, 0, 2), (40, 0, 2)],
            size=size,
        )
        run_a = add_completed_pred_run(
            db, workspace, "method-a", pred_a, video_name="clip", gt_video_path=gt_path, size=size
        )
        run_b = add_completed_pred_run(
            db, workspace, "method-b", pred_b, video_name="clip", gt_video_path=gt_path, size=size
        )
        sync_run_assets(db, workspace, run_a)
        sync_run_assets(db, workspace, run_b)
        body = {
            "name": "internal-interpolation-study",
            "public_title": "Interpolation study",
            "target_votes": 1,
            "seed": 17,
            "methods": [
                {"run_id": run_a, "label": "Method Alpha"},
                {"run_id": run_b, "label": "Method Beta"},
            ],
        }
        return run_a, run_b, body, (gt_path, pred_a, pred_b)

    def _item_campaign(
        self,
        workspace,
        db,
        *,
        gt_size=(12, 8),
        pred_a_size=(8, 8),
        pred_b_size=(16, 8),
    ):
        collection = ensure_collection(
            db,
            "videos/test4k",
            "videos-test4k-campaign-v2",
            {"source_kind": "folder", "video_group": "test4k"},
        )
        gt_path = write_mp4(
            workspace.root.parent / "videos" / "test4k" / "clip.mp4",
            [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0)],
            size=gt_size,
            fps=5,
        )
        gt_asset = upsert_asset(
            db,
            collection_id=int(collection["id"]),
            source_key="folder:test4k/clip.mp4",
            source_kind="folder",
            media_kind="video",
            role="gt",
            display_name="clip.mp4",
            original_name="clip.mp4",
            storage_path=gt_path,
            frame_count=5,
            width=gt_size[0],
            height=gt_size[1],
            fps=5,
        )
        item = ensure_canonical_gt_item(db, int(gt_asset["id"]))
        pred_a = write_mp4(
            workspace.runs_dir / "item-method-a" / "pred.mp4",
            [(0, 1, 0), (20, 1, 0), (40, 1, 0)],
            size=pred_a_size,
            fps=5,
        )
        pred_b = write_mp4(
            workspace.runs_dir / "item-method-b" / "pred.mp4",
            [(0, 0, 1), (20, 0, 1), (40, 0, 1)],
            size=pred_b_size,
            fps=5,
        )
        mapping = [0, 2, 4]
        run_a = add_completed_pred_run(
            db,
            workspace,
            "item-method-a",
            pred_a,
            video_name="clip",
            size=pred_a_size,
            source_frame_indices=mapping,
        )
        run_b = add_completed_pred_run(
            db,
            workspace,
            "item-method-b",
            pred_b,
            video_name="clip",
            size=pred_b_size,
            source_frame_indices=mapping,
        )
        for run_id in (run_a, run_b):
            bind_run_source(db, run_id, int(item["id"]), video_name="clip")
            sync_run_assets(db, workspace, run_id)
        self.assertEqual(len(list_item_predictions(db, int(item["id"]))["predictions"]), 2)
        body = {
            "name": "item-campaign",
            "public_title": "Item Campaign",
            "target_votes": 1,
            "media_item_ids": [int(item["id"])],
            "method_a": {"kind": "run", "run_id": run_a},
            "method_b": {"kind": "run", "run_id": run_b},
            "spatial_policy": {
                "mode": "smallest_pred",
                "filter": "lanczos",
                "allow_known_aspect_stretch": True,
            },
        }
        return item, body, (gt_path, pred_a, pred_b)

    def _freeze_frame_fixture(
        self,
        root: Path,
        *,
        frame_count: int = 2,
        gt_size: tuple[int, int] = (8, 8),
        pred_a_size: tuple[int, int] = (8, 8),
        pred_b_size: tuple[int, int] = (8, 8),
    ):
        from vfieval.alignment import plan_alignment

        sizes = {"gt": gt_size, "pred_a": pred_a_size, "pred_b": pred_b_size}
        colors = {"gt": (20, 0, 0), "pred_a": (0, 20, 0), "pred_b": (0, 0, 20)}
        sources = {}
        for slot in ("gt", "pred_a", "pred_b"):
            directory = root / "source-frames" / slot
            directory.mkdir(parents=True, exist_ok=True)
            frames = []
            for index in range(frame_count):
                frame = directory / f"{index:06d}.png"
                Image.new("RGB", sizes[slot], colors[slot]).save(frame)
                frames.append(frame)
            sources[slot] = frames
        temporal = {
            "mode": "exact",
            "reference_frame_count": frame_count,
            "frame_count": frame_count,
            "prediction_frame_counts": [frame_count, frame_count],
            "mapping_count": frame_count,
            "mapping_first": 0,
            "mapping_last": frame_count - 1,
            "mapping_sha256": "fixture",
            "fps": 5.0,
            "timestamps_verified": True,
            "timestamps_sha256": "fixture",
            "timestamp_tolerance_seconds": 0.001,
        }
        plan = plan_alignment(
            {
                "slot": "gt",
                "width": gt_size[0],
                "height": gt_size[1],
                "frame_count": frame_count,
                "fps": 5.0,
            },
            [
                {
                    "slot": "pred_a",
                    "width": pred_a_size[0],
                    "height": pred_a_size[1],
                    "frame_count": frame_count,
                    "fps": 5.0,
                },
                {
                    "slot": "pred_b",
                    "width": pred_b_size[0],
                    "height": pred_b_size[1],
                    "frame_count": frame_count,
                    "fps": 5.0,
                },
            ],
            temporal_summary=temporal,
        )
        return plan, sources

    def test_item_mode_publish_materializes_three_streams_and_frozen_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            item, body, source_paths = self._item_campaign(workspace, db)
            preview = preview_campaign_v2(db, workspace, body)
            self.assertEqual(preview["task_count"], 1)
            plan = preview["items"][0]["alignment_plan"]
            self.assertEqual(plan["target"]["width"], 8)
            self.assertEqual(plan["target"]["height"], 8)
            self.assertEqual(plan["temporal"]["frame_count"], 3)
            self.assertTrue(plan["sources"]["gt"]["aspect_changed"])
            self.assertTrue(plan["sources"]["pred_b"]["aspect_changed"])

            draft = create_campaign_v2(db, workspace, body)
            published = publish_campaign_v2(db, workspace, int(draft["id"]))
            self.assertEqual(published["status"], "published")
            self.assertEqual(published["tasks"], 1)
            package = workspace.evaluations_dir / str(draft["id"])
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            manifest_item = manifest["items"][0]
            self.assertEqual(manifest_item["media_item_id"], int(item["id"]))
            self.assertEqual(manifest_item["alignment_fingerprint"], plan["fingerprint"])
            self.assertEqual(manifest_item["alignment_plan"]["target"]["width"], 8)
            freeze_report = manifest_item["freeze_pipeline"]
            self.assertEqual(
                set(freeze_report["outputs"]),
                {"gt", "pred_a", "pred_b"},
            )
            self.assertTrue(
                all(output["mode"] for output in freeze_report["outputs"].values())
            )
            self.assertEqual(freeze_report["version"], "campaign-freeze-stream-v3")
            self.assertEqual(freeze_report["gop_policy"]["gop_frames"], 5)
            self.assertEqual(set(freeze_report["keyframe_probe"]), {"gt", "pred_a", "pred_b"})
            self.assertEqual(len(freeze_report["stability_policy_fingerprint"]), 64)
            self.assertEqual(
                freeze_report["stability_policy_fingerprint"],
                freeze_report["stability_policy"]["fingerprint"],
            )
            self.assertIn("total", freeze_report["timings"])
            self.assertFalse(list(package.rglob("*.png")))
            for method in manifest_item["methods"]:
                self.assertTrue((package / method["path"]).is_file())
                self.assertNotIn("diff", method)

            preparation = get_preparation_v2(db, int(draft["id"]))
            self.assertEqual(preparation["state"], "completed")
            self.assertEqual(preparation["report"]["current"], 1)
            self.assertEqual(preparation["report"]["total"], 1)
            self.assertEqual(preparation["report"]["stage"], "completed")
            self.assertEqual(preparation["report"]["overall_fraction"], 1.0)
            self.assertEqual(preparation["report"]["pipeline"], "campaign-freeze-stream-v3")

            frozen_assets = db.query(
                "SELECT * FROM media_assets WHERE source_kind = 'evaluation_package' ORDER BY id"
            )
            self.assertEqual(len(frozen_assets), 3)
            self.assertTrue(all(int(asset["width"]) == 8 for asset in frozen_assets))
            self.assertTrue(all(int(asset["height"]) == 8 for asset in frozen_assets))
            self.assertTrue(all(int(asset["frame_count"]) == 3 for asset in frozen_assets))
            frozen_members = db.query(
                """
                SELECT member_role, reusable_as_pred, producer_kind, spatial_origin_json
                FROM media_item_members
                WHERE item_id = ? AND member_role IN ('evaluation_gt', 'evaluation_pred')
                ORDER BY id
                """,
                (int(item["id"]),),
            )
            self.assertEqual(
                [row["member_role"] for row in frozen_members],
                ["evaluation_gt", "evaluation_pred", "evaluation_pred"],
            )
            self.assertTrue(all(int(row["reusable_as_pred"]) == 0 for row in frozen_members))
            self.assertTrue(all(row["producer_kind"] == "evaluation_package" for row in frozen_members))
            bindings = db.query(
                "SELECT frozen_member_id FROM evaluation_bindings_v2 WHERE frozen_member_id IS NOT NULL"
            )
            self.assertEqual(len(bindings), 2)
            analysis = campaign_analysis_v2(db, int(draft["id"]), bootstrap_samples=0)
            self.assertEqual(analysis["alignment"]["fingerprints"], [plan["fingerprint"]])

            cleanup = RunCleanupService(db, workspace, cache_grace_seconds=0)
            request = cleanup.request_delete(int(body["method_a"]["run_id"]))
            self.assertEqual(cleanup.process_request(int(request["id"]))["status"], "completed")
            for source in source_paths:
                source.unlink(missing_ok=True)
            for asset in frozen_assets:
                self.assertTrue(Path(str(asset["storage_path"])).exists())

    def test_video_freeze_streams_without_temporary_png_and_hashes_outputs_once(self) -> None:
        from vfieval.pipeline import evaluation_freeze as freeze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, sources = self._freeze_frame_fixture(root)
            output_dir = root / "frozen"
            eligibility = {
                slot: {
                    "eligible": False,
                    "reasons": ["fixture forces streaming"],
                    "probe": None,
                }
                for slot in ("gt", "pred_a", "pred_b")
            }
            _FakeRawVideoSink.instances = []
            real_digest = freeze._digest_path
            real_materialize = freeze.materialize_aligned_rgb
            with patch.object(
                freeze, "streaming_backend_available", return_value=True
            ), patch.object(
                freeze, "_collect_remux_eligibility", return_value=eligibility
            ), patch.object(
                freeze, "_RawVideoSink", _FakeRawVideoSink
            ), patch.object(
                freeze, "validate_frozen_video", return_value={}
            ) as validate, patch.object(
                freeze, "_digest_path", wraps=real_digest
            ) as digest, patch.object(
                freeze, "materialize_aligned_rgb", wraps=real_materialize
            ) as materialize:
                result = freeze.freeze_campaign_media(
                    plan,
                    sources,
                    output_dir,
                    media_kind="video",
                    fps=5.0,
                    ffmpeg=sys.executable,
                    ffprobe=sys.executable,
                )

            self.assertEqual(result["pipeline"], "streaming")
            self.assertEqual(set(result["artifacts"]), set(freeze.OUTPUT_SLOTS))
            self.assertTrue(
                all(artifact["mode"] == "stream" for artifact in result["artifacts"].values())
            )
            self.assertFalse(list(output_dir.rglob("*.png")))
            self.assertEqual(materialize.call_count, 2 * len(freeze.SOURCE_SLOTS))
            self.assertEqual(validate.call_count, len(freeze.OUTPUT_SLOTS))
            self.assertEqual(len(_FakeRawVideoSink.instances), len(freeze.OUTPUT_SLOTS))
            self.assertTrue(
                all(len(sink.frames) == 2 for sink in _FakeRawVideoSink.instances)
            )
            self.assertTrue(
                all(
                    sink.kwargs["threads"] == result["encoder_threads"]
                    for sink in _FakeRawVideoSink.instances
                )
            )

            digest_counts = Counter(Path(call.args[0]).resolve() for call in digest.call_args_list)
            artifact_paths = {
                Path(artifact["path"]).resolve() for artifact in result["artifacts"].values()
            }
            self.assertEqual(set(digest_counts), artifact_paths)
            self.assertTrue(all(digest_counts[path] == 1 for path in artifact_paths))

    def test_item_mode_refuses_a_legacy_video_encoder_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _item, body, _source_paths = self._item_campaign(workspace, db)
            draft = create_campaign_v2(db, workspace, body)
            with patch(
                "vfieval.evaluations_v2.freeze_campaign_media",
                side_effect=FreezeBackendUnavailable("fixture backend unavailable"),
            ):
                with self.assertRaisesRegex(
                    FreezeBackendUnavailable,
                    "refusing a legacy encoder fallback",
                ):
                    publish_campaign_v2(db, workspace, int(draft["id"]))

            self.assertFalse((workspace.evaluations_dir / str(draft["id"])).exists())
            self.assertEqual(get_preparation_v2(db, int(draft["id"]))["state"], "failed")

    def test_rotation_and_resize_force_streaming_and_disable_paired_pred_remux(self) -> None:
        from vfieval.pipeline import evaluation_freeze as freeze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, sources = self._freeze_frame_fixture(root, pred_b_size=(10, 8))
            source_media = {}
            for slot in freeze.SOURCE_SLOTS:
                path = root / f"{slot}.mp4"
                path.write_bytes(f"source-{slot}".encode("utf-8"))
                source_media[slot] = path
            source_digests = {
                slot: freeze._source_signature(path)["sha256"]
                for slot, path in source_media.items()
            }

            def probe(path, **_kwargs):
                slot = Path(path).stem
                width, height = (10, 8) if slot == "pred_b" else (8, 8)
                return {
                    "codec": "h264",
                    "pix_fmt": "yuv420p",
                    "width": width,
                    "height": height,
                    "rotation_degrees": 90.0 if slot == "gt" else 0.0,
                    "frame_count": 2,
                    "fps": 5.0,
                    "timestamps": [0.0, 0.2],
                    "frame_durations": [0.2, 0.2],
                    "keyframe_timestamps": [0.0],
                    "cfr": True,
                    "audio_stream_count": 0,
                }

            _FakeRawVideoSink.instances = []
            with patch.object(
                freeze, "streaming_backend_available", return_value=True
            ), patch.object(
                freeze, "probe_video_for_freeze", side_effect=probe
            ), patch.object(
                freeze, "_RawVideoSink", _FakeRawVideoSink
            ), patch.object(
                freeze, "validate_frozen_video", return_value={}
            ), patch.object(freeze, "remux_video") as remux:
                result = freeze.freeze_campaign_media(
                    plan,
                    sources,
                    root / "frozen",
                    media_kind="video",
                    fps=5.0,
                    source_media=source_media,
                    source_timestamps={slot: [0.0, 0.2] for slot in freeze.SOURCE_SLOTS},
                    expected_source_sha256=source_digests,
                    ffmpeg=sys.executable,
                    ffprobe=sys.executable,
                )

            self.assertEqual(result["pipeline"], "streaming")
            self.assertIn("rotation metadata is not zero", result["remux"]["gt"]["reasons"])
            self.assertIn(
                "source requires spatial normalization",
                result["remux"]["pred_b"]["reasons"],
            )
            self.assertIn(
                "paired prediction is not remux-eligible",
                result["remux"]["pred_a"]["reasons"],
            )
            self.assertTrue(
                all(artifact["mode"] == "stream" for artifact in result["artifacts"].values())
            )
            remux.assert_not_called()

    def test_pred_remux_failure_reencodes_both_predictions(self) -> None:
        from vfieval.pipeline import evaluation_freeze as freeze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, sources = self._freeze_frame_fixture(root)
            source_media = {}
            for slot in freeze.SOURCE_SLOTS:
                path = root / f"{slot}.mp4"
                path.write_bytes(f"source-{slot}".encode("utf-8"))
                source_media[slot] = path
            eligibility = {
                "gt": {"eligible": False, "reasons": ["stream GT"], "probe": None},
                "pred_a": {"eligible": True, "reasons": [], "probe": {}},
                "pred_b": {"eligible": True, "reasons": [], "probe": {}},
            }

            def flaky_remux(_source, target, **_kwargs):
                target = Path(target)
                target.write_bytes(b"remux-copy")
                if target.name == "method-b.mp4":
                    raise freeze.RemuxError("injected remux incompatibility")
                return target

            _FakeRawVideoSink.instances = []
            with patch.object(
                freeze, "streaming_backend_available", return_value=True
            ), patch.object(
                freeze, "_collect_remux_eligibility", return_value=eligibility
            ), patch.object(
                freeze, "remux_video", side_effect=flaky_remux
            ) as remux, patch.object(
                freeze, "_RawVideoSink", _FakeRawVideoSink
            ), patch.object(
                freeze, "validate_frozen_video", return_value={}
            ):
                result = freeze.freeze_campaign_media(
                    plan,
                    sources,
                    root / "frozen",
                    media_kind="video",
                    fps=5.0,
                    source_media=source_media,
                    ffmpeg=sys.executable,
                    ffprobe=sys.executable,
                )

            self.assertEqual(remux.call_count, 2)
            self.assertEqual(result["artifacts"]["pred_a"]["mode"], "stream")
            self.assertEqual(result["artifacts"]["pred_b"]["mode"], "stream")
            self.assertEqual(
                Path(result["artifacts"]["pred_a"]["path"]).read_bytes(),
                b"fake-campaign-mp4",
            )
            self.assertEqual(
                Path(result["artifacts"]["pred_b"]["path"]).read_bytes(),
                b"fake-campaign-mp4",
            )

    def test_pred_remux_fast_path_commits_as_a_pair(self) -> None:
        from vfieval.pipeline import evaluation_freeze as freeze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, sources = self._freeze_frame_fixture(root)
            source_media = {}
            for slot in freeze.SOURCE_SLOTS:
                path = root / f"{slot}.mp4"
                path.write_bytes(f"source-{slot}".encode("utf-8"))
                source_media[slot] = path
            eligibility = {
                "gt": {"eligible": False, "reasons": ["stream GT"], "probe": None},
                "pred_a": {"eligible": True, "reasons": [], "probe": {}},
                "pred_b": {"eligible": True, "reasons": [], "probe": {}},
            }

            def private_remux(_source, target, **_kwargs):
                target = Path(target)
                target.write_bytes(b"private-remux")
                return target

            _FakeRawVideoSink.instances = []
            with patch.object(
                freeze, "streaming_backend_available", return_value=True
            ), patch.object(
                freeze, "_collect_remux_eligibility", return_value=eligibility
            ), patch.object(
                freeze, "remux_video", side_effect=private_remux
            ) as remux, patch.object(
                freeze, "_RawVideoSink", _FakeRawVideoSink
            ), patch.object(
                freeze, "validate_frozen_video", return_value={}
            ):
                result = freeze.freeze_campaign_media(
                    plan,
                    sources,
                    root / "frozen",
                    media_kind="video",
                    fps=5.0,
                    source_media=source_media,
                    ffmpeg=sys.executable,
                    ffprobe=sys.executable,
                )

            self.assertEqual(result["pipeline"], "remux+stream")
            self.assertEqual(remux.call_count, 2)
            self.assertEqual(result["artifacts"]["pred_a"]["mode"], "remux")
            self.assertEqual(result["artifacts"]["pred_b"]["mode"], "remux")
            self.assertEqual(
                {sink.path.name for sink in _FakeRawVideoSink.instances},
                {"reference.mp4"},
            )
            for slot in ("pred_a", "pred_b"):
                frozen_path = Path(result["artifacts"][slot]["path"])
                self.assertEqual(frozen_path.read_bytes(), b"private-remux")
                self.assertFalse(frozen_path.samefile(source_media[slot]))

    def test_item_mode_uses_decoded_dimensions_when_probe_dimensions_are_swapped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _item, body, _source_paths = self._item_campaign(
                workspace,
                db,
                gt_size=(12, 8),
                pred_a_size=(10, 6),
                pred_b_size=(16, 8),
            )
            from vfieval.compare_inputs import inspect_compare_path as real_inspect

            def swapped_inspect(workspace_arg, path):
                observed = real_inspect(workspace_arg, path)
                observed["width"], observed["height"] = (
                    int(observed["height"]),
                    int(observed["width"]),
                )
                return observed

            with patch(
                "vfieval.compare_inputs.inspect_compare_path",
                side_effect=swapped_inspect,
            ):
                preview = preview_campaign_v2(db, workspace, body)
                row = preview["items"][0]
                self.assertEqual(
                    (row["reference"]["width"], row["reference"]["height"]),
                    (12, 8),
                )
                self.assertEqual(
                    (row["methods"]["a"]["width"], row["methods"]["a"]["height"]),
                    (10, 6),
                )
                self.assertEqual(
                    (row["methods"]["b"]["width"], row["methods"]["b"]["height"]),
                    (16, 8),
                )
                plan = row["alignment_plan"]
                self.assertEqual(
                    plan["sources"]["gt"]["original"],
                    {"width": 12, "height": 8},
                )
                self.assertEqual(
                    plan["sources"]["pred_a"]["original"],
                    {"width": 10, "height": 6},
                )
                self.assertEqual(
                    plan["sources"]["pred_b"]["original"],
                    {"width": 16, "height": 8},
                )
                self.assertEqual(
                    (plan["target"]["width"], plan["target"]["height"]),
                    (10, 6),
                )

                draft = create_campaign_v2(db, workspace, body)
                published = publish_campaign_v2(db, workspace, int(draft["id"]))

            self.assertEqual(published["status"], "published")
            self.assertEqual(published["tasks"], 1)
            frozen_assets = db.query(
                "SELECT width, height, storage_path FROM media_assets "
                "WHERE source_kind = 'evaluation_package' ORDER BY id"
            )
            self.assertEqual(len(frozen_assets), 3)
            self.assertTrue(
                all(
                    (int(asset["width"]), int(asset["height"])) == (10, 6)
                    for asset in frozen_assets
                )
            )
            for asset in frozen_assets:
                observed = real_inspect(workspace, Path(str(asset["storage_path"])))
                self.assertEqual((int(observed["width"]), int(observed["height"])), (10, 6))
            manifest = json.loads(
                (workspace.evaluations_dir / str(draft["id"]) / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["items"][0]["alignment_fingerprint"], plan["fingerprint"])

    def test_item_mode_publish_revalidates_strict_frame_mapping_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _item, body, _source_paths = self._item_campaign(workspace, db)
            draft = create_campaign_v2(db, workspace, body)
            bindings = db.query(
                """
                SELECT source_member_id FROM evaluation_bindings_v2 b
                JOIN evaluation_methods_v2 m ON m.id = b.method_id
                WHERE m.campaign_id = ? ORDER BY m.slot
                """,
                (int(draft["id"]),),
            )
            with db.connection() as conn:
                conn.executemany(
                    "UPDATE media_item_members SET temporal_mapping_json = ? WHERE id = ?",
                    [
                        (
                            json.dumps({"source_frame_indices": [0, 1, 2]}),
                            int(binding["source_member_id"]),
                        )
                        for binding in bindings
                    ],
                )
            with self.assertRaisesRegex(ValueError, "new Campaign from a fresh preview"):
                publish_campaign_v2(db, workspace, int(draft["id"]))
            self.assertEqual(get_campaign_v2(db, int(draft["id"]))["status"], "failed")
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM evaluation_tasks_v2")["count"]),
                0,
            )
            self.assertEqual(
                int(
                    db.get(
                        "SELECT COUNT(*) AS count FROM media_assets WHERE source_kind = 'evaluation_package'"
                    )["count"]
                ),
                0,
            )
            self.assertFalse((workspace.evaluations_dir / str(draft["id"])).exists())

    def test_item_mode_publish_preserves_frame_sequence_package_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = ensure_collection(
                db,
                "videos/frame-set",
                "videos-frame-set-campaign-v2",
                {"source_kind": "folder", "video_group": "frame-set"},
            )
            gt_dir = workspace.root.parent / "videos" / "frame-set" / "clip"
            gt_dir.mkdir(parents=True, exist_ok=True)
            for index, value in enumerate((0, 20, 40)):
                Image.new("RGB", (8, 8), (value, 0, 0)).save(gt_dir / f"{index:06d}.png")
            gt_asset = upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key="folder:frame-set/clip",
                source_kind="folder",
                media_kind="frame_sequence",
                role="gt",
                display_name="clip",
                original_name="clip",
                storage_path=gt_dir,
                frame_count=3,
                width=8,
                height=8,
                fps=5,
            )
            item = ensure_canonical_gt_item(db, int(gt_asset["id"]))
            method_keys: list[str] = []
            for slot, color in (("a", (0, 1, 0)), ("b", (0, 0, 1))):
                pred_dir = workspace.media_dir / f"frame-sequence-{slot}"
                pred_dir.mkdir(parents=True, exist_ok=True)
                for index in range(3):
                    Image.new("RGB", (8, 8), color).save(pred_dir / f"{index:06d}.png")
                pred_asset = upsert_asset(
                    db,
                    collection_id=int(collection["id"]),
                    source_key=f"upload:frame-sequence-{slot}",
                    source_kind="upload",
                    media_kind="frame_sequence",
                    role="pred",
                    display_name=f"frame-sequence-{slot}",
                    original_name=f"frame-sequence-{slot}",
                    storage_path=pred_dir,
                    frame_count=3,
                    width=8,
                    height=8,
                    fps=5,
                )
                method_key = f"external:frame-sequence-{slot}"
                register_external_prediction(
                    db, int(item["id"]), int(pred_asset["id"]), method_key=method_key
                )
                method_keys.append(method_key)
            body = {
                "name": "frame-sequence-campaign",
                "public_title": "Frame sequence campaign",
                "media_item_ids": [int(item["id"])],
                "method_a": {"kind": "external", "method_key": method_keys[0]},
                "method_b": {"kind": "external", "method_key": method_keys[1]},
            }
            published = publish_campaign_v2(
                db,
                workspace,
                int(create_campaign_v2(db, workspace, body)["id"]),
            )
            self.assertEqual(published["status"], "published")
            assets = db.query(
                "SELECT media_kind, storage_path FROM media_assets WHERE source_kind = 'evaluation_package'"
            )
            self.assertEqual(len(assets), 3)
            self.assertTrue(all(row["media_kind"] == "frame_sequence" for row in assets))
            self.assertTrue(
                all(len(list(Path(str(row["storage_path"])).glob("*.png"))) == 3 for row in assets)
            )

    def test_preview_groups_two_run_methods_and_requires_strict_common_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_a, run_b, body, _paths = self._two_runs(workspace, db)
            preview = preview_campaign_v2(db, workspace, body)
            self.assertEqual(preview["schema_version"], 2)
            self.assertEqual(preview["task_count"], 1)
            self.assertEqual(preview["videos"][0]["status"], "ready")
            self.assertEqual(preview["videos"][0]["reference"]["frame_count"], 3)
            self.assertEqual(preview["videos"][0]["methods"]["a"]["width"], 8)
            self.assertEqual(preview["videos"][0]["methods"]["b"]["fps"], 5)
            self.assertEqual([row["run_id"] for row in preview["methods"]], [run_a, run_b])

            outputs = list_run_outputs(db)
            self.assertEqual({row["run_id"] for row in outputs}, {run_a, run_b})
            self.assertTrue(all(row["videos"] for row in outputs))

    def test_preview_rejects_actual_stream_fps_when_asset_metadata_matches(self) -> None:
        """The selector must validate the stream, not only catalog metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            gt_path = write_mp4(
                workspace.runs_dir / "timestamp-shared" / "gt.mp4",
                [(0, 0, 0), (20, 0, 0), (40, 0, 0)],
                fps=5,
            )
            pred_a = write_mp4(
                workspace.runs_dir / "timestamp-a" / "pred.mp4",
                [(0, 2, 0), (20, 2, 0), (40, 2, 0)],
                fps=5,
            )
            pred_b = write_mp4(
                workspace.runs_dir / "timestamp-b" / "pred.mp4",
                [(0, 0, 2), (20, 0, 2), (40, 0, 2)],
                fps=10,
            )
            # Deliberately persist matching 5 fps catalog metadata for all
            # three assets.  The decoded timestamps of Pred B still differ.
            run_a = add_completed_pred_run(
                db, workspace, "timestamp-a", pred_a, video_name="clip", gt_video_path=gt_path, fps=5
            )
            run_b = add_completed_pred_run(
                db, workspace, "timestamp-b", pred_b, video_name="clip", gt_video_path=gt_path, fps=5
            )
            sync_run_assets(db, workspace, run_a)
            sync_run_assets(db, workspace, run_b)
            preview = preview_campaign_v2(
                db,
                workspace,
                {
                    "name": "timestamp-mismatch",
                    "public_title": "Timestamp mismatch",
                    "methods": [
                        {"run_id": run_a, "label": "A"},
                        {"run_id": run_b, "label": "B"},
                    ],
                },
            )
            row = preview["videos"][0]
            self.assertEqual(row["status"], "alignment_mismatch")
            self.assertFalse(row["selectable"])
            self.assertIn("fps", " ".join(row["reasons"]).lower())

    def test_preview_uses_observed_fps_when_catalog_metadata_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            gt_path = write_mp4(
                workspace.runs_dir / "observed-fps-shared" / "gt.mp4",
                [(0, 0, 0), (20, 0, 0), (40, 0, 0)],
                fps=5,
            )
            pred_a = write_mp4(
                workspace.runs_dir / "observed-fps-a" / "pred.mp4",
                [(0, 2, 0), (20, 2, 0), (40, 2, 0)],
                fps=5,
            )
            pred_b = write_mp4(
                workspace.runs_dir / "observed-fps-b" / "pred.mp4",
                [(0, 0, 2), (20, 0, 2), (40, 0, 2)],
                fps=10,
            )
            run_a = add_completed_pred_run(
                db, workspace, "observed-fps-a", pred_a, video_name="clip", gt_video_path=gt_path, fps=5
            )
            run_b = add_completed_pred_run(
                db, workspace, "observed-fps-b", pred_b, video_name="clip", gt_video_path=gt_path, fps=5
            )
            sync_run_assets(db, workspace, run_a)
            sync_run_assets(db, workspace, run_b)
            frames = [Path(f"frame-{index}.png") for index in range(3)]
            # Some decoders cannot expose per-frame timestamps.  Strict
            # selection must still use the source stream fps rather than the
            # stale matching metadata persisted above.
            from vfieval.compare_inputs import inspect_compare_path as real_inspect

            def stale_inspect(workspace_arg, path):
                info = real_inspect(workspace_arg, path)
                info["fps"] = 5.0
                return info

            with patch("vfieval.evaluations_v2.inspect_compare_path", side_effect=stale_inspect), patch(
                "vfieval.datasets._load_compare_source_frames",
                side_effect=[
                    (frames, 5.0, [None, None, None]),
                    (frames, 5.0, [None, None, None]),
                    (frames, 5.0, [None, None, None]),
                    (frames, 10.0, [None, None, None]),
                ],
            ):
                preview = preview_campaign_v2(
                    db,
                    workspace,
                    {
                        "name": "observed-fps-mismatch",
                        "public_title": "Observed FPS mismatch",
                        "methods": [
                            {"run_id": run_a, "label": "A"},
                            {"run_id": run_b, "label": "B"},
                        ],
                    },
                )
            row = preview["videos"][0]
            self.assertEqual(row["status"], "alignment_mismatch")
            self.assertFalse(row["selectable"])
            self.assertIn("fps", " ".join(row["reasons"]).lower())

    def test_preview_reports_decoded_timestamp_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            frames = [Path(f"timestamp-frame-{index}.png") for index in range(3)]
            with patch(
                "vfieval.datasets._load_compare_source_frames",
                side_effect=[
                    (frames, 5.0, [0.0, 0.2, 0.4]),
                    (frames, 5.0, [0.0, 0.2, 0.4]),
                    (frames, 5.0, [0.0, 0.2, 0.4]),
                    (frames, 5.0, [0.0, 0.1, 0.4]),
                ],
            ):
                preview = preview_campaign_v2(db, workspace, body)
            row = preview["videos"][0]
            self.assertEqual(row["status"], "alignment_mismatch")
            self.assertFalse(row["selectable"])
            self.assertIn("timestamps", " ".join(row["reasons"]).lower())

    def test_advanced_uploaded_methods_use_explicit_video_and_gt_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "External blind methods")
            root = workspace.media_dir / "external-v2"
            gt = self._upload_asset(
                db,
                collection["id"],
                write_mp4(root / "gt.mp4", [(0, 0, 0)] * 3),
                "gt",
                "external-gt",
            )
            pred_a = self._upload_asset(
                db,
                collection["id"],
                write_mp4(root / "a.mp4", [(1, 0, 0)] * 3),
                "pred",
                "external-a",
            )
            pred_b = self._upload_asset(
                db,
                collection["id"],
                write_mp4(root / "b.mp4", [(0, 1, 0)] * 3),
                "pred",
                "external-b",
            )
            body = {
                "name": "external",
                "public_title": "External study",
                "methods": [
                    {
                        "source_kind": "upload",
                        "label": "External A",
                        "videos": [
                            {
                                "video_name": "clip",
                                "asset_id": pred_a["id"],
                                "reference_asset_id": gt["id"],
                            }
                        ],
                    },
                    {
                        "source_kind": "upload",
                        "label": "External B",
                        "videos": [
                            {
                                "video_name": "clip",
                                "asset_id": pred_b["id"],
                                "reference_asset_id": gt["id"],
                            }
                        ],
                    },
                ],
            }
            preview = preview_campaign_v2(db, workspace, body)
            self.assertEqual(preview["ready_video_names"], ["clip"])
            campaign = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            self.assertEqual(campaign["status"], "published")

    def test_atomic_package_survives_run_media_cleanup_and_blind_api_hides_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, source_paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            published = publish_campaign_v2(db, workspace, campaign["id"])
            self.assertEqual(published["status"], "published")
            self.assertEqual(published["tasks"], 1)
            package = workspace.root / "evaluations" / str(campaign["id"])
            self.assertTrue((package / "manifest.json").is_file())
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["freeze_pipeline"], "campaign-freeze-legacy-copy-v1"
            )
            self.assertEqual(
                manifest["items"][0]["freeze_pipeline"]["version"],
                "campaign-freeze-legacy-copy-v1",
            )
            self.assertEqual(
                get_preparation_v2(db, int(campaign["id"]))["report"]["pipeline"],
                "campaign-freeze-legacy-copy-v1",
            )
            frozen = db.query(
                "SELECT * FROM media_assets WHERE source_kind = 'evaluation_package' ORDER BY id"
            )
            self.assertEqual(len(frozen), 3)
            self.assertTrue(all(Path(row["storage_path"]).exists() for row in frozen))
            with self.assertRaisesRegex(ValueError, "immutable"):
                soft_delete_asset(db, workspace, int(frozen[0]["id"]))
            self.assertEqual(get_asset(db, int(frozen[0]["id"]))["state"], "ready")

            token = str(published["public_token"])
            public = blind_payload(db, token)
            self.assertEqual(public["campaign"]["title"], "Interpolation study")
            self.assertIsNone(public["task"])
            first = blind_session(
                db,
                token,
                {"evaluator_id": "browser-uuid", "display_name": "Alice"},
                lease_seconds=120,
            )
            task = first["task"]
            self.assertIsNotNone(task)
            assignment_token = str(
                db.get(
                    "SELECT assignment_token FROM evaluation_assignments_v2 WHERE evaluator_id = ?",
                    ("browser-uuid",),
                )["assignment_token"]
            )
            serialized = str(task)
            self.assertNotIn("run_id", serialized)
            self.assertNotIn("asset_id", serialized)
            self.assertNotIn("Method Alpha", serialized)
            self.assertNotIn("Method Beta", serialized)
            again = blind_session(
                db,
                token,
                {"evaluator_id": "browser-uuid", "display_name": "Alice"},
                lease_seconds=120,
            )
            self.assertEqual(again["task"]["token"], task["token"])
            renewed = blind_heartbeat(
                db, token, task["token"], "browser-uuid", lease_seconds=300
            )
            self.assertGreater(renewed["lease_expires_at"], task["lease_expires_at"])
            _asset, media_path = blind_media_asset(
                db, workspace, token, task["token"], "left", assignment_token
            )
            self.assertTrue(media_path.is_relative_to(package))

            with db.connection() as conn:
                conn.execute(
                    "UPDATE evaluation_assignments_v2 SET state = 'expired', lease_expires_at = ? WHERE evaluator_id = ?",
                    (0, "browser-uuid"),
                )
            with self.assertRaisesRegex(ValueError, "lease expired"):
                blind_media_asset(
                    db, workspace, token, task["token"], "left", assignment_token
                )
            with db.connection() as conn:
                conn.execute(
                    "UPDATE evaluation_assignments_v2 SET state = 'leased', lease_expires_at = ? WHERE evaluator_id = ?",
                    (10**12, "browser-uuid"),
                )

            for source in source_paths:
                source.unlink()
            vote = blind_submit_vote(
                db,
                token,
                task["token"],
                "browser-uuid",
                {
                    "choice": "tie",
                    "reasons": ["temporal_stability"],
                    "confidence": "high",
                },
            )
            self.assertTrue(vote["progress"]["complete"])
            self.assertIn("results", vote)
            self.assertEqual(
                {row["label"] for row in vote["results"]["human"]["ranking"]},
                {"Method Alpha", "Method Beta"},
            )
            _asset, surviving_path = blind_media_asset(
                db, workspace, token, task["token"], "reference", assignment_token
            )
            self.assertTrue(surviving_path.exists())

            analysis_a = campaign_analysis_v2(db, campaign["id"], bootstrap_samples=50)
            analysis_b = campaign_analysis_v2(db, campaign["id"], bootstrap_samples=50)
            self.assertEqual(analysis_a, analysis_b)
            self.assertEqual(analysis_a["human"]["ranking"][0]["score"], 0.5)
            self.assertIsNone(analysis_a["combined_score"])
            exported = campaign_export_v2(db, campaign["id"])
            self.assertEqual(len(exported["methods"]), 2)
            self.assertEqual(len(exported["items"]), 1)
            self.assertEqual(len(exported["tasks"]), 1)
            self.assertEqual(len(exported["votes"]), 1)

    def test_frozen_package_is_a_private_snapshot_not_a_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, source_paths = self._two_runs(workspace, db)
            campaign = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            package = workspace.evaluations_dir / str(campaign["id"])
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            frozen_pred = package / manifest["items"][0]["methods"][0]["path"]
            source_pred = source_paths[1]
            self.assertFalse(frozen_pred.samefile(source_pred))
            before = frozen_pred.read_bytes()
            source_pred.write_bytes(b"source bytes changed after publish")
            self.assertEqual(frozen_pred.read_bytes(), before)

    def test_optional_ratings_map_through_swap_and_reviews_update_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            token = str(campaign["public_token"])
            session = blind_session(
                db,
                token,
                {"evaluator_id": "ratings-browser", "display_name": "Ratings Tester"},
            )
            task = session["task"]
            self.assertIsNotNone(task)
            self.assertEqual(task["frame_count"], 3)
            self.assertEqual(task["fps"], 5.0)
            self.assertAlmostEqual(task["duration_seconds"], 0.6)

            for invalid in (0.75, 5.25, 1.1, float("nan"), True, [4], {"value": 4}):
                with self.assertRaisesRegex(ValueError, "0.25 steps"):
                    blind_submit_vote(
                        db,
                        token,
                        task["token"],
                        "ratings-browser",
                        {"choice": "left", "left_rating": invalid},
                    )

            blind_submit_vote(
                db,
                token,
                task["token"],
                "ratings-browser",
                {
                    "choice": "left",
                    "left_rating": 4.25,
                    "right_rating": None,
                    "confidence": "medium",
                    "note": "first",
                },
            )
            stored = db.get(
                """
                SELECT v.*, a.side_swap
                FROM evaluation_votes_v2 v
                JOIN evaluation_assignments_v2 a ON a.id = v.assignment_id
                WHERE v.evaluator_id = ?
                """,
                ("ratings-browser",),
            )
            self.assertIsNotNone(stored)
            if int(stored["side_swap"]):
                self.assertIsNone(stored["rating_a"])
                self.assertEqual(float(stored["rating_b"]), 4.25)
            else:
                self.assertEqual(float(stored["rating_a"]), 4.25)
                self.assertIsNone(stored["rating_b"])

            listed = blind_reviews(db, token, "ratings-browser")
            self.assertTrue(listed["editable"])
            self.assertEqual(len(listed["reviews"]), 1)
            self.assertEqual(listed["reviews"][0]["vote"]["left_rating"], 4.25)
            self.assertIsNone(listed["reviews"][0]["vote"]["right_rating"])
            reviewed = blind_review_task(db, token, task["token"], "ratings-browser")
            self.assertTrue(reviewed["task"]["review"])
            self.assertFalse(reviewed["task"]["read_only"])
            self.assertEqual(reviewed["task"]["vote"]["choice"], "left")
            self.assertEqual(reviewed["task"]["fps"], task["fps"])
            self.assertEqual(
                reviewed["task"]["duration_seconds"], task["duration_seconds"]
            )
            serialized_reviews = json.dumps(
                {"listed": listed, "reviewed": reviewed}, sort_keys=True
            )
            for forbidden in (
                "method_id",
                "run_id",
                "model",
                "checkpoint",
                "asset_id",
                "assignment_id",
            ):
                self.assertNotIn(forbidden, serialized_reviews)
            blind_session(
                db,
                token,
                {"evaluator_id": "other-browser", "display_name": "Other Tester"},
            )
            with self.assertRaises(KeyError):
                blind_review_task(db, token, task["token"], "other-browser")

            blind_submit_vote(
                db,
                token,
                task["token"],
                "ratings-browser",
                {"choice": "right", "left_rating": None, "right_rating": 2.5},
            )
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM evaluation_votes_v2")["count"]),
                1,
            )
            updated = blind_review_task(db, token, task["token"], "ratings-browser")
            self.assertEqual(updated["task"]["vote"]["choice"], "right")
            self.assertIsNone(updated["task"]["vote"]["left_rating"])
            self.assertEqual(updated["task"]["vote"]["right_rating"], 2.5)

            analysis = campaign_analysis_v2(db, int(campaign["id"]), bootstrap_samples=0)
            self.assertEqual(sum(row["count"] for row in analysis["ratings"]["methods"]), 1)
            self.assertEqual(
                [value for row in analysis["ratings"]["methods"] for value in [row["mean"]] if value is not None],
                [2.5],
            )

            with db.connection() as conn:
                conn.execute(
                    """
                    UPDATE media_assets
                    SET fps = NULL
                    WHERE id = (
                        SELECT frozen_reference_asset_id
                        FROM evaluation_items_v2
                        WHERE campaign_id = ?
                    )
                    """,
                    (int(campaign["id"]),),
                )
            historical_review = blind_review_task(
                db, token, task["token"], "ratings-browser"
            )
            self.assertIsNone(historical_review["task"]["fps"])
            self.assertAlmostEqual(
                historical_review["task"]["duration_seconds"], 0.6
            )

            with db.connection() as conn:
                conn.execute(
                    """
                    UPDATE media_assets
                    SET fps = NULL
                    WHERE id IN (
                        SELECT frozen_reference_asset_id
                        FROM evaluation_items_v2
                        WHERE campaign_id = ?
                        UNION
                        SELECT bindings.frozen_asset_id
                        FROM evaluation_bindings_v2 AS bindings
                        JOIN evaluation_items_v2 AS items
                          ON items.id = bindings.item_id
                        WHERE items.campaign_id = ?
                    )
                    """,
                    (int(campaign["id"]), int(campaign["id"])),
                )
            unknown_timing_review = blind_review_task(
                db, token, task["token"], "ratings-browser"
            )
            self.assertIsNone(unknown_timing_review["task"]["fps"])
            self.assertIsNone(unknown_timing_review["task"]["duration_seconds"])

            close_campaign_v2(db, int(campaign["id"]))
            self.assertFalse(blind_reviews(db, token, "ratings-browser")["editable"])
            closed_review = blind_review_task(db, token, task["token"], "ratings-browser")
            self.assertTrue(closed_review["task"]["read_only"])
            with self.assertRaisesRegex(ValueError, "not accepting votes"):
                blind_submit_vote(
                    db,
                    token,
                    task["token"],
                    "ratings-browser",
                    {"choice": "tie"},
                )

    def test_superseded_preparation_claim_cannot_publish_or_clear_new_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            request_publish_campaign_v2(db, campaign["id"])
            with db.connection() as conn:
                conn.execute(
                    """
                    UPDATE evaluation_preparations_v2
                    SET state = 'running', claim_token = 'new-owner', updated_at = ?
                    WHERE campaign_id = ?
                    """,
                    (10**12, int(campaign["id"])),
                )
            with self.assertRaises(EvaluationConflict):
                publish_campaign_v2(
                    db,
                    workspace,
                    campaign["id"],
                    claim_token="stale-owner",
                )
            preparation = get_preparation_v2(db, campaign["id"])
            self.assertEqual(preparation["state"], "running")
            self.assertEqual(preparation["claim_token"], "new-owner")
            self.assertEqual(get_campaign_v2(db, campaign["id"])["status"], "preparing")
            self.assertFalse((workspace.evaluations_dir / str(campaign["id"])).exists())

    def test_preparation_progress_keeps_legacy_counters_with_frame_details(self) -> None:
        from vfieval.evaluations_v2 import _PreparationProgressReporter

        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = make_workspace(tmp)
            with patch("vfieval.evaluations_v2._update_preparation_progress") as update:
                reporter = _PreparationProgressReporter(
                    db,
                    campaign_id=17,
                    claim_token="owned-claim",
                    item_total=2,
                    min_interval_seconds=0,
                )
                reporter.start_item(1, "clip.mp4")
                reporter.callback(
                    {
                        "stage": "streaming_frames",
                        "frame_current": 2,
                        "frame_total": 4,
                        "pipeline": "stream",
                    }
                )
                reporter.finish_item({"encode_seconds": 0.25})

            reports = [call.args[3] for call in update.call_args_list]
            self.assertGreaterEqual(len(reports), 3)
            self.assertTrue(
                all(
                    report["phase"] == "validating_and_freezing"
                    and "current" in report
                    and report["total"] == 2
                    for report in reports
                )
            )
            frame_report = next(
                report for report in reports if report["stage"] == "streaming_frames"
            )
            self.assertEqual(frame_report["item_index"], 1)
            self.assertEqual(frame_report["item_name"], "clip.mp4")
            self.assertEqual(frame_report["frame_current"], 2)
            self.assertEqual(frame_report["frame_total"], 4)
            self.assertEqual(frame_report["overall_fraction"], 0.25)
            self.assertEqual(frame_report["current"], 0)
            completed = reports[-1]
            self.assertEqual(completed["current"], 1)
            self.assertEqual(completed["overall_fraction"], 0.5)
            self.assertEqual(completed["item_timings"], {"encode_seconds": 0.25})

    def test_persistent_preparation_request_can_be_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            requested = request_publish_campaign_v2(db, campaign["id"])
            self.assertEqual(requested["campaign"]["status"], "preparing")
            self.assertEqual(requested["preparation"]["state"], "queued")
            completed = run_pending_preparations(db, workspace)
            self.assertEqual(completed, [{"campaign_id": campaign["id"], "status": "published"}])
            self.assertEqual(get_preparation_v2(db, campaign["id"])["state"], "completed")

    def test_preparation_runner_serializes_concurrent_process_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            first_entered = threading.Event()
            release_first = threading.Event()
            state_lock = threading.Lock()
            active = 0
            max_active = 0
            calls = 0

            def fake_runner(*_args, **_kwargs):
                nonlocal active, max_active, calls
                with state_lock:
                    calls += 1
                    call_number = calls
                    active += 1
                    max_active = max(max_active, active)
                try:
                    if call_number == 1:
                        first_entered.set()
                        self.assertTrue(release_first.wait(5))
                    return [{"call": call_number}]
                finally:
                    with state_lock:
                        active -= 1

            with patch(
                "vfieval.evaluations_v2._run_pending_preparations_locked",
                side_effect=fake_runner,
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(run_pending_preparations, db, workspace)
                    self.assertTrue(first_entered.wait(5))
                    second = pool.submit(run_pending_preparations, db, workspace)
                    time.sleep(0.1)
                    self.assertFalse(second.done())
                    release_first.set()
                    self.assertEqual(first.result(timeout=5), [{"call": 1}])
                    self.assertEqual(second.result(timeout=5), [{"call": 2}])
            self.assertEqual(max_active, 1)

    def test_run_purge_keeps_published_frozen_campaign_playable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            published = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            request = service.request_delete(run_a)
            completed = service.process_request(int(request["id"]))
            self.assertEqual(completed["status"], "completed")
            self.assertIsNotNone(db.get_run(run_a)["deleted_at"])

            session = blind_session(
                db,
                str(published["public_token"]),
                {"evaluator_id": "purge-browser", "display_name": "Purge Tester"},
            )
            task = session["task"]
            self.assertIsNotNone(task)
            assignment_token = str(
                db.get(
                    "SELECT assignment_token FROM evaluation_assignments_v2 WHERE evaluator_id = ?",
                    ("purge-browser",),
                )["assignment_token"]
            )
            for side in ("reference", "left", "right"):
                asset, path = blind_media_asset(
                    db,
                    workspace,
                    str(published["public_token"]),
                    str(task["token"]),
                    side,
                    assignment_token,
                )
                self.assertEqual(asset["source_kind"], "evaluation_package")
                self.assertTrue(path.exists())

    def test_incomplete_published_campaign_can_be_deleted_and_run_purge_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = publish_campaign_v2(
                db,
                workspace,
                create_campaign_v2(db, workspace, body)["id"],
            )
            campaign_id = int(campaign["id"])
            package = workspace.evaluations_dir / str(campaign_id)
            (package / "manifest.json").unlink()

            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            request = service.request_delete(run_a)
            failed = service.process_request(int(request["id"]))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error"]["type"], CampaignDependencyError.__name__)
            self.assertEqual(failed["error"]["campaign_id"], campaign_id)
            self.assertEqual(failed["error"]["action"], "open_campaign")

            with self.assertRaisesRegex(ValueError, "confirm_destructive"):
                delete_campaign_v2(
                    db,
                    workspace,
                    campaign_id,
                    confirmed=True,
                )
            deleted = delete_campaign_v2(
                db,
                workspace,
                campaign_id,
                confirmed=True,
                destructive_confirmed=True,
            )
            self.assertTrue(deleted["deleted"])
            self.assertFalse(package.exists())
            self.assertIsNone(
                db.get("SELECT id FROM evaluation_campaigns_v2 WHERE id = ?", (campaign_id,))
            )
            self.assertEqual(
                db.query(
                    "SELECT id FROM media_assets WHERE source_kind = 'evaluation_package' "
                    "AND source_key GLOB ?",
                    (f"evaluation_package:{campaign_id}:*",),
                ),
                [],
            )

            retried = service.request_delete(run_a)
            completed = service.process_request(int(retried["id"]))
            self.assertEqual(completed["status"], "completed")
            self.assertIsNotNone(db.get_run(run_a)["deleted_at"])

    def test_campaign_delete_tombstone_cleanup_is_retryable_after_file_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            campaign_id = int(campaign["id"])
            package = workspace.evaluations_dir / str(campaign_id)
            package.mkdir(parents=True)
            (package / "broken.partial").write_bytes(b"partial")
            tombstone = workspace.evaluations_dir / ".delete-staging" / str(campaign_id)
            real_rmtree = shutil.rmtree

            def fail_tombstone_once(path, *args, **kwargs):
                if Path(path) == tombstone:
                    raise OSError("file is in use")
                return real_rmtree(path, *args, **kwargs)

            with patch("vfieval.evaluations_v2.shutil.rmtree", side_effect=fail_tombstone_once):
                deleted = delete_campaign_v2(
                    db,
                    workspace,
                    campaign_id,
                    confirmed=True,
                )
            self.assertTrue(deleted["cleanup_pending"])
            self.assertTrue(tombstone.exists())

            retried = delete_campaign_v2(
                db,
                workspace,
                campaign_id,
                confirmed=True,
            )
            self.assertTrue(retried["already_deleted"])
            self.assertFalse(retried["cleanup_pending"])
            self.assertFalse(tombstone.exists())

    def test_run_purge_freezes_readable_published_v1_campaign_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_a, run_b, _body, _paths = self._two_runs(workspace, db)
            reference_id = int(
                db.get(
                    "SELECT asset_id FROM run_media_assets WHERE run_id = ? AND role = 'gt' LIMIT 1",
                    (run_a,),
                )["asset_id"]
            )
            pred_a_id = int(
                db.get(
                    "SELECT asset_id FROM run_media_assets WHERE run_id = ? AND role = 'pred' LIMIT 1",
                    (run_a,),
                )["asset_id"]
            )
            pred_b_id = int(
                db.get(
                    "SELECT asset_id FROM run_media_assets WHERE run_id = ? AND role = 'pred' LIMIT 1",
                    (run_b,),
                )["asset_id"]
            )
            legacy = create_campaign(db, {"name": "legacy-protected", "target_votes": 1})
            for asset_id in (pred_a_id, pred_b_id):
                add_candidate(
                    db,
                    workspace,
                    int(legacy["id"]),
                    {
                        "reference_asset_id": reference_id,
                        "asset_id": asset_id,
                        "video_name": "clip",
                    },
                )
            self.assertEqual(
                publish_campaign(db, workspace, int(legacy["id"]))["status"],
                "published",
            )

            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            request = service.request_delete(run_a)
            completed = service.process_request(int(request["id"]))
            self.assertEqual(completed["status"], "completed")
            protected_reference = db.get(
                "SELECT source_kind, storage_path, state FROM media_assets WHERE id = ?",
                (reference_id,),
            )
            protected_pred = db.get(
                "SELECT source_kind, storage_path, state FROM media_assets WHERE id = ?",
                (pred_a_id,),
            )
            for asset in (protected_reference, protected_pred):
                self.assertEqual(asset["source_kind"], "evaluation_package")
                self.assertEqual(asset["state"], "ready")
                self.assertTrue(Path(str(asset["storage_path"])).exists())

    def test_failed_preparation_removes_staging_and_leaves_no_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            source_b = db.get(
                """
                SELECT ma.storage_path
                FROM evaluation_bindings_v2 b
                JOIN evaluation_methods_v2 m ON m.id = b.method_id
                JOIN media_assets ma ON ma.id = b.source_asset_id
                WHERE m.campaign_id = ? AND m.slot = 'b'
                """,
                (int(campaign["id"]),),
            )
            Path(str(source_b["storage_path"])).unlink()
            with self.assertRaises(FileNotFoundError):
                publish_campaign_v2(db, workspace, campaign["id"])
            failed = get_campaign_v2(db, campaign["id"])
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["tasks"], 0)
            self.assertFalse((workspace.root / "evaluations" / str(campaign["id"])).exists())
            self.assertFalse(any((workspace.root / "evaluations" / ".staging").iterdir()))

    def test_encoder_failure_cleans_partial_freeze_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _item, body, _paths = self._item_campaign(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)

            def fail_during_encode(_plan, _sources, output_dir, **kwargs):
                output_dir = Path(output_dir)
                (output_dir / "partial-encoder-output.mp4").write_bytes(b"partial")
                kwargs["progress_callback"](
                    {
                        "stage": "materializing",
                        "frame_current": 1,
                        "frame_total": 3,
                        "pipeline": "streaming",
                        "force": True,
                    }
                )
                raise RuntimeError("injected Campaign encoder failure")

            with patch(
                "vfieval.evaluations_v2.freeze_campaign_media",
                side_effect=fail_during_encode,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected Campaign encoder failure"):
                    publish_campaign_v2(db, workspace, int(campaign["id"]))

            failed = get_campaign_v2(db, int(campaign["id"]))
            preparation = get_preparation_v2(db, int(campaign["id"]))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(preparation["state"], "failed")
            self.assertEqual(preparation["report"]["stage"], "materializing")
            self.assertEqual(preparation["report"]["frame_current"], 1)
            self.assertIn("encoder failure", preparation["error"]["message"])
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM evaluation_tasks_v2")["count"]),
                0,
            )
            self.assertEqual(
                int(
                    db.get(
                        "SELECT COUNT(*) AS count FROM media_assets "
                        "WHERE source_kind = 'evaluation_package'"
                    )["count"]
                ),
                0,
            )
            self.assertFalse((workspace.evaluations_dir / str(campaign["id"])).exists())
            self.assertFalse(any((workspace.evaluations_dir / ".staging").iterdir()))

    def test_claim_loss_during_freeze_cleans_staging_without_clearing_new_owner(self) -> None:
        import threading
        from contextlib import nullcontext

        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _item, body, _paths = self._item_campaign(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            ownership_lost = threading.Event()

            def supersede_during_encode(_plan, _sources, output_dir, **kwargs):
                output_dir = Path(output_dir)
                (output_dir / "partial-old-owner.mp4").write_bytes(b"partial")
                with db.connection() as conn:
                    conn.execute(
                        """
                        UPDATE evaluation_preparations_v2
                        SET state = 'running', claim_token = 'new-owner', updated_at = ?
                        WHERE campaign_id = ?
                        """,
                        (10**12, int(campaign["id"])),
                    )
                ownership_lost.set()
                kwargs["cancel_check"]()
                self.fail("a superseded Campaign freeze must stop immediately")

            with patch(
                "vfieval.evaluations_v2._preparation_claim_heartbeat",
                return_value=nullcontext(ownership_lost),
            ), patch(
                "vfieval.evaluations_v2.freeze_campaign_media",
                side_effect=supersede_during_encode,
            ):
                with self.assertRaisesRegex(EvaluationConflict, "superseded"):
                    publish_campaign_v2(db, workspace, int(campaign["id"]))

            preparation = get_preparation_v2(db, int(campaign["id"]))
            self.assertEqual(get_campaign_v2(db, int(campaign["id"]))["status"], "preparing")
            self.assertEqual(preparation["state"], "running")
            self.assertEqual(preparation["claim_token"], "new-owner")
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM evaluation_tasks_v2")["count"]),
                0,
            )
            self.assertEqual(
                int(
                    db.get(
                        "SELECT COUNT(*) AS count FROM media_assets "
                        "WHERE source_kind = 'evaluation_package'"
                    )["count"]
                ),
                0,
            )
            self.assertFalse((workspace.evaluations_dir / str(campaign["id"])).exists())
            self.assertFalse(any((workspace.evaluations_dir / ".staging").iterdir()))

    def test_assignment_leases_limit_concurrent_votes_to_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            body["target_votes"] = 2
            campaign = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            token = campaign["public_token"]

            def join(index: int):
                return blind_session(
                    db,
                    token,
                    {"evaluator_id": f"browser-{index}", "display_name": f"User {index}"},
                )

            with ThreadPoolExecutor(max_workers=5) as pool:
                sessions = list(pool.map(join, range(5)))
            assigned = [
                (index, payload["task"])
                for index, payload in enumerate(sessions)
                if payload["task"] is not None
            ]
            self.assertEqual(len(assigned), 2)
            self.assertTrue(all(not payload["progress"]["complete"] for payload in sessions))

            def vote(pair):
                index, task = pair
                return blind_submit_vote(
                    db, token, task["token"], f"browser-{index}", {"choice": "tie"}
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(vote, assigned))
            count = db.get("SELECT COUNT(*) AS count FROM evaluation_votes_v2")
            self.assertEqual(int(count["count"]), 2)


if __name__ == "__main__":
    unittest.main()
