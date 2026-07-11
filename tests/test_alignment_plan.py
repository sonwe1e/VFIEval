from __future__ import annotations

import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from vfieval.alignment import (
    alignment_cache_key,
    materialize_aligned_frame,
    materialize_frame_sets,
    plan_alignment,
    validate_temporal_alignment,
)
from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.compare_inputs import resolve_compare_descriptor
from vfieval.media_assets import create_collection, upsert_asset
from vfieval.media_items import ensure_canonical_gt_item, register_external_prediction
from vfieval.pipeline.inference import _write_video_artifacts
from vfieval.datasets import scan_dataset


def _source(width: int, height: int, *, frames: int = 3, **extra: object) -> dict[str, object]:
    return {"width": width, "height": height, "frame_count": frames, "fps": 24.0, **extra}


class AlignmentPlanTests(unittest.TestCase):
    def test_single_prediction_uses_its_native_size_and_reports_transform(self) -> None:
        reference = _source(3840, 2160, slot="gt", asset_id=11)
        prediction = _source(1920, 1080, slot="pred_a", member_id=21, member_role="model_pred")

        plan = plan_alignment(reference, [prediction])

        self.assertEqual(plan["target"], {"width": 1920, "height": 1080, "source_slot": "pred_a"})
        self.assertEqual(plan["filter"], "lanczos")
        self.assertEqual(plan["sources"]["gt"]["direction"], "downscale")
        self.assertEqual(plan["sources"]["gt"]["scale_x"], 0.5)
        self.assertEqual(plan["sources"]["gt"]["scale_y"], 0.5)
        self.assertFalse(plan["sources"]["gt"]["aspect_changed"])
        self.assertEqual(plan["sources"]["pred_a"]["direction"], "none")
        self.assertEqual(plan, plan_alignment(reference, [prediction]))
        self.assertEqual(len(plan["fingerprint"]), 64)

    def test_two_prediction_tie_uses_max_edge_then_width_then_height(self) -> None:
        reference = _source(16, 12)
        pred_a = _source(8, 6, slot="pred_a")
        pred_b = _source(6, 8, slot="pred_b")

        plan = plan_alignment(reference, [pred_a, pred_b])

        self.assertEqual(plan["target"]["source_slot"], "pred_b")
        self.assertEqual((plan["target"]["width"], plan["target"]["height"]), (6, 8))
        self.assertEqual(plan["sources"]["pred_a"]["direction"], "mixed")
        self.assertTrue(plan["sources"]["pred_a"]["aspect_changed"])

    def test_external_aspect_change_requires_explicit_confirmation(self) -> None:
        reference = _source(16, 12)
        pred_a = _source(8, 6, slot="pred_a", member_role="model_pred")
        external = _source(16, 4, slot="pred_b", member_role="external_pred")

        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            plan_alignment(reference, [pred_a, external])

        confirmed = plan_alignment(
            reference,
            [pred_a, external],
            spatial_policy={"allow_external_aspect_stretch": True},
        )
        self.assertTrue(confirmed["sources"]["pred_b"]["aspect_changed"])
        self.assertTrue(confirmed["sources"]["pred_b"]["aspect_stretch_authorized"])

    def test_temporal_mapping_fps_and_timestamps_remain_strict(self) -> None:
        reference = _source(8, 8, frames=5, timestamps=[0.0, 0.1, 0.2, 0.3, 0.4])
        pred_a = _source(
            8,
            8,
            frames=3,
            source_frame_indices=[1, 2, 3],
            timestamps=[0.1, 0.2, 0.3],
        )
        pred_b = dict(pred_a)
        summary = validate_temporal_alignment(reference, [pred_a, pred_b])
        self.assertEqual(summary["mode"], "indexed")
        self.assertEqual(summary["frame_count"], 3)
        self.assertTrue(summary["timestamps_verified"])

        with self.assertRaisesRegex(ValueError, "same ordered"):
            validate_temporal_alignment(reference, [pred_a, {**pred_b, "source_frame_indices": [0, 1, 2]}])
        with self.assertRaisesRegex(ValueError, "fps"):
            validate_temporal_alignment(reference, [{**pred_a, "fps": 25.0}])
        with self.assertRaisesRegex(ValueError, "frame timestamps"):
            validate_temporal_alignment(reference, [{**pred_a, "timestamps": [0.1, 0.25, 0.3]}])

    def test_materializer_is_cached_and_rebuilds_after_gc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            gt_path = Path(tmp) / "gt.png"
            pred_path = Path(tmp) / "pred.png"
            Image.new("RGB", (8, 6), (255, 0, 0)).save(gt_path)
            Image.new("RGB", (4, 3), (0, 255, 0)).save(pred_path)
            plan = plan_alignment(
                _source(8, 6, frames=1, slot="gt"),
                [_source(4, 3, frames=1, slot="pred_a")],
            )

            output = materialize_aligned_frame(db, workspace, plan, "gt", gt_path)
            self.assertEqual(output.parent, workspace.root / "compare_cache")
            with Image.open(output) as image:
                self.assertEqual(image.size, (4, 3))
            key = alignment_cache_key(plan, "gt", gt_path)
            entry = db.get_cache_entry("compare_cache", key)
            self.assertIsNotNone(entry)
            self.assertEqual(entry["state"], "ready")

            output.unlink()
            rebuilt = materialize_aligned_frame(db, workspace, plan, "gt", gt_path)
            self.assertEqual(rebuilt, output)
            self.assertTrue(rebuilt.is_file())
            sets = materialize_frame_sets(
                db,
                workspace,
                plan,
                {"gt": [gt_path], "pred_a": [pred_path]},
            )
            self.assertEqual(sets["pred_a"], [pred_path.resolve()])

    def test_compare_video_writer_can_omit_reusable_pred_video(self) -> None:
        class RecordingDatabase:
            def __init__(self) -> None:
                self.kinds: list[str] = []

            def add_artifact(self, _job_id, _sample_id, kind, *_args, **_kwargs) -> None:
                self.kinds.append(str(kind))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frame_dir = Path(tmp) / "frames"
            frame_dir.mkdir()
            gt = frame_dir / "gt.png"
            pred = frame_dir / "pred.png"
            diff = frame_dir / "diff.png"
            for path, color in ((gt, (1, 2, 3)), (pred, (3, 2, 1)), (diff, (2, 0, 2))):
                Image.new("RGB", (4, 4), color).save(path)
            groups = {
                "clip": {
                    "video_name": "clip",
                    "fps": 24.0,
                    "frames": [
                        {
                            "order": 0,
                            "sample_name": "clip__A__000000",
                            "gt_path": gt,
                            "pred_path": pred,
                            "diff_path": diff,
                            "track_label": "A",
                            "track_key": "A",
                            "track_run_id": 7,
                            "track_artifact_id": 9,
                        }
                    ],
                }
            }
            db = RecordingDatabase()

            def fake_mp4(_frames: list[Path], path: Path, _fps: float) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"mp4")

            with patch("vfieval.pipeline.inference._write_mp4", side_effect=fake_mp4):
                _write_video_artifacts(db, 1, run_dir, groups, publish_pred_video=False)  # type: ignore[arg-type]

            self.assertNotIn("pred_video", db.kinds)
            self.assertIn("gt_video", db.kinds)
            self.assertIn("diff_video", db.kinds)
            manifest = json.loads((run_dir / "videos" / "clip" / "manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(manifest["tracks"][0]["pred_video"])
            self.assertFalse((run_dir / "videos" / "clip" / "A" / "pred.mp4").exists())

    def test_compare_dataset_materializes_the_shared_alignment_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            gt_dir = Path(tmp) / "gt"
            pred_dir = Path(tmp) / "pred"
            gt_dir.mkdir()
            pred_dir.mkdir()
            Image.new("RGB", (8, 6), (1, 2, 3)).save(gt_dir / "000000.png")
            Image.new("RGB", (4, 3), (3, 2, 1)).save(pred_dir / "000000.png")
            plan = plan_alignment(
                _source(8, 6, frames=1, slot="gt"),
                [_source(4, 3, frames=1, slot="pred_a")],
            )
            dataset_id = db.create_dataset(
                "aligned-compare",
                str(gt_dir),
                has_gt=True,
                source_type="compare",
                decode_mode="compare",
                metadata={
                    "reference_path": str(gt_dir),
                    "distorted_path": str(pred_dir),
                    "align_mode": "strict",
                    "alignment_plan": plan,
                },
            )

            self.assertEqual(scan_dataset(db, workspace, dataset_id), 1)
            sample = db.list_samples(dataset_id)[0]
            with Image.open(sample["gt_path"]) as image:
                self.assertEqual(image.size, (4, 3))
            with Image.open(sample["img1_path"]) as image:
                self.assertEqual(image.size, (4, 3))
            self.assertEqual(sample["metadata"]["alignment_fingerprint"], plan["fingerprint"])
            self.assertEqual(db.get_dataset(dataset_id)["metadata"]["alignment_plan"], plan)

    def test_media_item_descriptors_resolve_managed_members_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            collection = create_collection(db, "Upload")
            gt_dir = workspace.media_dir / "gt"
            pred_dir = workspace.media_dir / "pred"
            gt_dir.mkdir()
            pred_dir.mkdir()
            Image.new("RGB", (8, 6), (1, 2, 3)).save(gt_dir / "000000.png")
            Image.new("RGB", (4, 3), (3, 2, 1)).save(pred_dir / "000000.png")
            gt_asset = upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key="upload:gt",
                source_kind="upload",
                media_kind="frame_sequence",
                role="gt",
                display_name="GT",
                original_name="gt",
                storage_path=gt_dir,
                frame_count=1,
                width=8,
                height=6,
                fps=24.0,
            )
            pred_asset = upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key="upload:pred",
                source_kind="upload",
                media_kind="frame_sequence",
                role="pred",
                display_name="Pred",
                original_name="pred",
                storage_path=pred_dir,
                frame_count=1,
                width=4,
                height=3,
                fps=24.0,
            )
            item = ensure_canonical_gt_item(db, int(gt_asset["id"]))
            member = register_external_prediction(
                db,
                int(item["id"]),
                int(pred_asset["id"]),
                method_key="external:test",
                temporal_mapping={
                    "source_frame_indices": [0],
                    "fps": 24.0,
                    "timestamps": [0.0],
                },
                spatial_origin={"width": 4, "height": 3},
                aspect_stretch_confirmed=True,
            )

            reference = resolve_compare_descriptor(
                workspace,
                db,
                {"kind": "media_item", "item_id": item["id"]},
                role="reference",
            )
            distorted = resolve_compare_descriptor(
                workspace,
                db,
                {"kind": "media_item_member", "member_id": member["id"]},
                role="distorted",
            )

            self.assertEqual(reference["item_id"], item["id"])
            self.assertEqual(reference["member_role"], "canonical_gt")
            self.assertEqual(reference["fps"], 24.0)
            self.assertEqual(distorted["item_id"], item["id"])
            self.assertEqual(distorted["member_id"], member["id"])
            self.assertEqual(distorted["member_role"], "external_pred")
            self.assertEqual(distorted["producer_kind"], "external")
            self.assertEqual(distorted["source_frame_indices"], [0])
            self.assertEqual(distorted["temporal_timestamps"], [0.0])
            self.assertEqual(distorted["spatial_origin"], {"width": 4, "height": 3})
            self.assertTrue(distorted["allow_aspect_stretch"])


if __name__ == "__main__":
    unittest.main()
