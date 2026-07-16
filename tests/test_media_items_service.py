from __future__ import annotations

import tempfile
import unittest
import json
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.media_assets import bind_run_asset, create_collection, ensure_collection, upsert_asset
from vfieval.media_items import (
    bind_run_source,
    ensure_canonical_gt_item,
    list_media_item_groups,
    list_media_items,
    register_model_prediction,
    resolve_media_item_compare,
    sync_media_items,
)
from vfieval.pipeline.inference import run_inference_job

from v13_test_utils import (
    add_completed_pred_run,
    get_json,
    make_workspace,
    post_json,
    start_server,
    stop_server,
    write_mp4,
)


def _post_error(base_url: str, path: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as response:
        return int(response.code), json.loads(response.read().decode("utf-8"))
    raise AssertionError("request unexpectedly succeeded")


class MediaItemServiceTests(unittest.TestCase):
    def _canonical_gt(self, db, workspace, collection_name: str, source_key: str, file_name: str):
        collection = ensure_collection(
            db,
            f"videos/{collection_name}",
            f"videos-{collection_name}",
            {"source_kind": "folder", "video_group": collection_name},
        )
        path = workspace.root.parent / "videos" / collection_name / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_mp4(path, [(0, 0, 0), (20, 0, 0), (40, 0, 0)], size=(8, 8), fps=5)
        asset = upsert_asset(
            db,
            collection_id=collection["id"],
            source_key=source_key,
            source_kind="folder",
            media_kind="video",
            role="gt",
            display_name=file_name,
            original_name=file_name,
            storage_path=path,
            frame_count=3,
            width=8,
            height=8,
            fps=5,
        )
        return collection, asset, ensure_canonical_gt_item(db, asset["id"])

    def _completed_model_run(self, db, workspace, item_id: int, pred_asset: dict, *, tag: str = "default") -> int:
        model_id = db.upsert_model(f"item-service-model-{tag}", "dummy", None, 8, 8, {})
        dataset_root = workspace.root / f"item-service-dataset-{tag}"
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_id = db.create_dataset(f"item-service-dataset-{tag}", str(dataset_root), True)
        run_id = db.create_run(
            f"item-service-run-{tag}",
            model_id,
            dataset_id,
            8,
            8,
            1,
            "cpu",
            "fp32",
            [],
            metadata={"run_type": "model_inference"},
        )
        job_id = int(db.get_run(run_id)["inference_job_id"])
        if not db.mark_run_started(run_id, "running"):
            raise RuntimeError("test Run rejected inference start")
        claimed = db.claim_next_job(f"item-service-{tag}", ["inference"])
        if claimed is None or int(claimed["id"]) != job_id:
            raise RuntimeError("test inference Job could not be claimed")
        if not db.complete_run_inference(run_id, {}, {}, "completed"):
            raise RuntimeError("test Run rejected inference completion")
        if not db.complete_job(job_id, {}):
            raise RuntimeError("test inference Job rejected completion")
        bind_run_source(db, run_id, item_id, video_name="clip.mp4")
        bind_run_asset(db, run_id, pred_asset["id"], "pred", video_name="clip.mp4")
        return run_id

    def test_sync_keeps_same_named_gt_assets_in_separate_collections_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            first_collection, first_asset, first_item = self._canonical_gt(
                db, workspace, "set-a", "folder:set-a/clip.mp4", "clip.mp4"
            )
            second_collection, second_asset, second_item = self._canonical_gt(
                db, workspace, "set-b", "folder:set-b/clip.mp4", "clip.mp4"
            )

            report = sync_media_items(db)
            self.assertEqual(report["total"], 2)
            self.assertNotEqual(first_item["id"], second_item["id"])
            self.assertNotEqual(first_asset["id"], second_asset["id"])

            groups = list_media_item_groups(db)["groups"]
            self.assertEqual({row["id"] for row in groups}, {first_collection["id"], second_collection["id"]})
            first_page = list_media_items(db, first_collection["id"])
            self.assertEqual([row["id"] for row in first_page["items"]], [first_item["id"]])

    def test_compare_resolution_requires_reusable_model_member_of_the_same_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection, _gt, item = self._canonical_gt(
                db, workspace, "source", "folder:source/clip.mp4", "clip.mp4"
            )
            pred_path = workspace.runs_dir / "1" / "videos" / "clip" / "pred.mp4"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_bytes(b"pred")
            pred_asset = upsert_asset(
                db,
                collection_id=create_collection(
                    db, "Run outputs", metadata={"source_kind": "run_artifact"}
                )["id"],
                source_key="run_artifact:item-service-pred",
                source_kind="run_artifact",
                media_kind="video",
                role="pred",
                display_name="clip pred",
                original_name="pred.mp4",
                storage_path=pred_path,
                frame_count=3,
                width=8,
                height=8,
                fps=5,
            )
            run_id = self._completed_model_run(db, workspace, item["id"], pred_asset)
            member = register_model_prediction(
                db,
                run_id,
                item["id"],
                pred_asset["id"],
                temporal_mapping={"source_frame_indices": [0, 1, 2]},
                spatial_origin={"width": 8, "height": 8},
            )

            resolved = resolve_media_item_compare(db, workspace, item["id"], [member["id"]])
            self.assertEqual(resolved["reference"]["asset_id"], item["canonical_gt_asset_id"])
            self.assertEqual(resolved["members"][0]["member_id"], member["id"])
            self.assertEqual(resolved["members"][0]["path"], str(pred_path.resolve()))
            self.assertEqual(
                resolved["members"][0]["temporal_mapping"]["source_frame_indices"], [0, 1, 2]
            )

            _other_collection, _other_gt, other_item = self._canonical_gt(
                db, workspace, "other", "folder:other/clip.mp4", "clip.mp4"
            )
            with self.assertRaisesRegex(ValueError, "same media item"):
                resolve_media_item_compare(db, workspace, other_item["id"], [member["id"]])

    def _item_with_model_prediction(self, db, workspace, *, group: str, pred_size: tuple[int, int]):
        collection, _gt, item = self._canonical_gt(
            db, workspace, group, f"folder:{group}/clip.mp4", "clip.mp4"
        )
        run_collection = ensure_collection(
            db,
            "Run outputs",
            "run-outputs-item-service",
            {"source_kind": "run_artifact"},
        )
        pred_path = workspace.runs_dir / "model-pred" / group / "pred.mp4"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        write_mp4(pred_path, [(0, 1, 0), (20, 1, 0), (40, 1, 0)], size=pred_size, fps=5)
        pred_asset = upsert_asset(
            db,
            collection_id=run_collection["id"],
            source_key=f"run_artifact:{group}:pred",
            source_kind="run_artifact",
            media_kind="video",
            role="pred",
            display_name=f"{group} pred",
            original_name="pred.mp4",
            storage_path=pred_path,
            frame_count=3,
            width=pred_size[0],
            height=pred_size[1],
            fps=5,
        )
        # This helper is intentionally used once per temporary database; its
        # producer Run is a real completed model-inference Run, never a
        # Compare-derived shortcut.
        run_id = self._completed_model_run(db, workspace, item["id"], pred_asset, tag=group)
        member = register_model_prediction(
            db,
            run_id,
            item["id"],
            pred_asset["id"],
            method_key="model-a",
            temporal_mapping={"source_frame_indices": [0, 1, 2]},
            spatial_origin={"width": pred_size[0], "height": pred_size[1]},
        )
        return collection, item, member, run_id, pred_path

    def test_item_picker_http_routes_only_expose_bound_reusable_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection, item, member, _run_id, _pred_path = self._item_with_model_prediction(
                db, workspace, group="picker", pred_size=(8, 8)
            )
            server, thread, base_url = start_server(db, workspace)
            try:
                groups = get_json(base_url, "/api/media/item-groups?role=gt")["groups"]
                self.assertIn(collection["id"], {int(row["id"]) for row in groups})

                page = get_json(base_url, f"/api/media/items?group_id={collection['id']}&q=clip&page=1&page_size=10")
                self.assertEqual([int(row["id"]) for row in page["items"]], [int(item["id"])])

                predictions = get_json(base_url, f"/api/media/items/{item['id']}/predictions")["predictions"]
                self.assertEqual([int(row["member_id"]) for row in predictions], [int(member["id"])])
                self.assertTrue(predictions[0]["reusable_as_pred"])
                self.assertEqual(predictions[0]["producer_kind"], "model_inference")
            finally:
                stop_server(server, thread)

    def test_legacy_compare_descriptors_must_adapt_to_same_item_members(self) -> None:
        """Compatibility descriptors never reopen the old asset-only path."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            gt_a = Path(tmp) / "videos" / "anime" / "clip.mp4"
            gt_b = Path(tmp) / "videos" / "other" / "clip.mp4"
            gt_a.parent.mkdir(parents=True, exist_ok=True)
            gt_b.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_a, [(0, 0, 0), (20, 0, 0), (40, 0, 0)], size=(8, 8), fps=5)
            write_mp4(gt_b, [(0, 0, 0), (0, 20, 0), (0, 40, 0)], size=(8, 8), fps=5)
            pred_a = workspace.root / "legacy-a.mp4"
            pred_b = workspace.root / "legacy-b.mp4"
            unbound_pred = workspace.root / "legacy-unbound.mp4"
            write_mp4(pred_a, [(0, 1, 0), (20, 1, 0), (40, 1, 0)], size=(8, 8), fps=5)
            write_mp4(pred_b, [(0, 0, 1), (20, 0, 1), (40, 0, 1)], size=(8, 8), fps=5)
            write_mp4(unbound_pred, [(1, 0, 0), (1, 20, 0), (1, 40, 0)], size=(8, 8), fps=5)
            bound_run = add_completed_pred_run(
                db, workspace, "legacy-bound", pred_a, source_video_path=gt_a
            )
            other_run = add_completed_pred_run(
                db, workspace, "legacy-other", pred_b, source_video_path=gt_b
            )
            unbound_run = add_completed_pred_run(db, workspace, "legacy-unbound", unbound_pred)
            bound_member = db.get(
                """
                SELECT id, item_id, asset_id FROM media_item_members
                WHERE producer_run_id = ? AND member_role = 'model_pred'
                """,
                (bound_run,),
            )
            self.assertIsNotNone(bound_member)
            assert bound_member is not None

            legacy_reference = {"kind": "video_group", "group": "anime", "video": "clip.mp4"}
            server, thread, base_url = start_server(db, workspace)
            try:
                valid = post_json(
                    base_url,
                    "/api/preflight",
                    {
                        "run_type": "video_compare",
                        "reference": legacy_reference,
                        "distorted": [
                            {"kind": "run_artifact", "run_id": bound_run, "video": "clip"}
                        ],
                        "metrics": [],
                    },
                )
                self.assertTrue(valid["ok"], valid)
                self.assertEqual(valid["reference"]["item_id"], int(bound_member["item_id"]))
                self.assertEqual(valid["distorted_tracks"][0]["member_id"], int(bound_member["id"]))

                # A physical media_asset is still a compatibility descriptor,
                # but only when it resolves to the same reusable Item member.
                via_asset = post_json(
                    base_url,
                    "/api/preflight",
                    {
                        "run_type": "video_compare",
                        "reference": legacy_reference,
                        "distorted": [
                            {"kind": "media_asset", "asset_id": int(bound_member["asset_id"])}
                        ],
                        "metrics": [],
                    },
                )
                self.assertTrue(via_asset["ok"], via_asset)
                self.assertEqual(via_asset["distorted_tracks"][0]["member_id"], int(bound_member["id"]))

                cross_status, cross_error = _post_error(
                    base_url,
                    "/api/preflight",
                    {
                        "run_type": "video_compare",
                        "reference": legacy_reference,
                        "distorted": [
                            {"kind": "run_artifact", "run_id": other_run, "video": "clip"}
                        ],
                    },
                )
                self.assertEqual(cross_status, 400)
                self.assertIn("same media item", cross_error["error"]["message"])

                foreign_artifact = db.list_run_artifacts(other_run, kind="pred_video")[0]
                foreign_status, foreign_error = _post_error(
                    base_url,
                    "/api/preflight",
                    {
                        "run_type": "video_compare",
                        "reference": legacy_reference,
                        "distorted": [
                            {
                                "kind": "run_artifact",
                                "run_id": bound_run,
                                "artifact_id": int(foreign_artifact["id"]),
                            }
                        ],
                    },
                )
                self.assertEqual(foreign_status, 400)
                self.assertIn("does not belong", foreign_error["error"]["message"])

                unbound_status, unbound_error = _post_error(
                    base_url,
                    "/api/preflight",
                    {
                        "run_type": "video_compare",
                        "reference": legacy_reference,
                        "distorted": [
                            {"kind": "run_artifact", "run_id": unbound_run, "video": "clip"}
                        ],
                    },
                )
                self.assertEqual(unbound_status, 400)
                self.assertIn("no trustworthy media item binding", unbound_error["error"]["message"])

                path_status, path_error = _post_error(
                    base_url,
                    "/api/preflight",
                    {
                        "run_type": "video_compare",
                        "reference": {**legacy_reference, "path": "C:/client-supplied.mp4"},
                        "distorted": [
                            {"kind": "run_artifact", "run_id": bound_run, "video": "clip"}
                        ],
                    },
                )
                self.assertEqual(path_status, 400)
                self.assertIn("client-supplied paths", path_error["error"]["message"])

                model_id = db.upsert_model("legacy-compare", "dummy", None, 8, 8, {})
                dataset_id = db.create_dataset(
                    "legacy-compare-dataset", str(workspace.root / "legacy-compare-dataset"), True
                )
                compare_run = db.create_run(
                    "legacy-derived",
                    model_id,
                    dataset_id,
                    8,
                    8,
                    1,
                    "cpu",
                    "fp32",
                    [],
                    metadata={"run_type": "video_compare"},
                )
                derived_status, derived_error = _post_error(
                    base_url,
                    "/api/preflight",
                    {
                        "run_type": "video_compare",
                        "reference": legacy_reference,
                        "distorted": [
                            {"kind": "run_artifact", "run_id": compare_run, "video": "clip"}
                        ],
                    },
                )
                self.assertEqual(derived_status, 400)
                self.assertIn("video_compare Run", derived_error["error"]["message"])
            finally:
                stop_server(server, thread)

    def test_item_compare_normalizes_same_aspect_media_and_never_publishes_pred_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection, item, member, _source_run_id, _pred_path = self._item_with_model_prediction(
                db, workspace, group="compare", pred_size=(16, 16)
            )
            _other_collection, _other_item, _other_member, _other_run_id, _other_path = self._item_with_model_prediction(
                db, workspace, group="other", pred_size=(8, 8)
            )
            server, thread, base_url = start_server(db, workspace)
            try:
                compare_payload = {
                    "run_type": "video_compare",
                    "media_item_id": item["id"],
                    "pred_member_ids": [member["id"]],
                    "spatial_policy": {
                        "mode": "smallest_pred",
                        "filter": "lanczos",
                        "allow_known_aspect_stretch": True,
                    },
                    "metrics": [],
                }
                with patch("vfieval.server._start_local_inference_worker", return_value=None):
                    created = post_json(base_url, "/api/runs", compare_payload)
                compare_run_id = int(created["run_id"])
                compare_run = db.get_run(compare_run_id)
                dataset = db.get_dataset(int(compare_run["dataset_id"]))
                plan = (compare_run.get("metadata") or {}).get("alignment_plan") or (
                    dataset.get("metadata") or {}
                ).get("alignment_plan")
                self.assertIsInstance(plan, dict)
                self.assertEqual(plan["target"], {"width": 16, "height": 16, "source_slot": "pred_a"})
                self.assertEqual(plan["filter"], "lanczos")
                self.assertTrue(plan["fingerprint"])

                compare_job_id = int(compare_run["inference_job_id"])
                self.assertEqual(
                    int(db.claim_next_job("item-compare", ["inference"])["id"]),
                    compare_job_id,
                )
                compare_result = run_inference_job(db, workspace, compare_job_id)
                self.assertTrue(db.complete_job(compare_job_id, compare_result.__dict__))
                self.assertEqual(db.list_run_artifacts(compare_run_id, kind="pred_video"), [])
                self.assertTrue(db.list_run_artifacts(compare_run_id, kind="diff_video"))
                inputs = get_json(base_url, f"/api/runs/{compare_run_id}/compare-inputs")
                self.assertEqual([row["slot"] for row in inputs["inputs"]], ["gt", "pred_a"])
                self.assertEqual(inputs["alignment_plan"]["fingerprint"], plan["fingerprint"])
                with urllib.request.urlopen(
                    f"{base_url}/api/runs/{compare_run_id}/compare-inputs/pred_a/media?variant=aligned",
                    timeout=30,
                ) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers.get_content_type(), "video/mp4")
                    self.assertTrue(response.read(64))

                # The completed Compare has no reusable Pred member and is not
                # returned by the original Item's picker list.
                predictions = get_json(base_url, f"/api/media/items/{item['id']}/predictions")["predictions"]
                self.assertEqual([int(row["member_id"]) for row in predictions], [int(member["id"])])

                snapshot_path = workspace.runs_dir / str(compare_run_id) / "inputs" / "snapshot.mp4"
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                write_mp4(snapshot_path, [(0, 2, 0), (20, 2, 0), (40, 2, 0)], size=(16, 16), fps=5)
                snapshot_asset = upsert_asset(
                    db,
                    collection_id=ensure_collection(
                        db,
                        "Compare snapshots",
                        "compare-snapshots-item-service",
                        {"source_kind": "run_artifact"},
                    )["id"],
                    source_key=f"run_artifact:compare-snapshot:{compare_run_id}",
                    source_kind="run_artifact",
                    media_kind="video",
                    role="pred",
                    display_name="compare snapshot",
                    original_name="snapshot.mp4",
                    storage_path=snapshot_path,
                    frame_count=3,
                    width=16,
                    height=16,
                    fps=5,
                )
                with db.connection() as conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO media_item_members(
                            item_id, asset_id, member_role, producer_kind, producer_run_id,
                            method_key, reusable_as_pred, temporal_mapping_json,
                            spatial_origin_json, state, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, 'compare_snapshot', 'video_compare', ?, '', 0, '{}', '{}', 'ready', '{}', 1, 1)
                        """,
                        (int(item["id"]), int(snapshot_asset["id"]), compare_run_id),
                    )
                    snapshot_member_id = int(cursor.lastrowid)
                snapshot_status, _snapshot_error = _post_error(
                    base_url,
                    "/api/runs",
                    {**compare_payload, "pred_member_ids": [snapshot_member_id]},
                )
                self.assertEqual(snapshot_status, 400)

                status, error = _post_error(
                    base_url,
                    "/api/runs",
                    {**compare_payload, "media_item_id": _other_item["id"]},
                )
                self.assertEqual(status, 400)
                self.assertIn("same media item", error["error"]["message"])
            finally:
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
