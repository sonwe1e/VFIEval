from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from vfieval.input_identity import (
    INPUT_IDENTITY_SCHEMA,
    InputIdentityChanged,
    assert_input_identity_files_available,
    assert_input_identity_matches,
    build_checkpoint_identity,
    build_file_identity,
    build_run_input_identity,
    build_source_identity,
    compare_input_identities,
    resolved_checkpoint_relative_path,
)


class RunInputIdentityTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        models = root / "models"
        checkpoints = root / "checkpoints"
        videos = root / "videos"
        (checkpoints / "rife").mkdir(parents=True, exist_ok=True)
        (videos / "set-a").mkdir(parents=True, exist_ok=True)
        models.mkdir(parents=True, exist_ok=True)
        model = models / "rife.py"
        checkpoint = checkpoints / "rife" / "latest.pth"
        source = videos / "set-a" / "clip.mp4"
        if not model.exists():
            model.write_bytes(b"model-v1")
        if not checkpoint.exists():
            checkpoint.write_bytes(b"checkpoint-v1")
        if not source.exists():
            source.write_bytes(b"video-v1")
        return model, checkpoint, source

    def _identity(
        self,
        root: Path,
        *,
        checkpoint_request: str = "auto",
        request: dict | None = None,
    ) -> dict:
        model, checkpoint, source = self._fixture(root)
        model_identity = build_file_identity(
            model,
            trusted_root=root / "models",
            display_path="models/rife.py",
        )
        checkpoint_file = build_file_identity(
            checkpoint,
            trusted_root=root / "checkpoints",
            display_path="checkpoints/rife/latest.pth",
        )
        source_file = build_file_identity(
            source,
            trusted_root=root / "videos",
            display_path="videos/set-a/clip.mp4",
        )
        source_identity = build_source_identity(
            item_id=7,
            asset_id=11,
            qualified_name="set-a/clip.mp4",
            file_identity=source_file,
        )
        return build_run_input_identity(
            model=model_identity,
            checkpoint=build_checkpoint_identity(
                checkpoint_request,
                resolved=checkpoint_file,
            ),
            sources=[source_identity],
            request=request
            or {
                "model_file": "rife.py",
                "checkpoint": checkpoint_request,
                "video_groups": ["set-a"],
                "selected_videos": ["clip.mp4"],
                "batch_size": 4,
                "precision": "fp32",
                "metrics": ["lpips_vit_patch", "vmaf"],
            },
        )

    def test_fingerprint_is_stable_and_contains_no_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._identity(root)
            second = self._identity(
                root,
                request={
                    "metrics": ["lpips_vit_patch", "vmaf"],
                    "precision": "fp32",
                    "batch_size": 4,
                    "selected_videos": ["clip.mp4"],
                    "video_groups": ["set-a"],
                    "checkpoint": "auto",
                    "model_file": "rife.py",
                },
            )

            self.assertEqual(first["schema"], INPUT_IDENTITY_SCHEMA)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(len(first["fingerprint"]), 64)
            serialized = json.dumps(first, ensure_ascii=False)
            self.assertNotIn(root.as_posix(), serialized.replace("\\", "/"))

    def test_content_and_trusted_relative_path_changes_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self._identity(root)
            source = root / "videos" / "set-a" / "clip.mp4"
            source.write_bytes(b"video-v2-with-different-content")
            changed_content = self._identity(root)
            content_fields = {
                row["field"]
                for row in compare_input_identities(expected, changed_content)["differences"]
            }
            self.assertIn("sources[0].content.sha256", content_fields)
            self.assertNotEqual(expected["fingerprint"], changed_content["fingerprint"])

            renamed = root / "videos" / "set-a" / "renamed.mp4"
            source.rename(renamed)
            renamed_file = build_file_identity(
                renamed,
                trusted_root=root / "videos",
                display_path="videos/set-a/renamed.mp4",
            )
            renamed_identity = build_run_input_identity(
                model=changed_content["model"],
                checkpoint=changed_content["checkpoint"],
                sources=[
                    build_source_identity(
                        item_id=7,
                        asset_id=11,
                        qualified_name="set-a/renamed.mp4",
                        file_identity=renamed_file,
                    )
                ],
                request=changed_content["request"],
            )
            path_fields = {
                row["field"]
                for row in compare_input_identities(changed_content, renamed_identity)["differences"]
            }
            self.assertIn("sources[0].qualified_name", path_fields)
            self.assertIn("sources[0].content.relative_path", path_fields)

    def test_auto_checkpoint_keeps_request_and_exact_resolved_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            automatic = self._identity(root, checkpoint_request="auto")
            explicit = self._identity(root, checkpoint_request="rife/latest.pth")

            self.assertEqual(automatic["checkpoint"]["requested"], "auto")
            self.assertEqual(
                automatic["checkpoint"]["resolved"]["relative_path"],
                "rife/latest.pth",
            )
            self.assertEqual(
                resolved_checkpoint_relative_path(automatic),
                "rife/latest.pth",
            )
            self.assertEqual(
                automatic["checkpoint"]["resolved"],
                explicit["checkpoint"]["resolved"],
            )
            self.assertNotEqual(automatic["fingerprint"], explicit["fingerprint"])

    def test_requested_decode_backend_is_part_of_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_request = {
                "model_file": "rife.py",
                "checkpoint": "auto",
                "video_groups": ["set-a"],
                "selected_videos": ["clip.mp4"],
                "batch_size": 1,
                "precision": "fp32",
                "metrics": [],
                "evaluation_contract": "midpoint-triplet-v2",
            }
            ffmpeg = self._identity(root, request={**base_request, "decode_backend": "ffmpeg"})
            opencv = self._identity(root, request={**base_request, "decode_backend": "opencv"})
            self.assertNotEqual(ffmpeg["fingerprint"], opencv["fingerprint"])
            differences = {
                row["field"]
                for row in compare_input_identities(ffmpeg, opencv)["differences"]
            }
            self.assertEqual(differences, {"request.decode_backend"})

    def test_structured_differences_are_public_and_raise_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self._identity(root)
            actual = copy.deepcopy(expected)
            actual["request"]["precision"] = "fp16"
            actual["request"]["internal_note"] = str((root / "secret" / "file.txt").resolve())
            actual["request"][str((root / "secret-key").resolve())] = "private"

            comparison = compare_input_identities(expected, actual)
            self.assertFalse(comparison["matches"])
            rows = {row["field"]: row for row in comparison["differences"]}
            self.assertEqual(rows["request.precision"]["kind"], "changed")
            self.assertEqual(rows["request.internal_note"]["kind"], "added")
            self.assertEqual(rows["request.internal_note"]["actual"], "<redacted-path>")
            self.assertEqual(rows["request.<redacted-key>"]["actual"], "private")

            with self.assertRaises(InputIdentityChanged) as caught:
                assert_input_identity_matches(expected, actual)
            payload = caught.exception.public_payload()
            self.assertEqual(payload["type"], "InputIdentityChanged")
            self.assertEqual(payload["differences"], comparison["differences"])
            self.assertNotIn(str(root), json.dumps(payload))

    def test_missing_files_are_reported_as_public_identity_differences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self._identity(root)
            model, checkpoint, source = self._fixture(root)
            model.unlink()
            checkpoint.unlink()
            source.unlink()

            with self.assertRaises(InputIdentityChanged) as caught:
                assert_input_identity_files_available(
                    expected,
                    models_root=root / "models",
                    checkpoints_root=root / "checkpoints",
                    videos_root=root / "videos",
                )

            payload = caught.exception.public_payload()
            rows = {row["field"]: row for row in payload["differences"]}
            self.assertEqual(rows["model"]["kind"], "missing")
            self.assertEqual(rows["checkpoint.resolved"]["kind"], "missing")
            self.assertEqual(rows["sources[0].content"]["kind"], "missing")
            self.assertNotIn(str(root), json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
