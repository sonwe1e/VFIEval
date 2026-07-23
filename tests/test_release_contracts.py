from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from vfieval.diagnostics import release_info
from vfieval.release import package_release_metadata


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "contracts" / "public_surface.json").read_text(encoding="utf-8")
)


class ReleaseIdentityTests(unittest.TestCase):
    def test_source_checkout_has_explicit_release_identity(self) -> None:
        metadata = package_release_metadata()
        self.assertEqual(metadata["version"], "0.1.0")
        self.assertIn("build_id", metadata)
        self.assertIn("commit_sha", metadata)
        self.assertIn("source_date_epoch", metadata)

    def test_runtime_build_id_override_remains_supported(self) -> None:
        with patch.dict(os.environ, {"VFIEVAL_BUILD_ID": "runtime-override"}):
            payload = release_info()
        self.assertEqual(payload["build_id"], "runtime-override")
        self.assertIn("commit_sha", payload)


class PublicSurfaceContractTests(unittest.TestCase):
    def test_console_entrypoint_and_cli_commands_match_contract(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(CONTRACT["cli_entrypoint"], pyproject)
        cli_source = (ROOT / "src" / "vfieval" / "cli.py").read_text(encoding="utf-8")
        commands = sorted(set(re.findall(r'sub\.add_parser\("([^"]+)"', cli_source)))
        self.assertEqual(commands, CONTRACT["cli_commands"])

    def test_critical_routes_exist_and_are_navigable(self) -> None:
        server_source = (ROOT / "src" / "vfieval" / "server.py").read_text(encoding="utf-8")
        navigation = (ROOT / "NAVIGATION.md").read_text(encoding="utf-8")
        for route in CONTRACT["route_anchors"]:
            with self.subTest(route=route["name"]):
                self.assertIn(route["source"], server_source)
                self.assertIn(route["navigation"], navigation)

    def test_web_entrypoints_reference_packaged_local_assets(self) -> None:
        web_root = ROOT / "src" / "vfieval" / "web"
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('vfieval = ["_build_info.json", "web/*"]', pyproject)
        for relative in [*CONTRACT["web_entrypoints"], *CONTRACT["required_web_assets"]]:
            with self.subTest(asset=relative):
                self.assertTrue((web_root / relative).is_file())

        for entrypoint in CONTRACT["web_entrypoints"]:
            html = (web_root / entrypoint).read_text(encoding="utf-8")
            references = re.findall(r'(?:src|href)="\/([^"?#]+)', html)
            for reference in references:
                with self.subTest(entrypoint=entrypoint, reference=reference):
                    self.assertTrue((web_root / reference).is_file())

    def test_navigation_owns_release_and_public_surface_changes(self) -> None:
        navigation = (ROOT / "NAVIGATION.md").read_text(encoding="utf-8")
        self.assertIn("### 17. Release packaging + public-surface contracts", navigation)
        self.assertIn("tools/build_release.py", navigation)
        self.assertIn("tests/test_release_contracts.py", navigation)


if __name__ == "__main__":
    unittest.main()
