from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "src" / "vfieval" / "web"


class LegacyCampaignUiReadOnlyTests(unittest.TestCase):
    def test_main_application_has_no_hidden_campaign_v1_write_controls(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for control_id in (
            "evaluator-form",
            "campaign-form",
            "candidate-form",
            "refresh-evaluations",
            "evaluation-task",
            "campaign-analysis",
        ):
            with self.subTest(control_id=control_id):
                self.assertNotIn(f'id="{control_id}"', index)

        self.assertNotIn("/api/evaluation-tasks/", app)
        self.assertNotIn("createEvaluationCampaign", app)
        self.assertNotIn("addEvaluationCandidate", app)
        self.assertNotIn("submitEvaluationVote", app)

    def test_studio_keeps_legacy_read_export_and_archive_only(self) -> None:
        studio = (WEB_ROOT / "studio.js").read_text(encoding="utf-8")

        self.assertIn("历史 Campaign", studio)
        self.assertIn("/export", studio)
        self.assertIn("data-studio-legacy-archive", studio)
        self.assertNotIn("data-studio-legacy-discard", studio)
        self.assertNotIn("legacyCampaignDiscard", studio)
        self.assertNotIn("/discard", studio)


if __name__ == "__main__":
    unittest.main()
