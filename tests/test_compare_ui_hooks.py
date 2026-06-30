from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CompareUiHookTests(unittest.TestCase):
    def test_compare_layers_and_master_video_controls_are_wired(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(encoding="utf-8")
        styles = (ROOT / "src" / "vfieval" / "web" / "styles.css").read_text(encoding="utf-8")

        self.assertIn("compare_layers", app_js)
        self.assertIn("extra_layers", app_js)
        self.assertIn("data-compare-layer-kind", app_js)
        self.assertIn("data-compare-grid-columns", app_js)
        self.assertIn("data-master-video-play", app_js)
        self.assertIn("syncActiveVideos", app_js)
        self.assertIn("compare-layer-grid", styles)
        self.assertIn("--compare-grid-columns", styles)
        self.assertIn("grid-auto-flow: column", styles)


if __name__ == "__main__":
    unittest.main()
