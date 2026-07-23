from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "vfieval" / "web"


class FrontendSharedPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shared = (WEB / "shared.js").read_text(encoding="utf-8")
        cls.app = (WEB / "app.js").read_text(encoding="utf-8")
        cls.compare = (WEB / "compare.js").read_text(encoding="utf-8")
        cls.run_detail = (WEB / "run-detail.js").read_text(encoding="utf-8")
        cls.media = (WEB / "media.js").read_text(encoding="utf-8")
        cls.studio = (WEB / "studio.js").read_text(encoding="utf-8")
        cls.blind = (WEB / "blind.js").read_text(encoding="utf-8")
        cls.index = (WEB / "index.html").read_text(encoding="utf-8")
        cls.blind_html = (WEB / "blind.html").read_text(encoding="utf-8")

    def test_shared_bundle_loads_before_each_consumer(self) -> None:
        self.assertLess(
            self.index.index('<script src="/shared.js"></script>'),
            self.index.index('<script src="/app.js"></script>'),
        )
        ordered_scripts = (
            "/shared.js",
            "/app.js",
            "/compare.js",
            "/run-detail.js",
            "/media.js",
            "/studio.js",
        )
        offsets = [self.index.index(f'<script src="{script}"></script>') for script in ordered_scripts]
        self.assertEqual(offsets, sorted(offsets))
        self.assertLess(
            self.blind_html.index('<script src="/shared.js"'),
            self.blind_html.index('<script src="/blind.js"'),
        )

    def test_request_primitive_normalizes_errors_and_reports_recovery(self) -> None:
        for fragment in (
            "async function request(path, options)",
            "async function readResponse(response)",
            "function createError(options)",
            "function recoverySuggestion(errorLike)",
            "function reportDiagnostic(settings, error)",
            "error.code = code",
            "error.request_id = ids.request_id",
            "error.support_id = ids.support_id",
            "error.details = details",
            "error.recovery_suggestion",
            "settings.requireJsonSuccess",
            "settings.messageFormatter",
        ):
            self.assertIn(fragment, self.shared)

    def test_all_api_wrappers_delegate_to_the_shared_request_lifecycle(self) -> None:
        self.assertIn("return Shared.request(path", self.app)
        self.assertIn("return Shared.request(path", self.studio)
        self.assertIn("return Shared.request(path", self.blind)
        self.assertIn("_formatStorageCapacityError(data)", self.app)
        self.assertIn("timeoutMs: 15_000", self.blind)
        self.assertIn("requireJsonSuccess: true", self.blind)

    def test_create_flows_share_one_single_flight_implementation(self) -> None:
        self.assertIn("function createSingleFlight()", self.shared)
        self.assertIn(
            "const runCreationFlight = Shared.createSingleFlight();",
            self.app,
        )
        self.assertIn(
            "const compareCreationFlight = Shared.createSingleFlight();",
            self.app,
        )
        self.assertIn(
            "const campaignCreationFlight = Shared.createSingleFlight();",
            self.studio,
        )
        for source, flight in (
            (self.app, "runCreationFlight"),
            (self.compare, "compareCreationFlight"),
            (self.studio, "campaignCreationFlight"),
        ):
            self.assertIn(f"{flight}.tryLock()", source)
            self.assertIn(f"{flight}.release()", source)

    def test_storage_copy_and_submission_ids_are_centralized(self) -> None:
        for source in (
            self.app,
            self.compare,
            self.run_detail,
            self.media,
            self.studio,
            self.blind,
        ):
            self.assertNotIn("localStorage.", source)
            self.assertNotIn("navigator.clipboard", source)
            self.assertNotIn("crypto.randomUUID", source)
        for helper in (
            "function copyText(value)",
            "function storageGet(key, fallbackValue)",
            "function storageSet(key, value)",
            "function storageJsonGet(key, fallbackValue)",
            "function storageJsonSet(key, value)",
            "function createSubmissionId(fallbackPrefix)",
        ):
            self.assertIn(helper, self.shared)

    def test_domain_scripts_keep_one_state_and_defer_bootstrap_until_loaded(self) -> None:
        ownership = (
            (self.compare, "async function startCompareRun(event)"),
            (self.run_detail, "function renderRunDetail()"),
            (self.run_detail, "function renderMetricChart("),
            (self.media, "async function uploadExternalMedia(event)"),
            (self.media, "function renderMediaLibrary()"),
        )
        for source, marker in ownership:
            self.assertIn(marker, source)
            self.assertNotIn(marker, self.app)
        for source in (self.compare, self.run_detail, self.media):
            self.assertIsNone(re.search(r"^const state\s*=", source, flags=re.MULTILINE))
            self.assertIsNone(re.search(r"^async function api\(", source, flags=re.MULTILINE))
            self.assertIn("classic-script global", source)
        self.assertIn(
            'document.addEventListener("DOMContentLoaded", bootstrapApp, { once: true })',
            self.app,
        )


if __name__ == "__main__":
    unittest.main()
