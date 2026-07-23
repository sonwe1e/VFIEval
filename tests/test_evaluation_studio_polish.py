from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "vfieval" / "web"


class EvaluationStudioPolishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.studio_js = (WEB / "studio.js").read_text(encoding="utf-8")
        cls.studio_css = (WEB / "studio.css").read_text(encoding="utf-8")

    def function_source(self, name: str) -> str:
        match = re.search(
            rf"^  (?:async\s+)?function\s+{re.escape(name)}\s*\(",
            self.studio_js,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(match, f"studio.js is missing function {name}")
        assert match is not None
        following = re.search(
            r"^  (?:async\s+)?function\s+[A-Za-z_$][\w$]*\s*\(",
            self.studio_js[match.end():],
            flags=re.MULTILINE,
        )
        end = match.end() + following.start() if following is not None else len(self.studio_js)
        return self.studio_js[match.start():end]

    def test_item_picker_exposes_page_filtered_clear_and_selected_actions(self) -> None:
        renderer = self.function_source("renderItems")
        for hook in (
            "data-studio-select-page",
            "data-studio-select-filtered",
            "data-studio-clear-selection",
            "data-studio-only-selected",
            "data-studio-cancel-bulk",
        ):
            self.assertIn(hook, renderer)
        self.assertIn('aria-pressed="${studioState.itemOnlySelected', renderer)
        self.assertIn("selectedCachedItems()", self.function_source("itemView"))
        self.assertIn(".studio-item-toolbar", self.studio_css)

    def test_single_item_selection_updates_existing_cards_without_full_render(self) -> None:
        updater = self.function_source("updateItemSelectionUi")
        self.assertIn('querySelectorAll("[data-studio-item]")', updater)
        self.assertIn('classList.toggle("selected", selected)', updater)
        self.assertIn("count.textContent", updater)
        self.assertNotIn("innerHTML", updater)

        start = self.studio_js.index('const item = event.target.closest?.("[data-studio-item]")')
        end = self.studio_js.index(
            'if (event.target.matches?.("#studio-method-a-source',
            start,
        )
        checkbox_handler = self.studio_js[start:end]
        self.assertIn("applyItemSelectionChange()", checkbox_handler)
        self.assertNotIn("renderItems()", checkbox_handler)

    def test_filtered_select_all_uses_a_cancellable_server_snapshot(self) -> None:
        selector = self.function_source("selectAllFilteredItems")
        self.assertIn("new AbortController()", selector)
        self.assertIn("controller.signal", selector)
        self.assertIn("currentItemFilterSignature()", selector)
        self.assertIn('request("/api/media/item-selections"', selector)
        self.assertIn("studioState.itemSelectionToken = String(payload.selection_token", selector)
        self.assertIn("studioState.itemSelectionTokenTotal", selector)
        self.assertNotIn("for (let page =", selector)
        self.assertLess(
            selector.index("studioState.selectedItemIds.clear()"),
            selector.index("await applyItemSelectionChange()"),
        )
        cancel = self.function_source("cancelBulkItemSelection")
        self.assertIn("itemBulkController.abort()", cancel)
        self.assertIn("原有选择保持不变", cancel)

    def test_campaign_draft_is_versioned_and_reconciles_stale_items_and_methods(self) -> None:
        self.assertIn('const CAMPAIGN_DRAFT_KEY = "vfieval:campaign-draft:v1"', self.studio_js)
        self.assertIn("const CAMPAIGN_DRAFT_VERSION = 1", self.studio_js)
        self.assertIn("Shared.storageJsonSet(CAMPAIGN_DRAFT_KEY", self.function_source("saveCampaignDraft"))
        self.assertIn("Shared.storageRemove(CAMPAIGN_DRAFT_KEY)", self.function_source("clearCampaignDraft"))

        reconcile = self.function_source("reconcileDraftItems")
        self.assertIn('fetchItemPage(page, { pageSize: BULK_ITEM_PAGE_SIZE, query: "" })', reconcile)
        self.assertIn("studioState.selectedItemIds = foundIds", reconcile)
        self.assertIn("已删除或失效的 Item", reconcile)
        controls = self.function_source("applyCampaignDraftControls")
        self.assertIn("requestedKey", controls)
        self.assertIn("已不可用", controls)

        methods = self.function_source("loadMethodsForSelection")
        self.assertIn("METHOD_ITEM_CHUNK_SIZE", methods)
        self.assertIn("missing_item_ids: missing", methods)
        self.assertIn("complete: !missing.length", methods)

    def test_campaign_creation_reuses_submission_id_only_for_the_same_payload(self) -> None:
        identity = self.function_source("campaignSubmissionIdFor")
        self.assertIn("JSON.stringify(payload)", identity)
        self.assertIn("campaignSubmissionFingerprint !== fingerprint", identity)
        self.assertIn('Shared.createSubmissionId("studio")', identity)

        create = self.function_source("createCampaign")
        self.assertIn("const creationPayload = {", create)
        self.assertIn("campaignSubmissionIdFor(creationPayload)", create)
        self.assertIn("submission_id: submissionId", create)
        self.assertIn("resetCampaignSubmissionIdentity()", create)
        self.assertLess(
            create.index("campaignSubmissionIdFor(creationPayload)"),
            create.index("await request"),
        )
        self.assertLess(
            create.index("await request"),
            create.index("resetCampaignSubmissionIdentity()"),
        )

    def test_preparation_poll_surfaces_freshness_and_exponential_backoff(self) -> None:
        markup = self.function_source("preparationPollStatusMarkup")
        self.assertIn("上次成功刷新", markup)
        self.assertIn("状态可能已过期", markup)
        self.assertIn("preparationPollFailures", markup)
        poll = self.function_source("startPreparationPoll")
        self.assertIn("suppressDiagnostic: true", poll)
        self.assertIn("PREPARATION_POLL_BASE_MS * (2 **", poll)
        self.assertIn("PREPARATION_POLL_MAX_MS", poll)
        self.assertIn("studioState.preparationLastSuccessAt = Date.now()", poll)
        self.assertIn(".studio-refresh-state.stale", self.studio_css)


if __name__ == "__main__":
    unittest.main()
