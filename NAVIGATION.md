# NAVIGATION.md — Where To Change What

Purpose: locate the files, functions, DB methods, routes, and tests for a change
**without reading the whole tree**. Organized by *subsystem*, not by version.

How to use:
1. Find the subsystem row that matches your task.
2. Touch the listed files/functions. Line numbers are approximate (drift as code
   changes) — use the function name with your editor's symbol search, not the line.
3. Run the listed tests before declaring done.

Maintenance rule: when you add or move a feature, update its subsystem block here in
the same change. Do **not** add per-version sections — fold new work into the
subsystem it belongs to. Keep `CLAUDE.md` for architecture prose; keep this file for
lookup.

Search hygiene: `*.backup.*` files (13 in tree, ~92 in `archive/`) are gitignored but
still hit by filesystem Glob/Grep. Prefer `git grep` — it ignores them. To scope a
raw grep, add `--glob '!*.backup.*'`.

---

## Layer cheat-sheet (which file owns what)

| Layer | File | Size | Role |
|-------|------|------|------|
| HTTP routing + handlers | `src/vfieval/server.py` | ~3350 | GET/POST/DELETE dispatch, run/compare/evaluation orchestration |
| Data layer | `src/vfieval/db.py` | ~3100 | SQLite schema, migrations, all query methods |
| Main frontend | `src/vfieval/web/app.js` | ~4400 | inference, Compare, Runs, scoped result refresh; `index.html` + `styles.css` |
| Evaluation frontend | `src/vfieval/web/studio.js`, `blind.js` | ~800 | Studio/Media storage panels and isolated participant page |
| File entry points | `src/vfieval/file_inputs.py` | ~1400 | scan models/checkpoints/videos, preflight, thumbnails |
| Dataset scan | `src/vfieval/datasets.py` | ~1100 | video → triplet samples, `VideoEntry`, multi-group |
| Inference | `src/vfieval/pipeline/inference.py` | ~1400 | `run_inference_job`, prefetch/save pools |
| Post-processing | `src/vfieval/pipeline/postprocess.py` | ~240 | warp/blend/pred, `compose_interpolated` |
| Compare resolution | `src/vfieval/compare_inputs.py` | ~340 | descriptor → path, strict alignment |
| Metric health/assets | `src/vfieval/metrics/health.py` | ~1460 | manifests, downloads, availability |
| Metric execution | `src/vfieval/pipeline/metrics_runner.py`, `src/vfieval/pipeline/metric_jobs.py` | batched LPIPS scoring plus multi-device metric-wave queue/aggregation |
| Media catalog | `src/vfieval/media_assets.py` | ~860 | collections/assets, backfill, resolver, provenance relations |
| Uploads | `src/vfieval/uploads.py` | ~390 | resumable parts, hashes, ZIP validation, quotas/cleanup |
| Run purge + cache GC | `src/vfieval/run_cleanup.py` | ~775 | persistent purge requests, cache refs/leases, preview/confirmed GC |
| Blind evaluation V2 | `src/vfieval/evaluations_v2.py` | ~2500 | two-method Campaigns, frozen packages, opaque participant flow, analysis |
| Legacy blind evaluation | `src/vfieval/evaluations.py` | ~850 | schema v1 read/export compatibility |
| Execution profiles | `src/vfieval/performance.py` | ~150 | benchmark fingerprint and recommendation |
| Multi-device finalize | `src/vfieval/pipeline/finalize_runner.py` | ~150 | merge shard manifests, encode videos, queue metrics |

Route dispatch is a flat `if path == ...` chain in `server.py`: `do_GET` (L72),
`do_POST` (L218), `do_DELETE` (L413). To add a route, edit the relevant handler.

---

## Subsystems

### 1. Run Feedback + Statistics
User rating (1–5, 0.25 step) + free-text issue per run; content-scoped (video/track/model/checkpoint); stats view.
- **Data:** `db.py` `add_run_feedback` (L1254), `update_run_feedback` (L1293), `list_run_feedback` (L1333), `delete_run_feedback` (L1342), `list_all_feedback` (L1350), `feedback_stats` (L1380), `feedback_filter_options` (L1506)
- **API:** `server.py` `_parse_feedback_rating` (L1512), `_feedback_context` (L1532), `_create_run_feedback` (L1580), `_update_run_feedback` (L1612), `_feedback_overview` (L1655)
- **Routes:** `POST/GET /api/runs/{id}/feedback`, `DELETE /api/runs/{id}/feedback/{fid}` (do_DELETE L413), `GET /api/feedback`
- **Frontend:** `app.js` `renderRunFeedback` (L1772), `renderFeedbackItem` (L1825), `submitRunFeedback` (L2654), `deleteRunFeedback` (L2702), `loadStats` (L2711), `renderStats` (L2795), `ratingStars`/`formatRating`/`ratingOptions` (L1718–1743)
- **Table:** `run_feedback`; index `idx_run_feedback_run(run_id, created_at)`
- **Tests:** `tests/test_run_feedback.py`

### 2. Compare — multi-track (GT vs N preds)
- **Resolution:** `compare_inputs.py` `resolve_compare_descriptor` (L83), `_resolve_video_group_descriptor` (L113), `_resolve_run_artifact_descriptor` (L131), `validate_strict_alignment` (L219), `validate_strict_decoded_alignment` (L279)
- **API:** `server.py` `_create_video_compare_run` (L1029), `_dedupe_track_labels` (L1004), compare layer helpers `_compare_layer_payloads` (L2118), `_compare_track_rows` (L2157)
- **Routes:** `POST /api/runs` with `run_type:"video_compare"`
- **Frontend:** `app.js` `comparePayloadFromForm` (L705), `startCompareRun` (L1244), `renderCompareSelection` (L1077), `compareTrackLabel` (L499), `compareCompatibility` (L509)
- **Naming:** samples `{video_token}__{track_token}__{frame}`; dedup on sanitized token
- **Tests:** `tests/test_compare_multitrack.py`

### 3. Compare — source pickers (GT/Pred/flow/mask)
- **API:** `server.py` `_compare_sources` (L614), `_compare_gt_sources` (L629), `_compare_pred_sources` (L678), `_compare_layer_sources` (L761), `_source_pagination` (L736), `_source_page_payload` (L744)
- **Routes:** `GET /api/compare-sources/{gt,pred,flow,mask}` (do_GET L72; rejects client `path`)
- **Backing:** `file_inputs.list_video_groups` (L94), `db.list_runs` + `db.list_run_artifacts` (L1648)
- **Frontend:** `app.js` `loadCompareSources` (L561), `renderCompareGtCards` (L1012), `renderComparePredCards` (L1037), `compareSourcePager` (L550)
- **Tests:** `tests/test_compare_sources_api.py`, `tests/test_compare_ui_hooks.py`

### 4. Standard inference run creation
- **API:** `server.py` `_create_run_from_files` (L843), `_default_run_name` (L1254), `_reference_config` (L1191), `_resolve_execution_devices` (L1289), `_create_inference_shards` (L1305), `_partition_samples_by_video` (L1349)
- **Routes:** `POST /api/runs` (default), `POST /api/preflight`
- **Preflight/scan:** `file_inputs.py` `preflight_run` (L346), `resolve_video_selection` (L232), `_dry_run_model_file` (L926)
- **Dataset:** `datasets.py` `_resolve_video_entries` (L659), `VideoEntry` (L71)
- **Frontend:** `app.js` `payloadFromForm` (L673), `startRun` (L1277), `runPreflight` (L766), `renderPreflight` (L955)
- **Tests:** `tests/test_v2_runs.py`, `tests/test_v3_file_flow.py`, `tests/test_video_datasets.py`

### 5. Multi-group inference
- **Backing:** `datasets._resolve_video_entries` (single `video_group` vs `video_groups` list), `file_inputs.resolve_video_selection` (L232)
- **Frontend:** `app.js` `selectedGroupNames` (L149), `isMultiGroup` (L160), `renderGroupPicker` (L318), `renderGroupVideoTable` (L577), `selectedVideoNames` (L442)
- **Invariant:** single-group runs stay byte-identical to legacy (dataset root = group folder, bare clip names). Only multi-group roots at `videos/`.
- **Tests:** `tests/test_video_datasets.py`

### 6. Inference pipeline (worker hot path)
- **Entry:** `pipeline/inference.py` `run_inference_job` (L234), `_iter_prefetched_batches` (L1122)
- **Post-proc:** `pipeline/postprocess.py` `validate_model_outputs` (L10), `normalize_model_outputs` (L53), `backward_warp` (L104), `compose_interpolated` (L142)
- **Model load diag:** `models/utils.py` `load_state_dict_portable`; report → `run_dir/logs/model_load.log`
- **Concurrency knobs:** `payload["prefetch_workers"]` (default 2), `payload["save_workers"]` (default min(8, cpu))
- **Worker loop:** `worker.py`, `orchestration.py`
- **Tests:** `tests/test_end_to_end.py`, `tests/test_postprocess.py`

### 7. Model / checkpoint / video catalog (file entry points)
- **Scan:** `file_inputs.py` `list_model_files` (L45), `list_checkpoints` (L66), `list_video_groups` (L94)
- **Routes:** `GET /api/model-files`, `/api/checkpoints`, `/api/video-groups`, `/api/devices`
- **Frontend:** `app.js` `renderOptions` (L278), `renderCheckpointOptions` (L333), `renderDeviceOptions` (L374)
- **Model adapter loading:** `models/loader.py` (`file:` / `module:` / `dummy`)

### 8. Metrics (health, assets, execution)
- **Names/registry:** `metrics/names.py`, `metrics/registry.py`
- **Health/assets:** `metrics/health.py` (`metrics_health`, downloads via `prepare-metrics`)
- **Execution:** `pipeline/metrics_runner.py` (batched scoring/cache), `pipeline/metric_jobs.py` (multi-device wave creation/progress/aggregation); per-metric `metrics/feature.py` (lpips_*), `metrics/vmaf.py`, `metrics/cgvqm.py`
- **Routes:** `GET /api/metrics/health`
- **Frontend:** `app.js` `renderMetricOptions` (L261), `renderMetricHealthTable` (L870), `renderMetricEnvironmentPanel` (L914)
- **CLI:** `cli.py` `prepare-metrics [--check-only|--force]`
- **Tests:** `tests/test_metrics.py`

### 9. Run detail / timeline / sample APIs (scoped reads — perf-critical)
- **API:** `server.py` `_run_detail` (L1505), `_run_videos` (L1797), `_run_video_timeline` (L1839), `_run_sample_payload` (L2270), `_run_metric_summary` (L2332). Legacy full-load: `_run_timeline` (L1696, deprecated).
- **Routes:** `GET /api/runs/{id}` `/videos` `/videos/{name}/timeline` `/samples/{id}` `/metric-summary`; `/timeline` returns `X-Deprecated`
- **Backing (scoped):** `db.list_samples_by_video` (L615), `list_artifacts_by_sample` (L1937), `list_metrics_by_sample` (L2022)
- **Indices:** `idx_artifacts_sample(sample_id, kind)`, `idx_metric_results_sample(sample_id, metric_name)`, `idx_run_jobs_device(device)`
- **Rule:** these handlers must NOT call `_run_timeline` or iterate the whole run.
- **Freshness:** `runs.content_revision`; `db.bump_run_content_revision`. Artifact publication, metric completion, and artifact cleanup increment it; list/detail payloads expose it.
- **Frontend:** `app.js` `runContentRevisionChanged`, `invalidateRunResultCache`, `refreshRunsOnce`, `refreshRunResults`, `selectRun`, `loadRunVideoTimeline`, `loadSampleDetail`; requests carry generation/abort guards and preserve the current selection on refresh.
- **Tests:** `tests/test_sample_api_scope.py`, `tests/test_db_indices.py`, `tests/test_run_result_freshness_ui.py`

### 10. Run lifecycle (retry / cancel / delete / cleanup)
- **Service:** `run_cleanup.py` `RunCleanupService.request_delete`, `request_artifact_cleanup`, `process_pending`, `_purge_run`, `gc_preview`, `garbage_collect`; `register_run_cache_refs`, `cache_lease`
- **Data:** `db.py` purge request CRUD/claim/recovery methods, `mark_run_deleted_after_purge`, cache entry/ref/lease helpers, `bump_run_content_revision`
- **Routes:** `DELETE /api/runs/{id}` and legacy `/hide` return `202`; `POST /api/runs/{id}/cleanup-artifacts`, `/api/runs/batch-delete`; `GET /api/run-purge-requests/{id}`, `/api/storage/gc/preview`; `POST /api/storage/gc`
- **Loop:** `server.py` starts `RunCleanupService.run_forever`; embedded handlers pump pending requests so restart/test servers converge.
- **Frontend:** `app.js` `runPurgeState`, `renderRunPurgeNotice`, `deleteRun`, `batchDeleteRuns`, `cleanupRunArtifacts`; `studio.js` `previewStorageGc`, `executeStorageGc`
- **Invariants:** only `.vfieval/runs/{id}` is deleted; `deleted_at` is written after successful purge; shared cache needs zero active refs, zero leases, and expired grace; storage GC requires preview + `confirm=true`.
- **Tests:** `tests/test_run_cleanup.py`, `tests/test_v3_file_flow.py`

### 11. Artifact / file streaming
- **API:** `server.py` `_send_artifact` (L477), `_send_sample_file` (L491), `_send_file` (L504), `_parse_range_header` (L554)
- **Routes:** `GET /api/files/{artifact_id}?variant=`, `/api/samples/{id}/{slot}`, `/api/thumbnails/...`
- **Note:** honors HTTP byte ranges, 4 MiB chunks for video playback

### 12. Decode job
- **API:** `server.py` `_decode_progress_total` (L1467), `_selection_hash` (L1458)
- **Runner:** `pipeline/decode_runner.py`
- **Frontend:** `app.js` `renderDecodePanel` (L1579), `decodeBackendStatus` (L238)
- **Tests:** `tests/test_decode_progress.py`

### 13. Unified media catalog + resolver
- **Data/model:** `media_assets.py` `create_collection`, `upsert_asset`, `sync_folder_assets`, `sync_run_assets`, `resolve_asset_path`, `source_assets_to_video_payload`, `soft_delete_asset`, `media_audit`
- **Tables:** `media_collections`, `media_assets`, `media_asset_relations`, `run_media_assets`, `metric_asset_bindings`, `schema_migrations`; frozen Campaign media uses `source_kind='evaluation_package'`
- **Routes:** `GET/POST /api/media/collections`, `GET /api/media/assets`, `GET /api/media/sources`, `GET /api/media/run-outputs`, `GET/DELETE /api/media/assets/{id}`, `GET /api/media/assets/{id}/content`, `GET /api/media/audit`
- **Frontend:** `app.js` owns Sources/Uploads; `studio.js` `loadRunOutputs`, `renderDerivedRuns`, `renderPackages` owns Derived Runs and Evaluation Packages; `#view-media`
- **Visibility:** internal Run/evaluation collections are not user collections; deleted/cleaned Run outputs stay unavailable and must not be resurrected by catalog sync.
- **Compare:** `compare_inputs.resolve_compare_descriptor(kind="media_asset")`; primary picker submits asset ids
- **Tests:** `tests/test_media_catalog_uploads.py`, `tests/test_compare_multitrack.py`

### 14. External resumable uploads
- **Runner:** `uploads.py` `create_upload_session`, `receive_upload_part`, `complete_upload_session`, `delete_upload_session`, `cleanup_stale_uploads`, `_extract_frame_zip`
- **Tables:** `upload_sessions`, `upload_parts`
- **Routes:** `POST /api/uploads`, `GET/DELETE /api/uploads/{id}`, `PUT /api/uploads/{id}/parts/{index}`, `POST /api/uploads/{id}/complete`
- **Frontend:** `app.js` `uploadExternalMedia`, `Sha256Hasher`, `sha256File`; `#upload-form`
- **Storage:** `.vfieval/tmp/uploads/{session}` while incomplete; `.vfieval/media/{collection}/{asset_uuid}` after validation
- **Tests:** `tests/test_media_catalog_uploads.py`

### 15. Artifact profiles, benchmark, segments + finalize
- **Inference:** `pipeline/inference.py` `ARTIFACT_PROFILES`, `run_inference_job`, `_detach_tensors_to_cpu`, `_AsyncSavePipeline`, `_DeviceEventTimings`, `_NpuSmiSampler`
- **Sharding:** `orchestration.py` `partition_samples_by_video`, `_create_inference_shards`; continuous segments when videos are insufficient/skewed
- **Integrity:** `pipeline/artifact_integrity.py` validates required per-sample artifacts, shard manifest coverage, file existence, and canonical video counts before publication
- **Finalize:** `pipeline/finalize_runner.py` `run_finalize_job`; only finalize encodes shared videos and queues metrics, after strict pre/post integrity checks
- **Profiles:** `performance.py` `execution_profile_identity`, `record_execution_profile`, `recommend_execution_profile`; table `execution_profiles`
- **CLI:** `cli.py benchmark` (warmup/samples/repeats)
- **Frontend:** artifact profile and pool overrides in infer form; `renderPerformanceReport`, `renderExecutionProfileRecommendation`
- **Tests:** `tests/test_artifact_profiles.py`, `tests/test_artifact_integrity.py`

### 16. Blind evaluation V2 + frozen Campaign packages
- **Core:** `evaluations_v2.py` `list_run_outputs`, `preview_campaign_v2`, `create_campaign_v2`, `request_publish_campaign_v2`, `run_pending_preparations`, `publish_campaign_v2`, blind session/payload/media/heartbeat/vote functions, `campaign_analysis_v2`, `campaign_export_v2`; `pipeline/evaluation_freeze.py` owns bounded rawvideo/remux package materialization
- **Lifecycle:** draft → preparing → published → closed/archived; failed preparation can be retried. Publish deep-validates every selected GT/A/B item, builds staging, freezes under `.vfieval/evaluations/{campaign_id}`, writes a SHA-256 manifest, registers evaluation-package assets, then creates tasks atomically.
- **Tables:** `evaluation_campaigns_v2`, `evaluation_methods_v2`, `evaluation_items_v2`, `evaluation_bindings_v2`, `evaluation_preparations_v2`, `evaluation_tasks_v2`, `evaluation_assignments_v2`, `evaluation_votes_v2`, `evaluation_analysis_cache_v2`; shared identity table `evaluators`
- **Admin routes:** `GET /api/evaluation-campaigns`, `POST /api/evaluation-campaigns/v2/preview`, `POST /api/evaluation-campaigns/v2`, `GET /api/evaluation-campaigns/v2/{id}[/{analysis,export}]`, `POST /api/evaluation-campaigns/v2/{id}/{publish,close,archive}`
- **Participant routes:** `/evaluate/{opaque_token}`; `/api/blind/{token}`, `/session`, task-token `/media/{reference,left,right}`, `/vote`, `/heartbeat`
- **Frontend:** `studio.js` + `studio.css` for method picker, common-video matrix, preparation status, packages and organizer analysis; `blind.html` + `blind.js` + `blind.css` is isolated from the main navigation.
- **Privacy/concurrency:** participant payloads use opaque campaign/task/assignment/media URLs, stable side randomization, renewable assignment leases, and transactionally capped target votes. Results unlock only after the evaluator finishes all eligible tasks.
- **Analysis:** pairwise ties are half-wins, Bradley–Terry/bootstrap is deterministic, and human/objective results remain separate.
- **Legacy:** `evaluations.py` schema v1 stays read-only/exportable/archivable through `legacy_campaigns_readonly`; do not infer a V2 migration from labels.
- **Tests:** `tests/test_evaluation_campaigns_v2.py`, `tests/test_evaluation_studio_ui.py`, `tests/test_evaluation_campaigns.py`

---

## DB schema quick map
Tables: `models`, `datasets`, `samples`, `jobs`, `artifacts`, `metric_results`,
`metric_cache`, `experiments`, `runs`, `run_jobs`, `workers`, `run_feedback`,
`run_purge_requests`, `cache_entries`, `run_cache_refs`, `cache_leases`,
`media_collections`, `media_assets`, `media_asset_relations`, `run_media_assets`,
`metric_asset_bindings`, `upload_sessions`, `upload_parts`, `execution_profiles`,
`evaluators`, `evaluation_campaigns`, `evaluation_candidates`, `evaluation_tasks`,
`evaluation_votes`, `evaluation_campaigns_v2`, `evaluation_methods_v2`,
`evaluation_items_v2`, `evaluation_bindings_v2`, `evaluation_preparations_v2`,
`evaluation_tasks_v2`, `evaluation_assignments_v2`, `evaluation_votes_v2`,
`evaluation_analysis_cache_v2`, `schema_migrations`.
Core schema string + `_migrate` live in `db.py`; **adding a core column or index
requires editing both**. Campaign V2 schema and its idempotent setup live in
`evaluations_v2.py` `CAMPAIGN_V2_SCHEMA` / `ensure_v2_schema`.

## Frontend wiring
Main listeners are registered at the bottom of `app.js`: nav clicks, inference/Run
delegation, visibility-aware polling and result-cache invalidation. Evaluation Studio
exports `window.VFIEvalStudio` from `studio.js`; the participant page boots independently
from `blind.js` and must not depend on `app.js`. `api()` is the main-page request helper;
Studio and blind each keep a small private request wrapper.

## Test entry points
`python -m unittest discover -s tests` runs all. Per-subsystem tests are listed in
each block above. Model files in `models/test_*.py` are both fixtures and live UI
adapters — don't rename casually.
