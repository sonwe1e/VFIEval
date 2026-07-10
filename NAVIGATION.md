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
| HTTP routing + handlers | `src/vfieval/server.py` | ~2700 | GET/POST/DELETE dispatch, run/compare/feedback orchestration |
| Data layer | `src/vfieval/db.py` | ~2100 | SQLite schema, migrations, all query methods |
| Frontend | `src/vfieval/web/app.js` | ~3400 | all UI logic; `index.html` markup; `styles.css` |
| File entry points | `src/vfieval/file_inputs.py` | ~1400 | scan models/checkpoints/videos, preflight, thumbnails |
| Dataset scan | `src/vfieval/datasets.py` | ~1100 | video → triplet samples, `VideoEntry`, multi-group |
| Inference | `src/vfieval/pipeline/inference.py` | ~1400 | `run_inference_job`, prefetch/save pools |
| Post-processing | `src/vfieval/pipeline/postprocess.py` | ~240 | warp/blend/pred, `compose_interpolated` |
| Compare resolution | `src/vfieval/compare_inputs.py` | ~340 | descriptor → path, strict alignment |
| Metric health/assets | `src/vfieval/metrics/health.py` | ~1460 | manifests, downloads, availability |
| Metric execution | `src/vfieval/pipeline/metrics_runner.py` | ~280 | scores pred/gt after inference |
| Media catalog | `src/vfieval/media_assets.py` | ~760 | collections/assets, backfill, resolver, provenance relations |
| Uploads | `src/vfieval/uploads.py` | ~390 | resumable parts, hashes, ZIP validation, quotas/cleanup |
| Blind evaluation | `src/vfieval/evaluations.py` | ~850 | Campaigns, tasks, votes, ranking/export |
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
- **Execution:** `pipeline/metrics_runner.py`; per-metric `metrics/feature.py` (lpips_*), `metrics/vmaf.py`, `metrics/cgvqm.py`
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
- **Frontend:** `app.js` `renderRunDetail` (L1656), `selectRun` (L1410), `loadRunVideosPage` (L1393), `loadRunVideoTimeline` (L1619), `renderMetricChart` (L1958)
- **Tests:** `tests/test_sample_api_scope.py`, `tests/test_db_indices.py`

### 10. Run lifecycle (retry / cancel / delete / cleanup)
- **API:** `server.py` `_retry_run` (L1179), `_cleanup_run_artifacts` (L1232); batch at `POST /api/runs/batch-delete`
- **Data:** `db.py` `request_run_cancel` (L1145), `cancel_run` (L1199), `soft_delete_run` (L1228), `rename_run` (L1241), `mark_run_artifacts_cleaned` (L1533)
- **Routes:** `POST /api/runs/{id}` (action: cancel/hide/rename/cleanup/retry), `DELETE /api/runs/{id}`
- **Frontend:** `app.js` `renderRuns` (L1360), `refreshRunsOnly` (L1317), delegated click handler (L3119)

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
- **Tables:** `media_collections`, `media_assets`, `media_asset_relations`, `run_media_assets`, `metric_asset_bindings`, `schema_migrations`
- **Routes:** `GET/POST /api/media/collections`, `GET /api/media/assets`, `GET/DELETE /api/media/assets/{id}`, `GET /api/media/assets/{id}/content`, `GET /api/media/audit`
- **Frontend:** `app.js` `loadMediaLibrary`, `renderMediaLibrary`; `#view-media`
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
- **Finalize:** `pipeline/finalize_runner.py` `run_finalize_job`; only finalize encodes shared videos and queues metrics
- **Profiles:** `performance.py` `execution_profile_identity`, `record_execution_profile`, `recommend_execution_profile`; table `execution_profiles`
- **CLI:** `cli.py benchmark` (warmup/samples/repeats)
- **Frontend:** artifact profile and pool overrides in infer form; `renderPerformanceReport`, `renderExecutionProfileRecommendation`
- **Tests:** `tests/test_artifact_profiles.py`

### 16. Blind evaluation + Campaign analysis
- **Core:** `evaluations.py` evaluator/Campaign/candidate/task/vote CRUD, `presentation_for`, `next_task`, `campaign_analysis`, Bradley–Terry/bootstrap, CSV/JSON export
- **Tables:** `evaluators`, `evaluation_campaigns`, `evaluation_candidates`, `evaluation_tasks`, `evaluation_votes`
- **Routes:** `/api/evaluators/session`, `/api/evaluation-campaigns*`, `/api/evaluation-tasks/adhoc`, `/api/evaluation-tasks/{id}/votes`, task-side `/media/{reference,left,right}`
- **Frontend:** `loadEvaluations`, `renderCampaigns`, `renderEvaluationTask`, `renderCampaignAnalysis`; `#view-evaluations`
- **Privacy invariant:** participant task payloads expose opaque task-side URLs, never true asset/model/checkpoint/Run identity before voting
- **Tests:** `tests/test_evaluation_campaigns.py`

---

## DB schema quick map
Tables: `models`, `datasets`, `samples`, `jobs`, `artifacts`, `metric_results`,
`metric_cache`, `experiments`, `runs`, `run_jobs`, `workers`, `run_feedback`,
`media_collections`, `media_assets`, `media_asset_relations`, `run_media_assets`,
`metric_asset_bindings`, `upload_sessions`, `upload_parts`, `execution_profiles`,
`evaluators`, `evaluation_campaigns`, `evaluation_candidates`, `evaluation_tasks`,
`evaluation_votes`, `schema_migrations`.
Schema string + `_migrate` both live in `db.py` `init` (L291). **Adding a column or
index requires editing BOTH the SCHEMA string and `_migrate`.**

## Frontend wiring
Event listeners registered at bottom of `app.js` (L2909+): nav clicks, form submit,
delegated `click`/`change`/`submit`/`toggle` handlers (L3101–3399), visibility-pause
for the runs poll (L3399). `api()` helper at L88; `$()` = getElementById.

## Test entry points
`python -m unittest discover -s tests` runs all. Per-subsystem tests are listed in
each block above. Model files in `models/test_*.py` are both fixtures and live UI
adapters — don't rename casually.
