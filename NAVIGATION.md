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
| HTTP routing + handlers | `src/vfieval/server.py` | ~5900 | GET/POST/DELETE dispatch plus run/compare/evaluation composition; Runtime and Job request semantics live in the modules below |
| API query validation | `src/vfieval/api_validation.py` | ~75 | bounded query integers and shared pagination normalization for extracted/new routes |
| Runtime API | `src/vfieval/runtime_api.py` | ~85 | stable `/api/health` live/ready/reasons/queues/maintenance payload, including required JobSupervisor readiness |
| Internal Job API | `src/vfieval/job_api.py` | ~210 | Worker registration, Job create/claim, callback validation, and `worker_id` + `attempt` + `lease_token` fencing |
| Submission idempotency | `src/vfieval/submissions.py` | ~165 | submission fingerprinting, leased pending reservations, stale reconciliation, completed replay, and conflict semantics |
| Data layer | `src/vfieval/db.py` | ~6200 | SQLite schema, migrations, all query methods |
| Shared frontend primitives | `src/vfieval/web/shared.js` | ~380 | request lifecycle, normalized errors/recovery advice, safe storage/copy, submission IDs and single-flight locks |
| Main frontend shell | `src/vfieval/web/app.js` | ~2500 | shared state, navigation, inference workflow, delegated listeners and deferred bootstrap; `index.html` + `styles.css` |
| Compare frontend | `src/vfieval/web/compare.js` | ~600 | Item-first selection, Alignment Plan, preflight, submission and Studio handoff |
| Run Detail frontend | `src/vfieval/web/run-detail.js` | ~2800 | Run list/detail, timeline, feedback, stats, refresh guards and purge actions |
| Media frontend | `src/vfieval/web/media.js` | ~740 | Media Library, filters, upload/hash lifecycle, external Pred binding and catalog sync |
| Evaluation frontend | `src/vfieval/web/studio.js`, `blind.js` | ~4900 | Campaign Studio/packages and isolated participant page |
| File entry points | `src/vfieval/file_inputs.py` | ~2100 | scan models/checkpoints/videos, preflight, thumbnails |
| Large video selections | `src/vfieval/video_selection_tokens.py` | ~650 | hashed, expiring folder-video snapshots, paged membership, and exact preflight/Run expansion |
| Dataset scan | `src/vfieval/datasets.py` | ~1750 | video → triplet samples, `VideoEntry`, multi-group |
| Inference | `src/vfieval/pipeline/inference.py` | ~3000 | `run_inference_job`, prefetch/save pools |
| Post-processing | `src/vfieval/pipeline/postprocess.py` | ~220 | warp/blend/pred, `compose_interpolated` |
| Compare compatibility | `src/vfieval/compare_inputs.py` | ~540 | legacy descriptor → path validation |
| Item Compare + alignment | `src/vfieval/media_items.py`, `alignment.py` | ~1810 | Item/Member resolution, strict temporal identity, LANCZOS Alignment Plan |
| Large Item selections | `src/vfieval/selection_tokens.py` | ~310 | hashed, expiring Collection+query snapshots, paged reads, and aggregate method coverage without large `IN` lists |
| Metric health/assets | `src/vfieval/metrics/health.py` | ~1860 | manifests, downloads, availability |
| Metric execution | `src/vfieval/pipeline/metrics_runner.py`, `src/vfieval/pipeline/metric_jobs.py` | batched LPIPS scoring plus multi-device metric-wave queue/aggregation |
| Media catalog | `src/vfieval/media_assets.py` | ~1610 | collections/assets, backfill, resolver, provenance relations |
| Uploads | `src/vfieval/uploads.py` | ~450 | resumable parts, hashes, ZIP validation, quotas/cleanup |
| Run purge + cache GC | `src/vfieval/run_cleanup.py` | ~2240 | persistent purge requests, cache refs/leases, preview/confirmed GC |
| Blind evaluation V2 | `src/vfieval/evaluations_v2.py` | ~6710 | two-method Campaigns, frozen packages, opaque participant flow, analysis |
| Legacy blind evaluation | `src/vfieval/evaluations.py` | ~910 | schema v1 read/export compatibility |
| Execution profiles | `src/vfieval/performance.py` | ~150 | benchmark fingerprint and recommendation |
| Multi-device finalize | `src/vfieval/pipeline/finalize_runner.py` | ~290 | merge shard manifests, encode videos, queue metrics |
| Job dispatch + leases | `src/vfieval/job_api.py`, `src/vfieval/orchestration.py`, `src/vfieval/job_leases.py` | ~1050 | internal HTTP protocol, reusable role/device JobSupervisor, heartbeat renewal, stale-worker fencing and Run failure recovery |
| Storage reservations | `src/vfieval/storage_budget.py` | ~210 | Run/upload/Campaign disk budget and 507 capacity errors |
| Child-process cancellation | `src/vfieval/process_control.py` | ~90 | cooperative polling plus bounded terminate → kill for FFmpeg/VMAF/CGVQM |
| Runtime support | `src/vfieval/runtime_api.py`, `src/vfieval/diagnostics.py`, `runtime_logging.py` | ~800 | HTTP readiness assembly, queue/doctor/support diagnostics, and rotating structured logs |

Route dispatch is a flat `if path == ...` chain in `server.py`: `do_GET` (~L650),
`do_POST` (~L1130), `do_DELETE` (~L1700). To add a route, edit the relevant handler.

---

## Subsystems

### 1. Run Feedback + Statistics
User rating (1–5, 0.25 step) + free-text issue per run; content-scoped (video/track/model/checkpoint); stats view.
- **Data:** `db.py` `add_run_feedback` (L1254), `update_run_feedback` (L1293), `list_run_feedback` (L1333), `delete_run_feedback` (L1342), `list_all_feedback` (L1350), `feedback_stats` (L1380), `feedback_filter_options` (L1506)
- **API:** `server.py` `_parse_feedback_rating` (L1512), `_feedback_context` (L1532), `_create_run_feedback` (L1580), `_update_run_feedback` (L1612), `_feedback_overview` (L1655)
- **Routes:** `POST/GET /api/runs/{id}/feedback`, `DELETE /api/runs/{id}/feedback/{fid}` (do_DELETE L413), `GET /api/feedback`
- **Frontend:** `run-detail.js` `renderRunFeedback`, `renderFeedbackItem`, `submitRunFeedback`, `deleteRunFeedback`, `loadStats`, `renderStats`, `ratingStars`/`formatRating`/`ratingOptions`
- **Table:** `run_feedback`; index `idx_run_feedback_run(run_id, created_at)`
- **Tests:** `tests/test_run_feedback.py`

### 2. Compare — Item-first (GT vs 1–2 Pred Members)
- **Primary resolution:** `media_items.py` `list_media_items`, `list_item_predictions`, `resolve_media_item_compare`; `alignment.py` `validate_temporal_alignment`, `plan_alignment`, `materialize_aligned_frame`
- **Write boundary:** `media_items.py` `_upsert_member`, `_upsert_binding`, `replace_active_binding_member`; `db.py` media-identity migration validation plus `trg_media_*` / `trg_run_media_item_bindings_*` triggers prevent direct-SQL cross-Item, role, slot, canonical-GT, and invalid Compare-snapshot bindings
- **Compatibility:** `compare_inputs.py` keeps legacy descriptor resolution/readability; it is not the primary picker contract
- **API:** `server.py` `_create_video_compare_run` (~L3150), compare layer helpers `_compare_layer_payloads`, `_compare_track_rows`
- **Routes:** `POST /api/runs` with `run_type:"video_compare"`
- **Selection routes:** `GET /api/media/items`, `GET /api/media/items/{id}/predictions`; legacy `GET /api/compare-sources/{gt,pred,flow,mask}` remains readable
- **Frontend:** `compare.js` `comparePayloadFromForm`, `startCompareRun`, `renderCompareSelection`, `compareTrackLabel`, `renderCompareAlignmentSummary`
- **Invariant:** ordered source-frame indices, frame counts, FPS, and available timestamps are strict; one or two Preds determine the explicit fingerprinted LANCZOS target size
- **Naming:** samples `{video_token}__{track_token}__{frame}`; dedup on sanitized token
- **Tests:** `tests/test_compare_multitrack.py`, `tests/test_alignment_plan.py`, `tests/test_media_items_service.py`, `tests/test_media_identity_invariants.py`, `tests/test_compare_snapshot_cleanup.py`

### 3. Compare — source pickers (GT/Pred/flow/mask)
- **API:** `server.py` `_compare_sources` (L614), `_compare_gt_sources` (L629), `_compare_pred_sources` (L678), `_compare_layer_sources` (L761), `_source_pagination` (L736), `_source_page_payload` (L744)
- **Routes:** `GET /api/compare-sources/{gt,pred,flow,mask}` (do_GET L72; rejects client `path`)
- **Backing:** `file_inputs.list_video_groups` (L94), `db.list_runs` + `db.list_run_artifacts` (L1648)
- **Frontend:** `compare.js` `loadCompareSources`, `renderCompareGtCards`, `renderComparePredCards`, `compareSourcePager`
- **Tests:** `tests/test_compare_sources_api.py`, `tests/test_compare_ui_hooks.py`

### 4. Standard inference run creation
- **API:** `server.py` `_create_run_from_files` (L843), `_default_run_name` (L1254), `_reference_config` (L1191), `_resolve_execution_devices` (L1289), `_create_inference_shards` (L1305), `_partition_samples_by_video` (L1349)
- **Exact input identity:** `input_identity.py` builds portable model/checkpoint/source signatures, normalized request fingerprints, and public structured retry differences
- **Routes:** `POST /api/runs` (default), `POST /api/preflight` with `preflight_level=quick|deep`, and `POST /api/preflight/quick`; deep preflight returns `input_fingerprint` and mints a short-lived request-and-physical-input-bound token reused by matching Run creation
- **Large selection:** `video_selection_tokens.py` freezes server-resolved `videos/<group>/<file>` entries behind a random, hashed, 24-hour token. Preflight and Run creation accept `video_selection_token`, revalidate every group/path/catalog/content signature, then expand it to the exact legacy-compatible `selected_videos` list persisted in Run metadata. Explicit `selected_videos` remains a compatibility input and cannot be combined with a token.
- **Preflight/scan:** `file_inputs.py` `preflight_run` (L346), `resolve_video_selection` (L232), `_dry_run_model_file` (L926); quick preflight skips model construction/full cache identity, deep preflight owns the model contract check
- **Workload guard:** `workload.py` `estimate_workload` (pure JSON-safe memory/artifact estimate, risk reasons/fingerprint); `storage_budget.py` adds active reservations, safety margin, and pre-submit capacity enforcement
- **Dataset:** `datasets.py` `_resolve_video_entries` (L659), `VideoEntry` (L71)
- **Frontend:** `app.js` `payloadFromForm` (L673), `startRun` (deep/token/risk acknowledgement), `runPreflight` (automatic quick vs forced deep), `renderPreflight`/`renderWorkloadEstimate`
- **Tests:** `tests/test_v2_runs.py`, `tests/test_v3_file_flow.py`, `tests/test_video_selection_tokens.py`, `tests/test_video_datasets.py`, `tests/test_two_level_preflight.py`, `tests/test_run_reproducibility.py`, `tests/test_experiment_experience_ui.py`, `tests/test_workload.py`, `tests/test_input_identity.py`

### 5. Multi-group inference
- **Backing:** `datasets._resolve_video_entries` (single `video_group` vs `video_groups` list), `file_inputs.resolve_video_selection` (L232)
- **Selection snapshot:** `video_selection_tokens.create_video_selection_snapshot`, `video_selection_membership`, `resolve_video_selection_snapshot`, `expand_video_selection_payload`
- **Routes:** `POST /api/video-selections`, paged `GET /api/video-selections/{token}`, and `GET /api/video-groups/{group}/videos?...&video_selection_token=`. The group page returns only visible rows plus counts; it never serializes `all_video_names`, a full asset map, or the full filtered result.
- **Frontend:** `app.js` `selectedGroupNames`, `ensureVideoSelectionSnapshot`, `mutateVideoSelectionSnapshot`, `renderGroupPicker`, `renderGroupVideoTable`, `payloadFromForm`
- **Invariant:** single-group runs stay byte-identical to legacy (dataset root = group folder, bare clip names). Only multi-group roots at `videos/`.
- **Tests:** `tests/test_video_datasets.py`, `tests/test_video_selection_tokens.py`

### 6. Inference pipeline (worker hot path)
- **Entry:** `pipeline/inference.py` `run_inference_job` (L234), `_iter_prefetched_batches` (L1122)
- **Post-proc:** `pipeline/postprocess.py` `validate_model_outputs` (L10), `normalize_model_outputs` (L53), `backward_warp` (L104), `compose_interpolated` (L142)
- **Model load diag:** `models/utils.py` `load_state_dict_portable`; report → `run_dir/logs/model_load.log`
- **Concurrency knobs:** `payload["prefetch_workers"]` (default 2), `payload["save_workers"]` (default min(8, cpu))
- **Worker loop:** `worker.py`, `orchestration.py`; `JobSupervisor` scans queued work at server startup and reuses one bounded Worker per role/device slot; claims carry `worker_id` + `claim_attempt` + `lease_token`; `job_leases.py` renews those fenced heartbeats and owns bounded stale-worker recovery (`JobRecoveryService`); `orchestration.shutdown_worker_processes` gracefully terminates only locally spawned accelerator workers
- **Internal worker routes:** `job_api.py` owns validation for `POST /api/jobs/claim` and `POST /api/jobs/{id}/{progress,complete,fail,heartbeat}`; `server.py` only dispatches JSON. Every callback is fenced by `worker_id`, `attempt`, and `lease_token`.
- **Tests:** `tests/test_api_modules.py`, `tests/test_end_to_end.py`, `tests/test_postprocess.py`, `tests/test_job_leases.py`, `tests/test_job_supervisor.py`, `tests/test_orchestration_processes.py`

### 7. Model / checkpoint / video catalog (file entry points)
- **Scan:** `file_inputs.py` `list_model_files` (L45), `list_checkpoints` (L66), `list_video_groups` (L94)
- **Routes:** `GET /api/model-files`, `/api/checkpoints`, `/api/video-groups`, `/api/devices`
- **Paged videos:** `media_assets.list_folder_group_videos` performs SQL count/filter/sort/paging and returns at most 200 rows without a full-name list
- **Frontend:** `app.js` `renderOptions` (L278), `renderCheckpointOptions` (L333), `renderDeviceOptions` (L374)
- **Model adapter loading:** `models/loader.py` (`file:` / `module:` / `dummy`)

### 8. Metrics (health, assets, execution)
- **Names/registry:** `metrics/names.py`, `metrics/registry.py`
- **Health/assets:** `metrics/health.py` (`metrics_health`, downloads via `prepare-metrics`)
- **Execution:** `pipeline/metrics_runner.py` (batched scoring/cache and retry cache policy; consumes already-published Run assets and records adapter setup failures per metric unit), `pipeline/metric_jobs.py` (multi-device wave creation/progress/aggregation); per-metric `metrics/feature.py` (lpips_*), `metrics/vmaf.py`, `metrics/cgvqm.py` (bounded driver protocol diagnostics)
- **Routes:** `GET /api/metrics/health`, `POST /api/runs/{id}/metrics/retry`
- **Frontend:** `app.js` `renderMetricOptions`, `renderMetricHealthTable`, `renderMetricEnvironmentPanel`; `run-detail.js` owns Run Detail failed/unavailable metric retry actions
- **CLI:** `cli.py` `prepare-metrics [--check-only|--force]`
- **Tests:** `tests/test_metrics.py`

### 9. Run detail / timeline / sample APIs (scoped reads — perf-critical)
- **API:** `server.py` `_run_detail` (L1505), `_run_videos` (L1797), `_run_video_timeline` (L1839), `_run_sample_payload` (L2270), `_run_metric_summary` (L2332). Legacy full-load: `_run_timeline` (L1696, deprecated).
- **Routes:** `GET /api/runs/{id}` `/videos` `/videos/{name}/timeline` `/samples/{id}` `/metric-summary`; `POST /api/runs/{id}/metrics/retry`; `/timeline` returns `X-Deprecated`
- **Backing (scoped):** `db.list_samples_by_video` (L615), `list_artifacts_by_sample` (L1937), `list_metrics_by_sample` (L2022)
- **Run list paging:** `db.list_runs_page` performs one page query plus one batched `run_purge_requests` query for the visible Run IDs; do not restore per-row cleanup lookups.
- **Indices:** `idx_artifacts_sample(sample_id, kind)`, `idx_metric_results_sample(sample_id, metric_name)`, `idx_run_jobs_device(device)`
- **Rule:** these handlers must NOT call `_run_timeline` or iterate the whole run.
- **Freshness:** `runs.content_revision`; `db.bump_run_content_revision`. Artifact publication, metric completion, and artifact cleanup increment it; list/detail payloads expose it.
- **Storage diagnostic:** artifact publication records canonical/preview byte sizes in artifact metadata; completion summaries expose actual bytes and `_run_detail` compares them with the preflight workload budget without filesystem reads.
- **Frontend:** `run-detail.js` `runContentRevisionChanged`, `invalidateRunResultCache`, `refreshRunsOnce`, `refreshRunResults`, `selectRun`, `loadRunVideoTimeline`, `loadSampleDetail`; requests carry generation/abort guards and preserve the current selection on refresh. Paged Run payloads include a global `active_total`, keeping the two-second poll active even when work is outside the visible page/filter.
- **Tests:** `tests/test_sample_api_scope.py`, `tests/test_db_indices.py`, `tests/test_run_result_freshness_ui.py`
- **Runtime health/support:** `runtime_api.py` assembles `GET /api/health` from release/schema/storage/Job-lease/recovery/Supervisor state plus `CatalogSyncCoordinator.status()`, cache-catalog coordination state, and durable cleanup backlogs; it never claims work or walks storage. A production-supplied stopped or failed `JobSupervisor` makes readiness fail, while direct test/doctor use without a configured Supervisor remains explicit and optional. `cli.py doctor` opens SQLite read-only and validates the requested/default host/port, repeatable target devices, FFmpeg/`libx264`, metrics, database and storage with exit codes `0/2/1` for healthy/capability-unavailable/execution-failed; `diagnostics` creates a sanitized ZIP with selected Run/Campaign state and structured log tails. Covered by `tests/test_api_modules.py`, `tests/test_runtime_diagnostics.py`, `tests/test_doctor_capabilities.py`, `tests/test_support_diagnostics.py`.

### 10. Run lifecycle (retry / cancel / delete / cleanup)
- **Service:** `run_cleanup.py` `RunCleanupService.start_cache_coordination`, `cache_coordination_status`, `preview_run_purge`, `consume_run_purge_preview`, `request_delete`, `request_artifact_cleanup`, `process_pending`, `_purge_run`, `gc_preview`, `garbage_collect`; `register_run_cache_refs`, `cache_lease`
- **Data:** `db.py` purge request CRUD/claim/recovery methods, `mark_run_deleted_after_purge`, cache entry/ref/lease helpers, `bump_run_content_revision`
- **Retry/Clone:** `server.py` `_retry_run` performs exact identity validation and freezes the resolved checkpoint; `_clone_run` deliberately resolves current files and records identity differences
- **Routes:** `POST /api/runs/{id}/{retry,clone}`; `POST /api/run-purge/preview`; confirmed `DELETE /api/runs/{id}` and legacy `/hide` return `202`; confirmed `POST /api/runs/{id}/cleanup-artifacts`, `/api/runs/batch-delete`; `GET /api/run-purge-requests/{id}`, `/api/storage/gc/preview`; `POST /api/storage/gc`
- **Loop:** production `server.py` and embedded test harnesses explicitly start `RunCleanupService.run_forever`; HTTP request handlers never pump pending cleanup work. Handler construction only starts the versioned historical decode/Compare cache scan in a daemon coordinator. GC returns `cache_catalog_not_ready` until that scan succeeds; Run purge/cleanup waits for exact coordination. A successful scan writes `maintenance:cache-catalog-v1` to `schema_migrations`, so later restarts skip the directory walk; failures stay diagnostic and retryable.
- **Frontend:** `run-detail.js` `cloneRunWithCurrentInputs`, `requestRunPurgePreview`, `withRunPurgePreview`, `deleteRun`, `batchDeleteRuns`, `cleanupRunArtifacts`, `runPurgeState`, `renderRunPurgeNotice`; Campaign dependency failures expose a Studio entry; `studio.js` `previewStorageGc`, `executeStorageGc`
- **Invariants:** Run deletion/artifact cleanup requires a fresh one-use preview token bound to operation, exact Run IDs, lifecycle/dependency/cache state; only `.vfieval/runs/{id}` is deleted; `deleted_at` is written after successful purge; shared cache needs zero active refs, zero leases, and expired grace; storage GC requires preview + `confirm=true`.
- **Tests:** `tests/test_cache_coordination.py`, `tests/test_run_cleanup.py`, `tests/test_run_purge_preview.py`, `tests/test_run_reproducibility.py`, `tests/test_v3_file_flow.py`

### 11. Artifact / file streaming
- **API:** `server.py` `_send_artifact` (L477), `_send_sample_file` (L491), `_send_file` (L504), `_parse_range_header` (L554)
- **Routes:** `GET /api/files/{artifact_id}?variant=`, `/api/samples/{id}/{slot}`, `/api/thumbnails/...`
- **Note:** honors HTTP byte ranges, 4 MiB chunks for video playback

### 12. Decode job
- **API:** `server.py` `_decode_progress_total` (L1467), `_selection_hash` (L1458)
- **Runner:** `pipeline/decode_runner.py`
- **Frontend:** `run-detail.js` `renderDecodePanel`; `app.js` `decodeBackendStatus`
- **Tests:** `tests/test_decode_progress.py`

### 13. Unified media catalog + resolver
- **Data/model:** `media_assets.py` `create_collection`, `upsert_asset`, `sync_folder_assets`, `sync_run_assets`, `resolve_asset_path`, `source_assets_to_video_payload`, `soft_delete_asset`, `media_audit`
- **Coordination:** `catalog_sync.py` `CatalogSyncCoordinator` coalesces startup/explicit background reconciliation and exposes status/revision; catalog GET routes consume read-only SQLite snapshots
- **Tables:** `media_collections`, `media_assets`, `media_asset_relations`, `run_media_assets`, `metric_asset_bindings`, `schema_migrations`; frozen Campaign media uses `source_kind='evaluation_package'`
- **Routes:** `POST /api/media/sync`, `GET /api/media/sync/status`; `GET/POST /api/media/collections`, `GET /api/media/assets`, `GET /api/media/sources`, `GET /api/media/run-outputs`, `GET/DELETE /api/media/assets/{id}`, `GET /api/media/assets/{id}/content`, `GET /api/media/assets/{id}/thumbnail` (lazy generation), `GET /api/media/audit`
- **Frontend:** `media.js` `syncCatalogAndRefresh`/`waitForCatalogSync` plus Sources/Uploads; `studio.js` `loadRunOutputs`, `renderDerivedRuns`, `renderPackages` owns Derived Runs and Evaluation Packages; `#view-media`
- **Visibility:** internal Run/evaluation collections are not user collections; deleted/cleaned Run outputs stay unavailable and must not be resurrected by catalog sync.
- **Compare:** `compare_inputs.resolve_compare_descriptor(kind="media_asset")`; primary picker submits asset ids
- **Tests:** `tests/test_catalog_sync.py`, `tests/test_server_request_maintenance.py`, `tests/test_media_catalog_uploads.py`, `tests/test_compare_multitrack.py`

### 14. External resumable uploads
- **Runner:** `uploads.py` `create_upload_session`, `receive_upload_part`, `complete_upload_session`, `delete_upload_session`, `cleanup_stale_uploads`, `_extract_frame_zip`
- **Tables:** `upload_sessions`, `upload_parts`
- **Routes:** `POST /api/uploads`, `GET/DELETE /api/uploads/{id}`, `PUT /api/uploads/{id}/parts/{index}`, `POST /api/uploads/{id}/complete`
- **Frontend:** `media.js` `uploadExternalMedia`, `createSha256Worker`, `sha256File`; `#upload-form`
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
- **Core:** `evaluations_v2.py` `list_run_outputs`, `preview_campaign_v2`, `create_campaign_v2`, `request_publish_campaign_v2`, `run_pending_preparations`, `publish_campaign_v2`, `delete_campaign_v2`, blind session/payload/media/heartbeat/vote/review functions, `campaign_analysis_v2`, `campaign_objective_curve_v2`, `campaign_export_v2`; `pipeline/evaluation_freeze.py` owns three-stream bounded rawvideo/paired-remux package materialization
- **Large selections:** `selection_tokens.py` freezes one `group_id + q` result into a random, hashed, 24-hour SQLite snapshot. `POST /api/media/item-selections` returns only its token and total; `GET /api/media/item-selections/{token}` pages selected Items; `GET /api/media/methods?selection_token=...` returns aggregate coverage. Campaign preview/create accept either the token or explicit `media_item_ids` and revalidate every Item against its original ready GT Collection.
- **Lifecycle:** draft → preparing → published → closed/archived; failed preparation can be retried. The publish route reserves frozen-package disk capacity before persisting `preparing/queued`, so a 507 never leaves a partial request; after persistence, a response-side client disconnect does not fail the durable job. Publish deep-validates every selected GT/A/B item, builds staging, freezes under `.vfieval/evaluations/{campaign_id}`, revalidates all source content, writes a SHA-256 manifest, registers evaluation-package assets, then creates tasks atomically. New packages omit Diff; historical Diff entries remain readable. Encoder-side EPIPE is diagnosed with the sink name, FFmpeg exit code and stderr, and the preparation loop emits one sanitized terminal result line.
- **Tables:** `evaluation_campaigns_v2`, `evaluation_methods_v2`, `evaluation_items_v2`, `evaluation_bindings_v2`, `evaluation_preparations_v2`, `evaluation_tasks_v2`, `evaluation_assignments_v2`, `evaluation_votes_v2`, `evaluation_analysis_cache_v2`; shared identity table `evaluators`
- **Admin routes:** paged `GET /api/evaluation-campaigns?page=&page_size=&q=&status=` returns lightweight V2/legacy summaries through `list_campaign_summaries_page` without loading Items, bindings or analysis; `POST /api/evaluation-campaigns/v2/preview`, `POST /api/evaluation-campaigns/v2`, `GET /api/evaluation-campaigns/v2/{id}[/{analysis,export,objective-curve}]`, `POST /api/evaluation-campaigns/v2/{id}/{publish,close,archive}`, `DELETE /api/evaluation-campaigns/v2/{id}`
- **Participant routes:** `/evaluate/{opaque_token}`; `/api/blind/{token}`, `/session`, `/reviews`, `/reviews/{task_token}`, task-token `/media/{reference,left,right}`, `/vote`, `/heartbeat`. Draft/preparing/failed public payloads are deliberately incomplete and never expose task, method, Run or asset identity.
- **Frontend:** `studio.js` + `studio.css` for the server-filtered/paged Campaign summary list, separately loaded Campaign detail, method picker, common-video matrix, preparation status, packages and organizer analysis; only published/closed/archived Campaigns expose a share link. `blind.html` + `blind.js` + `blind.css` is isolated from the main navigation; preparing links show an auto-refreshing wait state, while draft/failed links never request reviews.
- **Privacy/concurrency:** participant payloads use opaque campaign/task/assignment/media URLs, stable side randomization, renewable assignment leases, and transactionally capped target votes. Results unlock only after the evaluator finishes all eligible tasks.
- **Analysis:** pairwise ties are half-wins, Bradley–Terry/bootstrap is deterministic, optional quarter-step A/B ratings are separate, objective metrics aggregate Item-first across latest producer-Run shards, and organizers can lazily compare the two LPIPS per-frame curves without recomputation; human/objective results remain separate.
- **Legacy:** `evaluations.py` schema v1 stays read-only/exportable/archivable; list pages use `legacy_campaign_summaries_readonly`, while `legacy_campaigns_readonly` remains the compatibility surface for existing lifecycle helpers. Do not infer a V2 migration from labels.
- **Tests:** `tests/test_evaluation_campaigns_v2.py`, `tests/test_evaluation_studio_ui.py`, `tests/test_evaluation_campaigns.py`, `tests/test_paged_listing_scale.py`

### 17. Release packaging + public-surface contracts
- **Build identity:** `src/vfieval/release.py`, `_build_info.json`; release wheels embed the full commit SHA and expose it through `/api/health`
- **Release tools:** `tools/build_release.py` performs deterministic double-build verification and writes a wheel SHA-256; `tools/release_smoke.py` exercises the installed CLI, packaged static assets, and a live service
- **CI:** `.github/workflows/quality.yml` `release-wheel` job installs and starts the wheel in an isolated environment after CPU tests pass
- **Contract manifest:** `contracts/public_surface.json` is the reviewed inventory for CLI commands, critical routes, and web entrypoints
- **Tests:** `tests/test_release_contracts.py`; update the manifest and this navigation block whenever those public surfaces intentionally change
- **Real-browser gate:** `playwright.config.mjs`, `tests/browser/fixture-server.mjs`, and
  `tests/browser/*.spec.mjs` exercise the packaged static pages in Chromium with
  route-level API fixtures; `npm run test:browser` covers submission single-flight,
  persistent diagnostics, Run polling state preservation, chart accessibility, and
  blind-media readiness/responsive behavior. The `browser-tests` CI job installs only
  Chromium after CPU tests pass.

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
`evaluation_analysis_cache_v2`, `media_items`, `media_item_members`,
`run_media_item_bindings`, `media_item_selection_snapshots`,
`media_item_selection_snapshot_items`, `video_selection_snapshots`,
`video_selection_snapshot_items`, `schema_migrations`.
Core schema string + `_migrate` live in `db.py`; **adding a core column or index
requires editing both**. Media identity triggers are installed only after
`_validate_media_identity_invariants` proves historical rows are consistent;
violations block the version record and report example row IDs rather than being
silently repaired. Campaign V2 schema and its idempotent setup live in
`evaluations_v2.py` `CAMPAIGN_V2_SCHEMA` / `ensure_v2_schema`.

## Frontend wiring
`index.html` loads classic scripts in dependency order:
`shared.js → app.js → compare.js → run-detail.js → media.js → studio.js`.
`app.js` owns the one shared state object, navigation, inference flow, delegated event
listeners and a `DOMContentLoaded` bootstrap; the three domain files add functions to
the same classic-script global environment without duplicating state or requests.
Evaluation Studio exports `window.VFIEvalStudio` from `studio.js`; the participant page
boots independently from `blind.js` and must not depend on `app.js`. `shared.js` loads before every entrypoint
and owns JSON/text response parsing, normalized request errors and recovery advice,
safe local storage/copy helpers, submission IDs, and reusable single-flight locks.
`api()` is the main-page request helper; Studio and blind each keep a small policy
wrapper around the shared request lifecycle.

## Test entry points
`python -m unittest discover -s tests` runs all. Per-subsystem tests are listed in
each block above. Model files in `models/test_*.py` are both fixtures and live UI
adapters — don't rename casually.
