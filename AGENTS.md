# AGENTS.md

DO NOT send optional commentary.

## Project Identity

VFIEval is a local inference, post-processing, artifact viewing, evaluation, and comparison platform for video frame interpolation. It is a practical VFI evaluation tool, not a training framework and not a generic admin dashboard.

Do not add training features. Do not implement PSNR.

## Project History

- V3 moved the primary workflow to "files and folders as entry points": users place model files in `models/` and videos in `videos/{group}/`, then run inference without manual database registration.
- V5 made metric timelines and bad-frame navigation first-class goals: users should locate weak frames through curves, then inspect only the selected sample's artifacts.
- V11 added the foundation for NPU-aware device handling, run deletion and cleanup, preview artifacts, checkpoint discovery, and lightweight artifact grouping.
- V12 closed real deployment loops: Ascend NPU multi-device inference, portable metric assets, strict device consistency, direct GT/Pred video comparison (single track, local-path input), 4K-safe lazy previews, per-sample error capture, byte-range artifact streaming, and clearer failure reporting.
- V13 turns Compare into a first-class evaluation surface: GT and Pred are selected from server-resident resources (`videos/` groups and completed-run pred artifacts), one GT can be compared against multiple Pred tracks (predA / predB / ...), flow / mask / warp layers can be shown side-by-side, and the timeline/sample APIs are restructured so a single page request does not load the entire run.
- V13.1 unblocked NPU/CUDA throughput and closed silent-failure gaps: the inference pipeline now overlaps decode / model / save via a prefetch pool and an async save pool, checkpoint loads produce a structured report with missing/unexpected keys, dry-run inspects flow/mask magnitudes to catch "weights never loaded" runs before the queue starts, and the last raw-string Compare descriptor path was removed from HTTP surface, JS payloads, and tests.
- The 0711 media revision replaces asset-first comparison with an exact semantic identity layer: GT selection is `Collection -> media_item`, reusable predictions are bound `media_item_members`, Compare accepts one or two predictions from that same Item, and Campaign V2 publishes normalized frozen Item packages.

## Recently Stabilized (V12)

These are done and must not regress:

- Ascend NPU single-machine sharding with `torch.npu.set_device` called before model construction in every shard worker.
- Preflight and worker-side checks that catch CPU/NPU device mismatch before long runs.
- Metric health reporting (`completed` / `skipped` / `failed` / `unavailable`) surfaced in summaries and timeline data; no silent substitution.
- Run Detail uses paged video APIs and lazy preview artifacts; original 4K assets load only on explicit user action.
- Per-sample exceptions are captured as `sample_error` artifacts instead of failing the whole run.
- `GET /api/files/{artifact_id}` honors HTTP byte ranges and streams in 4 MiB chunks for video playback.
- Run deletion and cleanup only touch `.vfieval/runs/{run_id}` and run-scoped metadata.
- Video-group Run creation must expose decoding as a `decode` job with frame progress, backend details, and fallback reasons before inference jobs are queued.
- Re-running the same video input must reuse `decode_cache`; only stale or missing cache entries should be decoded again, and UI must distinguish cache reuse from actual decoding.

## Current Priorities

Current work should prioritize, in order:

- Keep `media_items`, `media_item_members`, and `run_media_item_bindings` as the only semantic identity path for new Compare and Campaign selections. `media_assets` remains the physical-file catalog, not proof that two files depict the same GT.
- Keep the primary Compare picker GT-first: choose one source/upload Collection, then one canonical GT Item, then one or two reusable predictions bound to exactly that Item. Compare outputs, snapshots, frozen Campaign media, deleted Runs, and unbound historical Preds never become candidates.
- Keep temporal identity strict while applying explicit spatial normalization. `source_frame_indices`, frame counts, FPS, and available timestamps must agree; dimensions may differ and are normalized with LANCZOS to the sole Pred's size or the deterministic smaller-pixel-area Pred size.
- Preserve deletion-time Compare inputs. Before a source Run is purged, every dependent Compare binding must atomically switch to a private, non-reusable `compare_snapshot`; any preservation failure blocks the source purge.
- Keep Campaign V2 Item-first and fixed to two methods: one GT group, selected Items, method coverage matrix, per-Item Alignment Plan, then an atomically published normalized frozen package.
- Preserve sample/video API scope and frontend result freshness: paged reads, request cancellation/generation guards, monotonic `content_revision`, and in-place refresh must not regress.

Do not expand to remote workers, auth, object storage, or new metric families until the V13 loops above are reliable. Windows-specific stability work is deferred — V13 targets Linux NPU and CUDA servers.

## Non-Negotiable Scope

- Do not add training features. Do not implement PSNR.
- Primary UI must use files and folders as entry points.
- Primary UI must not contain model registration or dataset registration forms.
- The first successful user flow must require no manual IDs.
- Do not require users to type `model_id`, `dataset_id`, `job_id`, or `experiment_id` in primary workflows.
- Do not substitute unavailable metrics with a different score.
- Do not render all 4K artifacts at once.
- Every failed run must show a human-readable error in UI.
- Runs where the model outputs are structurally valid but semantically empty (flow ≈ 0 and mask ≈ constant, or NaN) must be surfaced as preflight warnings — do not let a "weights never loaded" run silently proceed to a gray-blend result.
- Compare's primary entry must use `media_item_id` plus one or two `pred_member_ids` from the GT-first media APIs. Raw string descriptors, typed paths, asset-only identity, cross-Item combinations, Compare-derived Preds, and unbound historical Preds are rejected.
- Every selected Pred tile must identify its method/Run and alignment slot. At most two Preds are accepted, and every member must be reusable and belong to the same canonical Item.
- The server never trusts a client path. Item and Member IDs resolve through server-owned `media_items`, `media_item_members`, `media_assets`, and Run state; legacy `{kind: "video_group"|"run_artifact"|"media_asset"}` descriptors are compatibility inputs only and must resolve to a valid same-Item binding before entering the new flow.
- Sample-level and video-level APIs (`/api/runs/{id}/videos`, `/videos/{name}/timeline`, `/samples/{id}`) must not iterate the entire run's samples / artifacts / metrics. Only the legacy `/api/runs/{id}/timeline` debug endpoint is allowed to materialize a full-run view.

## Primary User Flows

### Model Inference Flow

Users choose a model file, optional checkpoint, video group, video subset, output size, device, precision, and metrics. VFIEval decodes or reuses cached frames, runs the model, applies platform-owned post-processing, writes artifacts, runs available metrics, and shows timeline-centered results.

This flow must produce viewable `pred/gt/diff` artifacts when ground truth exists. It may also produce flow, mask, warp, blend, and `extra_*` artifacts.

### Direct GT/Pred Compare Flow

Users may compare existing GT and Pred outputs without running a model. The primary entry uses server-resident sources only:

- **GT group** is chosen first from a source/upload Collection such as `test4k`; search and paging then apply only inside that group.
- **GT Item** is one exact `media_item` whose `canonical_gt_asset_id` identifies one physical source. Same names or hashes in another Collection do not share identity or predictions.
- **Pred** is one or two reusable `media_item_members` bound to that same Item. New `model_inference` outputs and explicitly bound External Preds are eligible; Compare outputs, snapshots, frozen evaluation media, old unbound Runs, and invalid/deleted Runs are not.
- **External Pred** binding is an advanced action and must name the Item explicitly. An aspect-ratio-changing normalization requires explicit confirmation at binding or policy time.

Temporal alignment remains exact: ordered `source_frame_indices` must match across Preds, indices must be in range, and frame count, FPS, and available timestamps must agree. A resize can never repair a wrong video, frame mapping, or GT identity.

Spatial alignment is explicit rather than exact-dimension rejection. With one Pred, use that Pred's complete width and height. With two, use the complete dimensions of the lower-pixel-area Pred; ties compare maximum edge, width, height, then slot for determinism. Normalize GT and the other Pred with LANCZOS. The Alignment Plan records original/target dimensions, scale factors, `none/upscale/downscale/mixed`, aspect-ratio change, filter, temporal summary, and a fingerprint used by Compare/cache/metrics/Campaign.

The new Compare Run references the original GT/Pred Members, writes its bindings, Alignment Plan, Diff, metrics, and reports, and does **not** publish a reusable `pred_video`. Aligned inputs are rebuildable `compare_cache` materializations. If a source Run is about to be deleted, it is preserved under the dependent Compare Run as a non-reusable `compare_snapshot` before the source purge proceeds.

### Metric Environment Flow

Metric dependencies, weights, native evaluator binaries, and config files should be portable through `set/metrics/`. Missing assets must be reported as `unavailable` with a specific reason in SQLite and UI. Do not automatically download metric assets unless a future plan explicitly adds that behavior.

## File And Folder Entrypoints

- Models are discovered from `models/*.py`.
- Checkpoints are discovered from `checkpoints/{model_stem}/`.
- Video groups are discovered from `videos/*/`.
- Metric assets are discovered from `set/metrics/`.
- Run artifacts live under `.vfieval/runs/{run_id}/`.
- Repository layout, naming, and Git ownership rules live in `REPO_LAYOUT.md`. Keep real user models, videos, checkpoints, metric assets, SQLite files, runtime outputs, local tool state, and `*.backup.YYYYMMDD_HHMMSS` backups out of Git; keep generated `test_*` models, checkpoints, and videos tracked for clean-checkout tests.
- Generated `test_*` models, checkpoints, and videos are committed fixtures, not cleanup candidates. When session backups accumulate, archive `*.backup.YYYYMMDD_HHMMSS` files under `archive/file_backups/{timestamp}/` instead of leaving them scattered beside source files.

Legacy `models`, `datasets`, `jobs`, and `experiments` APIs may remain for compatibility, but those concepts must not pollute the primary UI.

## Model Contract

Models receive resized RGB `img0` and `img1` tensors in `BCHW`, value range `[0, 1]`, fixed `t=0.5`, on the inference device and dtype. Model files should expose:

```python
class Model:
    def infer(self, img0, img1):
        return {
            "flowt_0": flowt_0,
            "flowt_1": flowt_1,
            "mask0": mask0,
            "mask1": mask1,
        }
```

Tuple return `(flowt_0, flowt_1, mask0, mask1)` is allowed.

Model checkpoints should be portable across CPU, CUDA, and Ascend NPU workers. User model files should load checkpoint bytes on CPU first, then move the constructed network to the requested runtime device:

```python
state = torch.load(checkpoint_path, map_location="cpu")
network.load_state_dict(state["state_dict"] if "state_dict" in state else state)
network.to(device)
network.eval()
```

The helper `vfieval.models.utils.load_state_dict_portable(module, checkpoint_path, device)` is available for this pattern. Do not deserialize checkpoints directly onto a fixed device such as `npu:0` or `cuda:0`.

`flowt_0` and `flowt_1` are backward flow in resized pixel coordinates. They represent displacement from the target middle-frame pixel to the source pixel in `img0` or `img1`. `mask0` and `mask1` are logits, not probabilities.

Core outputs must match input batch, device, dtype, and channel count. Spatial resolution may be lower than the input; the platform must resize flow and mask to the inference resolution before warp/compose, scaling flow x/y magnitudes with width/height.

Extra visualization tensors are allowed and may be saved as `extra_*`, but comparison must focus on flow, mask, warp, blend, pred, and diff artifacts.

## Post-Processing Contract

The platform owns warp, sigmoid, blend, compose, visualization, artifact writing, and metric execution. Keep these formulas stable unless code, tests, and docs are updated together:

- `grid_sample(mode="bilinear", padding_mode="border", align_corners=True)`
- `mask0 = sigmoid(mask0_logits)`
- `blend = mask0 * warp0 + (1 - mask0) * warp1`
- `mask1 = sigmoid(mask1_logits)`
- `pred = mask1 * img1 + (1 - mask1) * blend`
- `pred`, `blend`, `warp`, and `diff` are clamped to `[0, 1]`
- `difference = abs(pred - gt)` when GT exists

All platform-created tensors used in post-processing, including grids and constants, must be created on the same device and compatible dtype as the input tensors.

## GT-First Compare API Surface

The primary picker uses the semantic Item APIs. They resolve server-managed IDs only and never accept a client filesystem path:

- `GET /api/media/item-groups?role=gt` — source/upload Collections that contain ready canonical GT Items.
- `GET /api/media/items?group_id=...&q=...&page=...` — Items inside the selected Collection; search never escapes that group.
- `GET /api/media/items/{item_id}/predictions` — reusable Pred Members bound to that exact Item.
- `GET /api/media/methods?item_id=...&item_id=...` — method coverage and missing-item matrix for Campaign selection.
- `GET /api/media/unbound-predictions` — management audit only; old or untrusted Pred artifacts listed here are not selectable.
- `POST /api/media/items/{item_id}/external-predictions` — advanced explicit External binding, including temporal/spatial provenance and aspect-stretch confirmation.

Run creation accepts this primary payload:

```json
{
  "run_type": "video_compare",
  "media_item_id": 12,
  "pred_member_ids": [81, 93],
  "spatial_policy": {
    "mode": "smallest_pred",
    "filter": "lanczos",
    "allow_known_aspect_stretch": true
  },
  "metrics": ["vmaf"]
}
```

The service resolves the Item and Members with `media_items.resolve_media_item_compare`, validates reusable roles and Run state, then constructs server-owned `media_item` / `media_item_member` descriptors. `alignment.validate_temporal_alignment` and `alignment.plan_alignment` own the shared temporal/spatial contract. Client `path` values and more than two Preds are rejected.

Compare detail uses `GET /api/runs/{id}/compare-inputs` and `GET /api/runs/{id}/compare-inputs/{slot}/media?variant=original|aligned`. Original playback follows the active binding; aligned playback is rebuilt from the source and fingerprinted cache when necessary.

`/api/compare-sources/{gt,pred,flow,mask}`, `video_group`, `run_artifact`, and `media_asset` descriptors remain only for legacy compatibility and historical reads. They are not the primary picker contract and must not make unbound or Compare-derived `pred_video` artifacts reusable. Historical V13 Compare Runs may retain their old track-scoped videos; newly created Item Compare Runs intentionally publish no `pred_video`.

## Timeline And Sample API Performance (V13)

- `_run_videos`, `_run_video_timeline`, and `_run_sample_payload` must not call `_run_timeline()` internally. They read only the rows they need.
- `samples`, `artifacts`, and `metric_results` must be queryable in sample-scoped and video-scoped chunks. Required indices (declared in the schema string in `db.py`):
  - `CREATE INDEX IF NOT EXISTS idx_artifacts_sample ON artifacts(sample_id, kind);`
  - `CREATE INDEX IF NOT EXISTS idx_metric_results_sample ON metric_results(sample_id, metric_name);`
  - `CREATE INDEX IF NOT EXISTS idx_run_jobs_device ON run_jobs(device);`
- New `db` helpers: `list_artifacts_by_sample(sample_id)`, `list_metrics_by_sample(sample_id)`, `list_samples_by_video(run_id, video_name)`. Prefer these over `list_run_artifacts` + Python filtering in any sample- or video-level handler.
- `GET /api/runs/{id}/timeline` remains as a compatibility/debug endpoint; it is the only endpoint allowed to materialize a full-run view, and its response carries `X-Deprecated: use /videos` to discourage UI consumption.

## Frontend Polling And Request Hygiene (V13)

- The 2-second `refreshRunsOnly` poll must pause while `document.hidden` is true and resume on `visibilitychange`. Terminal-status runs may be polled at a relaxed cadence (≥10s).
- `loadSampleDetail` and `loadRunVideoTimeline` must use an `AbortController` keyed to the current selection; switching sample/video aborts the previous in-flight request.
- `renderSamplePreview` and timeline tiles must show a skeleton/loading state, never a bare "loading..." text node that flashes on every render.
- Run Detail must render a `run-error-banner` whenever `run.error_json` is non-empty, and bind a "Retry this Run" action to the existing `_retry_run` endpoint.
- `runs.content_revision` is monotonic. Publishing artifacts, completing metrics, and cleaning artifacts must increment it and expose it in Run list/detail payloads.
- When the selected Run's revision changes, abort stale sample/timeline requests, invalidate only that Run's client caches, and reload videos/timeline/current sample while preserving video, frame, window, and metric selection. Request generations must prevent older responses from overwriting the refreshed state.
- A non-terminal Run with no published artifacts shows a skeleton and "artifacts are being generated" state. Show "no loadable artifacts" only after a fresh terminal-state query is still empty. Manual result refresh uses the same scoped invalidation path; never require a full browser reload.

## Compare UI Layout

- Source selection is a staged `GT Collection -> GT Item -> one or two bound Pred Members` flow. Changing group or Item clears incompatible Pred selection.
- The picker displays each prediction's method, producer Run, original dimensions, temporal mapping, and normalization summary. It never offers cross-Item or unbound candidates.
- Compare detail renders original/aligned GT and Pred inputs plus Diff/metrics from the aligned frames. Every tile identifies its slot and kind; a master playback control keeps the active videos synchronized.
- The Alignment Plan report and fingerprint remain visible in preflight and detail. Browser CSS scaling is presentation only and must never be used as metric or Diff input.
- Historical V13 multi-track layouts remain readable, including their `kind × track` extra-layer grid, but they are a legacy compatibility surface rather than the new source model.

## Inference Pipeline Concurrency

The inference hot loop is asynchronous by default. Do not reintroduce per-sample `.cpu()` calls or synchronous PNG writes on the main compute thread.

- `_iter_prefetched_batches` runs a `ThreadPoolExecutor` (default 2 workers, configurable via `payload["prefetch_workers"]`) that PIL-decodes `img0` / `img1` / GT in parallel and holds up to 2 batches ahead of the device. `pin_memory()` is applied when CUDA is available so the device transfer can be `non_blocking=True`.
- The main loop only does `.to(device, non_blocking=True)` → `model.predict` → `normalize` → `compose_interpolated` → a single `.detach().to("cpu")` per bundle, then hands off to the save pool. It must not touch the filesystem or SQLite.
- `_AsyncSavePipeline` runs a `ThreadPoolExecutor` sized `min(8, cpu_count())` (configurable via `payload["save_workers"]`). Each background sample writes 8 core PNGs, per-artifact previews, extra visualizations, difference, and `db.add_artifact` records. Because every write uses its own short-lived sqlite3 connection through `Database.connection()`, WAL-mode file locking is enough — no cross-thread `Lock` around DB writes.
- Progress updates coalesce to at most `total // 200` events; do not push per-sample updates from inside the save pool.

## Preflight Diagnostics

- Completed model-inference runs must store `run.result_json.output_health` and `run_dir/logs/output_health.log`, computed from real inference frames, so checkpoint-load success is not confused with useful flow/mask output.

`_dry_run_model_file` returns `{output_health, model_load}` and `preflight_run` promotes both into the response so the UI can surface silent-failure runs before they start.

- `output_health.stats` records `abs_mean` / `abs_max` / `nan_count` for `flowt_0`, `flowt_1`, and post-sigmoid `mask0` / `mask1`. If `abs_max < 1e-4` on both flows AND `std < 1e-3` on both masks, a warning is emitted (`"flow ≈ 0 且 mask ≈ constant"`) — this is the checkpoint-never-loaded signature (`sigmoid(0) = 0.5`).
- `model_load` is produced by `vfieval.models.utils.load_state_dict_portable`. It loads with `strict=False`, returns `{checkpoint_path, matched, total_in_checkpoint, missing_keys, unexpected_keys}`, and attaches the same dict to `module._last_load_report`. Any non-empty missing/unexpected list surfaces as a `CheckpointLoadReport` warning in `preflight` and as `run.result_json.model_load` after inference; `run_dir/logs/model_load.log` records the same list.
- The Run Detail UI (`renderModelLoadReport`) renders the summary line and switches to a warn banner when either list is non-empty.

## NPU And Multi-Device Rules

- If any inference shard fails, queued sibling shard jobs must be canceled and running siblings must stop at their next cancellation check.

NPU multi-device inference targets Ascend `torch_npu` and uses single-machine shard workers, not DDP.

- Use device ids like `npu:0`, `npu:1`, and `npu:7`.
- Each worker should own one NPU device and process a video-level or segment-level shard.
- Each NPU worker must call `torch.npu.set_device(index)` before model creation, dry-run, tensor staging, or inference.
- Preflight must run on the selected target device and catch CPU/NPU mismatch before the user starts a long run.
- Do not move tensors back to CPU during post-processing except for artifact encoding or metric input boundaries.
- CUDA support remains; multi-device priority is Ascend NPU, with CUDA as the secondary supported target.
- Cross-machine workers are future work and should use HTTP registration, claim, heartbeat, progress, complete, and fail APIs rather than direct SQLite access.

## Metrics

Only `lpips_vit_patch`, `lpips_convnext`, `vmaf`, and `cgvqm` are valid metrics. Missing metric manifests, required assets, driver commands, interpreters, system executables, config files, or bindings must produce `unavailable`, never substitute another score.

Per-sample metrics may be plotted as timeline curves. Video-level metrics such as VMAF or CGVQM must be shown as video-level summaries unless the adapter produces real per-sample values. Do not create fake per-frame points from video-level scores.

`lpips_vit_patch`, `lpips_convnext`, and `cgvqm` now resolve through `set/metrics/<metric>/manifest.json` with `input_mode`, `driver.command`, `required_files`, and optional `env`. Missing manifest or required files maps to `missing_weights`; missing driver executables or invalid manifest structure maps to `missing_evaluator`.

`vmaf` remains the first built-in real metric path. It resolves `ffmpeg` from `set/metrics/vmaf/manifest.json -> ffmpeg_path` first, then falls back to `PATH`; health payloads and run metadata should expose `implementation_mode`, `manifest_path`, `driver_command`, and `resolved_executable`.

`lpips_vit_patch` is an internal DINOv2 feature-distance metric using `dinov2_vits14_reg` by default. `lpips_convnext` is an internal timm ConvNeXt V2 tiny feature-distance metric using `convnextv2_tiny.fcmae_ft_in22k_in1k` by default. Both are sample-level, lower-is-better metrics; `prepare-metrics` downloads their default assets into `set/metrics/`, while missing Python packages remain `missing_dependency`.

`cgvqm` runs as a video-only wrapper around a local IntelLabs CGVQM checkout declared in `set/metrics/cgvqm/manifest.json`. `prepare-metrics` downloads the checkout and writes `run_cgvqm_vfieval.py`; the wrapper reads VFIEval JSON from stdin and writes `{status, value, details}` JSON to stdout.

Metric evaluation resolution must be explicit and part of health, result details, and cache keys: DINOv2 uses max edge 518 padded to 14, ConvNeXt uses max edge 288 padded to 32, and CGVQM uses temporary evaluation videos capped to long edge 720 without overwriting original artifacts.

Metric jobs inherit the run's inference device through `metric_device`. CUDA/NPU metric execution may be attempted, but a metric-side device or warmup failure must be recorded as `unavailable` with the device and reason; do not silently fall back to CPU.

Metric cache keys must include metric name, adapter version, metric config, manifest fingerprint, driver fingerprint, reference identity, and distorted identity. Reopening a Run Detail page must read SQLite and artifacts only; it must not trigger metric recomputation.

No-GT samples must be marked `skipped: no ground truth` for full-reference metrics.

## Timeline UI

Run Detail must be timeline-centered and lazy-loaded. Do not render every video player or every artifact image at once.

Primary UI should use paged video APIs and windowed timelines: `GET /api/runs/{id}/videos`, `GET /api/runs/{id}/videos/{video_name}/timeline`, and `GET /api/runs/{id}/samples/{sample_id}`. Keep `GET /api/runs/{id}/timeline` only as a compatibility/debug endpoint.

The default UI should load preview artifacts first. Original 4K or larger artifacts should load only after explicit user action. Clicking a metric curve point or worst-sample row should load only that sample's core artifacts.

GT/Pred/Diff video artifacts must be playable in the browser. `GET /api/files/{artifact_id}` must support HTTP byte ranges for video playback and return `Accept-Ranges: bytes`; Run Detail should use lightweight `<video controls preload="metadata">` controls rather than auto-loading full videos.

Core artifacts should be grouped so only the selected artifact group loads previews:

- Basic comparison: `gt`, `pred`, `diff`
- Model internals: `flowt_0`, `flowt_1`, `mask0`, `mask1`, `warp0`, `warp1`, `blend`
- Extra visualizations: collapsed `extra_*`

## Artifacts And Cleanup

A clean checkout must include generated test models and generated test videos. Every model-inference Run must produce viewable `pred/gt/diff` artifacts when ground truth exists. A new Item Compare Run reuses its bound GT/Pred inputs and publishes Diff/metrics/reports, not a reusable `pred_video`. Every failed Run must show a human-readable error in UI.

Run deletion is a persistent purge request, not a `deleted_at` toggle. `DELETE /api/runs/{id}` returns `202`; active Runs enter canceling, wait for every worker to stop, and only then purge. Write `deleted_at` only after every cleanup step succeeds. Failed requests retain a human-readable error/report, are retryable, and incomplete requests resume after server restart. Batch deletion creates one request per Run and continues past individual failures; legacy `hide` is only an alias for the same service.

Run deletion and artifact-only cleanup must use the same idempotent path-safety service and may delete only the exact trusted `.vfieval/runs/{run_id}` directory plus Run-scoped artifact/feedback metadata. Keep Run jobs, metric rows, purge reports, and formal blind-evaluation history. Do not delete source model files, checkpoints, source videos, `set/metrics/`, metric cache, video metadata, or thumbnail cache.

`decode_cache` and `compare_cache` are managed by `cache_entries`, `run_cache_refs`, and renewable leases. Removing one Run releases only its refs. A cache entry is GC-eligible only after its last valid Run ref is released, no lease is active, and the default 10-minute grace has expired. Shared caches must never be removed while another Run can reuse them.

Historical Run directories and orphan caches require `GET /api/storage/gc/preview` followed by an explicit confirmed `POST /api/storage/gc`. Preview must report bytes, blockers, active refs/leases, and affected Campaigns. Startup backfill/catalog sync may mark stale Run media unavailable but must never resurrect deleted or artifact-cleaned Run outputs.

Before purging a model-inference Run, cleanup must find dependent `compare_pred` bindings. It first materializes each source into `.vfieval/runs/{compare_run_id}/inputs/`, registers a non-reusable `compare_snapshot` Member, and atomically switches `active_member_id` while preserving `original_member_id`. All snapshots must succeed before any source bytes are removed; failure leaves the source Run intact and the purge retryable. Purging the dependent Compare Run removes its private snapshots.

Failed, canceled, invalid, and test runs must be removable from the UI. A canceled or failed run should not leave a partial output that the UI presents as complete.

## Campaign V2 And Frozen Evaluation Packages

- A Campaign V2 compares exactly two Pred methods. The primary method identity is a completed model-inference Run with reusable Members across selected Items; External Pred is an advanced source. Never auto-expand a Campaign to all Runs.
- Creation is GT-first: enter campaign details, choose one GT Collection, select one or more Items, inspect method coverage, choose A/B, then inspect each Item's Alignment Plan. A method missing any selected Item blocks publish; users may deselect that Item, but the service never silently skips it.
- Temporal validation uses the same strict contract as Compare. Spatial differences are normalized with the same deterministic `smallest_pred` LANCZOS policy; dimensions are not a failure by themselves. Every per-Item Alignment Plan fingerprint enters the manifest and analysis detail.
- Publish is a persistent preparation job. Deep-validate and materialize normalized GT/A/B (plus Diff/report) in staging, write a SHA-256 manifest under `.vfieval/evaluations/{campaign_id}`, register non-reusable `evaluation_gt` / `evaluation_pred` Members backed by `source_kind='evaluation_package'`, then atomically publish tasks. Failure removes staging and creates no partial task set. A failed preparation is retryable and resumes safely after restart.
- Published Campaign configuration, method/item bindings, and evaluation-package bytes are immutable. A Campaign may be closed or archived. Before destructive Run cleanup, freeze any still-Run-backed published Campaign media; if preservation fails, fail that Run purge before deleting its outputs.
- Participant UI lives only at `/evaluate/{opaque_token}` and must not load the admin navigation, Media catalog, or organizer analysis. Campaign/task/assignment/media URLs are opaque and must not reveal Run, model, checkpoint, method label, asset id, or task id before voting.
- Assignment leases are renewable and target-vote admission is transactionally capped. Left/right order is stable per task and evaluator. A participant sees named live results only after completing every eligible task; organizers may always see coverage and analysis.
- Human analysis uses pairwise half-wins for ties, Bradley–Terry, and deterministic bootstrap intervals. Objective metric summaries keep their own direction/status semantics; never combine subjective and objective scores.
- Schema v1 Campaigns remain read-only, exportable, and archivable. Do not infer migration from labels or model names; only an empty v1 draft may be explicitly discarded.

## Architecture

Keep SQLite as metadata and index storage. Keep artifacts on disk. Prefer dependency-light changes unless a dependency materially improves video decoding, NPU reliability, metric correctness, or artifact portability.

Primary execution remains split between the Web/API control plane, inference workers, and metric workers. Inference failure must not erase already-readable logs. Metric failure must not block viewing completed inference artifacts.

Future remote workers should be implemented through explicit HTTP worker APIs. Do not let remote workers mutate SQLite directly.

## Codex Maintenance Workflow

Every Codex session working on VFIEval should read this `AGENTS.md` before planning or modifying code. When a task reveals a stable project rule, deployment constraint, or recurring failure mode, update this file in the same change set after tests pass. Keep these updates concise and specific to VFIEval; do not turn this file into a changelog.

## Legacy V13 Compatibility Checklist

V13's asset/descriptor and multi-track contracts describe historical compatibility only. Do not use them as the primary Compare architecture after the 0711 media revision.

1. Preserve historical V13 Runs and their track-scoped `pred.mp4` / `diff.mp4` artifacts for reading; do not expose those Compare outputs as reusable Pred Members.
2. Keep `/api/compare-sources/{gt,pred,flow,mask}` and `{kind: "video_group"|"run_artifact"|"media_asset"}` only as wrappers for compatible, bound sources. New UI code uses the Item APIs and `media_item_id` / `pred_member_ids`.
3. Preserve V11/V12/V13 reliability: NPU sharding, previews, purge/GC, checkpoint diagnostics, per-sample errors, byte-range streaming, scoped timeline queries, and stale-request guards must not regress.
4. Keep the primary UI clean: no model/dataset registration, raw jobs, typed local paths, or unbound historical Preds in first-run, Compare, or Campaign workflows.

## V13.1 Fixes Landed

1. **Async inference throughput** — `pipeline/inference.py` was refactored around `_iter_prefetched_batches` and `_AsyncSavePipeline`. The main compute loop performs one device→host transfer per batch; PNG encoding, preview generation, and `db.add_artifact` inserts happen in the background save pool. Timing accounting now separates `decode` (prefetch wait), `model`, `post`, `save` (main-thread submit time only). `payload["save_workers"]` and `payload["prefetch_workers"]` allow per-run tuning.
2. **Silent-failure diagnostics** — `load_state_dict_portable` no longer discards `missing_keys` / `unexpected_keys`. `_dry_run_model_file` inspects flow magnitude and mask variance and returns diagnostics up through `preflight_run` / Run Detail. Any run whose checkpoint didn't attach is flagged before compute starts.
3. **Compare raw-path removal** — `web/index.html` no longer contains reference/distorted text inputs; `app.js::payloadFromForm` returns `null` unless a structured GT + Pred picker selection exists; `compare_inputs.resolve_compare_descriptor` rejects strings; `server.py` dispatches `video_compare` strictly on `run_type`. Six `tests/test_v3_file_flow.py` cases were rewritten to use `{kind: "video_group"}` / `{kind: "run_artifact"}` descriptors via `v13_test_utils`; `test_video_compare_rejects_raw_string_descriptors` asserts the new rejection path.

## Unified Media, Throughput, And Blind Evaluation Rules

- `media_collections` and `media_assets` are the physical catalog for folder files, uploads, Run artifacts, Compare snapshots, and frozen evaluation packages. Existing source files are not moved. Backfill is idempotent by `source_key`; every ready asset records SHA-256, media metadata, a server-managed path, and provenance.
- `media_items` is the semantic video identity layer. One Item belongs to one Collection and one exact `canonical_gt_asset_id`; do not merge by filename, stem, label, or SHA across folders. `media_item_members` connects physical assets as `canonical_gt/model_pred/external_pred/compare_snapshot/evaluation_gt/evaluation_pred`. `run_media_item_bindings` records Run source/output/Compare inputs and preserves original versus active Member across snapshots.
- Only a new model-inference `model_pred` and an explicitly Item-bound `external_pred` may set `reusable_as_pred=true`. Compare snapshots, Compare-derived media, Campaign package Members, old unbound Run outputs, deleted/cleaned Runs, and `video_compare` producers are never reusable.
- Media UI has `Sources / Uploads`, lazy `Derived Runs` grouped as Run → video → Track, and `Evaluation Packages`. Internal auto-generated Run/evaluation collections are not user directories. Derived/Compare/Campaign source queries include only valid, non-cleaned Run outputs.
- Primary inference sends `source_assets`, binds each source Item, and registers new Pred outputs back onto that Item with temporal/spatial provenance. Do not backfill legacy Run Preds as Members. Primary Compare sends `media_item_id` and one or two `pred_member_ids`; legacy descriptors remain compatibility wrappers only.
- Compare/Campaign alignment is temporally strict but spatially normalized. Never truncate, offset, reinterpret the GT, or repair a wrong frame map with resize. Use the shared deterministic LANCZOS Alignment Plan, record aspect stretch, and require explicit confirmation for External Pred aspect changes.
- Uploads are 8 MiB resumable parts with per-part and whole-file SHA-256. The client chooses Collection, role, alias, and media kind but never a path. Frame sequences are ZIP + explicit FPS; reject traversal, symlinks, unsafe expansion, invalid images, and mixed dimensions.
- Artifact profiles are stable: `evaluation` saves Pred/GT/Diff evaluation media without full internals; `diagnostic` adds Flow/Mask/Warp/Blend/extra; `benchmark` saves no media and runs no metrics.
- The save pipeline must keep device-to-host bundles and pending saves bounded, transfer one packed tensor bundle per batch, batch artifact inserts, and encode videos directly from the already-written sample frames.
- When device count exceeds video count or load is skewed, split long videos into continuous sample segments. Shards write frames plus manifests; a `finalize` job alone merges segments, encodes videos, publishes Run assets, and queues metrics.
- Blind evaluation uses a stable browser UUID plus display name, not authentication. Formal blind votes are independent of `run_feedback`; Run cleanup must not delete Campaigns, tasks, assignments, votes, ranking history, frozen package bytes, or protected uploads.
- Human and objective analysis remain separate. Human ranking uses pairwise half-wins for ties, Bradley–Terry, and deterministic bootstrap intervals. Objective summaries preserve metric direction and status semantics; never emit a combined subjective/objective score.

## Future Roadmap

- V14 should add remote worker orchestration through HTTP worker lifecycle APIs.
- V15+ may add portable workspace export bundles and cross-machine artifact sync.
- Windows-specific hardening (path-traversal nuances, encoding quirks, `_read_json` size limits, stuck-job reaper) is deferred until the V13 priorities ship. Linux NPU and CUDA servers are the only supported deployment targets in V13.

## Testing

Run `python -m unittest discover -s tests` and `git diff --check` before finalizing code changes. If only `AGENTS.md` changes, `git diff --check` and manual review are sufficient.

Coverage should include file discovery, checkpoint discovery, preflight, model interface failures, NPU device selection, CPU/NPU mismatch errors, video decode/cache behavior, post-processing contracts, run lifecycle, run deletion, artifact grouping, metric unavailable behavior, timeline data, direct GT/Pred comparison, and UI-relevant lazy result display.

Current additions to keep covered:

- `tests/test_media_items_service.py` — exact canonical Item identity, GT group paging, eligible Member filtering, same-Item enforcement, one/two-Pred validation, no new Compare `pred_video`, input detail/media routes, and External aspect confirmation.
- `tests/test_alignment_plan.py` — strict temporal mapping plus single/two-Pred deterministic target selection, LANCZOS report/fingerprint, up/down/mixed resize, External stretch rejection, and cache rebuild after GC.
- `tests/test_compare_snapshot_cleanup.py` — source Run deletion freezes dependent Compare inputs, atomically switches active bindings, removes snapshots with the Compare Run, and blocks purge when preservation fails.
- `tests/test_compare_multitrack.py` — historical V13 multi-track compatibility plus explicit spatial Alignment Plan behavior; historical track videos remain readable but are not new reusable source members.
- `tests/test_compare_sources_api.py` — legacy `/api/compare-sources/{gt,pred,flow,mask}` remains server-resolved, rejects client paths, and cannot expose invalid/unbound outputs as new primary sources.
- `tests/test_compare_ui_hooks.py` — GT-first group/Item/Member picker, one-to-two Pred payload, Alignment Plan rendering, and original/aligned input URLs.
- `tests/test_db_indices.py` — `idx_artifacts_sample`, `idx_metric_results_sample`, and `idx_run_jobs_device` exist after `db.connect()`.
- `tests/test_sample_api_scope.py` — `/api/runs/{id}/samples/{sample_id}` and `/videos/{name}/timeline` issue O(sample) / O(video) SQL queries, not full-run scans (asserted via `sqlite3.set_trace_callback`).
- `tests/test_media_catalog_uploads.py` — migration backup, idempotent folder backfill, `source_assets`, resumable/hash-checked ZIP upload, malicious ZIP rejection, HTTP Range, soft deletion, and aligned-GT provenance.
- `tests/test_artifact_profiles.py` — evaluation/diagnostic/benchmark outputs, bounded save queue, batched artifact writes, segment balancing, finalize scheduling, and optional NPU utilization parsing.
- `tests/test_evaluation_campaigns.py` — identity hiding, deterministic side randomization, vote upsert, 20 concurrent evaluators, coverage/provisional state, Bradley–Terry/bootstrap analysis, filters, closing, and protected media history.
- `tests/test_run_cleanup.py` — persistent delete/cleanup, restart recovery, path safety, shared cache refs/leases/grace, storage preview/confirmed GC, media invalidation, and Campaign preservation.
- `tests/test_run_result_freshness_ui.py` — `content_revision` invalidation, stale-response generations, scoped manual refresh, generating skeleton, and terminal empty state.
- `tests/test_evaluation_campaigns_v2.py` — Item-first two-method coverage, per-Item normalization/fingerprint, atomic frozen Members/packages, Run-delete survival, opaque assignments/media, vote caps, result reveal, method-level analysis, and v1 read-only compatibility.
- `tests/test_evaluation_studio_ui.py` — GT group -> Items -> method coverage wizard, alignment summaries, preparation controls, Media partitions, and isolated blind participant page wiring.
