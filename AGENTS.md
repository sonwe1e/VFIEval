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

## Current Priorities

V13 work should prioritize, in order:

- Server-resident Compare sources: list `videos/{group}/` GT clips and `kind='pred_video'` artifacts from completed runs through `/api/compare-sources/{gt,pred,flow,mask}`, so the primary UI no longer requires typed local paths.
- Multi-track Compare: one GT vs. N distorted tracks (each from a different Run's pred artifact), labeled per-track. Strict alignment validates each `(gt, distorted_i)` pair independently — any misaligned distorted fails the run.
- Side-by-side flow / mask / warp layers in the Compare Run Detail, laid out as a `kind × track` grid with an explicit column cap to prevent the multi-track view from going off-screen.
- Sample/video-level APIs that do not depend on a full-run timeline load: `_run_videos`, `_run_video_timeline`, and `_run_sample_payload` must read only the rows they need, backed by new `artifacts(sample_id, kind)` and `metric_results(sample_id, metric_name)` indices.
- Frontend polling and request hygiene: pause the 2-second runs poll when `document.hidden`, cancel in-flight sample/timeline requests with `AbortController` on selection change, surface preflight/run errors with a "Retry this Run" button bound to the existing `_retry_run`.

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
- Compare's primary entry must select GT from `videos/{group}/` and Pred from completed-run artifacts via the server APIs. Typed absolute paths remain only as a fallback under an "advanced" disclosure, never as the default form.
- A Compare run may carry multiple Pred tracks. Every Pred tile in the UI must display its `track_label` and `kind` so the user can tell which model produced which output.
- The server never trusts a `path` field sent by the client for Pred / flow / mask sources. The descriptor (`{kind, run_id, video, ...}`) is resolved server-side through `db.list_run_artifacts` and `file_inputs` helpers.
- Sample-level and video-level APIs (`/api/runs/{id}/videos`, `/videos/{name}/timeline`, `/samples/{id}`) must not iterate the entire run's samples / artifacts / metrics. Only the legacy `/api/runs/{id}/timeline` debug endpoint is allowed to materialize a full-run view.

## Primary User Flows

### Model Inference Flow

Users choose a model file, optional checkpoint, video group, video subset, output size, device, precision, and metrics. VFIEval decodes or reuses cached frames, runs the model, applies platform-owned post-processing, writes artifacts, runs available metrics, and shows timeline-centered results.

This flow must produce viewable `pred/gt/diff` artifacts when ground truth exists. It may also produce flow, mask, warp, blend, and `extra_*` artifacts.

### Direct GT/Pred Compare Flow

Users may compare existing GT and Pred outputs without running a model. The primary entry uses server-resident sources only:

- **GT** is a single source — one video file under `videos/{group}/` or a frame directory below it. Selection is two server-side picks (group, then video).
- **Pred** is one or more tracks. Each track points at a completed Run's `kind='pred_video'` artifact for a specific `video_name`, and carries a `track_label` (e.g. `ModelA`, `ModelB`) chosen by the user.
- **Extra layers** (`flow`, `mask`, `warp`) reuse the same per-track Run references. The platform reads each track's `flowt_*`, `mask*`, `warp*`, `blend` artifacts from `db.list_run_artifacts(run_id, kind=...)` and surfaces them as side-by-side tiles labeled by `(track_label, kind)`.

Strict alignment must validate every `(gt, distorted_i)` pair independently: any misaligned track fails the whole run with a per-track reason. Frame count, dimensions, and (when present) fps/timestamps must match. Do not silently truncate or offset external inputs. VFIEval-generated GT/Pred pairs are aligned by the run manifest; a mismatch there is a pipeline bug.

A typed-local-path fallback may exist under an "advanced" disclosure for ad-hoc cases, but it must not be the default form and must never appear in the primary Compare workflow.

### Metric Environment Flow

Metric dependencies, weights, native evaluator binaries, and config files should be portable through `set/metrics/`. Missing assets must be reported as `unavailable` with a specific reason in SQLite and UI. Do not automatically download metric assets unless a future plan explicitly adds that behavior.

## File And Folder Entrypoints

- Models are discovered from `models/*.py`.
- Checkpoints are discovered from `checkpoints/{model_stem}/`.
- Video groups are discovered from `videos/*/`.
- Metric assets are discovered from `set/metrics/`.
- Run artifacts live under `.vfieval/runs/{run_id}/`.

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

## Compare API Surface (V13)

Compare picker and run creation use the following server-only endpoints. They are read-only against `db` + `file_inputs`; they never trust client-provided filesystem paths.

- `GET /api/compare-sources/gt` — enumerates `videos/{group}/*` clips. Each row carries `{ group, video, path, frame_count, width, height, fps }`. Backed by `file_inputs.list_video_groups` plus the existing `_video_summary` helper.
- `GET /api/compare-sources/pred[?run_id=...]` — enumerates `kind='pred_video'` artifacts from completed (non-cleaned) runs. Each row carries `{ run_id, run_name, video, artifact_id, frame_count, width, height, fps, created_at }`. Metadata fields come from the artifact's `metadata_json`.
- `GET /api/compare-sources/flow?run_id=...&video=...` and `.../mask?...` — enumerate per-video flow / mask / warp artifact groups for a specific Run + video. Used to populate the optional extra layers in the picker.

Run creation accepts a structured Compare payload:

```json
{
  "run_type": "video_compare",
  "reference": { "kind": "video_group", "group": "anime", "video": "clip01.mp4" },
  "distorted": [
    { "kind": "run_artifact", "run_id": 12, "video": "clip01.mp4", "label": "ModelA" },
    { "kind": "run_artifact", "run_id": 17, "video": "clip01.mp4", "label": "ModelB" }
  ],
  "extra_layers": [
    { "source": "run_artifact", "run_id": 12, "kinds": ["flowt_0", "mask0"] }
  ],
  "metrics": ["lpips_vit_patch", "vmaf"]
}
```

The legacy string form (`reference: "<path>"`, `distorted: "<path>"`) remains accepted as a compatibility fallback for the advanced form, but is not surfaced as the default UI.

Server-side descriptor resolution lives in `compare_inputs.resolve_compare_descriptor(workspace, db, descriptor)`. It must reject any descriptor whose resolved disk path is not produced by `file_inputs` (for GT) or `db.list_run_artifacts` (for Pred / flow / mask). Path strings sent by the client are never used directly.

Compare run artifacts must be track-scoped on disk: `.vfieval/runs/{run_id}/videos/{video_name}/{track_label}/pred.mp4` and `.../diff.mp4`. The `gt.mp4` for that video is shared across tracks at `.vfieval/runs/{run_id}/videos/{video_name}/gt.mp4`. Each sample row's `name` encodes `{video_stem}__{track_label}__{frame_index:06d}` so per-track timeline filtering works without schema changes. Artifact metadata records `compare_track_label` and `compare_track_run_id` for every Pred / flow / mask / warp / blend artifact.

Strict alignment (`compare_inputs.validate_strict_alignment`) is applied per `(reference, distorted_i)` pair. The first failing track aborts the run with a per-track error message; partial Compare runs are not allowed.

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

## Compare UI Layout (V13)

- Multi-track Compare renders tiles in a `kind × track` grid. The user controls column count via a segmented control (2 / 3 / 4); overflow scrolls horizontally rather than reflowing tiles off-screen.
- Every tile must show its `track_label` and `kind` chip. Default `extra_*` group stays collapsed.
- Video tiles use `<video controls playsinline preload="metadata">` and remain paused on load. A master play control synchronizes `currentTime` and `play()` across all tiles in the active group so multi-track videos do not drift.
- Hovering a tile highlights the corresponding frame on the timeline; clicking a timeline point updates every tile in the grid.

## NPU And Multi-Device Rules

NPU multi-device inference targets Ascend `torch_npu` and uses single-machine shard workers, not DDP.

- Use device ids like `npu:0`, `npu:1`, and `npu:7`.
- Each worker should own one NPU device and process a video-level or segment-level shard.
- Each NPU worker must call `torch.npu.set_device(index)` before model creation, dry-run, tensor staging, or inference.
- Preflight must run on the selected target device and catch CPU/NPU mismatch before the user starts a long run.
- Do not move tensors back to CPU during post-processing except for artifact encoding or metric input boundaries.
- CUDA support remains; multi-device priority is Ascend NPU, with CUDA as the secondary supported target.
- Cross-machine workers are future work and should use HTTP registration, claim, heartbeat, progress, complete, and fail APIs rather than direct SQLite access.

## Metrics

Only `lpips_vit_patch`, `lpips_convnext`, `vmaf`, and `cgvqm` are valid metrics. Missing native evaluator assets, dependencies, weights, commands, config files, or bindings must produce `unavailable`, never substitute another score.

Per-sample metrics may be plotted as timeline curves. Video-level metrics such as VMAF or CGVQM must be shown as video-level summaries unless the adapter produces real per-sample values. Do not create fake per-frame points from video-level scores.

Metric cache keys must include metric name, adapter version, metric config, reference identity, and distorted identity. Reopening a Run Detail page must read SQLite and artifacts only; it must not trigger metric recomputation.

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

A clean checkout must include generated test models and generated test videos. Every run must produce viewable `pred/gt/diff` artifacts when ground truth exists. Every failed run must show a human-readable error in UI.

Run deletion and cleanup must only affect `.vfieval/runs/{run_id}` and related metadata. Do not delete source model files, source checkpoints, source videos, decode cache shared by other valid runs, or `set/metrics/` assets.

Failed, canceled, invalid, and test runs must be removable from the UI. A canceled or failed run should not leave a partial output that the UI presents as complete.

## Architecture

Keep SQLite as metadata and index storage. Keep artifacts on disk. Prefer dependency-light changes unless a dependency materially improves video decoding, NPU reliability, metric correctness, or artifact portability.

Primary execution remains split between the Web/API control plane, inference workers, and metric workers. Inference failure must not erase already-readable logs. Metric failure must not block viewing completed inference artifacts.

Future remote workers should be implemented through explicit HTTP worker APIs. Do not let remote workers mutate SQLite directly.

## Codex Maintenance Workflow

Every Codex session working on VFIEval should read this `AGENTS.md` before planning or modifying code. When a task reveals a stable project rule, deployment constraint, or recurring failure mode, update this file in the same change set after tests pass. Keep these updates concise and specific to VFIEval; do not turn this file into a changelog.

## V13 Execution Checklist

1. Preserve all V11 and V12 work: NPU sharding, preview artifacts, delete/cleanup, checkpoint discovery, per-sample error capture, byte-range artifact streaming, and timeline grouping must not regress.
2. Ship `/api/compare-sources/{gt,pred,flow,mask}` and the structured Compare payload before touching the picker UI; the new UI must consume the API rather than the legacy string form.
3. Make multi-track Compare reach `.vfieval/runs/{run_id}/videos/{video_name}/{track_label}/pred.mp4` on disk and surface labeled `(track, kind)` tiles in Run Detail. Strict alignment runs per track; any failure aborts the whole run with a per-track reason.
4. Add the three new indices (`idx_artifacts_sample`, `idx_metric_results_sample`, `idx_run_jobs_device`) to `db.py` and rewrite `_run_videos` / `_run_video_timeline` / `_run_sample_payload` so they no longer call `_run_timeline` internally.
5. Pause the frontend runs poll on `document.hidden`, abort stale sample/timeline requests with `AbortController`, and bind the `run-error-banner` retry to `_retry_run`.
6. Keep the primary UI clean: no model registration, dataset registration, raw jobs, or experiment administration in the first-run workflow. The Compare advanced fallback for typed local paths stays collapsed.

## Future Roadmap

- V14 should add remote worker orchestration through HTTP worker lifecycle APIs.
- V15+ may add portable workspace bundles, reproducible evaluation packages, and cross-machine artifact sync.
- Windows-specific hardening (path-traversal nuances, encoding quirks, `_read_json` size limits, stuck-job reaper) is deferred until the V13 priorities ship. Linux NPU and CUDA servers are the only supported deployment targets in V13.

## Testing

Run `python -m unittest discover -s tests` and `git diff --check` before finalizing code changes. If only `AGENTS.md` changes, `git diff --check` and manual review are sufficient.

Coverage should include file discovery, checkpoint discovery, preflight, model interface failures, NPU device selection, CPU/NPU mismatch errors, video decode/cache behavior, post-processing contracts, run lifecycle, run deletion, artifact grouping, metric unavailable behavior, timeline data, direct GT/Pred comparison, and UI-relevant lazy result display.

V13 additions to keep covered:

- `tests/test_compare_multitrack.py` — end-to-end Compare run with one GT and two Pred tracks. Asserts per-track `pred.mp4` / `diff.mp4` on disk, shared `gt.mp4`, sample names encode `{video}__{track}__{frame}`, and per-track metrics land in `metric_results` with `compare_track_label` metadata.
- `tests/test_compare_sources_api.py` — `/api/compare-sources/{gt,pred,flow,mask}` returns server-resident rows only and rejects any client-supplied `path`.
- `tests/test_db_indices.py` — `idx_artifacts_sample`, `idx_metric_results_sample`, and `idx_run_jobs_device` exist after `db.connect()`.
- `tests/test_sample_api_scope.py` — `/api/runs/{id}/samples/{sample_id}` and `/videos/{name}/timeline` issue O(sample) / O(video) SQL queries, not full-run scans (asserted via `sqlite3.set_trace_callback`).
