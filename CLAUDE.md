# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VFIEval is a local video frame interpolation (VFI) evaluation platform. It runs inference on user-provided model files against video datasets, performs post-processing (warp/blend/pred from flow+mask outputs), computes quality metrics, and serves results through a web UI. It does not train models and does not implement PSNR.

V13 makes Compare a first-class evaluation surface: GT and Pred are picked from server-resident resources (`videos/` groups and completed-run pred artifacts), one GT can be compared against multiple Pred tracks (`predA`, `predB`, ...), `flow`/`mask`/`warp` layers can be shown side-by-side, and sample/video APIs no longer materialize the entire run timeline per request. Linux NPU and CUDA servers are the only supported deployment targets in V13; Windows-specific hardening is deferred.

## Commands

```powershell
# Run tests
python -m unittest discover -s tests

# Run a single test
python -m unittest tests.test_end_to_end

# Compare-flow tests (V13)
python -m unittest tests.test_compare_multitrack tests.test_compare_sources_api tests.test_db_indices

# Run feedback tests (V13)
python -m unittest tests.test_run_feedback

# Start the server (main development workflow)
$env:PYTHONPATH='src'
python -m vfieval.cli init --workspace .vfieval
python -m vfieval.cli --workspace .vfieval serve --host 127.0.0.1 --port 8765

# Check metric asset status
python -m vfieval.cli --workspace .vfieval prepare-metrics --check-only

# Generate test video assets
python scripts\generate_test_assets.py

# Lint check (no dedicated linter configured, but check whitespace)
git diff --check
```

Set `$env:PYTHONPATH='src'` before running any vfieval commands outside of `pip install -e .`.

## Architecture

### Core Loop

1. **Web UI** (`src/vfieval/server.py`) — ThreadingHTTPServer serving a REST API and static files from `src/vfieval/web/`. The `POST /api/runs` endpoint is the main entry point for creating inference runs.

2. **Run Creation** (`server.py:_create_run_from_files`) — Resolves model file, video group, checkpoint, and device/precision. Upserts model and dataset records, scans video frames into triplet samples, creates a Run with linked inference Job(s), then spawns a local worker thread.

3. **Worker** (`src/vfieval/worker.py`) — Polls the SQLite job queue, claims jobs, dispatches to inference or metric runners. Supports `--device-filter` for multi-GPU/NPU sharding.

4. **Inference Pipeline** (`src/vfieval/pipeline/inference.py`) — Loads model via adapter, iterates samples in batches, calls `model.predict(img0, img1, 0.5)`, normalizes outputs, runs post-processing, saves visual artifacts (pred/gt/diff/flow/mask/warp/blend), assembles video artifacts.

5. **Post-processing** (`src/vfieval/pipeline/postprocess.py`) — Fixed pipeline: grid_sample warp → sigmoid mask → blend → pred. Models only return flow and mask logits.

6. **Metrics** (`src/vfieval/pipeline/metrics_runner.py`) — After inference completes, a metric job evaluates lpips_vit_patch, lpips_convnext, vmaf, or cgvqm against pred/gt pairs.

### Key Abstractions

- **Model Adapter** (`src/vfieval/models/loader.py`) — Loads models from `file:path.py` (user model files in `models/`), `module:factory`, or `"dummy"`. User models define `class Model` with `infer(img0, img1)` returning a dict or 4-tuple of `(flowt_0, flowt_1, mask0, mask1)`.

- **Database** (`src/vfieval/db.py`) — Single SQLite file at `.vfieval/vfieval.sqlite` with WAL mode. Tables: models, datasets, samples, jobs, artifacts, metric_results, metric_cache, experiments, runs, run_jobs, workers, run_feedback.

- **WorkspaceConfig** (`src/vfieval/config.py`) — Resolves all paths under `.vfieval/` (db, artifacts, runs, tmp).

- **Devices** (`src/vfieval/devices.py`) — Handles CUDA, NPU (torch_npu), and CPU device resolution, autocast, and precision support detection.

### Multi-GPU/NPU Execution

`execution_mode=multi_cuda|multi_npu` partitions samples by video across devices. Each shard becomes a separate inference job with `device_filter`. NPU shards spawn independent worker processes; CUDA shards use local threads.

### File Layout Conventions

- `models/*.py` — User model files (scanned by UI)
- `checkpoints/{model_stem}/` — Model weights
- `videos/{group_name}/` — Video datasets grouped by style/source
- `set/metrics/` — Metric asset manifests and weights
- `.vfieval/` — Workspace (SQLite DB, run outputs, decoded frame cache)

### Model Contract

Inputs: `img0, img1` are `[B,3,H,W]` RGB float tensors in `[0,1]`, already on device with correct dtype.

Outputs: dict with `flowt_0 [B,2,h,w]`, `flowt_1 [B,2,h,w]`, `mask0 [B,1,h,w]`, `mask1 [B,1,h,w]` (logits, not sigmoid). Flow is backward flow in pixel coordinates. Output resolution may differ from input — platform rescales.

### Supported Metrics

`lpips_vit_patch`, `lpips_convnext`, `vmaf`, `cgvqm` — defined in `src/vfieval/metrics/names.py`. Frame-level metrics (lpips_*) produce per-sample timeline curves. Video-level metrics (vmaf, cgvqm) produce per-video summaries only.

### Inference Pipeline Concurrency

`run_inference_job` uses three overlapping stages so the device does not sit idle waiting for CPU work:

- **Prefetch pool** (`_iter_prefetched_batches`): a small `ThreadPoolExecutor` (default 2 workers, configurable via `payload["prefetch_workers"]`) decodes `img0`/`img1` (and optional GT) via PIL on CPU, resizes to inference resolution, and holds up to 2 batches ahead. `pin_memory()` is applied when CUDA is available.
- **Main loop**: `.to(device, non_blocking=True)` → `model.predict` → `compose_interpolated` → single `.to("cpu")` for the whole bundle → submit to save pool. No per-sample `.cpu()` calls remain in the hot path.
- **Save pool** (`_AsyncSavePipeline`): a `ThreadPoolExecutor` sized `min(8, cpu_count())` (configurable via `payload["save_workers"]`) does PNG encoding, `preview/` thumbnails, and `db.add_artifact` inserts in the background. Each save-worker thread opens its own short-lived sqlite3 connection so WAL-mode file locking is sufficient — no cross-thread `threading.Lock` is needed.
- Progress updates are throttled to every `max(1, total // 200)` samples to keep SQLite writes off the hot path.

### Model Load + Output Health Diagnostics

- `vfieval.models.utils.load_state_dict_portable` loads with `strict=False`, returns a structured report (`checkpoint_path`, `matched`, `total_in_checkpoint`, `missing_keys`, `unexpected_keys`), and attaches it to `module._last_load_report`.
- `file_inputs._dry_run_model_file` returns `{output_health, model_load}`. Output-health checks flow `abs_max < 1e-4` and mask `std < 1e-3` to detect the "checkpoint not loaded" failure mode (`sigmoid(0) = 0.5` produces a mid-gray blend). NaN outputs are also flagged.
- `preflight_run` promotes these into `warnings` and the `model` field. Mismatched checkpoint keys surface as a `CheckpointLoadReport` warning with truncated missing/unexpected key lists.
- `run_inference_job` extracts `_last_load_report` from the loaded model, writes `run_dir/logs/model_load.log`, and includes `model_load` in `result_json`. The Run Detail UI renders a summary panel and warns on any mismatch.

### Compare API Surface (V13)

Compare uses server-resident sources only. Raw string descriptors are rejected — `resolve_compare_descriptor` requires a dict with a `kind` field.

- `GET /api/compare-sources/gt` — enumerates `videos/{group}/*` clips via `file_inputs.list_video_groups`. Each row: `{ group, video, path, frame_count, width, height, fps }`.
- `GET /api/compare-sources/pred[?run_id=...]` — enumerates `kind='pred_video'` artifacts from completed (non-cleaned) runs via `db.list_runs` + `db.list_run_artifacts`. Each row: `{ run_id, run_name, video, artifact_id, frame_count, width, height, fps, created_at }`.
- `GET /api/compare-sources/{flow,mask}?run_id=...&video=...` — enumerates per-video `flowt_*` / `mask*` / `warp*` / `blend` artifact groups for the picker.

Structured Compare payload accepted by `POST /api/runs` (`run_type: "video_compare"`):

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

Descriptor resolution lives in `compare_inputs.resolve_compare_descriptor(workspace, db, descriptor)` — the server never trusts a client-supplied `path` for Pred / flow / mask sources, and raw string descriptors are rejected with `"compare source descriptor must be an object with a 'kind' field"`. Multi-track Compare writes `pred.mp4` / `diff.mp4` under `.vfieval/runs/{run_id}/videos/{video_name}/{track_label}/`; the shared `gt.mp4` lives one level up. Sample names encode `{video_stem}__{track_label}__{frame_index:06d}` so per-track timelines work without schema changes.

### Timeline And Sample API Performance (V13)

- `_run_videos`, `_run_video_timeline`, and `_run_sample_payload` read only the rows they need — they do not call `_run_timeline()` internally.
- Required indices in `db.py` schema string:
  - `idx_artifacts_sample` on `artifacts(sample_id, kind)`
  - `idx_metric_results_sample` on `metric_results(sample_id, metric_name)`
  - `idx_run_jobs_device` on `run_jobs(device)`
- Sample- and video-level handlers use `db.list_artifacts_by_sample`, `db.list_metrics_by_sample`, and `db.list_samples_by_video`. `/api/runs/{id}/timeline` remains as a compatibility/debug endpoint and returns `X-Deprecated: use /videos`.

### Multi-Group Inference (V13)

One inference task can span multiple `videos/` groups. `datasets._resolve_video_entries` / `VideoEntry` and `file_inputs.resolve_video_selection` accept either a single `video_group` (legacy) or a `video_groups` list. Single-group runs stay byte-identical to legacy behavior — dataset root is the group folder, clip names stay bare — so cache and reference keys remain compatible. Only multi-group runs root at `videos/` and qualify clips as `group/file`. The infer form sends `video_group` (single) or `video_groups` (multi); the frontend uses a `#video-group-picker` multi-checkbox with per-group video tables. Default run name is `model-checkpoint-videogroup` (`server._default_run_name`).

### Run Feedback + Statistics (V13)

Runs carry user feedback (rating 1–5 and/or free-text issue). Table `run_feedback` (id, run_id FK CASCADE, username, rating INT nullable, issue TEXT, metadata_json, created_at) is in both SCHEMA and `_migrate`, indexed by `idx_run_feedback_run(run_id, created_at)`.

- DB methods: `add_run_feedback`, `list_run_feedback(run_id)`, `delete_run_feedback(run_id, feedback_id)` (scoped by run_id), `list_all_feedback`, `feedback_stats` (overall + rating_distribution + by_user + by_run).
- Server: `_create_run_feedback` validates rating 1–5 and requires a rating or issue; `_feedback_overview` wraps `feedback_stats` and adds recent entries. Routes: `POST/GET /api/runs/{id}/feedback`, `DELETE /api/runs/{id}/feedback/{fid}`, `GET /api/feedback`. `_run_detail` includes `run["feedback"]`.
- Frontend: `renderRunFeedback` panel in run detail, a "统计" nav view (`#view-stats`) with `loadStats`/`renderStats`, plus `submitRunFeedback`/`deleteRunFeedback`. Tests in `tests/test_run_feedback.py`.

### Compare Track-Label Dedup (V13)

Sample names are `{video_token}__{track_token}__{frame}` with `UNIQUE(dataset_id, name)` + `INSERT OR REPLACE`. Two selected preds sharing a sanitized track token silently overwrote each other (symptom: 1 GT + 1 pred instead of 2). `server._dedupe_track_labels(distorted_tracks)` bumps colliding labels to `{base}#{index+1}` (dedup on the sanitized token, not the raw label) before building `compare_tracks`. Regression test in `tests/test_compare_multitrack.py`.

## Testing Notes

Tests use `unittest` and create temporary workspaces with small synthetic images (PIL). The test suite imports from `src/` by inserting the path — no install needed for tests. Tests in `tests/` cover end-to-end inference+metrics, the v2 run API, video datasets, file-based flow, postprocessing, and metrics.

Model test files in `models/` (e.g., `test_average.py`, `test_img0.py`) are both test fixtures and real model files the UI scans.
