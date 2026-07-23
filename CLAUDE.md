# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Finding What To Change

Before reading source files to locate a change, consult `NAVIGATION.md`. It is a subsystem-indexed map: for each feature (Compare, Feedback, Inference, Metrics, Runs, ...) it lists the exact files, functions, line anchors, DB tables/methods, HTTP routes, and tests involved. Start there to jump straight to the relevant code instead of scanning the large files (`server.py`, `db.py`, `app.js` are each 2k–4.4k lines). Update `NAVIGATION.md` when you add or move a subsystem.

`README.md` describes product behavior for users; `AGENTS.md` defines stable contracts and working boundaries; `REPO_LAYOUT.md` defines folder policy. Keep this file for architecture prose and invariants, `NAVIGATION.md` for lookup — do not duplicate file/function-level detail here.

## Project Overview

VFIEval is a local video frame interpolation (VFI) evaluation platform. It runs inference on user-provided model files against video datasets, performs post-processing (warp/blend/pred from flow+mask outputs), computes quality metrics, and serves results through a web UI. It does not train models and does not implement PSNR.

Compare is a first-class evaluation surface: GT and Pred are picked from server-resident resources (Media Items, `videos/` groups, completed-run pred artifacts, uploaded media), one GT can be compared against up to two Pred tracks (`predA`, `predB`), and `flow`/`mask`/`warp` layers can be shown side-by-side. Sample/video APIs read only the rows they need; the full-run timeline endpoint exists only for compatibility/debug. Linux NPU and CUDA servers are the only supported deployment targets; Windows-specific hardening is deferred.

## Commands

```powershell
# Run tests
python -m unittest discover -s tests

# Run a single test
python -m unittest tests.test_end_to_end

# Compare-flow tests
python -m unittest tests.test_compare_multitrack tests.test_compare_sources_api tests.test_db_indices

# Run feedback tests
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

- **Database** (`src/vfieval/db.py`) — Single SQLite file at `.vfieval/vfieval.sqlite` with WAL mode. In addition to Run/job/artifact/metric tables it owns the media catalog, upload sessions, execution profiles, evaluator/Campaign/task/vote data, and repeatable schema migrations.

- **WorkspaceConfig** (`src/vfieval/config.py`) — Resolves all paths under `.vfieval/` (db, artifacts, runs, tmp, media, uploads, backups).

- **Devices** (`src/vfieval/devices.py`) — Handles CUDA, NPU (torch_npu), and CPU device resolution, autocast, and precision support detection.

### Multi-GPU/NPU Execution

`execution_mode=multi_cuda|multi_npu` partitions by video, then splits long videos into continuous sample segments when videos are insufficient or skewed. Each shard becomes an inference job with `device_filter`; shards write frames/manifests and one `finalize` job encodes videos before metrics. NPU shards spawn independent worker processes; CUDA shards use local threads.

### File Layout Conventions

- `models/*.py` — User model files (scanned by UI)
- `checkpoints/{model_stem}/` — Model weights
- `videos/{group_name}/` — Video datasets grouped by style/source
- `set/metrics/` — Metric asset manifests and weights (gitignored, produced by `prepare-metrics`)
- `.vfieval/` — Workspace (SQLite DB, run outputs, decoded frame cache)

### Model Contract

Inputs: `img0, img1` are `[B,3,H,W]` RGB float tensors in `[0,1]`, already on device with correct dtype.

Outputs: dict with `flowt_0 [B,2,h,w]`, `flowt_1 [B,2,h,w]`, `mask0 [B,1,h,w]`, `mask1 [B,1,h,w]` (logits, not sigmoid). Flow is backward flow in pixel coordinates. Output resolution may differ from input — platform rescales.

### Supported Metrics

`lpips_vit_patch`, `lpips_convnext`, `vmaf`, `cgvqm` — defined in `src/vfieval/metrics/names.py`. Frame-level metrics (lpips_*) produce per-sample timeline curves. Video-level metrics (vmaf, cgvqm) produce per-video summaries only. Metric assets are prepared by `prepare-metrics` into `set/metrics/`; health checks gate evaluation before any driver subprocess runs.

### Inference Pipeline Concurrency

`run_inference_job` uses three overlapping stages so the device does not sit idle waiting for CPU work:

- **Prefetch pool** (`_iter_prefetched_batches`): a small `ThreadPoolExecutor` (default 2 workers, configurable via `payload["prefetch_workers"]`) decodes `img0`/`img1` (and optional GT) via PIL on CPU, resizes to inference resolution, and holds up to 2 batches ahead. `pin_memory()` is applied when CUDA is available.
- **Main loop**: `.to(device, non_blocking=True)` → `model.predict` → `compose_interpolated` → single `.to("cpu")` for the whole bundle → submit to save pool. No per-sample `.cpu()` calls remain in the hot path.
- **Save pool** (`_AsyncSavePipeline`): a bounded `ThreadPoolExecutor` sized `min(8, cpu_count())` (configurable via `payload["save_workers"]`) does PNG encoding and previews. Artifact rows are buffered into batched SQLite transactions; queue backpressure bounds in-flight CPU tensors.
- Progress updates are throttled to every `max(1, total // 200)` samples to keep SQLite writes off the hot path.

### Model Load + Output Health Diagnostics

- `vfieval.models.utils.load_state_dict_portable` loads with `strict=False`, returns a structured report (`checkpoint_path`, `matched`, `total_in_checkpoint`, `missing_keys`, `unexpected_keys`), and attaches it to `module._last_load_report`.
- `file_inputs._dry_run_model_file` returns `{output_health, model_load}`. Output-health checks flow `abs_max < 1e-4` and mask `std < 1e-3` to detect the "checkpoint not loaded" failure mode (`sigmoid(0) = 0.5` produces a mid-gray blend). NaN outputs are also flagged.
- `preflight_run` promotes these into `warnings` and the `model` field. Mismatched checkpoint keys surface as a `CheckpointLoadReport` warning with truncated missing/unexpected key lists.
- `run_inference_job` extracts `_last_load_report` from the loaded model, writes `run_dir/logs/model_load.log`, and includes `model_load` in `result_json`. The Run Detail UI renders a summary panel and warns on any mismatch.

## Subsystem Invariants

Rules that are easy to break and not obvious from the code; for file/function-level detail see `NAVIGATION.md`.

- **Compare**: server-resident sources only — `resolve_compare_descriptor` requires a dict with a `kind` field and rejects raw string descriptors; the server never trusts a client-supplied `path` for Pred/flow/mask sources. Sample names encode `{video_stem}__{track_label}__{frame_index:06d}` under `UNIQUE(dataset_id, name)` + `INSERT OR REPLACE`, so colliding sanitized track tokens would silently overwrite each other — `server._dedupe_track_labels` bumps collisions on the sanitized token (not the raw label) before tracks are built. Multi-track Compare writes per-track `pred.mp4`/`diff.mp4` under `.vfieval/runs/{run_id}/videos/{video}/{track_label}/`; the shared `gt.mp4` lives one level up.
- **Scoped reads**: `_run_videos`, `_run_video_timeline`, and `_run_sample_payload` must not call `_run_timeline()` internally — they read only the rows they need. `/api/runs/{id}/timeline` stays as a compatibility/debug endpoint returning `X-Deprecated`. Required indices are listed in `NAVIGATION.md` §9; adding a core column or index requires editing both the schema string and `_migrate` in `db.py`.
- **Multi-group inference**: single-group runs stay byte-identical to legacy behavior (dataset root = group folder, bare clip names) so cache and reference keys remain compatible. Only multi-group runs root at `videos/` and qualify clips as `group/file`. The infer form sends `video_group` (single) or `video_groups` (multi).
- **Run feedback + statistics**: see `NAVIGATION.md` §1. Rating is 1–5 in 0.25 steps (rating or issue required), rows are content-scoped (video/track/model/checkpoint; Compare tracks chase back to the source Run), and run cleanup cascades feedback deletion.

## Testing Notes

Tests use `unittest` and create temporary workspaces with small synthetic images (PIL). The test suite imports from `src/` by inserting the path — no install needed for tests. Tests in `tests/` cover end-to-end inference+metrics, the v2 run API, video datasets, file-based flow, postprocessing, and metrics.

Model test files in `models/` (e.g., `test_average.py`, `test_img0.py`) are both test fixtures and real model files the UI scans.
