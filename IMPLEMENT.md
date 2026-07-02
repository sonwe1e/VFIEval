# IMPLEMENT

## [2026-07-03 01:18] CGVQM and Run Detail Usability Fixes

Discussed:
- The screenshot showed CGVQM unavailable because the wrapper stopped at `demo_cgvqm`, metric health panels consumed too much vertical space, video artifacts rendered as tiny black bars, and the metric curve looked too heavy.

Implemented:
- Updated the generated CGVQM wrapper and the current local wrapper to call IntelLabs `run_cgvqm(distorted, reference, cgvqm_type=CGVQM_2, device=..., patch_pool="max", patch_scale=4)`.
- Collapsed Metric Health behind a details disclosure while keeping a short unavailable summary visible.
- Moved Run Detail video artifacts into a dedicated strip and restored 16:9 video sizing.
- Reworked the metric chart with light grid lines, min/max labels, a thinner curve, and a clearer current-frame marker.

Files changed:
- `src/vfieval/metrics/health.py`
- `set/metrics/cgvqm/run_cgvqm_vfieval.py`
- `src/vfieval/web/app.js`
- `src/vfieval/web/styles.css`
- `tests/test_metrics.py`
- `tests/test_compare_ui_hooks.py`
- `tests/test_v3_file_flow.py`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Passed targeted metric/UI/file-flow tests, bundled Node syntax check for `app.js`, `python -m py_compile src\vfieval\metrics\health.py`, full `python -m unittest discover -s tests`, and `git diff --check`.

## [2026-07-03 00:49] Metric Asset Downloads and Evaluation Resolution

Discussed:
- The previous implementation left missing weights unresolved because `prepare-metrics` only wrote templates.
- The metric setup now needs to download assets into `set/metrics/` and lock each evaluation scheme to a known resolution before scoring.

Implemented:
- Made `prepare-metrics` download default DINOv2, ConvNeXt V2, and CGVQM assets, write runnable manifests, skip existing declared assets, and support `--force`.
- Kept `prepare-metrics --check-only` read-only and kept Python dependency installation out of scope.
- Added source URLs, fixed evaluation resolution, pad multiple, normalization, and CGVQM video long-edge fields to health and cache config.
- Added CGVQM temporary resize-video preparation capped to long edge 720 without modifying original artifacts.

Files changed:
- `src/vfieval/metrics/health.py`
- `src/vfieval/metrics/feature.py`
- `src/vfieval/metrics/cgvqm.py`
- `src/vfieval/cli.py`
- `tests/test_metrics.py`
- `tests/test_v3_file_flow.py`
- `README.md`
- `AGENTS.md`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Passed changed-file `py_compile`, targeted metric/file-flow/end-to-end tests, full `python -m unittest discover -s tests`, stale-doc scan, and `git diff --check`.

## [2026-07-03 00:27] Remaining Metric Adapters

Discussed:
- The user decided not to implement per-triplet VMAF and asked to complete the remaining metrics instead.
- The agreed defaults were DINOv2 ViT-S/14 registers for `lpips_vit_patch`, timm ConvNeXt V2 tiny for `lpips_convnext`, a lightweight local wrapper for IntelLabs CGVQM, local-only assets, and no silent CPU fallback when CUDA/NPU metric execution fails.

Implemented:
- Added internal feature-distance adapters for `lpips_vit_patch` and `lpips_convnext`, plus a CGVQM wrapper adapter that reads the existing manifest command contract.
- Reworked metric health and `prepare-metrics` manifests so DINOv2, ConvNeXt, and CGVQM report missing local repos, weights, Python packages, and wrapper commands clearly.
- Propagated `metric_device` from inference runs into metric jobs, cache keys, and metric adapter construction.
- Updated tests and docs for missing-assets behavior, CGVQM wrapper execution, cache invalidation, device failure handling, sample/video metric placement, and clean-checkout unavailable results.

Files changed:
- `src/vfieval/metrics/health.py`
- `src/vfieval/metrics/registry.py`
- `src/vfieval/metrics/feature.py`
- `src/vfieval/metrics/cgvqm.py`
- `src/vfieval/pipeline/metrics_runner.py`
- `src/vfieval/pipeline/inference.py`
- `src/vfieval/db.py`
- `src/vfieval/server.py`
- `tests/test_metrics.py`
- `tests/test_v3_file_flow.py`
- `tests/test_end_to_end.py`
- `README.md`
- `AGENTS.md`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Passed changed-file `py_compile`, passed `python -m unittest discover -s tests`, passed `git diff --check`, archived file backups, and confirmed no stray backup files remained outside the archive flow.

## [2026-07-03 00:04] Backup Archive Cleanup

Discussed:
- The repository root, `tests/`, and `src/` had accumulated many `*.backup.*` files from earlier sessions, which made real source files harder to scan.
- The committed `test_*` models and videos are intentional fixtures and should stay where the clean-checkout tests expect them, but local backup files needed a better long-term home.

Implemented:
- Added `scripts/archive_file_backups.py` to collect scattered `*.backup.*` files, move them into `archive/file_backups/{timestamp}/`, preserve repository-relative paths, and write a `manifest.json` for the archived session.
- Updated `REPO_LAYOUT.md` and `AGENTS.md` so future sessions keep committed `test_*` fixtures in place and archive backup files instead of leaving them beside source files.
- Used the new archive flow to move the current repository-local backup files out of the working source and test directories.

Files changed:
- `scripts/archive_file_backups.py`
- `REPO_LAYOUT.md`
- `AGENTS.md`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Archived 55 scattered backup files into `archive/file_backups/20260703_000417/`, verified `python scripts/archive_file_backups.py --dry-run` reports `moved_count = 0`, passed `python -m unittest discover -s tests`, and passed `git diff --check`.

## [2026-07-02 23:59] Portable Metric Driver and VMAF Smoke Path

Discussed:
- The metric stack still treated LPIPS/CGVQM as "available in theory" while the shipped adapter always raised `unavailable`, so the build needed a real portable execution contract instead of a placeholder native-binding story.
- VMAF needed to become the immediately usable path in the current build, including a direct smoke command and clearer health diagnostics about which `ffmpeg` binary is actually being used.

Implemented:
- Reworked `src/vfieval/metrics/health.py` around manifest-aware checks. LPIPS/CGVQM now validate `driver.command`, `required_files`, optional `env`, manifest shape, resolved executable, and cache fingerprints. VMAF now inspects an optional `set/metrics/vmaf/manifest.json`, resolves `ffmpeg_path` before `PATH`, and reports `implementation_mode`, `manifest_path`, `driver_command`, and `resolved_executable`.
- Replaced the placeholder `NativeMetric` behavior with a real manifest-command runner in `src/vfieval/metrics/native.py`, wired the registry to it, and added `vfieval smoke-metric --metric ... --reference ... --distorted ...` in `src/vfieval/cli.py`.
- Updated `src/vfieval/metrics/vmaf.py` to reuse the resolved health config and fixed Windows `libvmaf` log-path escaping so real `ffmpeg` runs can write JSON logs correctly.
- Updated the metric environment UI and documentation to show the richer diagnostics and explain that VMAF is the first immediately runnable metric while LPIPS/CGVQM require a manifest + driver + assets under `set/metrics/`.
- Added metric-focused tests for manifest health mapping, manifest-driver success/unavailable/failed stdout protocol, driver fingerprint cache invalidation, VMAF manifest overrides, CLI smoke execution, and file-flow metric metadata/API behavior.

Files changed:
- `src/vfieval/metrics/health.py`
- `src/vfieval/metrics/native.py`
- `src/vfieval/metrics/registry.py`
- `src/vfieval/metrics/vmaf.py`
- `src/vfieval/cli.py`
- `src/vfieval/web/app.js`
- `tests/test_metrics.py`
- `tests/test_v3_file_flow.py`
- `README.md`
- `AGENTS.md`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Passed targeted metric tests, passed `python -m unittest discover -s tests`, and passed `git diff --check`.

## [2026-07-02 23:16] 4K Test Fixture and Larger Dry-Run Probe

Discussed:
- The repository needed a built-in 4K video fixture so discovery and preflight can cover UHD metadata without pushing the default `test_style` path through 4K assets.
- The preflight dry-run probe was still using an `8x8` input, which was too small for some shape-sensitive model checks.

Implemented:
- Added a new generated test video group `videos/test_4k/` with one short `3840x2160` `mp4v` clip.
- Changed the dry-run probe input in `src/vfieval/file_inputs.py` from `(1, 3, 8, 8)` to `(1, 3, 128, 128)`.
- Added the probe shape to the dry-run cache key so the larger probe invalidates older in-process cache entries automatically.
- Extended `tests/test_v3_file_flow.py` to assert the larger probe reaches platform post-processing and that `test_4k` is discovered with UHD metadata and original-resolution preflight output.

Files changed:
- `scripts/generate_test_assets.py`
- `src/vfieval/file_inputs.py`
- `tests/test_v3_file_flow.py`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Generated the `videos/test_4k/` fixture, passed targeted dry-run and discovery tests, passed `python -m unittest discover -s tests`, and passed `git diff --check`.

## [2026-07-02 15:35] Runtime Output Diagnostics and Shard Failure Cleanup

Discussed:
- A completed run showed checkpoint loading success, but flow/mask previews still looked missing or empty.
- Local inspection showed the test checkpoint model produced zero flow and constant masks by design, so the platform needed runtime evidence that separates "checkpoint loaded" from "outputs are useful".

Implemented:
- Added `output_health` stats to completed model-inference results, computed from real inference frame bundles already on CPU.
- Wrote the same diagnostics to `logs/output_health.log` and surfaced the summary in Run Detail.
- Canceled queued sibling shard jobs after a run fails, and made already-running sibling shards stop at the next cancellation check.
- Created run directories and metadata before marking inference jobs as running, closing a cleanup race where UI could see `running` before `output_dir` existed.
- Added a request body size guard and generic 500 responses so raw server exception text is not sent to clients.

Files changed:
- `src/vfieval/pipeline/inference.py`
- `src/vfieval/db.py`
- `src/vfieval/server.py`
- `src/vfieval/web/app.js`
- `tests/test_v3_file_flow.py`
- `AGENTS.md`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Targeted tests and diff hygiene checks passed.

## [2026-07-01 23:20] Repository Layout and Git Rules

Discussed:
- The project folder structure was getting hard to reason about, especially which files are source, fixtures, local inputs, runtime output, backups, or private assets.

Implemented:
- Added `REPO_LAYOUT.md` with folder roles, naming rules, Git ownership rules, runtime-state rules, and check-ignore examples.
- Updated `.gitignore` so generated runtime state, real local inputs, metric assets, automatic backups, SQLite files, and local tool state stay out of Git.
- Kept generated test models, test checkpoints, and test videos visible to Git.
- Added a short AGENTS pointer so future VFIEval sessions keep repository hygiene aligned with `REPO_LAYOUT.md`.

Files changed:
- `.gitignore`
- `REPO_LAYOUT.md`
- `AGENTS.md`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Ignore behavior and diff hygiene checks passed.

## [2026-07-01 21:11] Usability and Efficiency Pass

Discussed:
- The project needed faster first-screen loading, better video and Compare selection, lower preflight churn, fewer Run Detail timeline queries, and updated docs.

Implemented:
- Added lightweight `GET /api/video-groups?summary=1` and kept full video details on the existing video-group videos endpoint.
- Added Compare source pagination and filters for GT and Pred sources.
- Added chunked batched artifact and metric DB helpers, then rewired video timeline payloads so overview uses batched metrics and artifact payloads stay scoped to the current window.
- Updated the web UI with video search/sort/pagination, lazy Compare source loading, editable track labels, default-collapsed extra layers, stale preflight aborts, and cached preflight payload keys.
- Updated README and CHANGELOG.

Files changed:
- `src/vfieval/file_inputs.py`
- `src/vfieval/server.py`
- `src/vfieval/db.py`
- `src/vfieval/web/app.js`
- `src/vfieval/web/styles.css`
- `tests/test_compare_sources_api.py`
- `tests/test_compare_ui_hooks.py`
- `tests/test_sample_api_scope.py`
- `README.md`
- `CHANGELOG.md`
- `IMPLEMENT.md`

Status:
- Complete. Full unittest suite and diff hygiene checks passed.
