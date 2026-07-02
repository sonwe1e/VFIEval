# IMPLEMENT

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
