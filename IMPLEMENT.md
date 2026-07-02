# IMPLEMENT

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
