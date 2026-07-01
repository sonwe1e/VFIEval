# VFIEval Repository Layout

This file defines what each top-level folder is for, how files should be named, and what belongs in Git.

## Source And Project Files

Commit these:

- `src/`: VFIEval package source, web assets, workers, pipeline, metrics, and server code.
- `tests/`: unit and integration tests.
- `scripts/`: small project scripts, including generated test asset scripts.
- `examples/`: example adapters and helper modules.
- Root project files: `.gitignore`, `AGENTS.md`, `CHANGELOG.md`, `CLAUDE.md`, `IMPLEMENT.md`, `README.md`, `REPO_LAYOUT.md`, and `pyproject.toml`.

Do not put generated runtime output in these folders.

## File Entry Points

VFIEval discovers user-facing inputs from fixed project folders:

- `models/*.py`: model adapters shown in the model picker.
- `checkpoints/{model_stem}/`: checkpoint files for a matching model stem.
- `videos/{group}/`: source videos grouped by style, project, or test set.
- `set/metrics/`: local metric manifests, weights, and evaluator assets.

The clean checkout keeps only generated test inputs:

- `models/test_*.py` and `models/_test_helpers.py`.
- `checkpoints/test_*/`.
- `videos/test_*/`.

Real model adapters, private videos, real checkpoints, and metric assets stay local by default. If a large asset must be shared, use a release artifact or Git LFS after an explicit decision.

## Runtime State

These folders and files are local only:

- `.vfieval/`: the default runtime workspace.
- `.vfieval-*/`: ad hoc test workspaces.
- `.vfieval/runs/{run_id}/`: run artifacts, logs, videos, previews, and per-sample images.
- `.vfieval/decode_cache/`, `.vfieval/tmp/`, `.vfieval/video_meta_cache/`, and `.vfieval/video_thumbnails/`.
- SQLite files such as `vfieval.sqlite`, `*.sqlite-wal`, and `*.sqlite-shm`.
- `archive/`: local project-level backup zips.
- `*.backup.YYYYMMDD_HHMMSS`: automatic file-level backups.
- `.env*`, `.agents/`, and `.claude/settings.local.json`.

Do not move runtime state into tracked source folders.

## Naming Rules

- Use `snake_case` for folders and files that VFIEval lists in the UI.
- Reserve `test_*` names for generated fixtures that are safe to commit.
- Name model adapters as `models/{model_stem}.py`.
- Store checkpoints under `checkpoints/{model_stem}/latest.pth` or `checkpoints/{model_stem}/epoch_0001.pth`.
- Store videos under `videos/{group}/{clip_name}.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, or another supported suffix.
- Keep platform artifact names stable: `pred`, `gt`, `diff`, `flowt_0`, `flowt_1`, `mask0`, `mask1`, `warp0`, `warp1`, and `blend`.

Avoid spaces in new filenames. Existing edge-case fixtures may keep spaces or Unicode when they are testing path handling.

## Git Checklist

Before pushing layout or ignore changes:

```powershell
git diff --check
git status --short
git check-ignore -v models/my_model.py
git check-ignore -v videos/anime/clip.mp4
git check-ignore -v checkpoints/my_model/latest.pth
git check-ignore -v .vfieval/vfieval.sqlite
git check-ignore -v CHANGELOG.md.backup.20260701_000000
```

Use `git check-ignore --no-index -v` when checking a path that is already tracked, such as `models/test_average.py` or `videos/test_style/blocks_motion.avi`.
