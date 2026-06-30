# AGENTS.md

DO NOT send optional commentary.

## Project Scope

VFIEval is a local inference, post-processing, artifact viewing, evaluation, and comparison platform for video frame interpolation. Do not add training features. Do not implement PSNR.

## Hard UI Rules

Primary UI must not contain model registration or dataset registration forms. Primary UI must use files and folders as entry points. Models are discovered from `models/*.py`. Video groups are discovered from `videos/*/`. The first successful user flow must require no manual IDs.

## Core Contracts

Models receive resized RGB `img0` and `img1` tensors in `BCHW`, value range `[0, 1]`, fixed `t=0.5`, on the inference device and dtype. Model files should expose `class Model: infer(self, img0, img1)`. They must return at least `flowt_0`, `flowt_1`, `mask0`, and `mask1`; tuple return `(flowt_0, flowt_1, mask0, mask1)` is allowed.

`flowt_0` and `flowt_1` are backward flow in resized pixel coordinates. `mask0` and `mask1` are logits, not probabilities. Extra visualization tensors are allowed and may be saved as `extra_*`, but comparison must focus on flow, mask, warp, blend, pred, and diff artifacts.

## Post-Processing

The platform owns warp, sigmoid, blend, compose, visualization, artifact writing, and metric execution. Keep `grid_sample(mode="bilinear", padding_mode="border", align_corners=True)`, sigmoid masks, and `pred = mask1 * img1 + (1 - mask1) * blend` stable unless tests and docs are updated together.

## Artifacts

A clean checkout must include generated test models and generated test videos. Every run must produce viewable `pred/gt/diff` artifacts when ground truth exists. Every failed run must show a human-readable error in UI.

## Metrics

Only `lpips_vit_patch`, `lpips_convnext`, `vmaf`, and `cgvqm` are valid metrics. Missing native evaluator assets, dependencies, or bindings must produce `unavailable`, never substitute another score. The primary UI may show metric selection, metric health, metric curves, video-level metric summaries, and worst-sample navigation, but must not add PSNR.

## Timeline UI

Run Detail must stay timeline-centered and lazy-loaded. Do not render every video player or every artifact image at once. Primary UI should use paged video APIs and windowed timelines: `GET /api/runs/{id}/videos`, `GET /api/runs/{id}/videos/{video_name}/timeline`, and `GET /api/runs/{id}/samples/{sample_id}`. Keep `GET /api/runs/{id}/timeline` only as a compatibility/debug endpoint. Extra model visualizations may be shown behind a collapsed section, but comparison focuses on flow, mask, warp, blend, pred, and diff.

## Architecture

Keep SQLite as metadata/index storage and artifacts on disk. Preserve legacy `models/datasets/jobs/experiments` APIs for compatibility, but do not let those concepts pollute the primary UI. Future remote workers should use HTTP registration, claim, heartbeat, progress, complete, and fail APIs rather than direct SQLite access. Prefer dependency-light changes unless a dependency materially improves video decoding, worker reliability, or metric correctness.

## Testing

Run `python -m unittest discover -s tests` and `git diff --check` before finalizing changes. Cover file discovery, preflight, model interface failures, video decode/cache behavior, post-processing contracts, run lifecycle, artifact grouping, and UI-relevant result display.
