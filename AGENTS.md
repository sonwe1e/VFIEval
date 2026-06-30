# AGENTS.md

DO NOT send optional commentary.

## Project Identity

VFIEval is a local inference, post-processing, artifact viewing, evaluation, and comparison platform for video frame interpolation. It is a practical VFI evaluation tool, not a training framework and not a generic admin dashboard.

Do not add training features. Do not implement PSNR.

## Project History

- V3 moved the primary workflow to "files and folders as entry points": users place model files in `models/` and videos in `videos/{group}/`, then run inference without manual database registration.
- V5 made metric timelines and bad-frame navigation first-class goals: users should locate weak frames through curves, then inspect only the selected sample's artifacts.
- V11 added the foundation for NPU-aware device handling, run deletion and cleanup, preview artifacts, checkpoint discovery, and lightweight artifact grouping.
- V12 focuses on closing real deployment loops: Ascend NPU multi-device inference, portable metric assets, strict device consistency, direct GT/Pred video comparison, 4K-safe lazy previews, and clearer failure reporting.

## Current Priorities

V12 work should prioritize:

- Stable single-machine Ascend NPU inference, including multi-worker sharding across devices like `npu:0` through `npu:7`.
- Early detection of CPU/NPU device mismatch through preflight and worker-side checks.
- Metric curves and metric health that explain whether a score is completed, skipped, failed, or unavailable.
- Lightweight Run Detail pages that do not render all videos or all 4K artifacts at once.
- Direct GT/Pred comparison mode for already-generated outputs, without loading a model.
- Safe run deletion and cleanup for failed, invalid, and test runs.

Do not expand broad Compare, remote workers, auth, object storage, or new metric families until the V12 loops above are reliable.

## Non-Negotiable Scope

- Do not add training features. Do not implement PSNR.
- Primary UI must use files and folders as entry points.
- Primary UI must not contain model registration or dataset registration forms.
- The first successful user flow must require no manual IDs.
- Do not require users to type `model_id`, `dataset_id`, `job_id`, or `experiment_id` in primary workflows.
- Do not substitute unavailable metrics with a different score.
- Do not render all 4K artifacts at once.
- Every failed run must show a human-readable error in UI.

## Primary User Flows

### Model Inference Flow

Users choose a model file, optional checkpoint, video group, video subset, output size, device, precision, and metrics. VFIEval decodes or reuses cached frames, runs the model, applies platform-owned post-processing, writes artifacts, runs available metrics, and shows timeline-centered results.

This flow must produce viewable `pred/gt/diff` artifacts when ground truth exists. It may also produce flow, mask, warp, blend, and `extra_*` artifacts.

### Direct GT/Pred Compare Flow

Users may compare existing GT and Pred videos or frame directories without running a model. This flow does not load model files and does not require flow, mask, warp, or blend artifacts.

External GT/Pred inputs must be strictly aligned by frame count, dimensions, and available fps/timestamp metadata. Do not silently truncate or offset external inputs. VFIEval-generated GT/Pred pairs must be aligned by the run manifest; a mismatch there is a pipeline bug.

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

## NPU And Multi-Device Rules

NPU multi-device inference targets Ascend `torch_npu` and uses single-machine shard workers, not DDP.

- Use device ids like `npu:0`, `npu:1`, and `npu:7`.
- Each worker should own one NPU device and process a video-level or segment-level shard.
- Each NPU worker must call `torch.npu.set_device(index)` before model creation, dry-run, tensor staging, or inference.
- Preflight must run on the selected target device and catch CPU/NPU mismatch before the user starts a long run.
- Do not move tensors back to CPU during post-processing except for artifact encoding or metric input boundaries.
- CUDA support may remain, but V12 multi-device priority is Ascend NPU.
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

## V12 Execution Checklist

1. Preserve existing V11 work: NPU helpers, preview artifacts, delete/cleanup, checkpoint discovery, and timeline grouping must not be reverted.
2. Harden NPU execution: preflight and worker inference must use the selected `npu:{id}` device consistently and fail early on device mismatch.
3. Make run cleanup safe: failed or unwanted runs can be deleted without touching source files or portable metric assets.
4. Make metrics explain themselves: completed, unavailable, failed, and skipped states must be visible in summaries and timeline data.
5. Keep 4K UI light: default to preview artifacts and load originals only on explicit user action.
6. Add or preserve direct GT/Pred compare rules: external comparisons are strict, while VFIEval-generated GT/Pred alignment is guaranteed by manifest.
7. Keep the primary UI clean: no model registration, dataset registration, raw jobs, or experiment administration in the first-run workflow.

## Future Roadmap

- V13 should implement multi-model Compare by reading existing run, timeline, and metric-summary data. It must not trigger recomputation just because a comparison page opens.
- V14 should add remote worker orchestration through HTTP worker lifecycle APIs.
- Long-term work may add portable workspace bundles, reproducible evaluation packages, and cross-machine artifact sync.

## Testing

Run `python -m unittest discover -s tests` and `git diff --check` before finalizing code changes. If only `AGENTS.md` changes, `git diff --check` and manual review are sufficient.

Coverage should include file discovery, checkpoint discovery, preflight, model interface failures, NPU device selection, CPU/NPU mismatch errors, video decode/cache behavior, post-processing contracts, run lifecycle, run deletion, artifact grouping, metric unavailable behavior, timeline data, direct GT/Pred comparison, and UI-relevant lazy result display.
