# AGENTS.md

## Project

VFIEval is a local platform for video frame interpolation inference, post-processing, artifact inspection, evaluation, and comparison. It is not a training framework or a generic admin dashboard.

- Do not add training features or PSNR.
- Keep files and folders as the primary entry points: `models/`, `checkpoints/`, `videos/`, and `set/metrics/`.
- Primary workflows must not require users to type internal IDs or register models and datasets manually.
- Keep SQLite for metadata and indices, and keep generated artifacts on disk.
- Remote workers, authentication, object storage, new metric families, and Windows-specific hardening are out of scope unless the task explicitly changes that scope.

## Working Style

- Consult `NAVIGATION.md` before searching large source files. Update it only when subsystem ownership or entry points change.
- Inspect the relevant code and tests, then make the smallest coherent change. Avoid unrelated refactors.
- Do not require brainstorming, a design document, a work log, a commit, or updates to `AGENTS.md`, `CHANGELOG.md`, or `IMPLEMENT.md` for routine work.
- Ask questions only when a high-impact product or compatibility decision cannot be resolved from the repository. Otherwise make a reasonable, stated assumption and proceed.
- Preserve user changes in a dirty worktree. Never discard or overwrite unrelated edits.
- Update user documentation when behavior, interfaces, setup, or operating procedures actually change.
- Treat source code and tests as the authority for current implementation details. This file defines stable boundaries; investigate any apparent conflict instead of mechanically copying an obsolete rule.

## Core Contracts

### Models and post-processing

Models receive resized RGB `img0` and `img1` tensors in `BCHW`, range `[0, 1]`, at `t=0.5`, on the requested device and dtype. A model returns `flowt_0`, `flowt_1`, `mask0`, and `mask1`, either as a dict or tuple. Flow is backward displacement in resized pixel coordinates; masks are logits.

- Load checkpoint bytes on CPU before moving the model to CUDA or NPU. Do not deserialize onto a fixed accelerator.
- Core outputs must match input batch, device, dtype, and channel requirements. Lower spatial resolution is allowed; resize flow and mask to inference resolution and scale flow magnitudes by the width and height ratios.
- The platform owns warp, sigmoid, blend, compose, visualization, artifact writing, and metrics.
- Preserve these formulas unless code, tests, and documentation change together:
  - `grid_sample(mode="bilinear", padding_mode="border", align_corners=True)`
  - `mask0 = sigmoid(mask0_logits)`
  - `blend = mask0 * warp0 + (1 - mask0) * warp1`
  - `mask1 = sigmoid(mask1_logits)`
  - `pred = mask1 * img1 + (1 - mask1) * blend`
  - clamp `pred`, `blend`, `warp`, and `diff` to `[0, 1]`
  - `diff = abs(pred - gt)` when GT exists
- Create post-processing tensors on the input device with a compatible dtype. Do not move intermediate tensors to CPU except at artifact encoding or metric boundaries.

### Media identity and Compare

`media_assets` is a physical-file catalog. Semantic identity for new Compare and Campaign selections comes only from `media_items`, `media_item_members`, and `run_media_item_bindings`.

- The primary Compare flow is `GT Collection -> GT Item -> one or two reusable Pred Members` bound to that exact Item.
- Accept `media_item_id` plus at most two `pred_member_ids`. Resolve all IDs server-side; never trust a client path.
- Only valid model-inference predictions and explicitly Item-bound External predictions are reusable. Compare outputs, snapshots, evaluation packages, deleted or invalid Runs, and unbound historical predictions are not candidates.
- Reject cross-Item combinations. Every selected prediction must expose its method or producer Run and alignment slot.
- Temporal identity is strict: ordered source-frame indices, frame counts, FPS, and available timestamps must agree. Resizing never repairs a temporal or semantic mismatch.
- Spatial normalization is explicit and uses LANCZOS. One Pred determines the target size; with two Preds use the deterministic lower-pixel-area size, with the established tie-break rules. Record the Alignment Plan and fingerprint.
- New Item Compare Runs reference their source Members and publish aligned Diff, metrics, and reports. They do not publish a reusable `pred_video`.
- Legacy descriptors and historical multi-track Compare Runs remain readable compatibility surfaces, not primary selection paths.

### Campaign V2

- Campaign V2 is Item-first and compares exactly two methods from one GT Collection across explicitly selected Items.
- Missing method coverage for any selected Item blocks publication; never silently skip an Item.
- Use the same strict temporal validation and deterministic spatial Alignment Plan as Compare.
- Publication must validate and materialize a normalized frozen package in staging, then publish tasks atomically. Failed preparation must not leave a partial task set.
- Published configuration, bindings, tasks, and frozen media are immutable. Run deletion must preserve any still-dependent published Campaign media or fail safely.
- Participant routes use opaque identifiers and must not expose method, Run, model, checkpoint, asset, or task identity before voting.
- Keep subjective and objective results separate. Human analysis uses pairwise half-wins for ties, Bradley-Terry, and deterministic bootstrap intervals.

## Runtime Reliability

### Devices and inference

- Ascend NPU workers use `npu:<index>` and call `torch.npu.set_device(index)` before model construction, dry-run, staging, or inference. CUDA remains supported.
- Preflight and workers must reject CPU/accelerator device mismatches before long runs.
- If an inference shard fails, cancel queued siblings and make running siblings stop at their next cancellation check.
- Keep decode, model execution, and saving overlapped. Do not reintroduce per-sample `.cpu()` calls, synchronous image writes, or SQLite writes on the main compute thread.
- Keep device-to-host bundles and pending save work bounded. Progress reporting must be coalesced rather than emitted per saved sample.
- Structurally valid but semantically empty output, including NaNs or near-zero flow with nearly constant masks, must produce a visible preflight warning.

### Metrics

Only `lpips_vit_patch`, `lpips_convnext`, `vmaf`, and `cgvqm` are valid metrics. Do not substitute another score when a metric is unavailable.

- Missing manifests, assets, dependencies, executables, configs, or bindings must be recorded as `unavailable` with a specific reason.
- Full-reference metrics on samples without GT are `skipped: no ground truth`.
- Do not fabricate per-frame points for video-level metrics.
- Metric resolution, implementation details, and relevant manifest/driver fingerprints belong in health, result details, and cache keys.
- Metric jobs inherit the Run device. A metric device failure is `unavailable`; do not silently fall back to CPU.
- Opening Run Detail reads existing SQLite rows and artifacts and must not recompute metrics.

### Artifacts, APIs, and UI

- Model-inference Runs with GT must produce viewable Pred, GT, and Diff artifacts. Failed Runs must show a human-readable error.
- Sample and video APIs must query only the requested scope. They must not call or reproduce the full-run timeline path. The legacy full timeline endpoint is debug-only and deprecated.
- Keep Run Detail paged, timeline-centered, and lazy. Load previews first; load original 4K media only after explicit user action.
- Video streaming must retain HTTP byte-range support. Do not render every player or artifact at once.
- Preserve request cancellation, generation guards, monotonic `content_revision`, scoped cache invalidation, and in-place refresh when selections or Run content change.
- A non-terminal Run without artifacts shows a generating skeleton. Show an empty-artifact state only after a fresh terminal query confirms it.

### Logs

Runtime logs that support diagnosis and recovery are part of the product, including worker failures, model-load reports, output-health reports, and inference failures. Preserve them when changing the related execution paths.

Do not create agent activity logs or require changelog-style narration for every edit. Add new runtime logging only when it materially improves diagnosis, recovery, or user-visible failure reporting, and avoid recording secrets or untrusted paths unnecessarily.

## Deletion and Storage Safety

- Run deletion is a persistent, retryable purge request. Active Runs must cancel and stop before cleanup. Mark deletion complete only after every cleanup step succeeds.
- Cleanup may remove only the exact trusted `.vfieval/runs/{run_id}` directory and Run-scoped artifact or feedback metadata. Never delete models, checkpoints, source videos, metric assets, formal evaluation history, or unrelated paths.
- Before deleting a source Run, preserve every dependent Compare input as a private, non-reusable `compare_snapshot` and atomically switch its active binding. Any preservation failure blocks the source purge.
- Shared decode and Compare caches are reference- and lease-managed. Release only the deleting Run's references; garbage-collect entries only after the last valid reference, lease, and grace period are gone.
- Historical Run directories and orphan caches require preview followed by explicit confirmed GC.
- Failed, canceled, invalid, and test Runs must remain removable without presenting partial artifacts as complete.

## Validation

- Run the narrowest relevant tests while developing.
- For broad or high-risk code changes, run `python -m unittest discover -s tests`.
- Run `git diff --check` before finalizing tracked changes.
- If only `AGENTS.md` changes, `git diff --check` plus manual review is sufficient.
- Do not run unrelated expensive tests solely to satisfy ceremony; expand coverage in proportion to regression risk.
