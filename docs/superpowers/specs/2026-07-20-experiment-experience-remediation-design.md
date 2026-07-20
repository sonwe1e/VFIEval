# VFIEval Experiment Experience Remediation Design

## Goal

Make file-based VFI experiments reproducible, predictable under aggressive multi-device settings, inexpensive to browse, and safely removable. Campaign V2 remains the primary blind-evaluation workflow. The work does not add training, PSNR, remote workers, authentication, object storage, or new metric families.

## Confirmed regressions

- Ordinary GET requests synchronously pump Run cleanup, while several Media GET routes rescan and hash the source library. Opening Media or Studio can start overlapping full scans and contend with active experiments.
- The default `multi_npu + fp32 + batch 64 + whole-group selection` is intentionally throughput-oriented, but preflight only probes one 128x128 sample and therefore cannot describe the actual workload risk.
- Initial model-inference preflight walks every selected video to count frames exactly. Run creation repeats preflight and decoding subsequently reads the inputs again.
- Metric health, including ffmpeg/libvmaf probes, is recomputed from common page and preflight paths.
- Run retry replays `checkpoint=auto` and live file paths, so it is not an exact reproduction after a model, checkpoint, or source file changes.
- Single-Run deletion and artifact cleanup are immediately destructive. Campaign tombstones are retryable through the API but are not represented as durable cleanup work after campaign metadata disappears.
- The current workspace contains about 10.9 GiB of generated data for 11 source videos: about 5.48 GiB of Run output and 4.62 GiB of decode cache. Per-sample GT is duplicated into each Run.
- The Run list exposes only the newest 100 records, and catalog startup uses all-or-nothing loading.

## Delivery 1: correctness and safety

### Reproducible Run identity

At Run creation, resolve and persist an `input_identity` containing:

- model relative path, size, mtime, and SHA256;
- requested checkpoint plus resolved checkpoint relative path, size, mtime, and SHA256;
- selected canonical source Asset and Item IDs, qualified video names, and content SHA256;
- normalized execution request and a stable fingerprint over all fields above.

Reference configuration and reference keys include source content identities. Model metadata includes model and checkpoint hashes so historical records do not imply that a later file at the same path is identical.

`POST /api/runs/{id}/retry` is exact: it uses the stored resolved checkpoint and source identities. Changed or missing inputs return `409 InputIdentityChanged` with structured changed fields. `POST /api/runs/{id}/clone` creates a new Run from current files and records the source Run plus identity differences. Metric-only retry remains based on frozen artifacts and is unchanged.

### Throughput-oriented workload guard

Keep whole-group selection, fp32, batch 64, and multi-accelerator defaults when matching hardware exists. If the configured accelerator family is absent, select the fastest detected valid mode instead of leaving an impossible default.

Preflight returns a workload report with effective device/precision, per-device batch, dimensions, selected samples, input tensor lower bound, prefetched host-memory lower bound, and a conservative artifact budget. A workload is high risk when either:

- the two input tensors consume at least 5% of reported per-device memory;
- prefetched decoded inputs consume at least 25% of reported available host memory; or
- device memory is unknown and per-device batch pixels exceed 16 million.

High risk does not change settings automatically. The UI presents the exact values and requires a per-creation confirmation whose fingerprint becomes invalid when the configuration changes. The server rejects an unacknowledged high-risk request.

### Destructive workflow safety

Run deletion, batch deletion, and artifact cleanup first return a preview containing Run names/statuses, dependency summary, estimated exclusive bytes, shared cache bytes, and a short-lived preview token. Confirmation is required before the persistent purge request is created. Failed and invalid Runs remain deletable.

Campaign deletion writes a durable cleanup request before removing campaign metadata. The deterministic quarantine path is retried by the background maintenance loop until removed. Studio shows pending/failed Campaign cleanup requests even after the Campaign itself is gone.

No HTTP GET performs cleanup. Production and tests use an explicit server-runtime maintenance loop. Existing BrokenPipe/client-reset handling remains scoped to response I/O and never converts an internal worker or filesystem failure into success.

## Delivery 2: remove redundant work

### Catalog coordination

Introduce one process-local catalog coordinator. Startup schedules a background reconciliation. `POST /api/media/sync` starts or joins the current reconciliation and `GET /api/media/sync/status` reports phase, progress, errors, and `catalog_revision`.

Collection, Asset, Item, source, and paged video GET routes read SQLite snapshots only. Source hashes are recalculated only when size or mtime changes. Thumbnails are generated when their URL is requested rather than while listing the library.

The frontend shares one catalog load promise across Media, Compare, external binding, and Studio. Item selection remains stable across server-side pagination and search; the browser does not fetch every page merely to populate a selection control.

### Two-level preflight

`POST /api/preflight/quick` performs request validation, device selection, cached media metadata, and workload estimation. Form changes call only this route.

`POST /api/preflight` performs the model contract dry-run and required deep checks, then returns a short-lived `preflight_token` tied to the normalized request and input stat/content fingerprint. Run creation revalidates those identities and reuses the result. Existing API clients may omit the token and receive the full deep check.

Model-inference preflight uses an exact decode manifest when present; otherwise it trusts ffprobe/container frame metadata with a warning that exact frame count will be established by decode. Compare and Campaign retain strict decoded temporal validation.

Metric health is cached by manifest, weights, driver, and executable fingerprints. Explicit health refresh invalidates it. Normal page loads and preflight do not repeatedly spawn ffmpeg introspection processes.

### Frontend request behavior

Catalog startup uses independently handled requests so metrics failure cannot hide models, videos, devices, or Runs. Checkpoints load only for the selected model. The initial Runs request is shared with polling.

Campaign preparation polling remains serialized and generation-guarded, pauses while hidden, and uses bounded failure backoff. Client cancellation prevents stale UI writes; deep preflight deduplication prevents an aborted response from causing duplicate server work.

## Delivery 3: storage and history

GT sample Artifacts reference managed decode-cache GT files instead of copying identical PNG bytes into every Run. Cache references and leases keep them available through Run Detail, metrics, Compare, and Campaign preparation. Deleting one Run releases only its reference; GC requires no valid reference or lease and the existing grace period.

Run Detail and deletion previews report exclusive Run bytes, shared cache bytes, and estimated reclaimable bytes. Creation shows the artifact budget. No generated experiment or cache is automatically deleted.

`GET /api/runs` gains opt-in server-side pagination and filters for text, status, Run type, and model while retaining the legacy list response when pagination parameters are absent. The UI exposes all history and polls only active Runs plus the visible page.

Rename the feedback-only statistics view to “主观反馈统计”. Campaign objective analysis and the two-series LPIPS curve remain separate objective surfaces.

Runtime diagnostics record catalog duration/revision, preflight stage timings and cache hits, cleanup backlog, and predicted versus actual Run bytes. Logs do not add agent activity or expose unnecessary paths.

## API compatibility

- New: `POST /api/media/sync`, `GET /api/media/sync/status`.
- New: `POST /api/preflight/quick`; deep preflight returns `preflight_token`, `input_fingerprint`, and `workload`.
- Extended: `POST /api/runs` accepts `preflight_token` and `risk_ack_fingerprint`; omission remains backward compatible.
- Changed semantics: Run retry is exact. New `POST /api/runs/{id}/clone` is the explicit live-file operation.
- New: Run purge/cleanup preview endpoints and confirmation tokens.
- Extended: Campaign DELETE returns a durable cleanup request ID and cleanup status is queryable.
- Extended: paged `GET /api/runs`; the unpaged response shape remains unchanged.

## Validation

Tests cover read-only GET behavior, background cleanup without client traffic, concurrent catalog-sync coalescing, no rescans from pagination, workload risk acknowledgement, preflight token reuse/invalidation, exact retry and clone differences, manifest/executable health cache invalidation, Campaign cleanup recovery across restart, shared GT reference lifetime, more than 100 Runs, and more than 200 Media Items.

Each delivery runs its narrow tests. The completed change runs `python -m unittest discover -s tests` and `git diff --check`.

## Accepted defaults

- Deliver the complete remediation in three independently testable batches.
- Preserve throughput-first behavior rather than silently reducing batch or selected videos.
- Make expensive or destructive consequences explicit and confirmed.
- Keep source folders as the primary entry point, SQLite as metadata/index storage, and artifacts on local disk.
- Preserve current uncommitted client-disconnect and Campaign reconciliation work.
