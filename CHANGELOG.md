# CHANGELOG

## [2026-07-02 23:59]
- Replaced the placeholder LPIPS/CGVQM metric adapter with a manifest-driven command runner so project-local manifests can declare `driver.command`, `required_files`, and optional environment variables without hardcoded evaluator bindings.
- Made VMAF resolve `ffmpeg` from `set/metrics/vmaf/manifest.json -> ffmpeg_path` before falling back to `PATH`, added richer metric health diagnostics plus a `vfieval smoke-metric` CLI, and fixed Windows `libvmaf` log-path escaping.
- Updated the metric environment panel, README, AGENTS guidance, and metric/file-flow tests to cover manifest-driver execution, VMAF diagnostics, and the new smoke path.
- Files affected: `src/vfieval/metrics/health.py`, `src/vfieval/metrics/native.py`, `src/vfieval/metrics/registry.py`, `src/vfieval/metrics/vmaf.py`, `src/vfieval/cli.py`, `src/vfieval/web/app.js`, `tests/test_metrics.py`, `tests/test_v3_file_flow.py`, `README.md`, `AGENTS.md`, `CHANGELOG.md`, `IMPLEMENT.md`.

## [2026-07-02 23:16]
- Added a dedicated `videos/test_4k/uhd_gradient_motion.mp4` fixture and updated test asset generation so clean checkouts include one short UHD clip without routing the default `test_style` path through 4K assets.
- Increased the model preflight dry-run probe from `(1, 3, 8, 8)` to `(1, 3, 128, 128)` and versioned the dry-run cache key with the probe shape so stale in-process results are not reused.
- Extended file-flow tests to verify the larger probe shape reaches post-processing and that the new 4K group is discovered with `3840x2160` metadata and original-resolution preflight output.
- Files affected: `scripts/generate_test_assets.py`, `src/vfieval/file_inputs.py`, `tests/test_v3_file_flow.py`, `CHANGELOG.md`, `IMPLEMENT.md`.

## [2026-07-02 15:35]
- Added runtime output-health diagnostics so completed inference runs record real-frame flow/mask stats and warn when outputs are empty even if checkpoint loading succeeded.
- Stopped failed multi-shard runs from starting queued sibling shards, and made running sibling shards cancel on the next status check.
- Created run directories before marking inference jobs as running so cleanup and retry UI never see a running run without its output directory.
- Added a 10 MiB request body limit and hid raw internal exception text from API 500 responses.
- Files affected: `src/vfieval/pipeline/inference.py`, `src/vfieval/db.py`, `src/vfieval/server.py`, `src/vfieval/web/app.js`, `tests/test_v3_file_flow.py`, `AGENTS.md`, `IMPLEMENT.md`, `CHANGELOG.md`.

## [2026-07-01 23:20]
- Added repository layout and Git ownership rules so source files, generated test fixtures, local user inputs, runtime state, metric assets, and automatic backups have clear homes.
- Updated ignore rules to keep real models, videos, checkpoints, metric assets, SQLite state, runtime outputs, local tool settings, and file-level backups out of Git while preserving generated test fixtures.
- Files affected: `.gitignore`, `REPO_LAYOUT.md`, `AGENTS.md`, `IMPLEMENT.md`, `CHANGELOG.md`.

## [2026-07-01 21:11]
- Improved first-screen usability and large-workspace efficiency: lightweight video-group summaries, paged/searchable video and Compare source pickers, cached preflight dry-runs, cancellable stale preflight requests, default-collapsed Compare extra layers, and corrected multi-track Compare source labels.
- Reduced Run Detail timeline SQL fan-out by adding chunked batched sample artifact/metric reads, window-scoped artifact loading, and reused video artifact rows.
- Files affected: `src/vfieval/file_inputs.py`, `src/vfieval/server.py`, `src/vfieval/db.py`, `src/vfieval/web/app.js`, `src/vfieval/web/styles.css`, `tests/test_compare_sources_api.py`, `tests/test_compare_ui_hooks.py`, `tests/test_sample_api_scope.py`, `README.md`, `IMPLEMENT.md`.

## V11 - 多卡 NPU、Run 删除与轻量预览

- 新增 `multi_npu` 执行模式，使用 `torch_npu` 检测 `npu:0`、`npu:1` 等设备，并沿用 `run_jobs` 按视频粒度拆分 inference shard。
- 预检查 dry-run 改为使用最终 device/dtype，NPU 不可用或模型返回 CPU tensor 时能提前暴露设备不匹配。
- 新增 Run 软删除和产物清理接口：`DELETE /api/runs/{id}`、`POST /api/runs/{id}/cleanup-artifacts`，运行记录默认隐藏已删除 Run。
- 核心图片 artifact 生成 `512px` preview，Run Detail 默认只加载当前样本、当前分组的预览图，点击才打开原图。
- Run Detail 将核心产物分为 `图像 / Flow / Mask / Warp`，避免一次性加载十张 4K 图片。
- API、README、AGENTS 和测试同步更新，继续保持不训练、不实现 PSNR。

## V10 - 低分辨率输出、Checkpoint 与单机多卡推理

- 支持模型返回低于输入分辨率的 `flowt_0`、`flowt_1`、`mask0`、`mask1`；平台会统一 resize 到推理分辨率，并按空间比例缩放 flow 后再执行 warp/blend/compose。
- 新增 `checkpoints/{model_stem}/` 权重目录扫描，UI 可选择 `none`、`auto` 或具体 checkpoint 文件，模型加载时可通过 `checkpoint_path` 接收权重路径。
- 新增 `/api/checkpoints` 与 `/api/devices`，用于前端刷新 checkpoint 列表和设备能力。
- 新增 UI 刷新按钮，模型、视频和 checkpoint 文件变化后无需刷新浏览器即可重新扫描。
- 新增单机多卡 `multi_cuda` 模式，Run 会按视频粒度拆分为多个 inference shard job，并通过 `run_jobs` 记录每个 shard 的设备和进度。
- 推理、指标与 Run Detail 读取逻辑支持多 inference job 聚合，确保 artifact、metric 和进度可以覆盖多卡拆分场景。
- 新增测试模型 `test_lowres.py` 与 `test_checkpoint.py`，并更新模型输出 shape 校验测试，覆盖低分辨率输出与 checkpoint dry-run。
- 更新 `README.md` 和 `AGENTS.md`，明确低分辨率输出、checkpoint、文件夹入口和多卡推理约束。

验证：

- `python -m unittest discover -s tests`
- `python -m py_compile src\vfieval\db.py src\vfieval\server.py src\vfieval\file_inputs.py src\vfieval\pipeline\postprocess.py src\vfieval\pipeline\inference.py src\vfieval\pipeline\metrics_runner.py src\vfieval\worker.py src\vfieval\models\loader.py`
- `git diff --check`
