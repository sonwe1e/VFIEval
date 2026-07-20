# VFIEval

VFIEval 是一个面向视频插帧模型的本地推理评测工具。当前阶段主流程是：放模型文件、放视频文件夹、选择部分或全部视频、点击开始、查看 `pred/gt/diff` 与核心中间产物，并可通过指标曲线定位问题帧。平台不做训练，不实现 PSNR。

## 快速开始

```powershell
$env:PYTHONPATH='src'
python -m vfieval.cli init --workspace .vfieval
python -m vfieval.cli --workspace .vfieval serve --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765`。

页面主流程只有两步：

- `新建推理`：选择模型文件、视频集、部分或全部视频、分辨率、设备和精度，然后开始推理。
- `运行记录`：查看推理/评测阶段、错误、输出目录、指标曲线、坏帧列表和按需加载的核心产物。

不需要手动注册模型、注册数据集，也不需要填写 `model_id`、`dataset_id` 或 `job_id`。

## 文件夹入口

模型文件放在项目根目录的 `models/`：

```text
models/
  test_average.py
  test_img0.py
  test_img1.py
  my_model.py
```

权重文件放在项目根目录的 `checkpoints/{model_stem}/`：

```text
checkpoints/
  my_model/
    latest.pth
    epoch_001.pth
```

视频按风格、作品或测试集放在 `videos/{视频集}/`：

```text
videos/
  test_style/
    gradient_motion.avi
    blocks_motion.avi
  anime_style/
    001.mp4
    002.mp4
```

Web UI 仍以文件夹为入口，并把耗时检查拆成目录同步、快速预检和启动前深度预检：

- `models/*.py` 作为模型下拉框。
- `checkpoints/{model_stem}/*` 作为权重下拉框，可选择不加载、自动最新或具体文件。
- `videos/*/` 作为视频集下拉框；首屏只读取分组摘要，具体视频在选择视频集后按页加载，缩略图只在浏览器实际请求对应图片 URL 时生成。
- 服务启动时会在后台协调一次 Media Catalog 回填；Media、Compare 和 Campaign 的普通列表、搜索和分页只读当前快照，不会在每次 GET 时重新扫描目录。点击 `刷新文件列表` 只同步文件入口；Media 页的 `刷新` 还会显式纳入历史 Run 资产。两者都会加入已在运行的同步、等待完成后原地刷新，不需要刷新浏览器页面。
- 表单变化只执行快速预检：校验受信任路径、选择、设备/精度和容器媒体信息，并给出样本量及工作量估算；它不会加载模型做 dry-run，也不会返回可用于创建 Run 的令牌。
- 点击开始时强制执行深度预检，检查模型输出契约、解码缓存和媒体容器信息，并为模型、权重和所选视频生成内容签名。响应中的 `input_fingerprint` 标识这组物理输入；短期有效的 `preflight_token` 同时绑定配置与这些输入。`POST /api/runs` 会重新校验并复用已验证哈希，任一文件变化、配置变化或令牌过期都必须重新预检。已完成的解码清单提供精确帧数；否则显示容器帧数估算警告，并由解码任务在推理前复核。
- 预检会显示有效设备/精度、单设备 batch、分辨率、输入张量下界、主机预取下界和产物空间预算。数字是资源规划下界，不包含模型参数、激活和分配器工作区；高风险配置必须在看过具体数值后确认，提交时携带匹配的 `risk_ack_fingerprint`。
- 视频列表默认全选，支持服务端分页、按文件名搜索、排序和缩略图，也可以只勾选本次要推理的视频。
- “开始对比”提交期间会立即锁定为单次请求，并显示预检查、创建 Run 和打开详情三个阶段；无需重复点击。

VFIEval 中视频的 `width`、`height` 和“分辨率”均指应用旋转元数据并自动旋转解码后的显示方向尺寸。编码层的原始宽高只保留用于诊断，不参与对齐或缩放判断；视频解码保持自动旋转开启。

默认测试模型和测试视频已经随工作区提供。也可以重新生成测试视频：

```powershell
python scripts\generate_test_assets.py
```

## 模型接口

每个模型文件建议定义：

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

也支持 tuple 返回：

```python
return flowt_0, flowt_1, mask0, mask1
```

Contract：

- `img0/img1` 是 RGB、`BCHW`、值域 `[0, 1]`、已 resize、固定 `t=0.5`。
- 输入张量已经在实际推理 device 上，dtype 为 `fp32/fp16/bf16`。
- `flowt_0/flowt_1` 是 resized 像素坐标系下的 backward flow，平台使用 `target_pixel + flow` 采样。
- `mask0/mask1` 是 logits，不能提前 sigmoid，不需要归一化。
- 输出必须与输入同 batch、同 device、同 dtype；`flow` 必须是 `[B,2,h,w]`，`mask` 必须是 `[B,1,h,w]`。
- `flow/mask` 可以低于输入分辨率，平台会先 resize 到推理分辨率。`flow_x` 按 `W / w` 缩放，`flow_y` 按 `H / h` 缩放；`mask` 作为 logits resize 后再 sigmoid。
- 模型可以额外返回可视化 tensor；平台会保存为 `extra_*` 图像，但核心对比只使用 flow、mask、warp、blend、pred、diff。

## 后处理

平台统一写死后处理：

- `grid_sample(mode="bilinear", padding_mode="border", align_corners=True)`
- `mask0 = sigmoid(mask0_logits)`
- `blend = mask0 * warp0 + (1 - mask0) * warp1`
- `mask1 = sigmoid(mask1_logits)`
- `pred = mask1 * img1 + (1 - mask1) * blend`
- `pred/blend/warp/diff` clamp 到 `[0, 1]`
- `difference = abs(pred - gt)`

每个 Run 的主要输出：

```text
.vfieval/runs/{run_id}/
  config.json
  model_info.json
  video_group_info.json
  logs/
    inference.log
  videos/
    {video_name}/
      manifest.json
      pred.mp4
      gt.mp4
      diff.mp4
      preview/
        pred.mp4
        gt.mp4
        diff.mp4
```

`Pred / GT / Difference` PNG 和 `pred / gt / diff` 视频是 `canonical-v1` 核心制品，尺寸由 Run 的输出分辨率决定。高级设置中的 `visualize_height / visualize_width` 只控制 LANCZOS 预览，不改变推理合成、指标、Compare/Campaign 输入或媒体目录；预览尺寸与 canonical 相同时不会重复写文件。`GET /api/files/{artifact_id}` 始终返回 canonical，`?variant=preview` 返回预览（没有独立预览时回到同一 canonical 文件）。4K Run 会保留完整分辨率 PNG 和视频，因此磁盘占用与编码耗时会明显高于低分辨率 Run。

Run Detail 默认不会一次性加载所有视频和所有图片。页面先读取 Run 的视频摘要并播放预览视频，再按当前视频加载最多 160 帧的细节 `timeline`；长视频同时显示固定桶数的全片 min–max/均值总览，点击总览或拖动全局滑条时再加载目标细节窗口。只有选中某个样本时，才调用 `GET /api/runs/{id}/samples/{sample_id}` 加载对应的预览 `GT / Pred / Diff / Flow / Mask / Warp / Blend`；原图和原视频只在点击“打开原图/原视频”后加载，原视频继续支持 HTTP Range。

Run 的产物、指标或清理状态发生变化时，服务端会递增 `content_revision`。前端检测到 revision 变化后会中止旧请求、失效当前 Run 的结果缓存，并原位恢复当前视频、帧位置、时间线窗口和指标选择；推理完成后不再需要整页刷新。详情页的 `刷新结果` 可强制执行同样的局部刷新。运行中尚未发布产物时显示“产物生成中”，只有终态重新查询后仍为空才显示无产物。

## Run 删除、缓存与存储清理

删除 Run 是持久化的异步清理流程，而不是只隐藏一行记录：

- 删除、批量删除和“只清产物”都会先调用 `POST /api/run-purge/preview`，展示 Campaign/Compare/活动 Job 依赖、Run 目录占用、共享缓存和预计可回收空间。预览返回的 `preview_token` 短期有效、一次性使用，并绑定操作类型、精确 Run 集合及当前状态；状态变化、令牌过期或选择变化时必须重新预览。
- `DELETE /api/runs/{id}?preview_token=...` 返回 `202 Accepted` 并创建 `run_purge_requests`。运行中的 Run 会先请求取消，worker 停止后才删除受信任的 `.vfieval/runs/{id}`、清理产物索引与 Run feedback，并把对应 Run 媒体标记为不可用。
- 只有全部步骤成功后才写入 `deleted_at`。失败请求保留错误与清理报告，可重试；服务重启后会继续处理 `requested/canceling/purging` 请求。
- `cleanup-artifacts` 走同一套幂等服务，但保留 Run 记录；它和批量删除都在 JSON body 中提交预览令牌。批量删除逐 Run 建立请求，单项失败不会中断其它项。
- `decode_cache` 与 `compare_cache` 通过 `cache_entries`、`run_cache_refs` 和活动 lease 管理。共享缓存只有在最后一个有效 Run 引用释放、没有活动 lease 且超过默认 10 分钟宽限期后才可回收。
- 指标缓存、视频元数据与缩略图缓存不会随单个 Run 删除。历史残留和孤儿缓存必须先在 Media 页生成存储清理预览，再显式确认 GC。

已发布的 Campaign 媒体会在破坏性清理前冻结到独立评测包，因此删除来源 Run 不会破坏正式盲评播放、投票或分析历史。

文件入口的模型推理 Run 提供两种重新执行语义：

- `重试` 是精确复现：固定使用原 Run 当时解析出的 checkpoint，并重新校验模型、checkpoint、所选 Item/Asset、源视频内容及执行配置。任一输入缺失或变化时返回 `409 InputIdentityChanged` 和结构化差异，不会悄悄改用当前文件；没有输入身份记录的旧 Run 应使用 Clone。
- `按当前输入 Clone` 重新解析磁盘上的当前模型、`auto` checkpoint 和源文件，创建新 Run，并记录来源 Run 及输入身份差异。需要测试最新文件时使用 Clone，不要把它当作精确重试。

## Direct GT/Pred Compare

`Run 类型` 选择 `双视频对比` 后，页面会按需加载 Compare 来源：

- GT 从 `videos/{group}/` 中选择，支持搜索和分页。
- Pred 从已完成 Run 的 `pred_video` artifact 中选择，并按当前 GT 视频自动过滤。
- 每个 Pred track 可以在表格里编辑显示标签，创建 Run 时写入 `label`。
- `flow/mask/warp/blend` extra layers 默认不选，展开 `Extra layers` 后再选择需要对照的层，避免样本详情一次加载过多预览。

Compare Run Detail 会按 track 显示 `pred/diff` 视频，GT 视频共享；样本级 extra layers 仍按 `kind × track` 横向网格展示。

## 指标曲线

高级设置里可以选择指标：

- `lpips_vit_patch`
- `lpips_convnext`
- `vmaf`
- `cgvqm`

不支持也不会实现 `PSNR`。

指标行为：

- 推理完成后进入评测阶段，Run Detail 会分别显示推理阶段和评测阶段。
- `GET /api/metrics/health` 会显示每个指标的依赖和资产状态。
- `lpips_vit_patch`、`lpips_convnext`、`cgvqm` 默认走内建适配器契约；缺少官方资产或绑定时写为 `unavailable`，不会替换成其它分数。
- 逐帧指标按 triplet 样本显示曲线；长视频使用“全片总览 + 160 帧细节窗口”，细节曲线不为每帧绘制按钮，悬停或点击曲线会定位到最近帧。
- `VMAF` / `CGVQM` 如果只能产生视频级结果，只显示视频级摘要，不伪造成逐帧曲线。
- “最差样本”列表按当前指标排序，点击后直接跳到对应帧。

准备或检查指标资源：

```powershell
python -m vfieval.cli --workspace .vfieval prepare-metrics --check-only
python -m vfieval.cli --workspace .vfieval prepare-metrics
python -m vfieval.cli --workspace .vfieval smoke-metric --metric vmaf --reference <gt_video> --distorted <pred_video>
```

### Metric setup details

`prepare-metrics` downloads missing default metric assets into `set/metrics/` and writes runnable manifests. It does not install Python packages or system binaries. `prepare-metrics --check-only` is read-only, and `GET /api/metrics/health` reports status, source URLs, fixed evaluation resolution, manifest paths, executable paths, input mode, and timeline support.

Current metric setup contract:

```text
set/
  metrics/
    lpips_vit_patch/
      manifest.json
    lpips_convnext/
      manifest.json
    vmaf/
      manifest.json
    cgvqm/
      manifest.json
```

- `lpips_vit_patch`: `prepare-metrics` downloads the DINOv2 checkout and ViT-S/14 registers weights into `set/metrics/lpips_vit_patch/`.
- `lpips_convnext`: `prepare-metrics` downloads the local ConvNeXt V2 tiny checkpoint into `set/metrics/lpips_convnext/`.
- `cgvqm`: `prepare-metrics` downloads the IntelLabs CGVQM checkout and writes the VFIEval JSON wrapper into `set/metrics/cgvqm/`. This remains a video-level metric and does not synthesize per-frame points.
- `vmaf`: this is the first metric that can run immediately in the current build. VFIEval checks `manifest.json -> ffmpeg_path`, then `set/metrics/vmaf/ffmpeg.exe` (or `ffmpeg`), then `ffmpeg` on `PATH`; the selected binary must expose the `libvmaf` filter.

推理和 Compare 发布到浏览器的 MP4 固定使用 H.264 (`libx264`)、`yuv420p`、偶数尺寸与 `faststart`。如果当前 `ffmpeg` 缺少 `libx264`，且 OpenCV 也无法写出 AVC，Run 会以明确错误失败，不会静默生成浏览器通常无法播放的 `mp4v` 产物。VMAF 可在 manifest 中指定独立的 FFmpeg；浏览器产物编码仍使用进程 `PATH` 上的 `ffmpeg`。

Current remaining-metric adapters:

- `lpips_vit_patch` is a sample-level DINOv2 feature-distance metric. Default backbone: `dinov2_vits14_reg`. Evaluation uses max edge `518`, preserves aspect ratio, and pads to a multiple of `14`.
- `lpips_convnext` is a sample-level ConvNeXt V2 feature-distance metric. Default backbone: `convnextv2_tiny.fcmae_ft_in22k_in1k`. Evaluation uses max edge `288`, preserves aspect ratio, and pads to a multiple of `32`.
- `cgvqm` is a video-level wrapper around a local IntelLabs CGVQM checkout. Evaluation videos are written under the metric work directory with long edge capped at `720`; original artifacts are not overwritten.
- Metric jobs inherit the Run's inference devices through `metric_device`. 多卡 Run 会为每张推理卡创建一个逐帧指标分片；视频级指标只在 leader 分片执行一次。LPIPS 在每个分片中只加载一次模型并批量处理图像对，默认每卡 batch 为 ViT `8`、ConvNeXt `32`；高级设置 `每卡评测 Batch Size` 可覆盖默认值，设备 OOM 时会在同一张卡上逐级减半，不回退 CPU。LPIPS 和 CGVQM 在模型构造前绑定请求的 `npu:<index>`；CUDA/NPU 评测失败时写入带设备及原因的 `unavailable`。
- Run Detail 的“重试失败/不可用指标”只重试最新状态仍为 `failed` 或 `unavailable` 的指标；重试会绕过旧失败缓存，但仍复用有效的 `completed` 结果，历史结果行不会删除。
- CGVQM 读取 driver stdout 中最后一个合法协议 JSON；非零退出、空输出、坏 JSON、超时和临时视频帧数异常会分别记录 return code、实际帧数以及有界 stdout/stderr，便于定位 `cgvqm failed: 1` 等问题。
- `prepare-metrics --check-only` is read-only. `prepare-metrics --force` replaces only VFIEval-declared metric assets, not unrelated user files. Python packages such as `timm`, `safetensors`, `av`, and `scipy` are never installed automatically.

Status interpretation:

- `missing_weights`: the expected manifest or referenced project-local assets are not present yet.
- `missing_evaluator`: the manifest may exist, but the declared driver command, interpreter, or system executable is still unavailable.
- `missing_dependency`: the Python package dependency for that metric is missing from the current environment.
- `available`: the current build can attempt to execute that metric without substituting another score.

Metric assets live under `set/metrics/` by default, or under `VFIEVAL_METRIC_ASSETS_DIR` when that environment variable is set. A migrated set can carry relative weights, model sources, wrappers, and a project-local VMAF ffmpeg binary; it cannot carry the Ascend runtime or Python wheels. CGVQM drivers use the same Python interpreter that started VFIEval, so compatible `torchvision`, `av`, and NPU packages must be installed in that environment. Missing dependencies, failed downloads, or unsupported evaluator devices are recorded as `unavailable`; VFIEval never substitutes a different metric score.

## 统一媒体资产与 External 上传

Media 页按用途分成三块：

- `Sources / Uploads`：`videos/` 中的规范来源与 External 上传。
- `Derived Runs`：按 `Run → 视频 → Track` 分组折叠显示有效 Pred 输出；已删除或已清产物的 Run 不会重新出现在目录、Compare 或 Campaign 来源中。
- `Evaluation Packages`：已发布 Campaign 的冻结评测包。自动生成的内部 Run Collection 不作为用户目录展示。

Media Catalog 以稳定 `asset_id` 统一索引源文件、上传、Run 视频产物和评测包。服务启动和用户显式刷新会通过一个进程内协调器幂等回填目录与历史 Run；并发刷新会合并为同一后台任务。常规资产、Compare 和 Campaign 列表只读 SQLite 快照，不触发隐式重扫。原文件不移动，Catalog 只保存服务端路径、SHA-256、媒体信息和 provenance。Run 清理把对应 `run_artifact` 标记为 `unavailable`，冻结的 `evaluation_package` 仍可播放。

External 视频和帧序列通过浏览器分片上传：

- 固定 8 MiB 分片，支持重复分片幂等和重新提交续传，单资产默认上限 50 GiB。
- 视频保留原扩展名；帧序列首版使用 ZIP 并必须填写 FPS。
- 每片及完整文件均校验 SHA-256；拒绝路径穿越、符号链接、危险解压比、无效图片和混合尺寸。
- 服务端写入 `.vfieval/media/{collection}/{asset_uuid}/`；客户端不能提交磁盘路径。
- 同一 Collection 内显示别名唯一。软删除被 Campaign/投票引用的资产时只禁用播放并保留分析记录。

Compare 主界面只提交 `media_asset` descriptor。External GT/Pred 必须在帧数、尺寸、FPS及可用时间戳上严格一致，不做自动截断、偏移或缩放。VFIEval 自产 Pred 可使用平台记录的 `source_frame_indices` 生成推理分辨率的 `aligned GT`，并记录 `generated_from / aligned_gt_of / pred_of` 关系。

## 产物档位与 Benchmark

- `evaluation`：默认保存 Pred/GT/Diff 无损帧、视频及指标所需输入，不全量保存内部层。
- `diagnostic`：额外保存 Flow/Mask/Warp/Blend/`extra_*`。
- `benchmark`：不保存媒体产物、不运行指标，只输出启动、稳态、端到端、队列、显存和设备阶段性能。

保存队列有界并施加背压，artifact 记录批量写入 SQLite。多卡任务在视频不足或负载不均时按连续 sample segment 分片；各 shard 只写帧和 manifest，`finalize` job 统一合并视频并在之后启动指标。

运行标准 benchmark（默认 warmup 10 batch、测量 200 个样本、重复 3 次）：

```powershell
python -m vfieval.cli --workspace .vfieval benchmark --model-file my_model.py --video-group anime_style --execution-mode multi_npu --device-id npu:0 --device-id npu:1 --precision fp16 --batch-size 4
```

最优 batch/prefetch/save 配置按模型哈希、权重哈希、分辨率、精度、设备型号/数量和产物档位写入 `execution_profiles`，后续 preflight 会展示建议，表单中的显式设置仍优先。

## 多人盲评与 Campaign V2

Evaluation Studio 集中管理创建、发布和组织者分析，并把参与页面独立出去。一个 V2 Campaign 固定比较两份 Pred 方法，主流程中的方法是一份完成 Run 的 Track；External Pred 只在高级入口使用。创建向导依次填写基本信息、选择 Run/Track A 与 B、查看共同视频矩阵并勾选视频、确认严格对齐和任务量，然后发布。

矩阵不要求手填 `video_name`，会明确显示缺失、GT 冲突以及帧数、尺寸、FPS、时间戳不一致。只有两种方法都有输出、共享同一规范 GT 且严格对齐的视频才能进入发布；不截断、不偏移、不静默缩放。

发布是可恢复的 preparation job：先在 staging 中逐视频深度校验，再生成 `.vfieval/evaluations/{campaign_id}` 与 SHA-256 manifest，最后把冻结文件登记为 `source_kind="evaluation_package"` 并原子发布任务。新包只冻结 GT、Pred A、Pred B 三路；旧包中的 Diff 文件和 manifest 字段继续可读。`campaign-freeze-stream-v3` 新编码流在保留原分辨率、CRF 18、H.264/yuv420p 和 faststart 的同时，使用约 1 秒固定 GOP、固定最小关键帧间隔并关闭场景切换关键帧。GT/A/B 只有在首关键帧接近首帧且最大关键帧间隔不超过 2 秒时才可 remux；indexed VFI 的 GT 仍按映射重编码，Pred A/B 还必须同时满足完整帧序列、H.264/yuv420p、零旋转、CFR、尺寸、FPS 和相对时间轴严格一致，任一路不合格或失败都会让两路一起回退编码。manifest 记录 GOP 策略、关键帧探测与策略指纹；旧 asset-mode 兼容草稿若按原媒体复制发布，会明确标记为 `campaign-freeze-legacy-copy-v1`，不会伪称满足 v3 GOP 策略。若 FFmpeg/ffprobe 无法保证 v3，Item 模式发布会明确失败而不使用旧编码器回退。解码缓存使用可信内容摘要、大小和 mtime 身份，最终发布前仍重新校验全部源内容；阶段进度会分别展示 alignment、身份校验、hash、decode-cache、remux/encode、输出校验和源稳定性耗时。同一服务进程一次只冻结一个 Campaign，Item 内仍使用有界并行。

Studio 保留 `current/total · phase` 兼容摘要，其中 `current/total` 表示已经完整冻结的 Item 数，不是当前 Item 的帧数。新冻结引擎还会显示当前 Item、阶段、帧进度、总进度、所用管线和阶段耗时；remux、探测或哈希等没有逐帧计数的阶段可能只更新阶段与总进度。发布后的配置和评测包不可变，可以关闭或归档；删除来源 Run 不影响播放。失败发布不会留下半成品任务或垃圾 draft，可从 Studio 重试。

参与者只打开独立的 `/evaluate/{opaque_token}` 页面，不加载主导航、Media 或管理分析。浏览器生成稳定 evaluator UUID，评测员填写显示名后领取带可续期 lease 的任务；任务、assignment 和媒体 URL 都使用 opaque token，投票前不暴露模型、checkpoint、Run、方法标签或真实 asset/task id。左右顺序按任务与评测员稳定随机。A 更好、B 更好或平局必须选择；A、B 可分别填写可空的 1–5 分（0.25 步长），并可记录置信度和备注。评分独立统计，不改变平局半胜和 Bradley–Terry 排名。

参与页默认使用大画幅重叠对比：GT 作为全宽参考单独显示，候选 A/B 在下方共享一个全宽画面，通过可拖动、可触摸和可键盘操作的竖向分隔线查看同一位置的差异。“完整视图”可切换为 GT、候选 A、候选 B 三段纵向排列；两种视图始终复用同三个媒体节点，不重新下载、不改变匿名左右映射，也不重置当前播放时间或帧序列索引。若浏览器不支持所需的 CSS 裁剪能力，页面会自动使用完整视图。

视频任务由任务范围内的 sticky 控制条统一控制播放、暂停、整数帧选择、播放速率和循环。分割线视图可选“两个视图同步”、“仅 GT”或“仅候选对比”；完整视图可选三路同步或仅 GT/A/B 任一路。候选对比以匿名 A 为时钟同步 A/B，三路同步以 GT 为时钟，单路播放不运行多路纠偏。切换视图或范围会在当前时钟位置对齐三路并保持暂停；拖动 A/B 分割线只改变显示比例，不会改变播放范围或媒体状态。所有选帧操作都暂停并把 GT/A/B 定位到同一帧；支持 `requestVideoFrameCallback` 的浏览器按实际展示帧校正，其他浏览器使用 `timeupdate` 回退。

时长不超过 30 秒且三路 `Content-Length` 合计不超过 256 MiB 的任务会先完整下载为任务级 Blob，显示匿名 GT/A/B 进度，再开放播放；其他任务使用 Range 流式预缓冲。首次播放会等待活动范围每路有连续 10 秒数据或已到末尾；播放中低于 1.5 秒时主动对齐暂停，各路恢复到 5 秒或末尾后继续。状态会区分具体匿名通道的取流不足与浏览器解码等待；60 秒无缓冲进展后停止自动尝试并提供重新加载。浏览器策略拒绝自动恢复时保持对齐暂停并提示点击继续；真实解码或媒体错误会阻止投票并提供重试。帧序列任务仍始终让三张图共享同一帧索引。

完成页会列出本人已评视频。已发布 Campaign 可点击条目回到同一个盲评界面，预填并修改原答案；关闭或归档后仍可匿名回看，但只能只读查看。

### 参与链接与局域网访问

在 Studio 中选择已发布的 Campaign 后复制“参与链接”。参与者不需要注册或密码：打开链接、填写显示名，即可领取任务；同一浏览器会复用其本机 evaluator 身份。

服务默认绑定 `127.0.0.1`，因此在本机 Studio 中显示的 `http://127.0.0.1:8765/evaluate/...` 只能由该服务器本机打开。若要让受控内网中的参与者访问，请以明确的内网监听地址启动：

```powershell
$env:PYTHONPATH='src'
python -m vfieval.cli --workspace .vfieval serve --host 0.0.0.0 --port 8765
```

然后从 `http://<服务器内网 IP>:8765` 打开 Studio，并重新复制参与链接；该链接会使用当前页面的来源地址。不要把此服务直接暴露到公网。API 中的 `share_url` 保持相对 `/evaluate/{opaque_token}`，由 Studio 在显示和复制时补全当前来源，避免服务端猜测可访问的主机名。

`0.0.0.0` 只是监听地址，不能作为发给参与者的主机地址。经另一台服务器做端口映射时，应使用参与者实际可访问的服务器 IP 或域名，并优先使用完整 TCP 端口转发。如果使用 HTTP 反向代理，必须同时转发 `/evaluate/`、`/blind.js`、`/blind.css` 和 `/api/blind/`；只转发参与页面会导致页面停在“准备中”。可先从参与者设备直接打开 `http://<映射地址>/blind.js` 和 `http://<映射地址>/api/blind/<token>`，两者都应返回 `200`。

参与者完成个人全部可评任务后才看到当前实名实时结果；组织者可始终查看覆盖率、方法级分析和每方法/每视频评分填写数、均值与中位数。人类结果使用平局半胜、Bradley–Terry 和固定种子 bootstrap 区间；客观指标先按 Item × 方法 × 指标汇总，再让各 Item 等权进入总体统计，并保留逐帧覆盖与真实 unavailable 原因。两类结果不合成总分。旧 schema v1 Campaign 保持只读，可导出和归档，不按标签猜测迁移。

## API

主流程端点：

- `GET /api/model-files`
- `GET /api/checkpoints?model_file=...`
- `GET /api/devices`
- `GET /api/video-groups`，支持 `summary=1` 只返回分组和数量
- `GET /api/video-groups/{name}/videos?page=&page_size=&q=&sort=`
- `POST /api/media/sync` 启动或加入后台目录同步；`GET /api/media/sync/status` 返回阶段、错误和 `catalog_revision`
- `GET/POST /api/media/collections`
- `GET /api/media/assets?collection_id=&role=&source_kind=&q=&page=`
- `GET /api/media/assets/{id}`
- `GET /api/media/assets/{id}/content`，支持 HTTP Range
- `GET /api/media/assets/{id}/thumbnail`，按需生成文件目录视频缩略图
- `GET /api/media/sources?role=gt|pred`，只返回 source/upload 资产
- `GET /api/media/run-outputs`，按 Run、视频和 Track 返回有效 Pred
- `GET /api/media/audit`
- `POST /api/uploads`、`PUT /api/uploads/{id}/parts/{index}`、`POST /api/uploads/{id}/complete`
- `GET/DELETE /api/uploads/{id}`
- `GET /api/evaluation-campaigns`，合并 V2 与只读 v1 列表
- `POST /api/evaluation-campaigns/v2/preview`
- `POST /api/evaluation-campaigns/v2`
- `GET /api/evaluation-campaigns/v2/{id}`、`/analysis`、`/export`
- `POST /api/evaluation-campaigns/v2/{id}/publish|close|archive`
- `GET /api/blind/{token}`、`POST /api/blind/{token}/session`
- `GET /api/blind/{token}/reviews?evaluator_id=`、`GET /api/blind/{token}/reviews/{task_token}?evaluator_id=`
- `POST /api/blind/{token}/tasks/{task_token}/vote|heartbeat`
- `GET /api/compare-sources/gt?page=&page_size=&q=&group=`
- `GET /api/compare-sources/pred?page=&page_size=&q=&video=&run_id=`
- `GET /api/video-thumbnails/{key}`
- `GET /api/metrics/health`
- `GET /api/health`，只读返回目录同步状态和 Run/Campaign 清理积压
- `POST /api/preflight`，通过 `preflight_level=quick|deep` 选择层级；`POST /api/preflight/quick` 是强制 quick 的别名；只有 deep 返回 `preflight_token`
- `POST /api/runs`，支持 `model_file + video_group`、可选 `preflight_token` 和高风险确认 `risk_ack_fingerprint`
- `POST /api/runs` 支持 `checkpoint`、`execution_mode=single|multi_cuda|multi_npu`、`devices=["cuda:0"]` / `devices=["npu:0"]`、`batch_size_per_device`
- `POST /api/runs/{id}/cancel`
- `POST /api/runs/{id}/retry` 精确复现已记录输入；输入变化时返回 `409 InputIdentityChanged`
- `POST /api/runs/{id}/clone` 使用当前文件创建新 Run
- `POST /api/run-purge/preview`，body 为 `request_type=delete_run|cleanup_artifacts` 与精确 `run_ids`
- `DELETE /api/runs/{id}?preview_token=...` 创建持久化清理请求并返回 `202`
- `GET /api/run-purge-requests/{request_id}`
- `POST /api/runs/{id}/cleanup-artifacts` 在 body 中携带 `preview_token`，使用同一幂等清理服务但保留 Run
- `POST /api/runs/batch-delete` 在 body 中携带精确 `run_ids` 及对应 `preview_token`
- `GET /api/storage/gc/preview`、`POST /api/storage/gc`（必须 `confirm=true`）
- `GET /api/runs`
- `GET /api/runs/{id}`
- `GET /api/runs/{id}/videos?page=&page_size=&q=`
- `GET /api/runs/{id}/videos/{video_name}/timeline?metric=&bucket_count=&window_start=&window_size=`
- `GET /api/runs/{id}/samples/{sample_id}`
- `GET /api/runs/{id}/artifacts`
- `GET /api/runs/{id}/timeline`
- `GET /api/runs/{id}/metric-summary`
- `POST /api/runs/{id}/metrics/retry`
- `GET /api/compare?run_id=1,2`
- `GET /api/compare/samples?run_id=1,2&video_name=...&frame_index=...`
- `POST /api/workers/register`
- `POST /api/jobs/{id}/heartbeat`

旧的 `models/datasets/jobs/experiments` API 仍保留用于兼容和调试，但不属于主 UI 流程。

## NPU、多卡与大图预览

- 多卡 NPU 使用 `torch_npu`，设备名为 `npu:0`、`npu:1` 等；`/api/devices` 会返回检测到的 NPU 列表。
- `execution_mode=multi_npu` 优先按视频拆分；视频数量不足或负载明显不均时把长视频切为连续 sample segment。每个 shard 绑定一个 NPU，只写帧与 manifest，随后由 `finalize` job 合并视频并启动指标。
- 手动启动 worker 时可以使用 `python -m vfieval.cli --workspace .vfieval worker --role inference --device-filter npu:0 --idle-timeout 120`，绑定后的 worker 只领取对应 NPU 的 shard。
- 启动前深度预检查会在最终 device/dtype 上执行模型 dry-run，能提前暴露 CPU/NPU tensor 不匹配；表单自动快速预检不会构造模型。
- 新建任务页首屏只读取模型/视频分组摘要、Run 摘要、设备和指标健康；视频列表、Compare 来源和预检查会在用户加载或选择后再请求。
- Run Detail 默认加载 `512px` 预览图，原图只在点击预览时打开；核心产物按 `图像 / Flow / Mask / Warp` 分组，避免一次性加载十张 4K 图。
- 删除 Run 会同时进入产物清理队列；若只想释放产物并保留记录，使用 `只清产物`。详情页会显示取消、清理、失败或完成状态。
- 推理产物发布、指标完成或产物清理会递增 `content_revision`；Run Detail 原位失效缓存并刷新，不需要刷新整个浏览器页面。
- 指标缓存 key 会绑定 metric 名称、适配器/资产版本、当前 evaluator 环境以及 GT/Pred 文件身份；更换 manifest、权重或可执行环境后不会继续复用旧的 `unavailable` 缓存结果。

每个推理 Run 会写入 `reference_key` 和 `reference_config`，用于后续确认多个模型或不同权重是否基于同一组 GT、同一视频子集、同一 frame step 和同一输出分辨率。未来 Compare 只比较相同 `reference_key` 的结果，默认最多展示 `GT + Pred A + Pred B`。

## 验收流程

```text
1. 启动项目
2. 打开页面
3. 进入新建推理
4. 下拉框里看到 test_average.py 和 test_img0.py
5. 下拉框里看到 test_style
6. 选择 test_average.py + test_style
7. 点击开始推理
8. 页面显示进度
9. 推理完成
10. 打开运行记录
11. 能直接看到 pred 视频、gt 视频、diff 视频
12. 再用 test_img0.py 跑一次
13. 能肉眼看到两个模型输出不同
```

## 测试

```powershell
python -m unittest discover -s tests
git diff --check
```
