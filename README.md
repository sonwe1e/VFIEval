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

Web UI 会自动扫描：

- `models/*.py` 作为模型下拉框。
- `checkpoints/{model_stem}/*` 作为权重下拉框，可选择不加载、自动最新或具体文件。
- `videos/*/` 作为视频集下拉框；首屏只读取分组摘要，具体视频、缩略图和缓存状态会在选择视频集后按页加载。
- 点击 `刷新文件列表` 后会重新扫描模型、权重、视频集和设备，不需要刷新浏览器页面。
- 选择稳定后自动预检查模型接口、已选视频、真实帧数、triplets、缓存状态和设备/精度；重复的模型 dry-run 会复用缓存，提交任务前仍会强制检查一次。
- 视频列表默认全选，支持服务端分页、按文件名搜索、排序和缩略图，也可以只勾选本次要推理的视频。

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
      pred_frames/
      gt_frames/
      diff_frames/
```

Run Detail 默认不会一次性加载所有视频和所有图片。页面先读取 Run 的视频摘要，再按当前视频加载窗口化 `timeline`；只有选中某个样本时，才调用 `GET /api/runs/{id}/samples/{sample_id}` 加载对应的 `GT / Pred / Diff / Flow / Mask / Warp / Blend`。

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
- 逐帧指标按 triplet 样本显示曲线；点击曲线点会定位到对应帧。
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
- `vmaf`: this is the first metric that can run immediately in the current build. Install `ffmpeg` with the `libvmaf` filter on `PATH`, or set `set/metrics/vmaf/manifest.json -> ffmpeg_path` to a project-local portable ffmpeg binary.

Current remaining-metric adapters:

- `lpips_vit_patch` is a sample-level DINOv2 feature-distance metric. Default backbone: `dinov2_vits14_reg`. Evaluation uses max edge `518`, preserves aspect ratio, and pads to a multiple of `14`.
- `lpips_convnext` is a sample-level ConvNeXt V2 feature-distance metric. Default backbone: `convnextv2_tiny.fcmae_ft_in22k_in1k`. Evaluation uses max edge `288`, preserves aspect ratio, and pads to a multiple of `32`.
- `cgvqm` is a video-level wrapper around a local IntelLabs CGVQM checkout. Evaluation videos are written under the metric work directory with long edge capped at `720`; original artifacts are not overwritten.
- Metric jobs inherit the Run's inference device through `metric_device`. If CUDA/NPU metric warmup fails, VFIEval records `unavailable` with the device and reason instead of falling back to CPU.
- `prepare-metrics --check-only` is read-only. `prepare-metrics --force` replaces only VFIEval-declared metric assets, not unrelated user files. Python packages such as `timm`, `safetensors`, `av`, and `scipy` are never installed automatically.

Status interpretation:

- `missing_weights`: the expected manifest or referenced project-local assets are not present yet.
- `missing_evaluator`: the manifest may exist, but the declared driver command, interpreter, or system executable is still unavailable.
- `missing_dependency`: the Python package dependency for that metric is missing from the current environment.
- `available`: the current build can attempt to execute that metric without substituting another score.

Metric assets live under `set/metrics/` by default, or under `VFIEVAL_METRIC_ASSETS_DIR` when that environment variable is set. Missing dependencies, failed downloads, or unsupported evaluator devices are recorded as `unavailable`; VFIEval never substitutes a different metric score.

## 统一媒体资产与 External 上传

Media Library 以稳定 `asset_id` 统一索引 `videos/` 文件、External 上传和 Run 视频产物。目录扫描与历史 Run 会幂等回填到 Catalog；原文件不移动，Catalog 只保存服务端路径、SHA-256、媒体信息和 provenance。Run 清理只把对应 `run_artifact` 资产标记为 `unavailable`，不会破坏源素材、Campaign、正式盲评投票或统计历史。

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

## 多人盲评与 Campaign

浏览器生成稳定 evaluator UUID，评测员只需填写显示名；这是可信局域网体验标识，不是身份认证。Campaign 发布前会校验所有 GT/候选资产及严格对齐关系，并为每个视频生成平衡 round-robin 候选对。左右顺序按 `(task, evaluator, seed)` 稳定随机化。

参与者接口只返回 task-side 播放 URL，投票前不返回模型、权重、Run 或真实 asset id。每名评测员对同一任务只有一行投票，可修改；选项为 A、B 或平局，并可记录清晰度、时序稳定、重影、伪影、运动自然度、置信度和备注。

分析页分别展示 Head-to-head 胜率、Bradley–Terry 排名、固定种子 1000 次 bootstrap 的 95% 区间、覆盖率/`provisional` 状态、评测员票数、一致率和质量原因。客观指标保持各自方向与 `completed/unavailable/failed/skipped` 语义，不生成主客观混合总分。Campaign 支持完整 CSV/JSON 导出。

## API

主流程端点：

- `GET /api/model-files`
- `GET /api/checkpoints?model_file=...`
- `GET /api/devices`
- `GET /api/video-groups`，支持 `summary=1` 只返回分组和数量
- `GET /api/video-groups/{name}/videos?page=&page_size=&q=&sort=`
- `GET/POST /api/media/collections`
- `GET /api/media/assets?collection_id=&role=&source_kind=&q=&page=`
- `GET /api/media/assets/{id}`
- `GET /api/media/assets/{id}/content`，支持 HTTP Range
- `GET /api/media/audit`
- `POST /api/uploads`、`PUT /api/uploads/{id}/parts/{index}`、`POST /api/uploads/{id}/complete`
- `GET/DELETE /api/uploads/{id}`
- `GET/POST /api/evaluation-campaigns`
- `POST /api/evaluation-campaigns/{id}/candidates|publish|close`
- `GET /api/evaluation-campaigns/{id}/next|analysis|export`
- `POST /api/evaluation-tasks/adhoc`
- `POST /api/evaluation-tasks/{id}/votes`
- `GET /api/compare-sources/gt?page=&page_size=&q=&group=`
- `GET /api/compare-sources/pred?page=&page_size=&q=&video=&run_id=`
- `GET /api/video-thumbnails/{key}`
- `GET /api/metrics/health`
- `POST /api/preflight`
- `POST /api/runs`，支持 `model_file + video_group`
- `POST /api/runs` 支持 `checkpoint`、`execution_mode=single|multi_cuda|multi_npu`、`devices=["cuda:0"]` / `devices=["npu:0"]`、`batch_size_per_device`
- `POST /api/runs/{id}/cancel`
- `POST /api/runs/{id}/retry`
- `DELETE /api/runs/{id}` 软删除 Run，默认运行记录不再显示
- `POST /api/runs/{id}/cleanup-artifacts` 清理 `.vfieval/runs/{run_id}` 产物，仅允许在 `completed / failed / canceled` 后执行
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
- 预检查会在最终 device/dtype 上执行模型 dry-run，能提前暴露 CPU/NPU tensor 不匹配。
- 新建任务页首屏只读取模型/视频分组摘要、Run 摘要、设备和指标健康；视频列表、Compare 来源和预检查会在用户加载或选择后再请求。
- Run Detail 默认加载 `512px` 预览图，原图只在点击预览时打开；核心产物按 `图像 / Flow / Mask / Warp` 分组，避免一次性加载十张 4K 图。
- 失败或不想看的 Run 可以删除记录；产物清理需要单独点击，避免误删结果。
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
