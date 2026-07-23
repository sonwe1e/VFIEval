# VFIEval

VFIEval 是面向视频插帧（VFI）模型的本地推理评测平台：对用户提供的模型文件在视频数据集上执行推理，产出插帧结果与核心中间产物（flow/mask/warp/blend），计算质量指标，并支持 GT/Pred 对比（Compare）与多人盲评（Campaign）。平台不做训练，不实现 PSNR。

主要能力：

- 文件入口式推理：放模型文件、权重与视频文件夹即可运行，不需要注册模型/数据集或填写内部 ID。
- 多卡 CUDA/NPU 分片推理，产物档位与 benchmark 性能指纹。
- 核心产物检视：pred/gt/diff 视频与帧级 flow/mask/warp/blend 预览、指标曲线与最差样本定位。
- Compare：同一 GT 与最多 2 份 Pred 的严格对比。
- 指标：`lpips_vit_patch`、`lpips_convnext`、`vmaf`、`cgvqm`。
- 统一媒体目录与分片 External 上传。
- Campaign V2 多人盲评：opaque 参与链接、平局半胜与 Bradley–Terry 排名。

## 目录

- [快速开始](#快速开始)
- [文件入口](#文件入口)
- [新建推理](#新建推理)
- [运行记录与结果](#运行记录与结果)
- [指标](#指标)
- [Compare 对比](#compare-对比)
- [媒体目录与上传](#媒体目录与上传)
- [多人盲评 Campaign V2](#多人盲评-campaign-v2)
- [Run 生命周期](#run-生命周期)
- [部署与运维](#部署与运维)
- [Benchmark 与性能配置](#benchmark-与性能配置)
- [模型接口契约](#模型接口契约)
- [后处理公式](#后处理公式)
- [API 参考](#api-参考)
- [验收流程](#验收流程)
- [测试](#测试)

## 快速开始

环境要求：Python ≥ 3.10；核心依赖 `torch>=2.1`、`numpy>=1.24`、`Pillow>=9.0`、`opencv-python>=4.5`（`pip install -e .`，或如下设置 `PYTHONPATH`）；NPU 部署需要可选依赖 `torch-npu>=2.1`。

```powershell
$env:PYTHONPATH='src'
python -m vfieval.cli --workspace .vfieval init
python -m vfieval.cli --workspace .vfieval serve --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765`。页面主流程只有两步：

- `新建推理`：选择模型文件、视频集、部分或全部视频、分辨率、设备和精度，然后开始推理。
- `运行记录`：查看推理/评测阶段、错误、输出目录、指标曲线、坏帧列表和按需加载的核心产物。

默认测试模型和测试视频已随工作区提供，也可以重新生成：

```powershell
python scripts\generate_test_assets.py
```

## 文件入口

模型文件放在项目根目录的 `models/`，权重放在 `checkpoints/{model_stem}/`，视频按风格、作品或测试集放在 `videos/{视频集}/`，指标资产放在 `set/metrics/`（由 `prepare-metrics` 生成，见[指标](#指标)）：

```text
models/
  test_average.py
  my_model.py
checkpoints/
  my_model/
    latest.pth
videos/
  test_style/
    gradient_motion.avi
  anime_style/
    001.mp4
```

Web UI 以文件夹为入口：

- `models/*.py` 作为模型下拉框；`checkpoints/{model_stem}/*` 作为权重下拉框，可选择不加载、自动最新或具体文件。
- `videos/*/` 作为视频集；首屏只读取分组摘要，具体视频在选择视频集后按页加载（支持搜索、排序、缩略图，缩略图按需生成）。
- 服务启动时在后台协调一次 Media Catalog 回填；`刷新文件列表` 只同步文件入口，Media 页的 `刷新` 还会显式纳入历史 Run 资产。两者都会合并进已在运行的同步任务，完成后原地刷新。

## 新建推理

表单要点：

- 支持单视频集或多视频集（多选时 Run 以 `videos/` 为根，clip 记为 `组/文件`；单组 Run 与旧行为完全一致）。视频列表默认全选，可只勾选本次要推理的视频。
- 检测到 NPU 时表单使用保守起点：默认多卡、精度 `auto`、总 batch 与每设备 batch 均为 `1`，可在看过深度预检后再提高吞吐。
- 产物档位：`evaluation`（默认，保存 Pred/GT/Diff 无损帧、视频及指标所需输入）、`diagnostic`（额外保存 Flow/Mask/Warp/Blend/`extra_*`）、`benchmark`（不保存媒体产物、不跑指标，只输出性能数据）。保存队列有界并施加背压，artifact 记录批量写入 SQLite。

两级预检：

- 表单变化只执行快速预检：校验受信任路径、选择、设备/精度和容器媒体信息，给出样本量与工作量估算；不加载模型做 dry-run，也不返回可创建 Run 的令牌。
- 点击开始时强制执行深度预检：检查模型输出契约、解码缓存和媒体容器信息，并为模型、权重和所选视频生成内容签名。响应中的 `input_fingerprint` 标识这组物理输入；短期有效的 `preflight_token` 同时绑定配置与输入。`POST /api/runs` 会重新校验并复用已验证哈希，任一文件变化、配置变化或令牌过期都必须重新预检。
- 已完成的解码清单提供精确帧数；否则显示容器帧数估算警告，并由解码任务在推理前复核。
- 预检显示设备与卡数、精度、解码后端、评测契约、显存/主机内存、磁盘余量和产物空间预算。数字是资源规划下界；高风险配置必须看过数值后确认，提交时携带匹配的 `risk_ack_fingerprint`。

全参考评测契约 `midpoint-triplet-v2`：`frame_step=s` 时只生成真实对称的 `(i, i+s, i+2s)` 三元组，N 帧产生 `N-2s` 个可信样本，末尾不再用钳制帧伪造 GT。缺少该契约或含历史 `clamped_boundary` 样本的 Run 仍可查看，但必须用当前版本重跑后才能创建或发布新的盲测 Campaign。

多卡执行：`execution_mode=multi_cuda|multi_npu` 优先按视频拆分，视频不足或负载明显不均时把长视频切为连续 sample segment。每个 shard 绑定一个设备、只写帧与 manifest，随后由 `finalize` job 统一编码视频并启动指标。

视频的 `width`、`height` 和“分辨率”均指应用旋转元数据并自动旋转解码后的显示方向尺寸。编码层原始宽高只保留用于诊断，不参与对齐或缩放判断。

## 运行记录与结果

Run 分为推理与评测两个阶段，Run Detail 分别显示。每个 Run 的主要输出：

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

`Pred / GT / Difference` PNG 和 `pred / gt / diff` 视频是 `canonical-v1` 核心制品，尺寸由 Run 的输出分辨率决定。高级设置中的 `visualize_height / visualize_width` 只控制 LANCZOS 预览，不改变推理合成、指标、Compare/Campaign 输入或媒体目录；预览尺寸与 canonical 相同时不会重复写文件。`GET /api/files/{artifact_id}` 始终返回 canonical，`?variant=preview` 返回预览。4K Run 保留完整分辨率 PNG 和视频，磁盘占用与编码耗时明显更高。

Run Detail 默认不一次性加载所有内容：先读取视频摘要并播放预览视频，再按当前视频加载最多 160 帧的细节 timeline（长视频同时显示固定桶数的全片 min–max/均值总览）；选中某个样本时才加载该样本的 `GT / Pred / Diff / Flow / Mask / Warp / Blend` 预览；原图和原视频只在点击后加载，原视频支持 HTTP Range。核心产物按 `图像 / Flow / Mask / Warp` 分组，默认加载 `512px` 预览图。

Run 的产物、指标或清理状态变化时服务端递增 `content_revision`。前端检测到变化后中止旧请求、失效当前 Run 的结果缓存，并原位恢复当前视频、帧位置、时间线窗口和指标选择，不需要整页刷新；详情页的 `刷新结果` 可强制执行同样的局部刷新。

推理和 Compare 发布到浏览器的 MP4 固定使用 H.264（`libx264`）、`yuv420p`、偶数尺寸与 `faststart`。如果当前 `ffmpeg` 缺少 `libx264` 且 OpenCV 也无法写出 AVC，Run 会以明确错误失败，不会静默生成浏览器通常无法播放的 `mp4v` 产物。

### 指标曲线

- 逐帧指标按 triplet 样本显示曲线；长视频使用“全片总览 + 160 帧细节窗口”，悬停或点击曲线定位到最近帧。
- `VMAF` / `CGVQM` 只产生视频级结果时只显示视频级摘要，不伪造成逐帧曲线。
- “最差样本”列表按当前指标排序，点击后直接跳到对应帧。

### 运行反馈与统计

每个 Run 可以提交 1.00–5.00 评分（0.25 步长）和/或问题描述。反馈自动关联内容范围：视频、track、模型和 checkpoint（Compare track 会追溯到产生该 Pred 的源 Run）。反馈支持编辑和删除；`统计` 视图按数据集、模型、checkpoint、视频、用户和 Run 汇总评分分布与均值，并可按这些维度过滤。

## 指标

高级设置里可以选择指标：`lpips_vit_patch`、`lpips_convnext`、`vmaf`、`cgvqm`。不支持也不会实现 `PSNR`。

- 推理完成后进入评测阶段。`GET /api/metrics/health` 显示每个指标的依赖、资产状态、来源 URL、固定评测分辨率、manifest 与可执行文件路径。
- 指标资产默认位于 `set/metrics/`（可用 `VFIEVAL_METRIC_ASSETS_DIR` 覆盖）。缺少官方资产或绑定时记为 `unavailable`，不会替换成其它分数。

准备或检查指标资源：

```powershell
python -m vfieval.cli --workspace .vfieval prepare-metrics --check-only
python -m vfieval.cli --workspace .vfieval prepare-metrics
python -m vfieval.cli --workspace .vfieval smoke-metric --metric vmaf --reference <gt_video> --distorted <pred_video>
```

`prepare-metrics` 把缺失的默认资产下载到 `set/metrics/` 并写入可运行的 manifest；它不安装 Python 包或系统二进制。`--check-only` 只读且不会创建或迁移工作区数据库；`--force` 只替换 VFIEval 声明的资产，不动无关用户文件。指标 CLI 统一用退出码 `0` 表示健康或完成、`2` 表示所需能力不可用、`1` 表示准备或执行失败。

各指标 setup：

- `vmaf`：开箱即用的首选指标。依次检查 `manifest.json -> ffmpeg_path`、`set/metrics/vmaf/ffmpeg.exe`（或 `ffmpeg`）、`PATH` 上的 `ffmpeg`；选中的二进制必须提供 `libvmaf` filter。可在 manifest 中指定独立 FFmpeg，浏览器产物编码仍使用 `PATH` 上的 `ffmpeg`。
- `lpips_vit_patch`：`prepare-metrics` 下载 DINOv2 checkout 与 ViT-S/14 register 权重。样本级特征距离指标，评测长边 `518`、保持宽高比、补齐到 `14` 的倍数。
- `lpips_convnext`：`prepare-metrics` 下载 ConvNeXt V2 tiny checkpoint。样本级特征距离指标，评测长边 `288`、保持宽高比、补齐到 `32` 的倍数。
- 两个特征距离适配器都要求权重名称与形状完整匹配，并在加载后执行确定性一致性检查；权重、实现指纹和缓存契约见 [LPIPS feature-distance adapters](docs/feature-distance-metrics.md)。
- `cgvqm`：`prepare-metrics` 下载 IntelLabs CGVQM checkout 并生成 VFIEval JSON 驱动脚本。视频级指标，评测视频长边上限 `720`，写入指标工作目录，不覆盖原产物。驱动使用启动 VFIEval 的同一 Python 解释器，因此该环境必须装好兼容的 `torchvision`、`av`（NPU 环境还需对应 NPU 包）。

健康状态含义：

- `missing_weights`：期望的 manifest 或项目内资产尚未就位。
- `missing_evaluator`：manifest 存在，但声明的驱动命令、解释器或系统可执行文件不可用。
- `missing_dependency`：该指标的 Python 依赖缺失（`timm`、`safetensors`、`av`、`scipy` 等从不自动安装）。
- `available`：当前构建可以尝试执行该指标，不会替换为其它分数。

执行口径：

- 指标任务通过 `metric_device` 继承 Run 的推理设备。多卡 Run 为每张推理卡创建一个逐帧指标分片；视频级指标只在 leader 分片执行一次。
- LPIPS 在每个分片中只加载一次模型并批量处理图像对，默认每卡 batch 为 ViT `8`、ConvNeXt `32`；高级设置 `每卡评测 Batch Size` 可覆盖，设备 OOM 时在同一张卡上逐级减半，不回退 CPU。
- LPIPS 和 CGVQM 在模型构造前绑定请求的 `npu:<index>`；CUDA/NPU 评测失败时写入带设备及原因的 `unavailable`。
- 指标缓存 key 绑定指标名称、适配器/资产版本、当前 evaluator 环境以及 GT/Pred 文件身份；更换 manifest、权重或可执行环境后不会复用旧的 `unavailable` 缓存结果。
- Run Detail 的“重试失败/不可用指标”只重试最新状态仍为 `failed` 或 `unavailable` 的指标：绕过旧失败缓存，但复用有效的 `completed` 结果，历史结果行不会删除。
- CGVQM 读取驱动 stdout 中最后一个合法协议 JSON；非零退出、空输出、坏 JSON、超时和临时视频帧数异常会分别记录 return code、实际帧数以及有界 stdout/stderr，便于定位 `cgvqm failed: 1` 等问题。

## Compare 对比

Compare 用于把同一 GT 与 1–2 份 Pred 做严格对比（页面最多选择两份 Pred）。

来源与选择：

- GT 与 Pred 在规范化的 Media Item 中选择：一个 Media Item 聚合同一规范 GT 的源视频与各模型 Pred 成员。主选择器提交 GT Item 与 1–2 个 Pred 成员，GT 和 Pred 必须属于同一 Item。
- 后端 `resolve_compare_descriptor` 支持五种 `kind`：`video_group`、`media_item`（GT 角色），`run_artifact`、`media_asset`、`media_item_member`（Pred 角色）。不接受裸字符串 descriptor，也不信任客户端提供的路径。
- `flow/mask/warp/blend` extra layers 默认不选，展开后再选择需要对照的层。

对齐与分辨率：

- 时间身份严格一致：GT 与每份 Pred 的有序源帧索引、帧数、FPS 与可用时间戳必须一致；不截断、不偏移，空间缩放也不能修复时间或语义不匹配。
- VFIEval 自产 Pred 可使用平台记录的 `source_frame_indices` 重建推理分辨率的 aligned GT，并记录 `generated_from / aligned_gt_of / pred_of` 关系；Compare Inputs 面板可切换 aligned/original。
- 空间尺寸由明确的 Alignment Plan 归一化并记录指纹：一份 Pred 时跟随该 Pred，两份 Pred 时按既定规则选择较小像素面积；所有缩放使用 LANCZOS。External Pred 的宽高比拉伸需要在表单中显式勾选同意。

Compare Run Detail：

- Compare Inputs 面板显示 GT 与每份 Pred 的来源素材（aligned/original 切换）。
- 视频播放器按 track 显示（`{track_label}/{kind}`）；帧预览把 GT 与所有 Pred track 排成一行；样本级 extra layers 按 `kind × track` 网格展示。
- 可以勾选指标（与推理 Run 相同的指标集），按 track 统计。

## 媒体目录与上传

Media 页按用途分成三块：

- `Sources / Uploads`：`videos/` 中的规范来源与 External 上传。
- `Derived Runs`：按 `Run → 视频 → Track` 分组折叠显示有效 Pred 输出；已删除或已清产物的 Run 不会重新出现在目录、Compare 或 Campaign 来源中。
- `Evaluation Packages`：已发布 Campaign 的冻结评测包。自动生成的内部 Run Collection 不作为用户目录展示。

Media Catalog 以稳定 `asset_id` 统一索引源文件、上传、Run 视频产物和评测包。服务启动和用户显式刷新通过进程内协调器幂等回填，并发刷新合并为同一后台任务；常规列表、搜索和分页只读 SQLite 快照，不触发隐式重扫。原文件不移动，Catalog 只保存服务端路径、SHA-256、媒体信息和 provenance。Run 清理把对应 `run_artifact` 标记为 `unavailable`，冻结的 `evaluation_package` 仍可播放。

External 视频和帧序列通过浏览器分片上传：

- 固定 8 MiB 分片，支持重复分片幂等和重新提交续传，单资产默认上限 50 GiB。
- 视频保留原扩展名；帧序列使用 ZIP 并必须填写 FPS。
- 每片及完整文件均校验 SHA-256；拒绝路径穿越、符号链接、危险解压比、无效图片和混合尺寸。
- 服务端写入 `.vfieval/media/{collection}/{asset_uuid}/`；客户端不能提交磁盘路径。同一 Collection 内显示别名唯一。软删除被 Campaign/投票引用的资产时只禁用播放并保留分析记录。

## 多人盲评 Campaign V2

Evaluation Studio 集中管理创建、发布和组织者分析，参与页面独立。一个 V2 Campaign 固定比较两份 Pred 方法，主流程中的方法是一份完成 Run 的 Track；External Pred 只在高级入口使用。创建向导依次填写基本信息、选择 Track A 与 B、查看共同视频矩阵并勾选视频、确认严格对齐和任务量，然后发布。矩阵会明确显示缺失、GT 冲突以及帧数、尺寸、FPS、时间戳不一致；只有两种方法都有输出、共享同一规范 GT 且严格对齐的视频才能进入发布。

发布是可恢复的 preparation job：先在 staging 中逐视频深度校验，再生成 `.vfieval/evaluations/{campaign_id}` 与 SHA-256 manifest，把冻结文件登记为 `source_kind="evaluation_package"` 并原子发布任务。发布请求先持久化 `preparing/queued` 再回包；回包期间浏览器或反向代理断线只按客户端断线处理，不会把已排队的 preparation 标成失败。失败发布不留下半成品任务或垃圾 draft，可从 Studio 重试。发布后的配置和评测包不可变，可以关闭或归档；删除来源 Run 不影响播放。

冻结评测包只冻结 GT、Pred A、Pred B 三路（新包不再产出 Diff），编码策略 `campaign-freeze-stream-v3`：H.264/yuv420p/faststart、CRF 18、约 1 秒固定 GOP，关闭场景切换关键帧和 B 帧。源 MP4 满足关键帧约束（首关键帧距首帧 ≤0.1 秒、最大关键帧间隔 ≤2 秒等）时 A/B 成对 remux，任一路不合格或失败则两路一起回退重编码；manifest 记录 GOP 策略与策略指纹，最终发布前重新校验全部源内容。Studio 进度显示 `current/total · phase`（已完整冻结的 Item 数）、当前 Item、帧进度、管线和阶段耗时。

参与页面：

- 参与者只打开独立的 `/evaluate/{opaque_token}` 页面，不加载主导航、Media 或管理分析。浏览器生成稳定 evaluator UUID，填写显示名后领取带可续期 lease 的任务；任务、assignment 和媒体 URL 全部使用 opaque token，投票前不暴露模型、checkpoint、Run、方法标签或真实 asset/task id。左右顺序按任务与评测员稳定随机。
- 投票规则：A 更好、B 更好或平局必须选择；A、B 可分别填写可空的 1–5 分（0.25 步长），并可记录置信度和备注。评分独立统计，不改变平局半胜和 Bradley–Terry 排名。
- 默认大画幅分割线对比：GT 全宽单独显示，候选 A/B 在下方共享一个全宽画面，通过可拖动、可触摸和可键盘操作的竖向分隔线查看同一位置的差异；“完整视图”切换为 GT、A、B 三段纵向排列。浏览器不支持所需 CSS 裁剪能力时自动使用完整视图。视频任务由 sticky 控制条统一控制播放、帧选择、速率和循环。
- 时长 ≤30 秒且三路合计 ≤256 MiB 的任务先完整下载为任务级 Blob 再开放播放；其他任务使用 Range 流式预缓冲，取流不足时对齐暂停并自动恢复，真实媒体错误会阻止投票并提供重试。
- 完成页列出本人已评视频；已发布 Campaign 可回看并修改原答案，关闭或归档后只读。

组织者分析：参与者完成个人全部可评任务后才看到实时结果；组织者可始终查看覆盖率、方法级分析和每方法/每视频评分填写数、均值与中位数。人类结果使用平局半胜、Bradley–Terry 和固定种子 bootstrap 区间；客观指标先按 Item × 方法 × 指标汇总，再让各 Item 等权进入总体统计，并保留逐帧覆盖与真实 unavailable 原因。两类结果不合成总分。旧 schema v1 Campaign 保持只读，可导出和归档，不按标签猜测迁移。

## Run 生命周期

文件入口的模型推理 Run 提供两种重新执行语义：

- `重试` 是精确复现：固定使用原 Run 当时解析出的 checkpoint，并重新校验模型、checkpoint、所选 Item/Asset、源视频内容及执行配置。任一输入缺失或变化时返回 `409 InputIdentityChanged` 和结构化差异，不会悄悄改用当前文件；没有输入身份记录的旧 Run 应使用 Clone。
- `按当前输入 Clone` 重新解析磁盘上的当前模型、`auto` checkpoint 和源文件，创建新 Run，并记录来源 Run 及输入身份差异。

删除与清理是持久化的异步流程：

- 删除、批量删除和“只清产物”都会先调用 `POST /api/run-purge/preview`，展示 Campaign/Compare/活动 Job 依赖、Run 目录占用、共享缓存和预计可回收空间。预览返回的 `preview_token` 短期有效、一次性使用，并绑定操作类型、精确 Run 集合及当前状态；状态变化、令牌过期或选择变化时必须重新预览。
- `DELETE /api/runs/{id}?preview_token=...` 返回 `202 Accepted` 并创建 `run_purge_requests`。运行中的 Run 会先请求取消，worker 停止后才删除受信任的 `.vfieval/runs/{id}`、清理产物索引与 Run feedback，并把对应 Run 媒体标记为不可用。只有全部步骤成功后才写入 `deleted_at`；失败请求保留错误与清理报告可重试，服务重启后继续处理。
- `cleanup-artifacts` 走同一套幂等服务但保留 Run 记录；批量删除逐 Run 建立请求，单项失败不中断其它项。
- `decode_cache` 与 `compare_cache` 通过 `cache_entries`、`run_cache_refs` 和活动 lease 管理：共享缓存只有在最后一个有效 Run 引用释放、没有活动 lease 且超过默认 10 分钟宽限期后才可回收。指标缓存、视频元数据与缩略图缓存不随单个 Run 删除；历史残留和孤儿缓存必须先在 Media 页生成存储清理预览，再显式确认 GC。

已发布的 Campaign 媒体会在破坏性清理前冻结到独立评测包，因此删除来源 Run 不会破坏正式盲评播放、投票或分析历史。

## 部署与运维

正式发布使用带完整 commit SHA 的可重复 wheel，并在隔离环境中验证 CLI、网页静态资源和服务启动。CPU 3.10/3.12 自动门禁、CUDA/NPU 真机要求、构建命令和发布证据清单见 [发布与环境验证](docs/release-validation.md)。

服务默认绑定 `127.0.0.1`。参与链接与局域网访问：

- 在 Studio 中选择已发布的 Campaign 后复制“参与链接”。参与者不需要注册或密码：打开链接、填写显示名即可领取任务；同一浏览器复用其本机 evaluator 身份。
- 要让受控内网中的参与者访问，以明确的内网监听地址启动（`serve --host 0.0.0.0 --port 8765`），然后从 `http://<服务器内网 IP>:8765` 打开 Studio 并重新复制参与链接（链接使用当前页面来源地址）。不要把此服务直接暴露到公网。API 中的 `share_url` 保持相对 `/evaluate/{opaque_token}`，由 Studio 显示和复制时补全当前来源。
- `0.0.0.0` 只是监听地址，不能作为发给参与者的主机地址。经另一台服务器做端口映射时应使用参与者实际可访问的 IP 或域名，并优先使用完整 TCP 端口转发。使用 HTTP 反向代理时必须同时转发 `/evaluate/`、`/blind.js`、`/blind.css` 和 `/api/blind/`；可先从参与者设备直接打开 `http://<映射地址>/blind.js` 和 `http://<映射地址>/api/blind/<token>`，两者都应返回 `200`。
- 使用 Windows 管理机向局域网转发完整服务时，按 [Windows TCP relay 部署与自检](docs/deployment-windows-relay.md) 配置。

环境检查与诊断：

```powershell
python -m vfieval.cli --workspace .vfieval doctor
python -m vfieval.cli --workspace .vfieval doctor --json
python -m vfieval.cli --workspace .vfieval doctor --host 127.0.0.1 --port 8765 --device cuda:0
python -m vfieval.cli --workspace .vfieval diagnostics --run-id 123
python -m vfieval.cli --workspace .vfieval diagnostics --campaign-id 45
```

`doctor` 会验证准备监听的 host/port、FFmpeg 与 `libx264`，并可用可重复的 `--device` 明确要求 `cuda:0`、`npu:0` 等目标设备；未传参数时使用与 `serve` 相同的默认监听目标，也可由 `VFIEVAL_HOST`、`VFIEVAL_PORT`、`VFIEVAL_DEVICE` 提供部署目标。`doctor` 和 `diagnostics` 以 SQLite 只读模式打开已有数据库，不会创建或迁移它；诊断命令只写用户指定的 ZIP（未指定时写入工作区 `tmp/diagnostics/`）。CLI 退出码 `0` 表示健康或完成、`2` 表示端口占用、目标设备缺失、FFmpeg/编码器或指标依赖等能力不可用、`1` 表示数据库、存储或检查执行失败。运行日志按 JSONL 轮转写入 `.vfieval/logs/`。页面内部错误会显示 `support_id` 与 `request_id`，可直接用它们定位服务端日志；诊断包会脱敏工作区路径与令牌字段。

多卡与 worker：

- 多卡 NPU 使用 `torch_npu`，设备名为 `npu:0`、`npu:1` 等；`/api/devices` 返回检测到的 NPU 列表。
- 手动启动 worker：`python -m vfieval.cli --workspace .vfieval worker --role inference --device-filter npu:0 --idle-timeout 120`，绑定后的 worker 只领取对应设备的 shard。
- 启动前深度预检会在最终 device/dtype 上执行模型 dry-run，能提前暴露 CPU/NPU tensor 不匹配。
- Run、上传和盲测发布统一预留磁盘安全余量；容量不足会在写盘前返回 `507 StorageCapacityError`，不会留下半发布任务。

主要环境变量：

| 变量 | 作用 |
| --- | --- |
| `VFIEVAL_WORKSPACE` | 工作区根目录（默认 `.vfieval`） |
| `VFIEVAL_PROJECT_ROOT` | 项目根目录（默认工作区父目录），影响文件入口与 `set/metrics` 位置 |
| `VFIEVAL_METRIC_ASSETS_DIR` | 覆盖指标资产目录（默认 `<项目根>/set/metrics`） |
| `VFIEVAL_MODELS_DIR` / `VFIEVAL_CHECKPOINTS_DIR` / `VFIEVAL_VIDEOS_DIR` | 覆盖模型、权重、视频入口目录 |
| `VFIEVAL_VIDEO_FFMPEG` | 覆盖产物编码使用的 ffmpeg 路径 |
| `VFIEVAL_METRIC_DEVICE` | 覆盖指标评测设备 |
| `VFIEVAL_MIN_FREE_BYTES` | 存储安全余量的磁盘剩余阈值 |
| `VFIEVAL_HTTP_MAX_THREADS` | HTTP 服务线程上限（默认 64） |
| `VFIEVAL_UPLOAD_MAX_BYTES` / `VFIEVAL_UPLOAD_MAX_FILES` / `VFIEVAL_UPLOAD_STALE_SECONDS` | 上传大小、文件数与过期会话清理调参 |
| `VFIEVAL_COLLECTION_QUOTA_BYTES` | 上传 Collection 配额（默认 500 GiB） |
| `VFIEVAL_BUILD_ID` | 构建标识（诊断信息中显示，默认 `development`） |

## Benchmark 与性能配置

运行标准 benchmark（默认 warmup 10 batch、测量 200 个样本、重复 3 次）：

```powershell
python -m vfieval.cli --workspace .vfieval benchmark --model-file my_model.py --video-group anime_style --execution-mode multi_npu --device-id npu:0 --device-id npu:1 --precision fp16 --batch-size 4
```

最优 batch/prefetch/save 配置按模型哈希、权重哈希、分辨率、精度、设备型号/数量和产物档位写入 `execution_profiles`。后续 preflight 只展示历史建议；必须点击“应用这组建议”才会写入表单，不会静默覆盖当前设置。

## 模型接口契约

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

## 后处理公式

平台统一写死后处理：

- `grid_sample(mode="bilinear", padding_mode="border", align_corners=True)`
- `mask0 = sigmoid(mask0_logits)`
- `blend = mask0 * warp0 + (1 - mask0) * warp1`
- `mask1 = sigmoid(mask1_logits)`
- `pred = mask1 * img1 + (1 - mask1) * blend`
- `pred/blend/warp/diff` clamp 到 `[0, 1]`
- `difference = abs(pred - gt)`

## API 参考

主流程端点：

- 文件入口：`GET /api/model-files`、`GET /api/checkpoints?model_file=...`、`GET /api/devices`、`GET /api/video-groups`（`summary=1` 只返回分组和数量）、`GET /api/video-groups/{name}/videos?page=&page_size=&q=&sort=`
- 预检与 Run：`POST /api/preflight`（`preflight_level=quick|deep`）、`POST /api/preflight/quick`；`POST /api/runs`（支持 `model_file + video_group`/`video_groups`、`checkpoint`、`execution_mode=single|multi_cuda|multi_npu`、`devices`、`batch_size_per_device`、`preflight_token`、`risk_ack_fingerprint`）；`POST /api/runs/{id}/cancel|retry|clone|rename`
- Run 结果：`GET /api/runs`、`GET /api/runs/{id}`、`/videos?page=&page_size=&q=`、`/videos/{video_name}/timeline?metric=&bucket_count=&window_start=&window_size=`、`/samples/{sample_id}`、`/artifacts`、`/metric-summary`、`/timeline`（兼容/调试，返回 `X-Deprecated`）；`POST /api/runs/{id}/metrics/retry`
- Run 反馈：`POST/GET /api/runs/{id}/feedback`、`POST /api/runs/{id}/feedback/{fid}`（编辑）、`DELETE /api/runs/{id}/feedback/{fid}`、`GET /api/feedback`（统计与过滤项）
- 删除与 GC：`POST /api/run-purge/preview`（body 为 `request_type=delete_run|cleanup_artifacts` 与精确 `run_ids`）、`DELETE /api/runs/{id}?preview_token=...`（返回 `202`）、`GET /api/run-purge-requests/{id}`、`POST /api/runs/{id}/cleanup-artifacts`、`POST /api/runs/batch-delete`（body 携带 `preview_token`）、`GET /api/storage/gc/preview`、`POST /api/storage/gc`（必须 `confirm=true`）
- Compare：`GET /api/compare-sources/{gt,pred,flow,mask}`、`GET /api/runs/{id}/compare-inputs`、`GET /api/runs/{id}/compare-inputs/{slot}/media?variant=original|aligned`、`GET /api/compare?run_id=1,2`、`GET /api/compare/samples?run_id=1,2&video_name=...&frame_index=...`
- 媒体与上传：`POST /api/media/sync`（启动或加入后台同步）、`GET /api/media/sync/status`、`GET/POST /api/media/collections`、`GET /api/media/assets?collection_id=&role=&source_kind=&q=&page=`、`GET /api/media/assets/{id}`、`/{id}/content`（支持 Range）、`/{id}/thumbnail`、`GET /api/media/sources?role=gt|pred`、`GET /api/media/run-outputs`、`GET /api/media/audit`、`GET /api/media/items`、`GET /api/media/item-groups`、`GET /api/media/items/{id}/predictions`、`GET /api/media/items/{id}/external-predictions`、`GET /api/media/methods`、`GET /api/media/unbound-predictions`；`POST /api/uploads`、`PUT /api/uploads/{id}/parts/{index}`、`POST /api/uploads/{id}/complete`、`GET/DELETE /api/uploads/{id}`
- 盲评（管理端）：`GET /api/evaluation-campaigns`（合并 V2 与只读 v1）、`POST /api/evaluation-campaigns/v2/preview`、`POST /api/evaluation-campaigns/v2`、`GET /api/evaluation-campaigns/v2/{id}`（及 `/{analysis,export,preparation,objective-curve}`）、`POST /api/evaluation-campaigns/v2/{id}/publish|close|archive`、`DELETE /api/evaluation-campaigns/v2/{id}`、`GET /api/evaluation-cleanup-requests`（及 `/{id}/retry`）
- 盲评（参与者）：`GET /api/blind/{token}`、`POST /api/blind/{token}/session`、`GET /api/blind/{token}/reviews?evaluator_id=`、`GET /api/blind/{token}/reviews/{task_token}?evaluator_id=`、`POST /api/blind/{token}/tasks/{task_token}/vote|heartbeat`、`GET /api/blind/{token}/tasks/{task_token}/media/{reference,left,right}`、`GET /api/evaluators/session`
- 指标与健康：`GET /api/metrics/health`、`GET /api/health`；后者以 `live` 表示进程存活、以 `ready`/`ok` 表示可接收工作，并返回 `reasons`、队列年龄/消费者、租约、目录同步及 Run/Campaign 清理状态
- 资产与文件：`GET /api/files/{artifact_id}?variant=preview`、`GET /api/sample-files/{sample_id}/{img0|img1|gt}`、`GET /api/video-thumbnails/{key}`
- Worker 协议（内部）：`POST /api/workers/register`、`POST /api/jobs/claim`、`POST /api/jobs/{id}/heartbeat|progress|complete|fail`

旧的 `models/datasets/jobs/experiments/dashboard` API 与 v1 Campaign 管理路由仍保留用于兼容和调试，但不属于主 UI 流程。

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
