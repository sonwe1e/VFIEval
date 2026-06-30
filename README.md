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
- `videos/*/` 作为视频集下拉框。
- 选择后自动预检查模型接口、已选视频、真实帧数、triplets、缓存状态和设备/精度。
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
- 输出必须与输入同 batch、同 H/W、同 device、同 dtype。
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
```

## API

主流程端点：

- `GET /api/model-files`
- `GET /api/video-groups`
- `GET /api/video-groups/{name}/videos?page=&page_size=&q=&sort=`
- `GET /api/video-thumbnails/{key}`
- `GET /api/metrics/health`
- `POST /api/preflight`
- `POST /api/runs`，支持 `model_file + video_group`
- `POST /api/runs/{id}/cancel`
- `POST /api/runs/{id}/retry`
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
