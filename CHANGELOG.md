# CHANGELOG

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
