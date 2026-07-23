# VFIEval 发布与环境验证

VFIEval 的正式发布物是带完整 Git commit SHA 的 wheel。发布门禁不仅检查源码，还必须证明同一提交可生成字节一致的 wheel，并在隔离环境中通过 CLI、静态资源和服务启动冒烟。CPU 门禁由 CI 重复执行；CUDA/NPU 仍需在目标硬件上完成发布前验证。

---

## 支持与验证矩阵

| 运行环境 | Python | 核心依赖策略 | 自动门禁 | 发布前硬件门禁 |
|---|---|---|---|---|
| Linux CPU | 3.10、3.12 | `pyproject.toml` 核心依赖 | Ruff、完整单测、JS 语法、wheel 双构建、隔离安装与服务冒烟 | 不需要 |
| Linux CUDA | 3.10、3.12 | 选择与驱动匹配的官方 PyTorch CUDA wheel；记录 `pip freeze` 和驱动信息 | CPU 兼容门禁 | `doctor`、真实小型 Run、视频编码、指标不可用语义、取消 |
| Linux Ascend NPU | 目标部署 Python；必须使用彼此兼容的 PyTorch、`torch-npu`、CANN 组合 | 依部署清单锁定完整版本，不依赖宽松的 `>=` 自动解析生产环境 | CPU 兼容门禁 | 每张目标 NPU 的预检、连续多卡 Run、重启续跑、取消和指标设备失败 |

项目本身支持 Python 3.10 及以上，但“可安装”不等于“已在加速器上验证”。CUDA/NPU 发布必须保存目标机的 `doctor --json`、驱动/运行时版本、wheel SHA-256 和实际 Run 结果；常规 CI 不把 CPU 结果冒充为加速器证据。

---

## 构建正式 wheel

从干净且已提交的工作树执行：

```bash
python -m pip install -e ".[release]"
python tools/build_release.py --require-clean --verify-reproducible --out-dir dist
```

构建器使用提交时间作为 `SOURCE_DATE_EPOCH`，把完整 commit SHA 写入 wheel 内的 `_build_info.json`，连续构建两次并比较 wheel 的 SHA-256。成功后 `dist/` 同时包含 wheel 和对应 `.sha256` 文件；源码工作树中的 `_build_info.json` 会恢复为 `development`，不会留下生成性修改。

`GET /api/health` 的 `release.build_id` 和 `release.commit_sha` 必须等于构建提交。`VFIEVAL_BUILD_ID` 只保留为受控部署覆盖项，正常 wheel 发布不依赖它才能识别版本。

---

## 全新环境冒烟

发布候选应在新的虚拟环境中从 wheel 安装，而不是从源码目录或 editable install 启动：

```bash
python -m venv .release-venv
.release-venv/bin/python -m pip install dist/vfieval-*.whl
cd /tmp
/path/to/repo/.release-venv/bin/python /path/to/repo/tools/release_smoke.py \
  --expect-build-id <完整 commit SHA> \
  --forbid-source-root /path/to/repo/src
```

冒烟会验证四件事：安装后的 `vfieval` CLI 可解析命令；服务能从空工作区启动；wheel 中由 `index.html` 引用的全部本地 JavaScript 与 CSS 可被真实 HTTP 请求读取；健康接口返回嵌入 wheel 的提交身份。CI 会创建不继承系统包的全新虚拟环境，从 wheel 安装 VFIEval 及其声明依赖并执行 `pip check`，避免 editable install 或宿主环境掩盖缺包问题。

---

## 发布验收清单

1. `python -m ruff check src tests tools`、`python -m unittest discover -s tests`、全部网页脚本语法检查和 `git diff --check` 通过。
2. `tools/build_release.py --require-clean --verify-reproducible` 成功，wheel 与 `.sha256` 一并保存。
3. 全新环境冒烟返回 `status: ok`，健康接口的 commit SHA 与发布提交一致。
4. CUDA/NPU 目标机分别保存带部署监听目标和每张目标卡的 `doctor --host <host> --port <port> --device cuda:0 --json` 或 `--device npu:0 --json`；多卡时重复 `--device`。确认退出码为 `0` 后，再完成计划要求的真实推理、重启调度、取消、编码和指标失败验证。退出码 `2` 代表端口、设备、FFmpeg/`libx264` 或指标能力不可用，退出码 `1` 代表数据库、存储或探测执行失败。
5. 修改 CLI 命令、关键路由、网页入口或静态资源时，同步更新 `contracts/public_surface.json` 与 `NAVIGATION.md`；`tests/test_release_contracts.py` 会阻止入口静默漂移。
