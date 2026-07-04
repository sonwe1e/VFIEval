from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.pipeline.inference import _write_mp4
from vfieval.server import _make_handler


def make_workspace(tmp: str | Path) -> tuple[WorkspaceConfig, Database]:
    workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
    workspace.ensure()
    db = Database(workspace.db_path)
    db.init()
    return workspace, db


def write_mp4(path: Path, colors: list[tuple[int, int, int]], size: tuple[int, int] = (8, 8), fps: float = 5.0) -> Path:
    frame_dir = path.parent / f"{path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, color in enumerate(colors):
        frame_path = frame_dir / f"{index:06d}.png"
        Image.new("RGB", size, color).save(frame_path)
        frames.append(frame_path)
    _write_mp4(frames, path, fps)
    return path


def add_completed_pred_run(
    db: Database,
    workspace: WorkspaceConfig,
    name: str,
    pred_video_path: Path,
    video_name: str = "clip",
    sample_count: int = 3,
    size: tuple[int, int] = (8, 8),
    fps: float = 5.0,
    gt_video_path: Path | None = None,
    source_video_path: Path | None = None,
    source_frame_indices: list[int] | None = None,
    frame_step: int | None = None,
) -> int:
    model_id = db.upsert_model(f"model-{name}", "dummy", None, size[1], size[0], {"source": "test"})
    dataset_root = workspace.root / f"dataset-{name}"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_id = db.create_dataset(f"dataset-{name}", str(dataset_root), True, metadata={"source": "test"})
    sample_id = db.add_sample(
        dataset_id,
        f"{video_name}_000000",
        str(pred_video_path),
        str(pred_video_path),
        str(pred_video_path),
        {"video_name": video_name, "video_file": f"{video_name}.mp4", "frame_index": 0, "sample_index": 0, "fps": fps},
    )
    run_id = db.create_run(
        name,
        model_id,
        dataset_id,
        size[1],
        size[0],
        1,
        "cpu",
        "fp32",
        [],
        metadata={"output_dir": str(workspace.runs_dir / name)},
    )
    job_id = int(db.get_run(run_id)["inference_job_id"])
    pred_metadata = {"video_name": video_name, "frames": sample_count, "width": size[0], "height": size[1], "fps": fps}
    # Preds produced after the source-clip-GT change carry the mapping Compare
    # uses to reconstruct a pred-aligned GT from the source clip. Legacy preds
    # (these fields omitted) fall back to strict trim-to-common.
    if source_video_path is not None:
        pred_metadata["source_video_path"] = str(source_video_path.resolve())
    if source_frame_indices is not None:
        pred_metadata["source_frame_indices"] = list(source_frame_indices)
    if frame_step is not None:
        pred_metadata["frame_step"] = int(frame_step)
    db.add_artifact(
        job_id,
        None,
        "pred_video",
        str(pred_video_path),
        "video/mp4",
        pred_metadata,
    )
    if gt_video_path is not None:
        db.add_artifact(
            job_id,
            None,
            "gt_video",
            str(gt_video_path),
            "video/mp4",
            {"video_name": video_name, "frames": sample_count, "width": size[0], "height": size[1], "fps": fps},
        )
    flow_path = workspace.root / f"{name}-flow.png"
    mask_path = workspace.root / f"{name}-mask.png"
    Image.new("RGB", size, (10, 20, 30)).save(flow_path)
    Image.new("RGB", size, (30, 20, 10)).save(mask_path)
    db.add_artifact(job_id, sample_id, "flowt_0", str(flow_path), "image/png", {"sample": f"{video_name}_000000"})
    db.add_artifact(job_id, sample_id, "mask0", str(mask_path), "image/png", {"sample": f"{video_name}_000000"})
    db.complete_run_inference(run_id, {"output_dir": str(workspace.runs_dir / name)}, db.summarize_artifacts(job_id), "completed")
    return run_id


def start_server(db: Database, workspace: WorkspaceConfig):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def get_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_run(base_url: str, run_id: int) -> dict:
    deadline = time.time() + 30
    while time.time() < deadline:
        run = get_json(base_url, f"/api/runs/{run_id}")
        if run["status"] in {"completed", "failed", "canceled"}:
            return run
        time.sleep(0.25)
    raise AssertionError(f"run {run_id} did not finish")
