from __future__ import annotations

import argparse
import json
from pathlib import Path

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_dataset
from vfieval.db import Database
from vfieval.metrics import METRIC_NAMES
from vfieval.metrics.health import metrics_health, prepare_metric_asset_manifest
from vfieval.server import run_server
from vfieval.worker import WorkerOptions, run_worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vfieval")
    parser.add_argument("--workspace", default=".vfieval", help="VFIEval workspace directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    model = sub.add_parser("register-model")
    model.add_argument("--name", required=True)
    model.add_argument("--adapter", required=True)
    model.add_argument("--checkpoint")
    model.add_argument("--height", type=int, required=True)
    model.add_argument("--width", type=int, required=True)
    model.add_argument("--metadata-json", default="{}")

    dataset = sub.add_parser("create-dataset")
    dataset.add_argument("--name", required=True)
    dataset.add_argument("--root", required=True)
    dataset.add_argument("--has-gt", action="store_true")
    dataset.add_argument("--no-gt", action="store_true")
    dataset.add_argument("--source-type", choices=["frames", "video"], default="frames")
    dataset.add_argument("--decode-mode", choices=["frames", "video_gt_triplets", "video_pairs"])
    dataset.add_argument("--frame-step", type=int, default=1)
    dataset.add_argument("--max-frames", type=int)
    dataset.add_argument("--video-glob", default="*")
    dataset.add_argument("--metadata-json", default="{}")

    scan = sub.add_parser("scan-dataset")
    scan.add_argument("--dataset-id", type=int, required=True)

    enqueue = sub.add_parser("enqueue-inference")
    enqueue.add_argument("--model-id", type=int, required=True)
    enqueue.add_argument("--dataset-id", type=int, required=True)
    enqueue.add_argument("--height", type=int, required=True)
    enqueue.add_argument("--width", type=int, required=True)
    enqueue.add_argument("--batch-size", type=int, default=1)
    enqueue.add_argument("--device", default="auto")
    enqueue.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    enqueue.add_argument("--metric", action="append", choices=METRIC_NAMES, default=[])

    metric = sub.add_parser("enqueue-metrics")
    metric.add_argument("--inference-job-id", type=int, required=True)
    metric.add_argument("--dataset-id", type=int, required=True)
    metric.add_argument("--metric", action="append", choices=METRIC_NAMES, required=True)

    prepare_metrics = sub.add_parser("prepare-metrics")
    prepare_metrics.add_argument("--check-only", action="store_true")

    worker = sub.add_parser("worker")
    worker.add_argument("--role", choices=["decode", "inference", "metric", "all"], default="all")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--poll-interval", type=float, default=5.0)
    worker.add_argument("--worker-id")
    worker.add_argument("--device-filter")
    worker.add_argument("--idle-timeout", type=float)

    jobs = sub.add_parser("jobs")
    jobs.add_argument("--limit", type=int, default=50)

    args = parser.parse_args(argv)
    workspace = WorkspaceConfig.from_root(args.workspace)
    workspace.ensure()
    db = Database(workspace.db_path)
    db.init()

    if args.command == "init":
        print(json.dumps({"workspace": str(workspace.root), "db": str(workspace.db_path)}, indent=2))
        return 0

    if args.command == "serve":
        run_server(db=db, workspace=workspace, host=args.host, port=args.port)
        return 0

    if args.command == "register-model":
        model_id = db.register_model(
            name=args.name,
            adapter=args.adapter,
            checkpoint_path=args.checkpoint,
            input_height=args.height,
            input_width=args.width,
            metadata=json.loads(args.metadata_json),
        )
        print(json.dumps({"model_id": model_id}, indent=2))
        return 0

    if args.command == "create-dataset":
        if args.has_gt and args.no_gt:
            raise SystemExit("--has-gt and --no-gt are mutually exclusive")
        metadata = json.loads(args.metadata_json)
        metadata.update(
            {
                "frame_step": args.frame_step,
                "max_frames": args.max_frames,
                "video_glob": args.video_glob,
            }
        )
        dataset_id = db.create_dataset(
            name=args.name,
            root_path=str(Path(args.root).resolve()),
            has_gt=not args.no_gt,
            source_type=args.source_type,
            decode_mode=args.decode_mode,
            metadata=metadata,
        )
        print(json.dumps({"dataset_id": dataset_id}, indent=2))
        return 0

    if args.command == "scan-dataset":
        count = scan_dataset(db, workspace, args.dataset_id)
        print(json.dumps({"dataset_id": args.dataset_id, "samples": count}, indent=2))
        return 0

    if args.command == "enqueue-inference":
        payload = {
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "height": args.height,
            "width": args.width,
            "batch_size": args.batch_size,
            "device": args.device,
            "precision": args.precision,
            "metrics": args.metric,
        }
        job_id = db.create_job("inference", payload)
        print(json.dumps({"job_id": job_id, "kind": "inference"}, indent=2))
        return 0

    if args.command == "enqueue-metrics":
        payload = {
            "inference_job_id": args.inference_job_id,
            "dataset_id": args.dataset_id,
            "metric_names": args.metric,
        }
        job_id = db.create_job("metric", payload)
        print(json.dumps({"job_id": job_id, "kind": "metric"}, indent=2))
        return 0

    if args.command == "prepare-metrics":
        result = metrics_health(workspace) if args.check_only else prepare_metric_asset_manifest(workspace)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "worker":
        run_worker(
            db,
            workspace,
            WorkerOptions(
                role=args.role,
                once=args.once,
                poll_interval=args.poll_interval,
                worker_id=args.worker_id,
                device_filter=args.device_filter,
                idle_timeout=args.idle_timeout,
            ),
        )
        return 0

    if args.command == "jobs":
        print(json.dumps(db.list_jobs(limit=args.limit), indent=2))
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
