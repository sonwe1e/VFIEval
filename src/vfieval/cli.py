from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_dataset
from vfieval.db import Database
from vfieval.diagnostics import create_diagnostics_bundle, run_doctor
from vfieval.metrics import METRIC_NAMES, create_metric
from vfieval.metrics.base import MetricUnavailable
from vfieval.metrics.health import metrics_health, prepare_metric_asset_manifest
from vfieval.server import _create_run_from_files, run_server
from vfieval.runtime_logging import close_runtime_logging, configure_runtime_logging, log_event
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
    prepare_metrics.add_argument("--force", action="store_true")

    smoke_metric = sub.add_parser("smoke-metric")
    smoke_metric.add_argument("--metric", choices=METRIC_NAMES, required=True)
    smoke_metric.add_argument("--reference", required=True)
    smoke_metric.add_argument("--distorted", required=True)
    smoke_metric.add_argument("--work-dir")

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--model-file", required=True)
    benchmark.add_argument("--video-group", required=True)
    benchmark.add_argument("--checkpoint", default="none")
    benchmark.add_argument("--device", default="npu:0")
    benchmark.add_argument("--device-id", action="append", dest="devices", default=[])
    benchmark.add_argument("--execution-mode", choices=["single", "multi_npu", "multi_cuda"], default="single")
    benchmark.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--warmup-batches", type=int, default=10)
    benchmark.add_argument("--samples", type=int, default=200)
    benchmark.add_argument("--repeats", type=int, default=3)

    worker = sub.add_parser("worker")
    worker.add_argument("--role", choices=["decode", "inference", "finalize", "metric", "all"], default="all")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--poll-interval", type=float, default=5.0)
    worker.add_argument("--worker-id")
    worker.add_argument("--device-filter")
    worker.add_argument("--idle-timeout", type=float)

    jobs = sub.add_parser("jobs")
    jobs.add_argument("--limit", type=int, default=50)

    doctor = sub.add_parser("doctor", help="check devices, media tools, metrics, database, and storage")
    doctor.add_argument("--json", action="store_true", dest="json_output")

    diagnostics = sub.add_parser("diagnostics", help="create a sanitized support bundle")
    selection = diagnostics.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id", type=int)
    selection.add_argument("--campaign-id", type=int)
    diagnostics.add_argument("--output")

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
        result = metrics_health(workspace) if args.check_only else prepare_metric_asset_manifest(workspace, force=args.force)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "smoke-metric":
        metric = create_metric(args.metric, workspace)
        reference = Path(args.reference).resolve()
        distorted = Path(args.distorted).resolve()
        work_dir = Path(args.work_dir).resolve() if args.work_dir else workspace.tmp_dir / "smoke-metric" / args.metric
        try:
            result = metric.evaluate(reference, distorted, work_dir)
            payload = {"status": result.status, "value": result.value, "details": result.details}
            code = 0
        except MetricUnavailable as exc:
            payload = {"status": "unavailable", "value": None, "details": {"reason": str(exc)}}
            code = 0
        except Exception as exc:
            payload = {
                "status": "failed",
                "value": None,
                "details": {"reason": str(exc), "type": type(exc).__name__},
            }
            code = 1
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return code

    if args.command == "benchmark":
        reports = []
        for repeat in range(max(1, int(args.repeats))):
            created = _create_run_from_files(
                db,
                workspace,
                {
                    "run_type": "model_inference",
                    "name": f"benchmark-{Path(args.model_file).stem}-{repeat + 1}",
                    "model_file": args.model_file,
                    "video_group": args.video_group,
                    "checkpoint": args.checkpoint,
                    "device": args.device,
                    "devices": args.devices,
                    "execution_mode": args.execution_mode,
                    "precision": args.precision,
                    "batch_size": args.batch_size,
                    "batch_size_per_device": args.batch_size,
                    "artifact_profile": "benchmark",
                    "benchmark_warmup_batches": args.warmup_batches,
                    "benchmark_samples": args.samples,
                    "metrics": [],
                },
            )
            run_id = int(created["run_id"])
            while True:
                run = db.get_run(run_id)
                if run["status"] in {"completed", "failed", "canceled"}:
                    break
                time.sleep(0.25)
            if run["status"] != "completed":
                raise RuntimeError(f"benchmark run {run_id} failed: {run.get('error')}")
            reports.append({"run_id": run_id, "performance": (run.get("result") or {}).get("performance") or {}})
        best = max(
            reports,
            key=lambda row: float((row.get("performance") or {}).get("steady_state_fps") or 0.0),
        )
        print(json.dumps({"repeats": reports, "best": best}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "worker":
        configure_runtime_logging(workspace, filename=f"worker-{os.getpid()}.jsonl")
        log_event(20, "worker.started", "VFIEval worker started", role=args.role, device=args.device_filter)
        try:
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
        finally:
            log_event(20, "worker.stopped", "VFIEval worker stopped", role=args.role, device=args.device_filter)
            close_runtime_logging()
        return 0

    if args.command == "jobs":
        print(json.dumps(db.list_jobs(limit=args.limit), indent=2))
        return 0

    if args.command == "doctor":
        report = run_doctor(db, workspace)
        if args.json_output:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        else:
            status = "PASS" if report["ok"] else "FAIL"
            print(f"VFIEval doctor: {status}")
            for name, check in report["checks"].items():
                check_status = str(check.get("status") or "info").upper() if isinstance(check, dict) else "INFO"
                reason = str(check.get("reason") or "") if isinstance(check, dict) else ""
                print(f"  {check_status:7} {name}{': ' + reason if reason else ''}")
            print("Use --json for the complete machine-readable report.")
        return 0 if report["ok"] else 1

    if args.command == "diagnostics":
        bundle = create_diagnostics_bundle(
            db,
            workspace,
            run_id=args.run_id,
            campaign_id=args.campaign_id,
            output=args.output,
        )
        print(json.dumps({"diagnostics_bundle": str(bundle)}, indent=2, ensure_ascii=False))
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
