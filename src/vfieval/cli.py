from __future__ import annotations

import argparse
import json
import os
import sqlite3
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


class _ReadOnlyDatabase(Database):
    """Open an existing workspace database without creating or migrating it."""

    def connect(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise FileNotFoundError(f"VFIEval database does not exist: {self.db_path}")
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


def _metrics_exit_code(payload: dict) -> int:
    if payload.get("errors"):
        return 1
    health = payload.get("health") if isinstance(payload.get("health"), dict) else payload
    metrics = health.get("metrics") if isinstance(health, dict) else {}
    statuses = {
        str(row.get("status") or "")
        for row in (metrics or {}).values()
        if isinstance(row, dict)
    }
    if "invalid_assets" in statuses:
        return 1
    if any(status != "available" for status in statuses):
        return 2
    return 0


def _doctor_exit_code(report: dict) -> int:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if summary.get("errors"):
        return 1
    if summary.get("unavailable"):
        return 2
    return 0 if report.get("ok") else 1


def _tcp_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


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
    doctor.add_argument("--host", default=os.getenv("VFIEVAL_HOST", "127.0.0.1"))
    doctor.add_argument(
        "--port",
        type=_tcp_port,
        default=os.getenv("VFIEVAL_PORT", "8765"),
    )
    doctor.add_argument(
        "--device",
        action="append",
        dest="devices",
        default=[
            value.strip()
            for value in os.getenv("VFIEVAL_DEVICE", "").split(",")
            if value.strip()
        ],
        help="required target device; repeat for multiple targets (for example cuda:0 or npu:0)",
    )

    diagnostics = sub.add_parser("diagnostics", help="create a sanitized support bundle")
    selection = diagnostics.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id", type=int)
    selection.add_argument("--campaign-id", type=int)
    diagnostics.add_argument("--output")

    args = parser.parse_args(argv)
    workspace = WorkspaceConfig.from_root(args.workspace)
    db: Database | None
    if args.command in {"doctor", "diagnostics"}:
        db = _ReadOnlyDatabase(workspace.db_path)
    elif args.command in {"prepare-metrics", "smoke-metric"}:
        db = None
    else:
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
        return _metrics_exit_code(result)

    if args.command == "smoke-metric":
        metric = create_metric(args.metric, workspace)
        reference = Path(args.reference).resolve()
        distorted = Path(args.distorted).resolve()
        work_dir = Path(args.work_dir).resolve() if args.work_dir else workspace.tmp_dir / "smoke-metric" / args.metric
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = metric.evaluate(reference, distorted, work_dir)
            payload = {"status": result.status, "value": result.value, "details": result.details}
            code = 2 if result.status == "unavailable" else 1 if result.status == "failed" else 0
        except MetricUnavailable as exc:
            payload = {"status": "unavailable", "value": None, "details": {"reason": str(exc)}}
            code = 2
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
        assert db is not None
        try:
            report = run_doctor(
                db,
                workspace,
                host=args.host,
                port=args.port,
                target_devices=args.devices,
            )
        except Exception as exc:
            report = {
                "ok": False,
                "checks": {},
                "summary": {
                    "errors": ["doctor"],
                    "unavailable": [],
                    "warnings": [],
                },
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        code = _doctor_exit_code(report)
        if args.json_output:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        else:
            status = "PASS" if code == 0 else "UNAVAILABLE" if code == 2 else "FAIL"
            print(f"VFIEval doctor: {status}")
            for name, check in report["checks"].items():
                check_status = str(check.get("status") or "info").upper() if isinstance(check, dict) else "INFO"
                reason = str(check.get("reason") or "") if isinstance(check, dict) else ""
                print(f"  {check_status:7} {name}{': ' + reason if reason else ''}")
            print("Use --json for the complete machine-readable report.")
        return code

    if args.command == "diagnostics":
        assert db is not None
        try:
            bundle = create_diagnostics_bundle(
                db,
                workspace,
                run_id=args.run_id,
                campaign_id=args.campaign_id,
                output=args.output,
            )
        except (OSError, sqlite3.Error, KeyError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 1
        print(json.dumps({"diagnostics_bundle": str(bundle)}, indent=2, ensure_ascii=False))
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
