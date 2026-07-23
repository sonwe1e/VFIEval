from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _read_url(url: str, *, timeout: float = 2.0) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return (
            int(response.status),
            response.read(),
            str(response.headers.get("Content-Type") or ""),
        )


def _wait_for_health(base_url: str, process: subprocess.Popen[bytes]) -> dict:
    deadline = time.monotonic() + 30.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vfieval serve exited early with code {process.returncode}")
        try:
            status, body, _content_type = _read_url(f"{base_url}/api/health")
            if status == 200:
                payload = json.loads(body.decode("utf-8"))
                if isinstance(payload, dict) and payload.get("live") is True:
                    return payload
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"vfieval serve did not become live: {last_error}")


def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test an installed VFIEval wheel")
    parser.add_argument("--expect-build-id", required=True)
    parser.add_argument(
        "--package-root",
        help="optional wheel-only import root for local smoke without a new dependency environment",
    )
    parser.add_argument(
        "--forbid-source-root",
        help="fail when vfieval resolves from this source tree instead of the installed wheel",
    )
    args = parser.parse_args()

    clean_env = os.environ.copy()
    if args.package_root:
        clean_env["PYTHONPATH"] = str(Path(args.package_root).resolve())
    else:
        clean_env.pop("PYTHONPATH", None)
    origin_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, vfieval; print(pathlib.Path(vfieval.__file__).resolve())",
        ],
        env=clean_env,
        check=False,
        capture_output=True,
        text=True,
    )
    if origin_result.returncode != 0:
        raise RuntimeError(f"installed package import failed: {origin_result.stderr}")
    package_origin = Path(origin_result.stdout.strip()).resolve()
    if args.package_root and not package_origin.is_relative_to(Path(args.package_root).resolve()):
        raise RuntimeError(f"vfieval resolved outside the wheel import root: {package_origin}")
    if args.forbid_source_root and package_origin.is_relative_to(Path(args.forbid_source_root).resolve()):
        raise RuntimeError(f"vfieval resolved from the source checkout: {package_origin}")

    help_result = subprocess.run(
        [sys.executable, "-m", "vfieval.cli", "--help"],
        env=clean_env,
        check=False,
        capture_output=True,
        text=True,
    )
    if help_result.returncode != 0 or "prepare-metrics" not in help_result.stdout:
        raise RuntimeError(f"installed CLI help failed: {help_result.stderr}")

    with tempfile.TemporaryDirectory(prefix="vfieval-wheel-smoke-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vfieval.cli",
                "--workspace",
                str(workspace),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=root,
            env=clean_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            ),
            start_new_session=os.name != "nt",
        )
        try:
            health = _wait_for_health(base_url, process)
            actual_build_id = str((health.get("release") or {}).get("build_id") or "")
            if actual_build_id != args.expect_build_id:
                raise RuntimeError(
                    f"installed build id mismatch: expected {args.expect_build_id}, got {actual_build_id}"
                )
            _status, index, index_type = _read_url(f"{base_url}/")
            if b"/app.js" not in index or "text/html" not in index_type:
                raise RuntimeError("installed wheel did not serve the main HTML entrypoint")
            asset_paths = sorted(
                {
                    match.decode("utf-8")
                    for match in re.findall(rb'(?:src|href)="(/[^"?#]+)', index)
                    if not match.startswith(b"//")
                }
            )
            if not asset_paths:
                raise RuntimeError("installed wheel main entrypoint references no local assets")
            for asset_path in asset_paths:
                _status, body, content_type = _read_url(f"{base_url}{asset_path}")
                if not body:
                    raise RuntimeError(f"installed wheel served an empty asset: {asset_path}")
                if asset_path.endswith(".js") and "javascript" not in content_type:
                    raise RuntimeError(
                        f"installed wheel served the wrong content type for {asset_path}: "
                        f"{content_type}"
                    )
                if asset_path.endswith(".css") and "css" not in content_type:
                    raise RuntimeError(
                        f"installed wheel served the wrong content type for {asset_path}: "
                        f"{content_type}"
                    )
        finally:
            _stop_process_tree(process)

    print(
        json.dumps(
            {
                "status": "ok",
                "build_id": args.expect_build_id,
                "package_origin": str(package_origin),
                "cli": True,
                "static": True,
                "static_asset_count": len(asset_paths),
                "server": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
