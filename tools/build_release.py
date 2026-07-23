from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_INFO_PATH = REPO_ROOT / "src" / "vfieval" / "_build_info.json"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit_sha() -> str:
    candidate = (
        os.getenv("VFIEVAL_BUILD_COMMIT")
        or os.getenv("GITHUB_SHA")
        or _run_git("rev-parse", "HEAD")
    ).strip().lower()
    if not COMMIT_RE.fullmatch(candidate):
        raise RuntimeError("release commit must be a full 40-character Git SHA")
    return candidate


def _source_date_epoch(commit_sha: str) -> int:
    candidate = os.getenv("SOURCE_DATE_EPOCH", "").strip()
    if candidate:
        epoch = int(candidate)
    else:
        epoch = int(_run_git("show", "-s", "--format=%ct", commit_sha))
    if epoch <= 0:
        raise RuntimeError("SOURCE_DATE_EPOCH must be a positive Unix timestamp")
    return epoch


def _project_version() -> str:
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if not match:
        raise RuntimeError("project.version is missing from pyproject.toml")
    return match.group(1)


def _assert_clean() -> None:
    changed = _run_git("status", "--porcelain", "--untracked-files=no")
    if changed:
        raise RuntimeError("tracked worktree changes must be committed before a release build")


def _wheel_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_once(output_dir: Path, *, commit_sha: str, epoch: int, version: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = BUILD_INFO_PATH.read_bytes()
    metadata = {
        "build_id": commit_sha,
        "commit_sha": commit_sha,
        "source_date_epoch": epoch,
        "version": version,
    }
    try:
        BUILD_INFO_PATH.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env["SOURCE_DATE_EPOCH"] = str(epoch)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
    finally:
        BUILD_INFO_PATH.write_bytes(original)

    wheels = sorted(output_dir.glob("vfieval-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one VFIEval wheel in {output_dir}, found {len(wheels)}")
    return wheels[0]


def build_release(
    output_dir: Path,
    *,
    require_clean: bool,
    verify_reproducible: bool,
) -> tuple[Path, str]:
    if require_clean:
        _assert_clean()
    commit_sha = _commit_sha()
    epoch = _source_date_epoch(commit_sha)
    version = _project_version()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vfieval-release-") as temporary:
        first_dir = Path(temporary) / "first"
        first_wheel = _build_once(first_dir, commit_sha=commit_sha, epoch=epoch, version=version)
        first_digest = _wheel_digest(first_wheel)
        if verify_reproducible:
            second_wheel = _build_once(
                Path(temporary) / "second",
                commit_sha=commit_sha,
                epoch=epoch,
                version=version,
            )
            second_digest = _wheel_digest(second_wheel)
            if first_digest != second_digest:
                raise RuntimeError(
                    "wheel build is not reproducible: "
                    f"{first_digest} != {second_digest}"
                )

        destination = output_dir / first_wheel.name
        shutil.copy2(first_wheel, destination)
        destination.with_suffix(destination.suffix + ".sha256").write_text(
            f"{first_digest}  {destination.name}\n",
            encoding="ascii",
        )
    return destination, first_digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a commit-identified VFIEval wheel")
    parser.add_argument("--out-dir", default="dist")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--verify-reproducible", action="store_true")
    args = parser.parse_args()
    wheel, digest = build_release(
        Path(args.out_dir),
        require_clean=bool(args.require_clean),
        verify_reproducible=bool(args.verify_reproducible),
    )
    print(json.dumps({"wheel": str(wheel), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
