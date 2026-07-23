from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any


CancelCheck = Callable[[], None]


def run_cancellable(
    command: Sequence[str | PathLike[str]],
    *,
    input_text: str | None = None,
    timeout: float,
    cancel_check: CancelCheck,
    env: Mapping[str, str] | None = None,
    cwd: str | PathLike[str] | None = None,
    poll_interval: float = 0.2,
) -> subprocess.CompletedProcess[str]:
    """Run a captured text subprocess while honoring a cooperative cancel check."""

    process = subprocess.Popen(
        [str(value) for value in command],
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env) if env is not None else None,
        cwd=cwd,
    )
    deadline = time.monotonic() + max(0.0, float(timeout))
    pending_input = input_text
    try:
        while True:
            cancel_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stdout, stderr = terminate_process(process)
                raise subprocess.TimeoutExpired(
                    [str(value) for value in command],
                    timeout,
                    output=stdout,
                    stderr=stderr,
                )
            try:
                stdout, stderr = process.communicate(
                    input=pending_input,
                    timeout=min(max(0.01, float(poll_interval)), remaining),
                )
                return subprocess.CompletedProcess(
                    [str(value) for value in command],
                    int(process.returncode or 0),
                    stdout,
                    stderr,
                )
            except subprocess.TimeoutExpired:
                pending_input = None
    except BaseException:
        if process.poll() is None:
            terminate_process(process)
        raise


def terminate_process(
    process: subprocess.Popen[Any],
    *,
    terminate_timeout: float = 2.0,
    kill_timeout: float = 2.0,
) -> tuple[Any, Any]:
    """Terminate, then kill a child if needed, returning any captured output."""

    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        return process.communicate(timeout=max(0.01, float(terminate_timeout)))
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            return process.communicate(timeout=max(0.01, float(kill_timeout)))
        except (OSError, subprocess.TimeoutExpired):
            return "", ""
