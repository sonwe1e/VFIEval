from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from vfieval.metrics.base import MetricResult, MetricUnavailable


class CommandMetric:
    def __init__(self, name: str, env_var: str):
        self.name = name
        self.env_var = env_var

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        template = os.getenv(self.env_var)
        if not template:
            raise MetricUnavailable(
                f"{self.name} requires {self.env_var}. "
                "Point it to an evaluator command using {reference}, {distorted}, and optional {output}."
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        output_path = work_dir / f"{self.name}.json"
        command = template.format(
            reference=str(reference),
            distorted=str(distorted),
            output=str(output_path),
        )
        completed = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.name} command failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        if output_path.exists():
            data = json.loads(output_path.read_text(encoding="utf-8"))
            score = data.get("score")
            details = {k: v for k, v in data.items() if k != "score"}
        else:
            stdout = completed.stdout.strip()
            score = float(stdout)
            details = {"stdout": stdout}

        return MetricResult(status="completed", value=float(score), details=details)
