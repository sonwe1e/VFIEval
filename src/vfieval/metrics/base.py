from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class MetricResult:
    status: str
    value: float | None
    details: dict[str, object]


class MetricUnavailable(RuntimeError):
    pass


class MetricAdapter(Protocol):
    name: str

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        ...
