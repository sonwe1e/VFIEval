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
    def __init__(self, message: str, details: dict[str, object] | None = None):
        super().__init__(message)
        self.details = dict(details or {})


class MetricBatchOutOfMemory(MetricUnavailable):
    """A same-device batch can be retried with fewer image pairs."""

    pass


class MetricAdapter(Protocol):
    name: str

    def evaluate(self, reference: Path, distorted: Path, work_dir: Path) -> MetricResult:
        ...
