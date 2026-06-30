from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path
    db_path: Path
    artifacts_dir: Path
    runs_dir: Path
    tmp_dir: Path

    @classmethod
    def from_root(cls, root: str | os.PathLike[str] | None = None) -> "WorkspaceConfig":
        resolved_root = Path(root or os.getenv("VFIEVAL_WORKSPACE", ".vfieval")).resolve()
        return cls(
            root=resolved_root,
            db_path=resolved_root / "vfieval.sqlite",
            artifacts_dir=resolved_root / "artifacts",
            runs_dir=resolved_root / "runs",
            tmp_dir=resolved_root / "tmp",
        )

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
