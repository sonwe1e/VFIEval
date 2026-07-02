from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re


BACKUP_PATTERN = re.compile(r".+\.backup\.\d{8}_\d{6}$")


@dataclass
class ArchivedBackup:
    source: str
    destination: str
    size: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive scattered *.backup.* files into archive/file_backups.")
    parser.add_argument("--root", default=".", help="Repository root. Defaults to the current directory.")
    parser.add_argument(
        "--destination-root",
        default="archive/file_backups",
        help="Archive folder relative to --root. Defaults to archive/file_backups.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would move without changing files.")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    archive_root = (root / args.destination_root).resolve()
    session_dir = archive_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    backups = list(find_backup_files(root))
    archived: list[ArchivedBackup] = []
    empty_directories: set[Path] = set()

    for source in backups:
        relative = source.relative_to(root)
        destination = session_dir / relative
        archived.append(
            ArchivedBackup(
                source=str(source),
                destination=str(destination),
                size=source.stat().st_size,
            )
        )
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            empty_directories.add(source.parent)

    if not args.dry_run:
        for directory in sorted(empty_directories, key=lambda item: len(item.parts), reverse=True):
            _remove_empty_parents(directory, stop=root)
        _write_manifest(session_dir, archived)

    print(
        json.dumps(
            {
                "root": str(root),
                "destination_root": str(archive_root),
                "dry_run": bool(args.dry_run),
                "moved_count": len(archived),
                "session_dir": str(session_dir),
                "files": [asdict(item) for item in archived],
            },
            indent=2,
        )
    )
    return 0


def find_backup_files(root: Path) -> list[Path]:
    archive_root = (root / "archive").resolve()
    backups: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if archive_root in path.parents:
            continue
        if BACKUP_PATTERN.fullmatch(path.name):
            backups.append(path)
    backups.sort()
    return backups


def _write_manifest(session_dir: Path, archived: list[ArchivedBackup]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "archived_at": datetime.now().isoformat(timespec="seconds"),
                "count": len(archived),
                "files": [asdict(item) for item in archived],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _remove_empty_parents(directory: Path, stop: Path) -> None:
    current = directory
    while current != stop and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


if __name__ == "__main__":
    raise SystemExit(main())
