from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts
from vfieval.runtime_logging import runtime_logger


TERMINAL_RUN_STATUSES = {"completed", "failed", "canceled"}
CACHE_GRACE_SECONDS = 10 * 60
CACHE_BUILD_LOCK_TTL_SECONDS = 5 * 60
CACHE_BUILD_LOCK_WAIT_SECONDS = 15 * 60.0
CACHE_BUILD_LOCK_POLL_SECONDS = 0.1
PURGE_CLAIM_STALE_SECONDS = 5 * 60
RUN_PURGE_PREVIEW_TTL_SECONDS = 5 * 60
_BACKFILL_LOCK = threading.Lock()
_BACKFILLED_DATABASES: set[str] = set()
_SAFE_SNAPSHOT_SLOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class RunPurgePreviewError(ValueError):
    """A stable, API-friendly failure raised by Run purge preview validation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


class DecodeCacheBuildLock:
    """A producer token plus an asynchronous ownership-loss signal."""

    def __init__(self, cache_key: str, owner_token: str, lost: threading.Event, ttl_seconds: float) -> None:
        self.cache_key = cache_key
        self.owner_token = owner_token
        self.lost = lost
        self.ttl_seconds = ttl_seconds


@contextmanager
def _purge_claim_heartbeat(
    db: Database,
    request_id: int,
    claim_token: str,
) -> Iterator[threading.Event]:
    """Keep a durable purge claim alive while a filesystem delete may block."""
    stop = threading.Event()
    lost = threading.Event()

    def renew() -> None:
        interval = max(1.0, min(30.0, PURGE_CLAIM_STALE_SECONDS / 4.0))
        while not stop.wait(interval):
            try:
                if not db.heartbeat_run_purge_request(request_id, claim_token):
                    lost.set()
                    return
            except Exception:
                # A transient SQLite lock is not ownership loss. The next
                # renewal arrives well before the stale-claim timeout.
                continue

    thread = threading.Thread(target=renew, name="vfieval-purge-lease", daemon=True)
    thread.start()
    try:
        yield lost
    finally:
        stop.set()
        thread.join(timeout=2.0)


@contextmanager
def _immediate_transaction(db: Database) -> Iterator[Any]:
    """Run a SQLite write transaction with an explicit rollback boundary."""
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _path_size(path: Path) -> int:
    try:
        if path.is_symlink():
            return int(path.lstat().st_size)
        if path.is_file():
            return int(path.stat().st_size)
        if not path.is_dir():
            return 0
    except OSError:
        return 0
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not (root_path / name).is_symlink()]
        for name in files:
            child = root_path / name
            try:
                total += int(child.lstat().st_size)
            except OSError:
                continue
    return total


def _path_reclaimable_size(path: Path) -> int:
    """Estimate bytes released by unlinking this exact path tree.

    Run-local GT hard links share their data blocks with the managed decode
    cache.  Their logical size still belongs in ordinary inventory totals, but
    deleting one Run link does not reclaim those blocks while another link
    exists.  Keep purge reports honest without changing cache size accounting.
    """

    try:
        if path.is_symlink():
            return int(path.lstat().st_size)
        if path.is_file():
            stat = path.stat()
            return int(stat.st_size) if int(getattr(stat, "st_nlink", 1)) <= 1 else 0
        if not path.is_dir():
            return 0
    except OSError:
        return 0
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not (root_path / name).is_symlink()]
        for name in files:
            child = root_path / name
            try:
                stat = child.lstat()
                if int(getattr(stat, "st_nlink", 1)) <= 1:
                    total += int(stat.st_size)
            except OSError:
                continue
    return total


def _path_state(path: Path) -> dict[str, Any]:
    """Return a cheap, deterministic tree signature plus conservative byte totals."""

    digest = hashlib.sha256()
    logical_bytes = 0
    reclaimable_bytes = 0
    exists = path.exists() or path.is_symlink()
    if not exists:
        digest.update(b"missing")
        return {
            "exists": False,
            "logical_bytes": 0,
            "reclaimable_bytes": 0,
            "fingerprint": digest.hexdigest(),
        }

    def record(child: Path, relative: str) -> None:
        nonlocal logical_bytes, reclaimable_bytes
        try:
            stat = child.lstat()
        except OSError as exc:
            digest.update(f"error:{relative}:{type(exc).__name__}".encode("utf-8"))
            return
        is_link = child.is_symlink()
        is_file = child.is_file() and not is_link
        size = int(stat.st_size) if is_file or is_link else 0
        links = int(getattr(stat, "st_nlink", 1))
        logical_bytes += size
        if is_link or (is_file and links <= 1):
            reclaimable_bytes += size
        digest.update(
            json.dumps(
                {
                    "path": relative,
                    "mode": int(stat.st_mode),
                    "size": size,
                    "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
                    "links": links,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    record(path, ".")
    if path.is_dir() and not path.is_symlink():
        for root, dirs, files in os.walk(path, followlinks=False):
            root_path = Path(root)
            dirs.sort()
            files.sort()
            symlink_dirs = [name for name in dirs if (root_path / name).is_symlink()]
            dirs[:] = [name for name in dirs if name not in symlink_dirs]
            for name in symlink_dirs:
                child = root_path / name
                record(child, child.relative_to(path).as_posix())
            for name in files:
                child = root_path / name
                record(child, child.relative_to(path).as_posix())
    return {
        "exists": True,
        "logical_bytes": logical_bytes,
        "reclaimable_bytes": reclaimable_bytes,
        "fingerprint": digest.hexdigest(),
    }


def _path_content_digest(path: Path) -> str:
    """Hash a file or frame tree without following links outside the source."""
    if path.is_symlink():
        raise ValueError("Compare snapshot source must not be a symbolic link")
    resolved = path.resolve()
    digest = hashlib.sha256()
    if resolved.is_file():
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Compare snapshot source does not exist: {resolved}")
    for child in sorted(resolved.rglob("*"), key=lambda value: value.relative_to(resolved).as_posix()):
        if child.is_symlink():
            raise ValueError("Compare snapshot frame trees must not contain symbolic links")
        if not child.is_file():
            continue
        relative = child.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _materialize_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    """Atomically materialize a private Compare input snapshot.

    The staging path is a distinct file/directory.  It may use a Linux COW
    reflink, but never a hard link: a source artifact must not share a writable
    inode with a dependent Compare Run.
    """
    if source.is_symlink():
        raise ValueError("Compare snapshot source must not be a symbolic link")
    source = source.resolve()
    destination = destination.resolve()
    source_digest = _path_content_digest(source)
    if destination.exists():
        if _path_content_digest(destination) != source_digest:
            raise ValueError(f"existing Compare snapshot differs from its source: {destination}")
        return {
            "storage_path": str(destination),
            "content_sha256": source_digest,
            "size_bytes": _path_size(destination),
            "method": "reused",
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        _clone_or_copy_managed_path(source, staging)
        if _path_content_digest(staging) != source_digest:
            raise IOError("Compare snapshot verification failed")
        os.replace(staging, destination)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
        elif staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass
        raise
    return {
        "storage_path": str(destination),
        "content_sha256": source_digest,
        "size_bytes": _path_size(destination),
        "method": "reflink_or_private_copy",
    }


def _safe_snapshot_slot(slot: str, binding_id: int) -> str:
    value = str(slot or "").strip()
    if not value:
        return f"binding-{int(binding_id)}"
    if not _SAFE_SNAPSHOT_SLOT_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe Compare input slot: {value!r}")
    return value


def _clone_or_copy_file(source: Path, target: Path) -> None:
    """Create a private file snapshot, preferring a copy-on-write reflink.

    A hard link is intentionally not used: a subsequent in-place write to a
    Run artifact would otherwise mutate the dependent Compare input too.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    cloned = False
    if os.name != "nt":
        try:
            import fcntl  # Linux-only optional reflink acceleration.

            # FICLONE from linux/fs.h creates a new inode with copy-on-write
            # extents, keeping the snapshot private if either file is edited.
            with source.open("rb") as source_handle, target.open("xb") as target_handle:
                fcntl.ioctl(target_handle.fileno(), 0x40049409, source_handle.fileno())
            shutil.copystat(source, target, follow_symlinks=False)
            cloned = True
        except (ImportError, OSError):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
    if not cloned:
        shutil.copy2(source, target, follow_symlinks=False)


def _clone_or_copy_managed_path(source: Path, target: Path) -> None:
    """Copy a Run-owned media path without following a symlink boundary."""
    if source.is_symlink():
        raise ValueError("Compare dependency snapshot refuses symlink media")
    if source.is_file():
        _clone_or_copy_file(source, target)
        return
    if not source.is_dir():
        raise FileNotFoundError(f"Compare dependency source is unavailable: {source}")

    target.mkdir(parents=True, exist_ok=False)
    for child in sorted(source.rglob("*")):
        if child.is_symlink():
            raise ValueError("Compare dependency snapshot refuses symlink media")
        relative = child.relative_to(source)
        destination = target / relative
        if child.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            _clone_or_copy_file(child, destination)


def _json_object(raw: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(str(raw or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _cache_root(workspace: WorkspaceConfig, cache_type: str) -> Path:
    if cache_type == "decode_cache":
        return (workspace.root / "decode_cache").resolve()
    if cache_type == "compare_cache":
        return (workspace.root / "compare_cache").resolve()
    raise ValueError(f"unsupported cache type: {cache_type}")


def _trusted_cache_path(
    workspace: WorkspaceConfig,
    cache_type: str,
    cache_key: str,
    path: str | Path,
) -> Path:
    root = _cache_root(workspace, cache_type)
    resolved = Path(path).resolve()
    if not _is_relative_to(resolved, root) or resolved == root:
        raise ValueError(f"{cache_type} entry resolved outside its managed cache root")
    if cache_type == "decode_cache" and resolved.parent != root:
        raise ValueError("decode cache entries must be direct children of decode_cache")
    if cache_type == "compare_cache" and resolved.parent != root:
        raise ValueError("compare cache entries must be direct children of compare_cache")
    expected_key = resolved.name if cache_type == "decode_cache" else resolved.stem
    if expected_key != str(cache_key):
        raise ValueError("cache key does not match its managed storage path")
    return resolved


def _cache_entry_from_path(
    db: Database,
    workspace: WorkspaceConfig,
    cache_type: str,
    cache_key: str,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _trusted_cache_path(workspace, cache_type, cache_key, path)
    exists = resolved.is_dir() if cache_type == "decode_cache" else resolved.is_file()
    try:
        last_used_at = float(resolved.stat().st_mtime) if exists else utc_ts()
    except OSError:
        last_used_at = utc_ts()
    entry = db.upsert_cache_entry(
        cache_type,
        cache_key,
        resolved,
        state="ready" if exists else "missing",
        size_bytes=_path_size(resolved),
        metadata=metadata or {},
        last_used_at=last_used_at,
        gc_after=last_used_at + CACHE_GRACE_SECONDS,
    )
    if entry.get("state") == "deleting":
        raise RuntimeError(f"{cache_type} entry is being garbage-collected; retry cache use")
    return entry


def _path_cache_descriptor(workspace: WorkspaceConfig, raw_path: str | None) -> tuple[str, str, Path] | None:
    if not raw_path:
        return None
    path = Path(str(raw_path)).resolve()
    for cache_type in ("decode_cache", "compare_cache"):
        root = _cache_root(workspace, cache_type)
        if not _is_relative_to(path, root) or path == root or path.parent != root:
            continue
        cache_key = path.name if cache_type == "decode_cache" else path.stem
        return cache_type, cache_key, path
    return None


def register_run_cache_refs(
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int,
    *,
    grace_seconds: float = CACHE_GRACE_SECONDS,
) -> dict[str, int]:
    """Index the decode/aligned-GT caches actually referenced by one Run."""
    run = db.get_run(int(run_id))
    if run.get("deleted_at") is not None or run.get("artifact_cleaned_at") is not None:
        released = db.release_run_cache_refs(int(run_id), grace_seconds=grace_seconds)
        return {"added": 0, "released": len(released), "total": 0}

    rows = db.query(
        """
        SELECT metadata_json, img0_path, img1_path, gt_path
        FROM samples
        WHERE dataset_id = ?
        """,
        (int(run["dataset_id"]),),
    )
    descriptors: dict[tuple[str, str], Path] = {}
    for row in rows:
        try:
            metadata = json.loads(str(row.get("metadata_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        cache_key = str(metadata.get("cache_key") or "").strip()
        if cache_key:
            descriptors[("decode_cache", cache_key)] = _cache_root(workspace, "decode_cache") / cache_key
        for field in ("img0_path", "img1_path", "gt_path"):
            descriptor = _path_cache_descriptor(workspace, row.get(field))
            if descriptor is not None:
                cache_type, path_key, path = descriptor
                descriptors[(cache_type, path_key)] = path

    entry_ids: list[int] = []
    for (cache_type, cache_key), path in sorted(descriptors.items()):
        entry = _cache_entry_from_path(
            db,
            workspace,
            cache_type,
            cache_key,
            path,
            metadata={"indexed_from_run_id": int(run_id)},
        )
        entry_ids.append(int(entry["id"]))
    return db.replace_run_cache_refs(
        int(run_id),
        entry_ids,
        grace_seconds=grace_seconds,
    )


@contextmanager
def cache_lease(
    db: Database,
    workspace: WorkspaceConfig,
    cache_type: str,
    cache_key: str,
    path: str | Path,
    *,
    ttl_seconds: float = 6 * 60 * 60,
) -> Iterator[dict[str, Any]]:
    """Protect an in-use cache entry from concurrent storage GC."""
    entry = _cache_entry_from_path(
        db,
        workspace,
        cache_type,
        cache_key,
        Path(path),
        metadata={"lease_created_by": "runtime"},
    )
    lease_id = uuid.uuid4().hex
    db.acquire_cache_lease(int(entry["id"]), lease_id, ttl_seconds=ttl_seconds)
    stop = threading.Event()

    def renew() -> None:
        # Keep a long-running decode/resize from silently outliving its lease.
        # A failed renewal is retried on the next interval; the foreground cache
        # operation still owns normal error handling and releases on exit.
        interval = max(1.0, min(60.0, float(ttl_seconds) / 3.0))
        while not stop.wait(interval):
            try:
                db.acquire_cache_lease(int(entry["id"]), lease_id, ttl_seconds=ttl_seconds)
            except Exception:
                continue

    thread = threading.Thread(target=renew, name="vfieval-cache-lease", daemon=True)
    thread.start()
    try:
        yield entry
    except BaseException:
        raise
    else:
        # The cache path may have been created while this lease was active.
        # Refresh the physical catalog before release so a preflight-only
        # decode is immediately visible as a ready cache entry, even though it
        # has no Run reference yet.
        _cache_entry_from_path(
            db,
            workspace,
            cache_type,
            cache_key,
            Path(path),
            metadata=dict(entry.get("metadata") or {}),
        )
    finally:
        stop.set()
        thread.join(timeout=2.0)
        db.release_cache_lease(int(entry["id"]), lease_id)


@contextmanager
def decode_cache_build_lock(
    db: Database,
    cache_key: str,
    *,
    ttl_seconds: float = CACHE_BUILD_LOCK_TTL_SECONDS,
    wait_timeout_seconds: float = CACHE_BUILD_LOCK_WAIT_SECONDS,
    poll_interval_seconds: float = CACHE_BUILD_LOCK_POLL_SECONDS,
    on_wait: Callable[[], None] | None = None,
) -> Iterator[DecodeCacheBuildLock]:
    """Claim one cross-process decode-cache producer slot.

    ``cache_lease`` protects a physical entry from GC; this lock instead
    coordinates who may construct it.  Waiters poll at a bounded cadence and
    revalidate the final manifest after acquiring the slot, so a completed
    winner is reused rather than decoded again.
    """
    key = str(cache_key).strip()
    if not key:
        raise ValueError("decode cache build lock requires a cache key")
    ttl = max(1.0, float(ttl_seconds))
    timeout = max(0.0, float(wait_timeout_seconds))
    poll = max(0.01, float(poll_interval_seconds))
    owner_token = uuid.uuid4().hex
    started = time.monotonic()
    while not db.claim_decode_cache_build_lock(key, owner_token, ttl_seconds=ttl):
        if time.monotonic() - started >= timeout:
            raise TimeoutError(
                "decode cache is being built by another process; retry the input check"
            )
        if on_wait is not None:
            on_wait()
        time.sleep(poll)

    stop = threading.Event()
    lost = threading.Event()

    def renew() -> None:
        interval = max(0.25, min(60.0, ttl / 3.0))
        while not stop.wait(interval):
            try:
                if not db.renew_decode_cache_build_lock(key, owner_token, ttl_seconds=ttl):
                    lost.set()
                    return
            except Exception:
                # A transient SQLite contention error is not proof that
                # ownership was lost. The fenced publish step verifies it
                # synchronously before touching the final directory.
                continue

    thread = threading.Thread(target=renew, name="vfieval-decode-cache-build", daemon=True)
    thread.start()
    try:
        yield DecodeCacheBuildLock(key, owner_token, lost, ttl)
    finally:
        stop.set()
        thread.join(timeout=2.0)
        db.release_decode_cache_build_lock(key, owner_token)


class RunCleanupService:
    """Persistent, idempotent Run artifact cleanup and cache GC coordinator."""

    def __init__(
        self,
        db: Database,
        workspace: WorkspaceConfig,
        *,
        cache_grace_seconds: float = CACHE_GRACE_SECONDS,
        purge_preview_ttl_seconds: float = RUN_PURGE_PREVIEW_TTL_SECONDS,
    ) -> None:
        self.db = db
        self.workspace = workspace
        self.cache_grace_seconds = max(0.0, float(cache_grace_seconds))
        self.purge_preview_ttl_seconds = max(1.0, float(purge_preview_ttl_seconds))
        self._gc_preview_lock = threading.Lock()
        self._gc_preview_tokens: dict[str, dict[str, Any]] = {}
        self._run_purge_preview_lock = threading.Lock()
        self._run_purge_preview_tokens: dict[str, dict[str, Any]] = {}

    def ensure_backfilled(self) -> dict[str, int]:
        key = str(self.db.db_path.resolve())
        with _BACKFILL_LOCK:
            if key in _BACKFILLED_DATABASES:
                return {"physical_entries": 0, "run_refs": 0, "released_refs": 0}
            report = self.backfill_cache_catalog()
            _BACKFILLED_DATABASES.add(key)
            return report

    def run_forever(self, stop_event: threading.Event, poll_interval: float = 1.0) -> None:
        """Resume persistent purge requests until the server asks the loop to stop."""
        self.ensure_backfilled()
        while not stop_event.is_set():
            try:
                self.process_pending(limit=100)
            except Exception as exc:
                # Individual purge failures are persisted by process_request.
                # This guard keeps an unexpected coordinator error from killing
                # automatic cleanup for every other Run.
                runtime_logger().exception(
                    "run cleanup loop failed",
                    extra={
                        "event": "run_cleanup.loop_failed",
                        "details": {"error_type": type(exc).__name__},
                    },
                )
            stop_event.wait(max(0.1, float(poll_interval)))

    def backfill_cache_catalog(self) -> dict[str, int]:
        physical_entries = 0
        for cache_type in ("decode_cache", "compare_cache"):
            root = _cache_root(self.workspace, cache_type)
            if not root.exists():
                continue
            for child in root.iterdir():
                if cache_type == "decode_cache" and child.name.endswith(".partial") and child.is_dir():
                    # An interrupted decode leaves a direct ``<key>.partial``
                    # directory. It has no Run ref, so catalog it as failed and
                    # immediately eligible for the explicit storage-GC flow.
                    cache_key = child.name
                    try:
                        resolved = _trusted_cache_path(self.workspace, cache_type, cache_key, child)
                        last_used_at = float(resolved.stat().st_mtime)
                        self.db.upsert_cache_entry(
                            cache_type,
                            cache_key,
                            resolved,
                            state="failed",
                            size_bytes=_path_size(resolved),
                            metadata={"backfilled": True, "partial": True},
                            last_used_at=last_used_at,
                            gc_after=last_used_at,
                        )
                        physical_entries += 1
                    except (OSError, ValueError):
                        pass
                    continue
                if cache_type == "decode_cache" and not child.is_dir():
                    continue
                if cache_type == "compare_cache" and not child.is_file():
                    continue
                cache_key = child.name if cache_type == "decode_cache" else child.stem
                _cache_entry_from_path(
                    self.db,
                    self.workspace,
                    cache_type,
                    cache_key,
                    child,
                    metadata={"backfilled": True},
                )
                physical_entries += 1

        run_refs = 0
        active_run_ids = {
            int(row["id"])
            for row in self.db.query(
                """
                SELECT id FROM runs
                WHERE deleted_at IS NULL AND artifact_cleaned_at IS NULL
                ORDER BY id
                """
            )
        }
        for run_id in sorted(active_run_ids):
            result = register_run_cache_refs(
                self.db,
                self.workspace,
                run_id,
                grace_seconds=self.cache_grace_seconds,
            )
            run_refs += int(result.get("total") or 0)

        released_refs = 0
        inactive_rows = self.db.query(
            """
            SELECT DISTINCT rcr.run_id
            FROM run_cache_refs rcr
            JOIN runs r ON r.id = rcr.run_id
            WHERE rcr.released_at IS NULL
              AND (r.deleted_at IS NOT NULL OR r.artifact_cleaned_at IS NOT NULL)
            """
        )
        for row in inactive_rows:
            released_refs += len(
                self.db.release_run_cache_refs(
                    int(row["run_id"]),
                    grace_seconds=self.cache_grace_seconds,
                )
            )

        now = utc_ts()
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE media_assets
                SET state = 'unavailable', updated_at = ?
                WHERE source_kind = 'run_artifact'
                  AND id IN (
                      SELECT rma.asset_id
                      FROM run_media_assets rma
                      JOIN runs r ON r.id = rma.run_id
                      WHERE r.deleted_at IS NOT NULL OR r.artifact_cleaned_at IS NOT NULL
                  )
                """,
                (now,),
            )
        return {
            "physical_entries": physical_entries,
            "run_refs": run_refs,
            "released_refs": released_refs,
        }

    def preview_run_purge(
        self,
        request_type: str,
        run_ids: Iterable[int],
    ) -> dict[str, Any]:
        """Preview one exact Run deletion/cleanup selection and mint a one-use token.

        The token is process-local and deliberately short lived.  It binds the
        normalized Run ID set, operation type, Run lifecycle rows, active Jobs,
        dependency bindings, cache references, and exact managed Run-directory
        metadata observed here.
        """

        operation = self._normalize_run_purge_type(request_type)
        selected_ids = self._normalize_run_ids(run_ids)
        self.ensure_backfilled()
        for run_id in selected_ids:
            # Keep the cache accounting exact even for historical databases
            # whose Run references predate the cache catalog.
            register_run_cache_refs(
                self.db,
                self.workspace,
                run_id,
                grace_seconds=self.cache_grace_seconds,
            )
        preview, state_fingerprint = self._build_run_purge_preview(operation, selected_ids)
        now = utc_ts()
        expires_at = now + self.purge_preview_ttl_seconds
        token = uuid.uuid4().hex
        preview.update(
            {
                "generated_at": now,
                "expires_at": expires_at,
                "preview_token": token,
            }
        )
        with self._run_purge_preview_lock:
            # Retain recently expired entries long enough for callers to get a
            # specific ``expired_preview`` response instead of an ambiguous miss.
            cutoff = now - self.purge_preview_ttl_seconds
            self._run_purge_preview_tokens = {
                key: value
                for key, value in self._run_purge_preview_tokens.items()
                if float(value.get("expires_at") or 0) >= cutoff
            }
            self._run_purge_preview_tokens[token] = {
                "request_type": operation,
                "run_ids": selected_ids,
                "state_fingerprint": state_fingerprint,
                "expires_at": expires_at,
                "preview": preview,
            }
        return preview

    def consume_run_purge_preview(
        self,
        preview_token: str | None,
        *,
        request_type: str,
        run_ids: Iterable[int],
    ) -> dict[str, Any]:
        """Consume and validate a Run purge preview immediately before mutation."""

        token = str(preview_token or "").strip()
        if not token:
            raise RunPurgePreviewError(
                "missing_preview",
                "Run deletion or artifact cleanup requires a fresh preview token",
            )
        operation = self._normalize_run_purge_type(request_type)
        selected_ids = self._normalize_run_ids(run_ids)
        with self._run_purge_preview_lock:
            snapshot = self._run_purge_preview_tokens.pop(token, None)
        if snapshot is None:
            raise RunPurgePreviewError(
                "missing_preview",
                "Run purge preview token is missing or was already consumed; preview again",
            )
        if float(snapshot.get("expires_at") or 0) < utc_ts():
            raise RunPurgePreviewError(
                "expired_preview",
                "Run purge preview token expired; preview the current Run state again",
            )
        if str(snapshot.get("request_type") or "") != operation:
            raise RunPurgePreviewError(
                "preview_mismatch",
                "Run purge preview was created for a different operation",
            )
        if tuple(snapshot.get("run_ids") or ()) != selected_ids:
            raise RunPurgePreviewError(
                "preview_mismatch",
                "Run purge preview does not match the exact selected Run IDs",
            )
        current_preview, current_fingerprint = self._build_run_purge_preview(
            operation, selected_ids
        )
        state_changed = current_fingerprint != str(snapshot.get("state_fingerprint") or "")
        if state_changed and not self._active_delete_preview_compatible(
            snapshot.get("preview") or {}, current_preview
        ):
            raise RunPurgePreviewError(
                "stale_preview",
                "Run, dependency, artifact, or cache state changed after preview; preview again",
            )
        return {
            "validated": True,
            "validated_at": utc_ts(),
            "preview_token": token,
            "request_type": operation,
            "run_ids": list(selected_ids),
            "preview": current_preview if state_changed else snapshot["preview"],
            "state_changed_after_preview": state_changed,
        }

    @staticmethod
    def _active_delete_preview_compatible(
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        """Allow expected progress drift while deleting an active Run.

        Active workers update the Run row, Job heartbeat, and managed output
        tree continuously. Requiring those volatile values to remain byte-for-
        byte identical makes a preview token impossible to consume, even when
        the user confirms immediately. The relaxed path is intentionally
        narrow: it applies only when every originally previewed Run was active,
        keeps the exact operation and Run IDs, and still rejects any Campaign
        or Compare dependency change. Cleanup scope remains the trusted
        ``runs/{id}`` directory, and the deletion service rechecks workers and
        preserves dependencies before unlinking anything.
        """

        if (
            str(previous.get("request_type") or "") != "delete_run"
            or str(current.get("request_type") or "") != "delete_run"
        ):
            return False
        previous_ids = [int(value) for value in previous.get("run_ids") or []]
        current_ids = [int(value) for value in current.get("run_ids") or []]
        if not previous_ids or previous_ids != current_ids:
            return False
        previous_runs = {
            int(row.get("run_id") or 0): row for row in previous.get("runs") or []
        }
        current_runs = {
            int(row.get("run_id") or 0): row for row in current.get("runs") or []
        }
        if set(previous_runs) != set(previous_ids) or set(current_runs) != set(previous_ids):
            return False
        for run_id in previous_ids:
            before = previous_runs[run_id]
            after = current_runs[run_id]
            if (
                str(before.get("status") or "") in TERMINAL_RUN_STATUSES
                or bool(before.get("deleted"))
                or bool(after.get("deleted"))
                or not bool(before.get("allowed"))
                or not bool(after.get("allowed"))
            ):
                return False
            before_dependencies = dict(before.get("dependencies") or {})
            after_dependencies = dict(after.get("dependencies") or {})
            # Job IDs and their heartbeat/status are expected to change while
            # cancellation races inference/finalization. Semantic dependencies
            # are not expected to change and still fence the token.
            before_dependencies.pop("active_job_ids", None)
            after_dependencies.pop("active_job_ids", None)
            if before_dependencies != after_dependencies:
                return False
        return True

    @staticmethod
    def _normalize_run_purge_type(request_type: str) -> str:
        operation = str(request_type or "").strip()
        if operation not in {"delete_run", "cleanup_artifacts"}:
            raise ValueError("request_type must be delete_run or cleanup_artifacts")
        return operation

    @staticmethod
    def _normalize_run_ids(run_ids: Iterable[int]) -> tuple[int, ...]:
        try:
            selected = tuple(sorted({int(value) for value in run_ids}))
        except (TypeError, ValueError) as exc:
            raise ValueError("run_ids must contain exact integer Run IDs") from exc
        if not selected or selected[0] <= 0:
            raise ValueError("run_ids must contain at least one positive Run ID")
        return selected

    def _trusted_preview_run_dir(self, run_id: int) -> Path:
        runs_root = self.workspace.runs_dir.resolve()
        lexical = self.workspace.runs_dir / str(int(run_id))
        if lexical.is_symlink():
            raise ValueError(f"Run {run_id} output directory is a symbolic link")
        resolved = lexical.resolve()
        if (
            not _is_relative_to(resolved, runs_root)
            or resolved.parent != runs_root
            or resolved.name != str(int(run_id))
        ):
            raise ValueError(f"Run {run_id} output directory is outside the managed runs root")
        return resolved

    def _build_run_purge_preview(
        self,
        request_type: str,
        run_ids: tuple[int, ...],
    ) -> tuple[dict[str, Any], str]:
        placeholders = ",".join("?" for _ in run_ids)
        run_rows = self.db.query(
            f"""
            SELECT id, name, status, content_revision, deleted_at,
                   artifact_cleaned_at, updated_at
            FROM runs
            WHERE id IN ({placeholders})
            ORDER BY id
            """,
            run_ids,
        )
        found_ids = {int(row["id"]) for row in run_rows}
        missing = [run_id for run_id in run_ids if run_id not in found_ids]
        if missing:
            raise KeyError(f"run {missing[0]} not found")

        cache_rows = self.db.query(
            f"""
            SELECT rcr.run_id, ce.id, ce.cache_type, ce.cache_key,
                   ce.storage_path, ce.state, ce.size_bytes, ce.updated_at,
                   ce.gc_after, ce.deleted_at
            FROM run_cache_refs rcr
            JOIN cache_entries ce ON ce.id = rcr.cache_entry_id
            WHERE rcr.run_id IN ({placeholders})
              AND rcr.released_at IS NULL
              AND ce.deleted_at IS NULL AND ce.state != 'deleted'
            ORDER BY rcr.run_id, ce.id
            """,
            run_ids,
        )
        cache_entry_ids = sorted({int(row["id"]) for row in cache_rows})
        all_refs: dict[int, set[int]] = {entry_id: set() for entry_id in cache_entry_ids}
        active_leases: dict[int, list[dict[str, Any]]] = {entry_id: [] for entry_id in cache_entry_ids}
        if cache_entry_ids:
            cache_placeholders = ",".join("?" for _ in cache_entry_ids)
            for row in self.db.query(
                f"""
                SELECT cache_entry_id, run_id
                FROM run_cache_refs
                WHERE cache_entry_id IN ({cache_placeholders}) AND released_at IS NULL
                ORDER BY cache_entry_id, run_id
                """,
                cache_entry_ids,
            ):
                all_refs[int(row["cache_entry_id"])].add(int(row["run_id"]))
            now = utc_ts()
            for row in self.db.query(
                f"""
                SELECT cache_entry_id, lease_id, expires_at
                FROM cache_leases
                WHERE cache_entry_id IN ({cache_placeholders}) AND expires_at > ?
                ORDER BY cache_entry_id, lease_id
                """,
                (*cache_entry_ids, now),
            ):
                active_leases[int(row["cache_entry_id"])].append(
                    {
                        "lease_id": str(row["lease_id"]),
                        "expires_at": float(row["expires_at"]),
                    }
                )

        selected_set = set(run_ids)
        cache_by_run: dict[int, list[dict[str, Any]]] = {run_id: [] for run_id in run_ids}
        unique_caches: dict[int, dict[str, Any]] = {}
        for raw in cache_rows:
            row = dict(raw)
            entry_id = int(row["id"])
            refs = sorted(all_refs.get(entry_id) or set())
            cache = {
                "id": entry_id,
                "cache_type": str(row.get("cache_type") or ""),
                "cache_key": str(row.get("cache_key") or ""),
                "state": str(row.get("state") or ""),
                "size_bytes": int(row.get("size_bytes") or 0),
                "active_run_refs": refs,
                "active_leases": len(active_leases.get(entry_id) or []),
                "shared": len(refs) > 1,
                "shared_with_unselected": bool(set(refs) - selected_set),
            }
            cache_by_run[int(row["run_id"])].append(cache)
            unique_caches[entry_id] = {
                **cache,
                "storage_path": str(row.get("storage_path") or ""),
                "updated_at": row.get("updated_at"),
                "gc_after": row.get("gc_after"),
                "deleted_at": row.get("deleted_at"),
                "leases": active_leases.get(entry_id) or [],
            }

        public_runs: list[dict[str, Any]] = []
        state_runs: list[dict[str, Any]] = []
        campaign_ids: set[int] = set()
        compare_run_ids: set[int] = set()
        active_job_ids: set[int] = set()
        for raw_run in run_rows:
            run = dict(raw_run)
            run_id = int(run["id"])
            run_dir = self._trusted_preview_run_dir(run_id)
            directory = _path_state(run_dir)
            dependencies, dependency_state = self._run_purge_dependencies(run_id)
            campaign_ids.update(int(value) for value in dependencies["campaign_ids"])
            compare_run_ids.update(int(value) for value in dependencies["compare_run_ids"])
            active_job_ids.update(int(value) for value in dependencies["active_job_ids"])
            referenced_caches = cache_by_run[run_id]
            referenced_cache_bytes = sum(int(row["size_bytes"]) for row in referenced_caches)
            shared_cache_bytes = sum(
                int(row["size_bytes"]) for row in referenced_caches if row["shared"]
            )
            exclusive_cache_bytes = sum(
                int(row["size_bytes"]) for row in referenced_caches if not row["shared"]
            )
            reason = "ready"
            allowed = True
            if request_type == "cleanup_artifacts" and str(run.get("status") or "") not in TERMINAL_RUN_STATUSES:
                allowed = False
                reason = "run_not_terminal"
            elif request_type == "cleanup_artifacts" and dependencies["active_job_ids"]:
                allowed = False
                reason = "active_worker"
            elif request_type == "cleanup_artifacts" and run.get("artifact_cleaned_at") is not None:
                allowed = False
                reason = "artifacts_already_cleaned"
            elif request_type == "delete_run" and run.get("deleted_at") is not None:
                allowed = False
                reason = "run_already_deleted"
            public_runs.append(
                {
                    "run_id": run_id,
                    "name": str(run.get("name") or ""),
                    "status": str(run.get("status") or ""),
                    "allowed": allowed,
                    "reason": reason,
                    "artifact_cleaned": run.get("artifact_cleaned_at") is not None,
                    "deleted": run.get("deleted_at") is not None,
                    "dependencies": dependencies,
                    "cache_entry_ids": [int(row["id"]) for row in referenced_caches],
                    "bytes": {
                        "run_directory_bytes": int(directory["logical_bytes"]),
                        "exclusive_run_bytes": int(directory["reclaimable_bytes"]),
                        "referenced_cache_bytes": referenced_cache_bytes,
                        "shared_cache_bytes": shared_cache_bytes,
                        "exclusive_cache_bytes": exclusive_cache_bytes,
                        # Purging a Run only unlinks its managed directory. Cache
                        # bytes remain protected until refs, leases, and grace all clear.
                        "estimated_reclaimable_bytes": int(directory["reclaimable_bytes"]),
                    },
                }
            )
            state_runs.append(
                {
                    **run,
                    "directory": directory,
                    "dependencies": dependency_state,
                    "cache_entry_ids": [int(row["id"]) for row in referenced_caches],
                }
            )

        unique_cache_rows = list(unique_caches.values())
        referenced_cache_bytes = sum(int(row["size_bytes"]) for row in unique_cache_rows)
        shared_cache_bytes = sum(
            int(row["size_bytes"]) for row in unique_cache_rows if row["shared"]
        )
        external_shared_cache_bytes = sum(
            int(row["size_bytes"])
            for row in unique_cache_rows
            if row["shared_with_unselected"]
        )
        exclusive_cache_bytes = sum(
            int(row["size_bytes"])
            for row in unique_cache_rows
            if not row["shared_with_unselected"]
        )
        exclusive_run_bytes = sum(int(row["bytes"]["exclusive_run_bytes"]) for row in public_runs)
        preview = {
            "request_type": request_type,
            "run_ids": list(run_ids),
            "runs": public_runs,
            "summary": {
                "run_count": len(public_runs),
                "allowed_run_count": sum(1 for row in public_runs if row["allowed"]),
                "run_directory_bytes": sum(
                    int(row["bytes"]["run_directory_bytes"]) for row in public_runs
                ),
                "exclusive_run_bytes": exclusive_run_bytes,
                "referenced_cache_bytes": referenced_cache_bytes,
                "shared_cache_bytes": shared_cache_bytes,
                "shared_with_unselected_cache_bytes": external_shared_cache_bytes,
                "exclusive_cache_bytes": exclusive_cache_bytes,
                "estimated_reclaimable_bytes": exclusive_run_bytes,
                "potential_cache_bytes_after_grace": exclusive_cache_bytes,
                "dependencies": {
                    "campaign_ids": sorted(campaign_ids),
                    "compare_run_ids": sorted(compare_run_ids),
                    "active_job_ids": sorted(active_job_ids),
                },
            },
        }
        state = {
            "request_type": request_type,
            "run_ids": run_ids,
            "runs": state_runs,
            "caches": sorted(unique_cache_rows, key=lambda row: int(row["id"])),
        }
        state_fingerprint = hashlib.sha256(
            json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return preview, state_fingerprint

    def _run_purge_dependencies(
        self,
        run_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        jobs = self._active_jobs(run_id)
        active_jobs = [
            {
                "job_id": int(row["job_id"]),
                "kind": str(row.get("kind") or ""),
                "status": str(row.get("status") or ""),
                "updated_at": row.get("updated_at"),
            }
            for row in jobs
        ]
        campaign_ids = self._run_campaign_ids(run_id)
        preparing_campaign_ids = self._preparing_campaign_ids(run_id)
        campaign_state = [
            {
                "version": "v1",
                "campaign_id": int(row["campaign_id"]),
                "status": str(row.get("status") or ""),
                "updated_at": row.get("updated_at"),
            }
            for row in self.db.query(
                """
                SELECT DISTINCT c.id AS campaign_id, c.status, c.updated_at
                FROM run_media_assets rma
                JOIN evaluation_candidates ec
                  ON ec.asset_id = rma.asset_id OR ec.reference_asset_id = rma.asset_id
                JOIN evaluation_campaigns c ON c.id = ec.campaign_id
                WHERE rma.run_id = ?
                ORDER BY c.id
                """,
                (int(run_id),),
            )
        ]
        if self.db.get(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evaluation_methods_v2'"
        ) is not None:
            campaign_state.extend(
                {
                    "version": "v2",
                    "campaign_id": int(row["campaign_id"]),
                    "status": str(row.get("status") or ""),
                    "updated_at": row.get("updated_at"),
                }
                for row in self.db.query(
                    """
                    SELECT DISTINCT c.id AS campaign_id, c.status, c.updated_at
                    FROM evaluation_methods_v2 m
                    JOIN evaluation_campaigns_v2 c ON c.id = m.campaign_id
                    WHERE m.source_run_id = ?
                    ORDER BY c.id
                    """,
                    (int(run_id),),
                )
            )
        purge_requests = self.db.query(
            """
            SELECT id, request_type, status, attempt_count, updated_at
            FROM run_purge_requests
            WHERE run_id = ?
            ORDER BY id
            """,
            (int(run_id),),
        )
        compare_bindings: list[dict[str, Any]] = []
        if self.db.get(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_media_item_bindings'"
        ) is not None:
            compare_bindings = self.db.query(
                """
                SELECT rib.id AS binding_id, rib.run_id AS compare_run_id,
                       rib.item_id, rib.slot, rib.active_member_id,
                       rib.updated_at
                FROM run_media_item_bindings rib
                JOIN media_item_members mim ON mim.id = rib.active_member_id
                JOIN runs dependent ON dependent.id = rib.run_id
                WHERE mim.producer_run_id = ?
                  AND rib.binding_role = 'compare_pred'
                  AND dependent.deleted_at IS NULL
                  AND dependent.artifact_cleaned_at IS NULL
                ORDER BY rib.run_id, rib.id
                """,
                (int(run_id),),
            )
        compare_run_ids = sorted({int(row["compare_run_id"]) for row in compare_bindings})
        public = {
            "campaign_ids": campaign_ids,
            "preparing_campaign_ids": preparing_campaign_ids,
            "compare_run_ids": compare_run_ids,
            "compare_binding_count": len(compare_bindings),
            "active_job_ids": [int(row["job_id"]) for row in active_jobs],
        }
        state = {
            **public,
            "active_jobs": active_jobs,
            "campaigns": campaign_state,
            "compare_bindings": compare_bindings,
            "purge_requests": purge_requests,
        }
        return public, state

    def request_delete(self, run_id: int) -> dict[str, Any]:
        self.ensure_backfilled()
        register_run_cache_refs(
            self.db,
            self.workspace,
            int(run_id),
            grace_seconds=self.cache_grace_seconds,
        )
        request = self.db.request_run_purge(int(run_id), "delete_run")
        if request.get("status") == "completed":
            return request
        run = self.db.get_run(int(run_id))
        if str(run.get("status") or "") not in TERMINAL_RUN_STATUSES:
            self.db.request_run_cancel(int(run_id))
            return self.db.update_run_purge_request(
                int(request["id"]),
                "canceling",
                report={"phase": "waiting_for_workers", "active_job_ids": []},
                error={},
            )
        self.db.cancel_queued_run_jobs(int(run_id), "Run is queued for deletion")
        return self.db.get_run_purge_request_by_id(int(request["id"]))

    def request_artifact_cleanup(self, run_id: int) -> dict[str, Any]:
        self.ensure_backfilled()
        register_run_cache_refs(
            self.db,
            self.workspace,
            int(run_id),
            grace_seconds=self.cache_grace_seconds,
        )
        run = self.db.get_run(int(run_id))
        if str(run.get("status") or "") in TERMINAL_RUN_STATUSES:
            self.db.cancel_queued_run_jobs(int(run_id), "Run artifacts are being cleaned")
        active_jobs = self._active_jobs(int(run_id))
        if str(run.get("status") or "") not in TERMINAL_RUN_STATUSES or active_jobs:
            raise ValueError(
                "cleanup-artifacts is only allowed after a run is completed, failed, or canceled and all workers have stopped"
            )
        request = self.db.request_run_purge(int(run_id), "cleanup_artifacts")
        return request

    def process_pending(self, limit: int = 100) -> list[dict[str, Any]]:
        self.ensure_backfilled()
        self.db.recover_stale_run_purge_requests(utc_ts() - PURGE_CLAIM_STALE_SECONDS)
        result = []
        for request in self.db.list_pending_run_purge_requests(limit=limit):
            result.append(self.process_request(int(request["id"])))
        return result

    def process_request(self, request_id: int) -> dict[str, Any]:
        request = self.db.get_run_purge_request_by_id(int(request_id))
        if request["status"] in {"completed", "failed", "purging"}:
            return request
        run_id = int(request["run_id"])
        run = self.db.get_run(run_id)
        active_jobs = self._active_jobs(run_id)
        request_type = str(request["request_type"])

        if request_type == "delete_run" and (
            str(run.get("status") or "") not in TERMINAL_RUN_STATUSES or active_jobs
        ):
            if str(run.get("status") or "") not in TERMINAL_RUN_STATUSES:
                self.db.request_run_cancel(run_id)
                run = self.db.get_run(run_id)
                active_jobs = self._active_jobs(run_id)
                if str(run.get("status") or "") == "cancel_requested" and not active_jobs:
                    self.db.cancel_run(run_id, {"message": "Run canceled for deletion", "type": "RunCanceled"})
                    run = self.db.get_run(run_id)
            if str(run.get("status") or "") not in TERMINAL_RUN_STATUSES or active_jobs:
                return self.db.update_run_purge_request(
                    int(request["id"]),
                    "canceling",
                    report={
                        "phase": "waiting_for_workers",
                        "active_job_ids": [int(row["job_id"]) for row in active_jobs],
                    },
                    error={},
                )

        if request_type == "cleanup_artifacts" and (
            str(run.get("status") or "") not in TERMINAL_RUN_STATUSES or active_jobs
        ):
            return self.db.update_run_purge_request(
                int(request["id"]),
                "failed",
                error={"type": "RunBusy", "message": "run workers are still active"},
            )

        claim_token = uuid.uuid4().hex
        if not self.db.claim_run_purge_request(int(request["id"]), claim_token):
            return self.db.get_run_purge_request_by_id(int(request["id"]))
        with _purge_claim_heartbeat(self.db, int(request["id"]), claim_token) as claim_lost:
            try:
                report = self._purge_run(run_id, delete_run=request_type == "delete_run")
                if claim_lost.is_set():
                    raise RuntimeError("run purge claim was lost before cleanup completed")
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                for key in ("code", "campaign_id", "campaign_ids", "action"):
                    value = getattr(exc, key, None)
                    if value is not None:
                        error[key] = value
                try:
                    return self.db.update_run_purge_request(
                        int(request["id"]),
                        "failed",
                        error=error,
                        expected_claim_token=claim_token,
                    )
                except RuntimeError:
                    return self.db.get_run_purge_request_by_id(int(request["id"]))
            try:
                return self.db.update_run_purge_request(
                    int(request["id"]),
                    "completed",
                    report=report,
                    error={},
                    reclaimed_bytes=int(report.get("reclaimed_bytes") or 0),
                    expected_claim_token=claim_token,
                )
            except RuntimeError:
                return self.db.get_run_purge_request_by_id(int(request["id"]))

    def _active_jobs(self, run_id: int) -> list[dict[str, Any]]:
        return self.db.list_run_associated_jobs(int(run_id), statuses=("running",))

    def _prepare_protected_campaign_media(self, run_id: int) -> dict[str, Any]:
        try:
            from vfieval.evaluations_v2 import protect_campaign_media_for_run
        except ModuleNotFoundError:
            return {"protected": 0, "campaign_ids": []}
        return protect_campaign_media_for_run(self.db, self.workspace, int(run_id))

    def _prepare_compare_input_snapshots(self, run_id: int) -> dict[str, Any]:
        """Privately freeze active reusable Item Compare inputs before purge.

        File snapshots are completed first.  The new asset, non-reusable
        ``compare_snapshot`` member, dependent Run asset binding, and
        ``active_member_id`` transition then commit in one SQLite transaction.
        ``original_member_id`` remains untouched for provenance.
        """
        source_run = self.db.get_run(int(run_id))
        source_run_type = str(
            (source_run.get("metadata") or {}).get("run_type") or "model_inference"
        )
        if source_run_type != "model_inference":
            return {"protected": 0, "compare_run_ids": [], "snapshots": []}
        if self.db.get(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_media_item_bindings'"
        ) is None:
            return {"protected": 0, "compare_run_ids": [], "snapshots": []}
        dependencies = self.db.query(
            """
            SELECT
                rib.id AS binding_id, rib.run_id AS compare_run_id,
                rib.item_id, rib.slot, rib.metadata_json AS binding_metadata_json,
                mim.id AS source_member_id, mim.asset_id AS source_asset_id,
                mim.method_key AS source_method_key,
                mim.temporal_mapping_json, mim.spatial_origin_json,
                mim.metadata_json AS member_metadata_json,
                a.media_kind, a.display_name, a.original_name, a.storage_path,
                a.mime_type, a.frame_count, a.width, a.height, a.fps,
                cr.metadata_json AS compare_run_metadata_json
            FROM run_media_item_bindings rib
            JOIN media_item_members mim ON mim.id = rib.active_member_id
            JOIN media_assets a ON a.id = mim.asset_id
            JOIN runs cr ON cr.id = rib.run_id
            WHERE rib.binding_role = 'compare_pred'
              AND mim.producer_run_id = ?
              AND mim.member_role = 'model_pred'
              AND mim.producer_kind = 'model_inference'
              AND mim.reusable_as_pred = 1
              AND mim.state = 'ready' AND mim.deleted_at IS NULL
              AND a.source_kind = 'run_artifact' AND a.role = 'pred'
              AND a.state = 'ready' AND a.deleted_at IS NULL
              AND cr.deleted_at IS NULL AND cr.artifact_cleaned_at IS NULL
              AND EXISTS (
                  SELECT 1 FROM run_media_assets source_output
                  WHERE source_output.run_id = ?
                    AND source_output.asset_id = a.id
                    AND source_output.role = 'pred'
              )
            ORDER BY rib.run_id, rib.item_id, rib.slot, rib.id
            """,
            (int(run_id), int(run_id)),
        )
        if not dependencies:
            return {"protected": 0, "compare_run_ids": [], "snapshots": []}

        runs_root = self.workspace.runs_dir.resolve()
        source_run_dir = (runs_root / str(int(run_id))).resolve()
        if not _is_relative_to(source_run_dir, runs_root) or source_run_dir.parent != runs_root:
            raise ValueError("source Run output directory is outside the managed runs root")

        plans: list[dict[str, Any]] = []
        created_dirs: list[tuple[Path, Path]] = []
        try:
            for dependency in dependencies:
                binding_id = int(dependency["binding_id"])
                compare_run_id = int(dependency["compare_run_id"])
                if compare_run_id == int(run_id):
                    raise ValueError("a Run cannot snapshot a Compare input into itself")
                if str(
                    _json_object(dependency.get("compare_run_metadata_json")).get("run_type") or ""
                ) != "video_compare":
                    raise ValueError(
                        f"dependent Run {compare_run_id} has a compare_pred binding but is not a video_compare Run"
                    )
                media_kind = str(dependency.get("media_kind") or "")
                if media_kind not in {"video", "frame_sequence"}:
                    raise ValueError(f"unsupported source media kind for Compare snapshot: {media_kind}")

                raw_source = str(dependency.get("storage_path") or "").strip()
                if not raw_source:
                    raise ValueError(f"Compare binding {binding_id} source has no managed storage path")
                source_candidate = Path(raw_source)
                if source_candidate.is_symlink():
                    raise ValueError("Compare snapshot source must not be a symbolic link")
                source_path = source_candidate.resolve()
                if (
                    not _is_relative_to(source_path, source_run_dir)
                    or source_path == source_run_dir
                    or not source_path.exists()
                    or not (source_path.is_file() or source_path.is_dir())
                ):
                    raise ValueError(
                        f"Compare binding {binding_id} source is outside its trusted Run directory or unavailable"
                    )

                compare_run_dir = (runs_root / str(compare_run_id)).resolve()
                if not _is_relative_to(compare_run_dir, runs_root) or compare_run_dir.parent != runs_root:
                    raise ValueError("dependent Compare output directory is outside the managed runs root")
                slot = str(dependency.get("slot") or "")
                safe_slot = _safe_snapshot_slot(slot, binding_id)
                inputs_root = (compare_run_dir / "inputs" / safe_slot).resolve()
                snapshot_dir = (inputs_root / f"snapshot-{uuid.uuid4().hex}").resolve()
                suffix = source_path.suffix if source_path.is_file() else ""
                destination = (snapshot_dir / f"member-{int(dependency['source_member_id'])}{suffix}").resolve()
                if (
                    not _is_relative_to(inputs_root, compare_run_dir)
                    or not _is_relative_to(snapshot_dir, inputs_root)
                    or snapshot_dir.parent != inputs_root
                    or not _is_relative_to(destination, snapshot_dir)
                    or destination.parent != snapshot_dir
                ):
                    raise ValueError("Compare snapshot destination escaped its dependent Run directory")

                created_dirs.append((snapshot_dir, inputs_root))
                materialized = _materialize_snapshot(source_path, destination)
                plans.append(
                    {
                        **dependency,
                        "source_path": source_path,
                        "destination": destination,
                        "materialized": materialized,
                        "binding_metadata": _json_object(dependency.get("binding_metadata_json")),
                        "member_metadata": _json_object(dependency.get("member_metadata_json")),
                    }
                )

            now = utc_ts()
            snapshots: list[dict[str, Any]] = []
            with _immediate_transaction(self.db) as conn:
                for plan in plans:
                    binding_id = int(plan["binding_id"])
                    compare_run_id = int(plan["compare_run_id"])
                    source_member_id = int(plan["source_member_id"])
                    source_asset_id = int(plan["source_asset_id"])
                    item_id = int(plan["item_id"])
                    slot = str(plan.get("slot") or "")
                    current = conn.execute(
                        """
                        SELECT cr.metadata_json
                        FROM run_media_item_bindings rib
                        JOIN media_item_members mim ON mim.id = rib.active_member_id
                        JOIN media_assets a ON a.id = mim.asset_id
                        JOIN runs cr ON cr.id = rib.run_id
                        WHERE rib.id = ? AND rib.run_id = ? AND rib.item_id = ?
                          AND rib.binding_role = 'compare_pred' AND rib.slot = ?
                          AND rib.active_member_id = ?
                          AND mim.producer_run_id = ?
                          AND mim.member_role = 'model_pred'
                          AND mim.producer_kind = 'model_inference'
                          AND mim.reusable_as_pred = 1
                          AND mim.state = 'ready' AND mim.deleted_at IS NULL
                          AND a.id = ? AND a.source_kind = 'run_artifact'
                          AND a.role = 'pred' AND a.state = 'ready' AND a.deleted_at IS NULL
                          AND cr.deleted_at IS NULL AND cr.artifact_cleaned_at IS NULL
                        """,
                        (
                            binding_id,
                            compare_run_id,
                            item_id,
                            slot,
                            source_member_id,
                            int(run_id),
                            source_asset_id,
                        ),
                    ).fetchone()
                    if current is None or str(
                        _json_object(current["metadata_json"]).get("run_type") or ""
                    ) != "video_compare":
                        raise RuntimeError("Compare input binding changed during snapshot")

                    source_key = f"run_artifact:compare-snapshot:{compare_run_id}:{binding_id}:{uuid.uuid4().hex}"
                    provenance = {
                        "compare_snapshot": True,
                        "compare_run_id": compare_run_id,
                        "source_run_id": int(run_id),
                        "source_member_id": source_member_id,
                        "source_asset_id": source_asset_id,
                        "binding_id": binding_id,
                        "slot": slot,
                    }
                    asset_metadata = {
                        "compare_snapshot": True,
                        "immutable_input": True,
                        "source_run_id": int(run_id),
                        "source_member_id": source_member_id,
                        "source_asset_id": source_asset_id,
                        "binding_id": binding_id,
                        "copy_method": str(plan["materialized"]["method"]),
                    }
                    asset_cur = conn.execute(
                        """
                        INSERT INTO media_assets(
                            collection_id, source_key, source_kind, media_kind, role,
                            display_name, original_name, state, content_sha256,
                            size_bytes, storage_path, mime_type, frame_count, width,
                            height, fps, provenance_json, metadata_json,
                            created_at, updated_at, deleted_at
                        ) VALUES (?, ?, 'run_artifact', ?, 'pred', ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        (
                            None,
                            source_key,
                            str(plan["media_kind"]),
                            f"Compare input snapshot {binding_id}",
                            str(plan.get("original_name") or Path(plan["destination"]).name)[:500],
                            str(plan["materialized"]["content_sha256"]),
                            int(plan["materialized"]["size_bytes"]),
                            str(plan["destination"]),
                            str(plan.get("mime_type") or "application/octet-stream"),
                            int(plan.get("frame_count") or 0),
                            int(plan.get("width") or 0),
                            int(plan.get("height") or 0),
                            float(plan["fps"]) if plan.get("fps") is not None else None,
                            json.dumps(provenance, sort_keys=True, ensure_ascii=False),
                            json.dumps(asset_metadata, sort_keys=True, ensure_ascii=False),
                            now,
                            now,
                        ),
                    )
                    snapshot_asset_id = int(asset_cur.lastrowid)
                    member_metadata = {
                        "snapshot_reason": "source_run_cleanup",
                        "source_run_id": int(run_id),
                        "source_member_id": source_member_id,
                        "source_asset_id": source_asset_id,
                        "binding_id": binding_id,
                        "storage_path": str(plan["destination"]),
                        "content_sha256": str(plan["materialized"]["content_sha256"]),
                    }
                    member_cur = conn.execute(
                        """
                        INSERT INTO media_item_members(
                            item_id, asset_id, member_role, producer_kind, producer_run_id,
                            method_key, reusable_as_pred, temporal_mapping_json,
                            spatial_origin_json, state, metadata_json,
                            created_at, updated_at, deleted_at
                        ) VALUES (?, ?, 'compare_snapshot', 'video_compare', ?, ?, 0, ?, ?, 'ready', ?, ?, ?, NULL)
                        """,
                        (
                            item_id,
                            snapshot_asset_id,
                            compare_run_id,
                            str(plan.get("source_method_key") or ""),
                            json.dumps(_json_object(plan.get("temporal_mapping_json")), sort_keys=True, ensure_ascii=False),
                            json.dumps(_json_object(plan.get("spatial_origin_json")), sort_keys=True, ensure_ascii=False),
                            json.dumps(member_metadata, sort_keys=True, ensure_ascii=False),
                            now,
                            now,
                        ),
                    )
                    snapshot_member_id = int(member_cur.lastrowid)
                    self._bind_compare_snapshot_asset(
                        conn,
                        compare_run_id=compare_run_id,
                        source_asset_id=source_asset_id,
                        snapshot_asset_id=snapshot_asset_id,
                        binding_id=binding_id,
                        source_member_id=source_member_id,
                        binding_metadata=plan["binding_metadata"],
                        member_metadata=plan["member_metadata"],
                        slot=slot,
                        now=now,
                    )
                    switched = conn.execute(
                        """
                        UPDATE run_media_item_bindings
                        SET active_member_id = ?, updated_at = ?
                        WHERE id = ? AND run_id = ? AND item_id = ?
                          AND binding_role = 'compare_pred' AND slot = ?
                          AND active_member_id = ?
                        """,
                        (
                            snapshot_member_id,
                            now,
                            binding_id,
                            compare_run_id,
                            item_id,
                            slot,
                            source_member_id,
                        ),
                    )
                    if int(switched.rowcount or 0) != 1:
                        raise RuntimeError("Compare input binding changed during snapshot")
                    snapshots.append(
                        {
                            "binding_id": binding_id,
                            "compare_run_id": compare_run_id,
                            "item_id": item_id,
                            "slot": slot,
                            "source_member_id": source_member_id,
                            "snapshot_member_id": snapshot_member_id,
                            "snapshot_asset_id": snapshot_asset_id,
                            "storage_path": str(plan["destination"]),
                            "size_bytes": int(plan["materialized"]["size_bytes"]),
                            "copy_method": str(plan["materialized"]["method"]),
                        }
                    )
        except Exception:
            for snapshot_dir, inputs_root in reversed(created_dirs):
                if snapshot_dir.parent != inputs_root or not _is_relative_to(snapshot_dir, inputs_root):
                    continue
                try:
                    if snapshot_dir.is_symlink() or snapshot_dir.is_file():
                        snapshot_dir.unlink()
                    elif snapshot_dir.exists():
                        shutil.rmtree(snapshot_dir)
                except OSError:
                    pass
            raise

        return {
            "protected": len(snapshots),
            "compare_run_ids": sorted({int(row["compare_run_id"]) for row in snapshots}),
            "snapshots": snapshots,
        }

    @staticmethod
    def _bind_compare_snapshot_asset(
        conn: Any,
        *,
        compare_run_id: int,
        source_asset_id: int,
        snapshot_asset_id: int,
        binding_id: int,
        source_member_id: int,
        binding_metadata: dict[str, Any],
        member_metadata: dict[str, Any],
        slot: str,
        now: float,
    ) -> None:
        video_name = str(binding_metadata.get("video_name") or member_metadata.get("video_name") or "")
        track_label = str(binding_metadata.get("track_label") or slot or "")
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT * FROM run_media_assets
                WHERE run_id = ? AND asset_id = ? AND role = 'pred'
                ORDER BY video_name, track_label
                """,
                (compare_run_id, source_asset_id),
            ).fetchall()
        ]
        if video_name:
            rows = [row for row in rows if str(row.get("video_name") or "") == video_name]
        if track_label:
            rows = [row for row in rows if str(row.get("track_label") or "") == track_label]
        source_row = rows[0] if len(rows) == 1 else None
        if source_row is not None:
            video_name = str(source_row.get("video_name") or "")
            track_label = str(source_row.get("track_label") or "")
            model_name = str(source_row.get("model_name") or "")
            checkpoint = str(source_row.get("checkpoint") or "")
            metadata = _json_object(source_row.get("metadata_json"))
        else:
            model_name = ""
            checkpoint = ""
            metadata = {}
        metadata.update(
            {
                "input": True,
                "compare_snapshot": True,
                "binding_id": int(binding_id),
                "source_member_id": int(source_member_id),
                "source_asset_id": int(source_asset_id),
            }
        )
        conn.execute(
            """
            INSERT INTO run_media_assets(
                run_id, asset_id, role, video_name, track_label,
                model_name, checkpoint, metadata_json, created_at
            ) VALUES (?, ?, 'pred', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, asset_id, role, video_name, track_label)
            DO UPDATE SET model_name = excluded.model_name,
                          checkpoint = excluded.checkpoint,
                          metadata_json = excluded.metadata_json
            """,
            (
                compare_run_id,
                snapshot_asset_id,
                video_name,
                track_label,
                model_name,
                checkpoint,
                json.dumps(metadata, sort_keys=True, ensure_ascii=False),
                now,
            ),
        )
        if source_row is not None:
            conn.execute(
                """
                DELETE FROM run_media_assets
                WHERE run_id = ? AND asset_id = ? AND role = 'pred'
                  AND video_name = ? AND track_label = ?
                """,
                (compare_run_id, source_asset_id, video_name, track_label),
            )

    def _purge_run(self, run_id: int, *, delete_run: bool) -> dict[str, Any]:
        run = self.db.get_run(int(run_id))
        if str(run.get("status") or "") not in TERMINAL_RUN_STATUSES:
            raise ValueError("run did not reach a terminal state before purge")
        active_jobs = self._active_jobs(int(run_id))
        if active_jobs:
            raise ValueError("run workers are still active")
        self.db.cancel_queued_run_jobs(int(run_id), "Run is being purged")

        # Freeze every published Campaign dependency before the first destructive
        # action. The Campaign subsystem raises if it cannot preserve a source.
        campaign_report = self._prepare_protected_campaign_media(int(run_id))
        # Compare normally references the original reusable Pred. Preserve each
        # active dependency inside the Compare Run before deleting that source.
        compare_snapshot_report = self._prepare_compare_input_snapshots(int(run_id))
        register_run_cache_refs(
            self.db,
            self.workspace,
            int(run_id),
            grace_seconds=self.cache_grace_seconds,
        )

        runs_root = self.workspace.runs_dir.resolve()
        run_dir = (runs_root / str(int(run_id))).resolve()
        if not _is_relative_to(run_dir, runs_root) or run_dir.parent != runs_root:
            raise ValueError("run output directory is outside workspace runs directory")
        reclaimed_bytes = _path_reclaimable_size(run_dir)
        if run_dir.exists():
            shutil.rmtree(run_dir)

        self.db.mark_run_artifacts_cleaned(int(run_id))
        released_entries = self.db.release_run_cache_refs(
            int(run_id),
            grace_seconds=self.cache_grace_seconds,
        )
        if delete_run:
            self.db.mark_run_deleted_after_purge(int(run_id))
        return {
            "run_id": int(run_id),
            "delete_run": bool(delete_run),
            "artifact_cleaned": True,
            "deleted": bool(delete_run),
            "output_dir": str(run_dir),
            "reclaimed_bytes": reclaimed_bytes,
            "cache_refs_released": released_entries,
            "cache_gc_grace_seconds": self.cache_grace_seconds,
            "campaign_media": campaign_report,
            "compare_input_snapshots": compare_snapshot_report,
        }

    def gc_preview(
        self,
        entry_ids: Iterable[int] | None = None,
        run_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        self.ensure_backfilled()
        selected = {int(value) for value in entry_ids} if entry_ids is not None else None
        selected_runs = {int(value) for value in run_ids} if run_ids is not None else None
        now = utc_ts()
        caches: list[dict[str, Any]] = []
        inventory = self.db.cache_gc_inventory()
        by_cache_key = {
            (str(entry.get("cache_type") or ""), str(entry.get("cache_key") or "")): entry
            for entry in inventory
        }
        for raw_entry in inventory:
            entry = dict(raw_entry)
            entry_id = int(entry["id"])
            if selected is not None and entry_id not in selected:
                continue
            refs = int(entry.get("active_run_refs") or 0)
            leases = int(entry.get("active_leases") or 0)
            base_entry: dict[str, Any] | None = None
            cache_type = str(entry.get("cache_type") or "")
            cache_key = str(entry.get("cache_key") or "")
            if cache_type == "decode_cache" and cache_key.endswith(".partial"):
                base_entry = by_cache_key.get((cache_type, cache_key[: -len(".partial")]))
                if base_entry is not None:
                    # The active decoder leases the final cache key while it
                    # writes the sibling ``.partial`` directory.  Surface that
                    # dependency in preview and block its deletion.
                    refs = max(refs, int(base_entry.get("active_run_refs") or 0))
                    leases = max(leases, int(base_entry.get("active_leases") or 0))
                    entry["partial_base_cache_entry_id"] = int(base_entry["id"])
                    entry["active_run_refs"] = refs
                    entry["active_leases"] = leases
            gc_after = entry.get("gc_after")
            if str(entry.get("state") or "") == "deleting" or (
                base_entry is not None and str(base_entry.get("state") or "") == "deleting"
            ):
                reason = "deletion_in_progress"
            elif refs:
                reason = "referenced_by_active_runs"
            elif leases:
                reason = "active_lease"
            elif gc_after is None or float(gc_after) > now:
                reason = "grace_period"
            else:
                reason = "eligible"
            campaign_ids = set(self._cache_campaign_ids(entry_id))
            if base_entry is not None:
                campaign_ids.update(self._cache_campaign_ids(int(base_entry["id"])))
            caches.append(
                {
                    **entry,
                    "eligible": reason == "eligible",
                    "reason": reason,
                    "campaign_ids": sorted(campaign_ids),
                }
            )
        runs = self._run_gc_inventory(selected_runs)
        eligible_caches = [row for row in caches if row["eligible"]]
        eligible_runs = [row for row in runs if row["eligible"]]
        blocked = [
            {"kind": "cache", "id": int(row["id"]), "reason": row["reason"]}
            for row in caches
            if not row["eligible"]
        ] + [
            {"kind": "run", "id": row.get("run_id"), "path": row["path"], "reason": row["reason"]}
            for row in runs
            if not row["eligible"]
        ]
        cache_bytes = sum(int(row.get("size_bytes") or 0) for row in caches)
        run_bytes = sum(int(row.get("size_bytes") or 0) for row in runs)
        preview_token = uuid.uuid4().hex
        with self._gc_preview_lock:
            cutoff = now - 30 * 60
            self._gc_preview_tokens = {
                token: payload
                for token, payload in self._gc_preview_tokens.items()
                if float(payload.get("created_at") or 0) >= cutoff
            }
            self._gc_preview_tokens[preview_token] = {
                "created_at": now,
                "entry_ids": {int(row["id"]) for row in eligible_caches},
                "run_ids": {
                    int(row["directory_id"])
                    for row in eligible_runs
                    if row.get("directory_id") is not None
                },
            }
        return {
            "generated_at": now,
            "preview_token": preview_token,
            "grace_seconds": self.cache_grace_seconds,
            "runs": runs,
            "caches": caches,
            "entries": caches,
            "blocked": blocked,
            "summary": {
                "run_entries": len(runs),
                "run_bytes": run_bytes,
                "eligible_run_entries": len(eligible_runs),
                "eligible_run_bytes": sum(int(row.get("size_bytes") or 0) for row in eligible_runs),
                "cache_entries": len(caches),
                "cache_bytes": cache_bytes,
                "eligible_cache_entries": len(eligible_caches),
                "eligible_cache_bytes": sum(int(row.get("size_bytes") or 0) for row in eligible_caches),
                "entries": len(runs) + len(caches),
                "bytes": run_bytes + cache_bytes,
                "eligible_entries": len(eligible_runs) + len(eligible_caches),
                "eligible_bytes": sum(int(row.get("size_bytes") or 0) for row in eligible_runs)
                + sum(int(row.get("size_bytes") or 0) for row in eligible_caches),
                "campaign_ids": sorted(
                    {cid for row in caches for cid in row["campaign_ids"]}
                    | {cid for row in runs for cid in row["campaign_ids"]}
                ),
            },
        }

    def garbage_collect(
        self,
        *,
        confirmed: bool,
        entry_ids: Iterable[int] | None = None,
        run_ids: Iterable[int] | None = None,
        preview_token: str | None = None,
        require_preview_token: bool = False,
    ) -> dict[str, Any]:
        if not confirmed:
            raise ValueError("storage GC requires an explicit preview and confirm=true")
        if require_preview_token:
            token = str(preview_token or "").strip()
            with self._gc_preview_lock:
                snapshot = self._gc_preview_tokens.pop(token, None)
            if snapshot is None or float(snapshot.get("created_at") or 0) < utc_ts() - 30 * 60:
                raise ValueError("storage GC preview token is missing or expired; preview again")
            requested_entries = (
                {int(value) for value in entry_ids} if entry_ids is not None else snapshot["entry_ids"]
            )
            requested_runs = (
                {int(value) for value in run_ids} if run_ids is not None else snapshot["run_ids"]
            )
            if not requested_entries.issubset(snapshot["entry_ids"]):
                raise ValueError("storage GC cache selection was not present in the confirmed preview")
            if not requested_runs.issubset(snapshot["run_ids"]):
                raise ValueError("storage GC Run selection was not present in the confirmed preview")
            entry_ids = sorted(requested_entries)
            run_ids = sorted(requested_runs)
        preview = self.gc_preview(entry_ids, run_ids)
        deleted_runs: list[dict[str, Any]] = []
        deleted_caches: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for row in preview["runs"]:
            if not row["eligible"]:
                continue
            try:
                path = Path(str(row["path"])).resolve()
                runs_root = self.workspace.runs_dir.resolve()
                if not _is_relative_to(path, runs_root) or path.parent != runs_root:
                    raise ValueError("historical run directory resolved outside the managed runs root")
                run_id = row.get("run_id")
                if run_id is None:
                    reclaimed_bytes = _path_size(path)
                    if path.exists():
                        shutil.rmtree(path)
                    report = {
                        "run_id": None,
                        "path": str(path),
                        "reclaimed_bytes": reclaimed_bytes,
                        "orphan": True,
                    }
                else:
                    run = self.db.get_run(int(run_id))
                    report = self._purge_run(
                        int(run_id),
                        delete_run=run.get("deleted_at") is not None,
                    )
                    report["path"] = str(path)
                deleted_runs.append(report)
            except Exception as exc:
                failed.append(
                    {
                        "kind": "run",
                        "run_id": row.get("run_id"),
                        "path": row["path"],
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        for entry in preview["caches"]:
            if not entry["eligible"]:
                continue
            entry_id = int(entry["id"])
            try:
                claimed = self.db.claim_cache_entry_for_gc(entry_id)
                if claimed is None:
                    skipped.append(
                        {
                            "kind": "cache",
                            "id": entry_id,
                            "reason": "became_referenced_or_leased_after_preview",
                        }
                    )
                    continue
                path = _trusted_cache_path(
                    self.workspace,
                    str(claimed["cache_type"]),
                    str(claimed["cache_key"]),
                    str(claimed["storage_path"]),
                )
                reclaimed_bytes = _path_size(path)
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                self.db.mark_cache_entry_state(
                    entry_id,
                    "deleted",
                    size_bytes=0,
                    metadata={
                        **(entry.get("metadata") or {}),
                        "gc_deleted_at": utc_ts(),
                        "gc_reclaimed_bytes": reclaimed_bytes,
                    },
                )
                deleted_caches.append(
                    {
                        "id": entry_id,
                        "cache_type": entry["cache_type"],
                        "cache_key": entry["cache_key"],
                        "reclaimed_bytes": reclaimed_bytes,
                    }
                )
            except Exception as exc:
                self.db.mark_cache_entry_state(
                    entry_id,
                    "failed",
                    metadata={
                        **(entry.get("metadata") or {}),
                        "gc_error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                )
                failed.append(
                    {
                        "kind": "cache",
                        "id": entry_id,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        return {
            "deleted": deleted_caches,
            "deleted_caches": deleted_caches,
            "deleted_runs": deleted_runs,
            "failed": failed,
            "skipped": skipped,
            "reclaimed_bytes": sum(int(row["reclaimed_bytes"]) for row in deleted_caches)
            + sum(int(row.get("reclaimed_bytes") or 0) for row in deleted_runs),
            "preview": preview["summary"],
        }

    def _run_gc_inventory(self, selected_run_ids: set[int] | None) -> list[dict[str, Any]]:
        runs_root = self.workspace.runs_dir.resolve()
        if not runs_root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for child in sorted(runs_root.iterdir(), key=lambda value: value.name):
            if not child.is_dir() or not child.name.isdigit():
                continue
            run_id = int(child.name)
            if selected_run_ids is not None and run_id not in selected_run_ids:
                continue
            raw_run = self.db.get("SELECT * FROM runs WHERE id = ?", (run_id,))
            if raw_run is not None and raw_run.get("deleted_at") is None and raw_run.get("artifact_cleaned_at") is None:
                continue
            reason = "eligible"
            if raw_run is not None:
                status = str(raw_run.get("status") or "")
                if status not in TERMINAL_RUN_STATUSES:
                    reason = "run_not_terminal"
                elif self._active_jobs(run_id):
                    reason = "active_worker"
                elif self._preparing_campaign_ids(run_id):
                    reason = "campaign_preparing"
            rows.append(
                {
                    "run_id": run_id if raw_run is not None else None,
                    "directory_id": run_id,
                    "path": str(child.resolve()),
                    "size_bytes": _path_size(child),
                    "state": "orphan" if raw_run is None else (
                        "deleted" if raw_run.get("deleted_at") is not None else "artifact_cleaned"
                    ),
                    "status": raw_run.get("status") if raw_run is not None else None,
                    "campaign_ids": self._run_campaign_ids(run_id) if raw_run is not None else [],
                    "eligible": reason == "eligible",
                    "reason": reason,
                }
            )
        return rows

    def _run_campaign_ids(self, run_id: int) -> list[int]:
        rows = self.db.query(
            """
            SELECT DISTINCT ec.campaign_id
            FROM run_media_assets rma
            JOIN evaluation_candidates ec
              ON ec.asset_id = rma.asset_id OR ec.reference_asset_id = rma.asset_id
            WHERE rma.run_id = ?
            ORDER BY ec.campaign_id
            """,
            (int(run_id),),
        )
        campaign_ids = {int(row["campaign_id"]) for row in rows}
        if self.db.get(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evaluation_methods_v2'"
        ):
            campaign_ids.update(
                int(row["campaign_id"])
                for row in self.db.query(
                    "SELECT DISTINCT campaign_id FROM evaluation_methods_v2 WHERE source_run_id = ?",
                    (int(run_id),),
                )
            )
        return sorted(campaign_ids)

    def _preparing_campaign_ids(self, run_id: int) -> list[int]:
        if not self.db.get(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evaluation_methods_v2'"
        ):
            return []
        rows = self.db.query(
            """
            SELECT DISTINCT m.campaign_id
            FROM evaluation_methods_v2 m
            JOIN evaluation_campaigns_v2 c ON c.id = m.campaign_id
            WHERE m.source_run_id = ? AND c.status = 'preparing'
            ORDER BY m.campaign_id
            """,
            (int(run_id),),
        )
        return [int(row["campaign_id"]) for row in rows]

    def _cache_campaign_ids(self, cache_entry_id: int) -> list[int]:
        rows = self.db.query(
            """
            SELECT DISTINCT ec.campaign_id
            FROM run_cache_refs rcr
            JOIN run_media_assets rma ON rma.run_id = rcr.run_id
            JOIN evaluation_candidates ec
              ON ec.asset_id = rma.asset_id OR ec.reference_asset_id = rma.asset_id
            WHERE rcr.cache_entry_id = ?
            ORDER BY ec.campaign_id
            """,
            (int(cache_entry_id),),
        )
        campaign_ids = {int(row["campaign_id"]) for row in rows}
        if self.db.get(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evaluation_methods_v2'"
        ):
            campaign_ids.update(
                int(row["campaign_id"])
                for row in self.db.query(
                    """
                    SELECT DISTINCT m.campaign_id
                    FROM run_cache_refs rcr
                    JOIN evaluation_methods_v2 m ON m.source_run_id = rcr.run_id
                    WHERE rcr.cache_entry_id = ?
                    """,
                    (int(cache_entry_id),),
                )
            )
        return sorted(campaign_ids)
