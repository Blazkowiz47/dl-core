"""Build reusable plain-tar offsets with bounded, coordinated workers."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import signal
import tarfile
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_PROGRESS_INTERVAL = 5.0
_WORKER_STOP: Any = None


@dataclass
class TarIndex:
    """Offsets and file identity returned by one indexing task."""

    fingerprint: list[int]
    samples: list[Any]
    cached: bool


def _fingerprint(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def _cached_index(path: Path, index_root: Path | None) -> TarIndex | None:
    if index_root is None:
        return None
    index_path = index_root / f"{hashlib.sha256(str(path).encode()).hexdigest()}.json"
    fingerprint = _fingerprint(path)
    try:
        saved = json.loads(index_path.read_text(encoding="utf-8"))
        if saved["version"] != 1 or saved["fingerprint"] != fingerprint:
            return None
        samples = saved["samples"]
        if not isinstance(samples, list):
            return None
        for key, members in samples:
            if not isinstance(key, str) or not isinstance(members, dict):
                return None
            for extension, offsets in members.items():
                if (
                    not isinstance(extension, str)
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(value) is not int or value < 0 for value in offsets)
                    or sum(offsets) > fingerprint[2]
                ):
                    return None
        if _fingerprint(path) != fingerprint:
            return None
        return TarIndex(fingerprint, samples, True)
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextmanager
def _locked(paths: list[Path], stopped: Any) -> Iterator[None]:
    """Acquire one available OS lock; never unlink a shared lock file."""
    handle: BinaryIO | None = None
    while handle is None:
        if stopped.is_set():
            raise CancelledError("Tar indexing stopped")
        for path in paths:
            candidate = path.open("a+b")
            try:
                if os.name == "nt":
                    import msvcrt

                    if os.fstat(candidate.fileno()).st_size == 0:
                        candidate.write(b"\0")
                        candidate.flush()
                    candidate.seek(0)
                    msvcrt.locking(candidate.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                candidate.close()
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
            else:
                handle = candidate
                break
        if handle is None:
            stopped.wait(0.05)
    try:
        yield
    finally:
        # Closing releases POSIX flock and Windows byte-range locks, including
        # on exceptions. The OS also releases them if the process dies.
        handle.close()


def _scan_index(path: Path, fingerprint: list[int], stopped: Any) -> list[Any]:
    """Scan headers only; payload offsets refer to an uncompressed file."""
    grouped: dict[str, dict[str, list[int]]] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            # Offsets are retained below; TarInfo objects need not accumulate.
            archive.members.clear()
            if stopped.is_set():
                raise CancelledError("Tar indexing stopped")
            if not member.isfile():
                continue
            if member.sparse is not None:
                raise ValueError("Indexed tar does not support sparse members")
            match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", member.name)
            if match is None:
                continue
            key, extension = match.group(1), match.group(2).lower()
            members = grouped.setdefault(key, {})
            if extension in members:
                raise ValueError(
                    f"Duplicate tar member extension {extension!r} for sample {key!r}"
                )
            members[extension] = [member.offset_data, member.size]
    if _fingerprint(path) != fingerprint:
        raise RuntimeError(f"Tar changed while building its index: {path}")
    return list(grouped.items())


def _load_or_build_index(
    path: Path, index_root: Path | None, lock_root: Path, slots: int, stopped: Any
) -> TarIndex:
    cached = _cached_index(path, index_root)
    if cached is not None:
        return cached
    digest = hashlib.sha256(str(path).encode()).hexdigest()
    with _locked([lock_root / f"shard-{digest}.lock"], stopped):
        # A different rank may have published while we waited for this shard.
        cached = _cached_index(path, index_root)
        if cached is not None:
            return cached
        with _locked(
            [lock_root / f"slot-{slot}.lock" for slot in range(slots)], stopped
        ):
            fingerprint = _fingerprint(path)
            samples = _scan_index(path, fingerprint, stopped)
            if stopped.is_set():
                raise CancelledError("Tar indexing stopped")
            if index_root is not None:
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=index_root,
                        suffix=".tmp",
                        delete=False,
                    ) as handle:
                        temporary = Path(handle.name)
                        json.dump(
                            {
                                "version": 1,
                                "fingerprint": fingerprint,
                                "samples": samples,
                            },
                            handle,
                            separators=(",", ":"),
                        )
                    os.replace(temporary, index_root / f"{digest}.json")
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            return TarIndex(fingerprint, samples, False)


def _init_index_worker(stopped: Any) -> None:
    global _WORKER_STOP
    _WORKER_STOP = stopped
    # The parent handles Ctrl-C and requests cooperative cleanup of all tasks.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _index_worker(
    path: Path, index_root: Path | None, lock_root: Path, slots: int
) -> TarIndex:
    return _load_or_build_index(path, index_root, lock_root, slots, _WORKER_STOP)


def iter_tar_indexes(
    paths: list[Path],
    index_root: Path | None,
    index_workers: int,
    logger: logging.Logger,
    label: str,
    stats: dict[str, Any],
) -> Iterator[TarIndex]:
    """Yield indexes in input order while reporting completion out of order."""
    ranks = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    workers = min(max(1, index_workers // ranks), len(paths), os.cpu_count() or 1)
    started = time.monotonic()
    reported = started
    stats.update(shards=len(paths), cached=0, built=0, workers=workers, seconds=0.0)
    logger.info(
        "Tar indexes [%s]: starting %d shards with %d workers (shared build limit %d)",
        label,
        len(paths),
        workers,
        index_workers,
    )
    lock_root = (
        index_root / ".locks"
        if index_root is not None
        else Path(tempfile.gettempdir())
        / (
            "dl-core-tar-index-locks-"
            + hashlib.sha256(str(Path.home()).encode()).hexdigest()[:16]
        )
    )
    lock_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    stopped = context.Event() if workers > 1 else threading.Event()
    executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
    queue: deque[TarIndex | Future[TarIndex]] = deque()
    pending: set[Future[TarIndex]] = set()
    remaining = iter(paths)
    exhausted = False
    try:
        while queue or not exhausted:
            # Bound queued tasks and completed results as well as active scans.
            while not exhausted and len(queue) < max(1, 2 * workers):
                path = next(remaining, None)
                if path is None:
                    exhausted = True
                    break
                cached = _cached_index(path, index_root)
                if cached is not None:
                    stats["cached"] += 1
                    queue.append(cached)
                    continue
                if executor is None:
                    if workers > 1:
                        executor = ProcessPoolExecutor(
                            max_workers=workers,
                            mp_context=context,
                            initializer=_init_index_worker,
                            initargs=(stopped,),
                        )
                    else:
                        # Keep the parent responsive to signals and progress
                        # while retaining sequential scans and in-process hooks.
                        executor = ThreadPoolExecutor(max_workers=1)
                future = (
                    executor.submit(
                        _index_worker, path, index_root, lock_root, index_workers
                    )
                    if workers > 1
                    else executor.submit(
                        _load_or_build_index,
                        path,
                        index_root,
                        lock_root,
                        index_workers,
                        stopped,
                    )
                )
                pending.add(future)
                queue.append(future)
            done, _ = wait(
                pending,
                timeout=0
                if queue and (isinstance(queue[0], TarIndex) or queue[0].done())
                else 1.0,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                result = future.result()
                stats["cached" if result.cached else "built"] += 1
                pending.remove(future)
            now = time.monotonic()
            if now - reported >= _PROGRESS_INTERVAL:
                logger.info(
                    "Tar indexes [%s]: %d/%d ready (%d cached, %d built), %.1fs elapsed",
                    label,
                    stats["cached"] + stats["built"],
                    len(paths),
                    stats["cached"],
                    stats["built"],
                    now - started,
                )
                reported = now
            while queue and (isinstance(queue[0], TarIndex) or queue[0] not in pending):
                item = queue.popleft()
                yield item if isinstance(item, TarIndex) else item.result()
    except BaseException:
        stopped.set()
        for future in pending:
            future.cancel()
        logger.warning("Tar indexes [%s]: stopped before setup completed", label)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        stats["seconds"] = time.monotonic() - started
    logger.info(
        "Tar indexes [%s]: complete %d/%d (%d cached, %d built), %.1fs elapsed",
        label,
        len(paths),
        len(paths),
        stats["cached"],
        stats["built"],
        stats["seconds"],
    )
