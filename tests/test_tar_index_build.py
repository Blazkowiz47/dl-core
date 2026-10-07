"""Verify ordered index builds, process coordination, and signal cleanup."""

from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import os
import signal
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

from dl_core.datasets import IndexedTarDataset, TarShardWrapper
from dl_core.datasets import _tar_index

_real_index_init = _tar_index._init_index_worker


class _Wrapper(TarShardWrapper):
    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        return file_dict


def _write_tar(path: Path, count: int = 3) -> None:
    with tarfile.open(path, "w") as archive:
        for number in range(count):
            for extension in ("png", "json"):
                payload = f"{path.stem}:{number}:{extension}".encode()
                member = tarfile.TarInfo(f"sample-{number}.{extension}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def _coordinated_builder(
    paths: list[Path], index_root: Path, barrier: Any, state: Any, results: Any
) -> None:
    os.environ["LOCAL_WORLD_SIZE"] = "4"
    original = _tar_index._scan_index

    def scan(path: Path, fingerprint: list[int], stopped: Any) -> list[Any]:
        with state.get_lock():
            state[0] += 1
            state[1] = max(state[1], state[0])
            state[2] += 1
        try:
            time.sleep(0.2)
            return original(path, fingerprint, stopped)
        finally:
            with state.get_lock():
                state[0] -= 1

    _tar_index._scan_index = scan
    barrier.wait(timeout=45)
    dataset = IndexedTarDataset(paths, index_dir=index_root, index_workers=2)
    results.put(dataset.index_stats)
    dataset.close()


def _slow_scan(path: Path, fingerprint: list[int], stopped: Any) -> list[Any]:
    path.with_suffix(".started").touch()
    stopped.wait(45)
    raise _tar_index.CancelledError("Stopped synthetic scan")


def _slow_index_initializer(stopped: Any) -> None:
    _real_index_init(stopped)
    _tar_index._scan_index = _slow_scan


def _interrupted_builder(paths: list[Path], index_root: Path, results: Any) -> None:
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    _tar_index._init_index_worker = _slow_index_initializer
    try:
        IndexedTarDataset(paths, index_dir=index_root, index_workers=2)
    except KeyboardInterrupt:
        children = len(mp.active_children())
        _tar_index._init_index_worker = _real_index_init
        # A new constructor can acquire every lock released during cancellation.
        recovered = IndexedTarDataset(paths, index_dir=index_root, index_workers=1)
        results.put({"children": children, "recovered": len(recovered)})
        recovered.close()


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2", None])
def test_index_workers_must_be_positive_integers(workers: Any) -> None:
    with pytest.raises(ValueError, match="index_workers"):
        IndexedTarDataset([], index_workers=workers)


def test_parallel_indexes_match_serial_order_and_reuse(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    paths = [tmp_path / f"shard-{number}.tar" for number in range(5)]
    records = []
    for number, path in enumerate(paths):
        _write_tar(path, number + 2)
        records.append(
            {
                "path": str(path),
                "source_name": "a" if number % 2 else "b",
                "source_weight": 2 if number % 2 else 1,
                "tag": number,
            }
        )
    records[0]["sample_keys"] = ["sample-1"]
    records.append({"path": "absent.tar", "source_weight": 0})
    serial = IndexedTarDataset(records, index_dir=tmp_path / "serial", index_workers=1)
    parallel = IndexedTarDataset(
        records, index_dir=tmp_path / "parallel", index_workers=2
    )
    assert parallel._samples == serial._samples
    assert parallel.shards == serial.shards
    assert parallel.source_indices == serial.source_indices
    assert parallel.shard_totals == serial.shard_totals
    assert [parallel[i] for i in range(len(parallel))] == [
        serial[i] for i in range(len(serial))
    ]
    assert parallel.index_stats["built"] == 5
    assert parallel.index_stats["cached"] == 0
    assert not list((tmp_path / "parallel").glob("*.tmp"))
    for dataset in (serial, parallel):
        dataset.close()

    monkeypatch.setattr(
        _tar_index,
        "ProcessPoolExecutor",
        lambda **kwargs: pytest.fail("Warm cache started a pool"),
    )
    monkeypatch.setattr(
        tarfile,
        "open",
        lambda *args, **kwargs: pytest.fail("Warm index rescanned a tar"),
    )
    # Eligibility changes do not invalidate the generic member-offset cache.
    records[0]["sample_keys"] = ["sample-0"]
    warm = IndexedTarDataset(records, index_dir=tmp_path / "parallel", index_workers=2)
    assert warm[0]["key"] == "sample-0"
    assert warm.index_stats["cached"] == 5
    assert warm.index_stats["built"] == 0
    warm.close()


def test_invalid_index_schema_rebuilds_and_empty_selection_logs(
    tmp_path: Path, caplog: Any
) -> None:
    path = tmp_path / "shard.tar"
    _write_tar(path)
    root = tmp_path / "indexes"
    IndexedTarDataset([path], index_dir=root).close()
    index_path = next(root.glob("*.json"))
    saved = json.loads(index_path.read_text())
    saved["samples"] = [["sample", {"png": [0, path.stat().st_size + 1]}]]
    index_path.write_text(json.dumps(saved))
    rebuilt = IndexedTarDataset([path], index_dir=root)
    assert len(rebuilt) == 3
    assert rebuilt.index_stats["built"] == 1
    rebuilt.close()
    with caplog.at_level(logging.INFO):
        empty = IndexedTarDataset([], index_label="validation")
    assert empty.index_stats["workers"] == 0
    assert "complete 0/0 (0 cached, 0 built)" in caplog.text


def test_wrapper_options_and_progress_while_waiting(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    path = tmp_path / "shard.tar"
    _write_tar(path)
    original = _tar_index._scan_index

    def slow_scan(path: Path, fingerprint: list[int], stopped: Any) -> list[Any]:
        time.sleep(0.1)
        return original(path, fingerprint, stopped)

    monkeypatch.setattr(_tar_index, "_scan_index", slow_scan)
    monkeypatch.setattr(_tar_index, "_PROGRESS_INTERVAL", 0)
    # Also report before the scan completes, rather than only on completion.
    original_wait = _tar_index.wait
    monkeypatch.setattr(
        _tar_index,
        "wait",
        lambda fs, **kwargs: original_wait(
            fs, timeout=0.01, return_when=kwargs["return_when"]
        ),
    )
    wrapper = _Wrapper(
        {
            "auto_split": False,
            "indexed_tar": {"index_dir": str(tmp_path / "indexes"), "index_workers": 1},
        }
    )
    with caplog.at_level(logging.INFO):
        dataset = wrapper.build_indexed_dataset(
            [{"name": "val", "shards": [str(path)]}], "validation"
        )
    assert dataset.index_stats["workers"] == 1
    assert "Tar indexes [validation]: starting 1 shards" in caplog.text
    assert "0/1 ready (0 cached, 0 built)" in caplog.text
    assert "complete 1/1 (0 cached, 1 built)" in caplog.text
    dataset.close()


@pytest.mark.parametrize("shared_shard", [True, False])
def test_processes_share_build_slots_and_do_not_duplicate_indexes(
    tmp_path: Path, shared_shard: bool
) -> None:
    context = mp.get_context("spawn")
    paths = [tmp_path / f"shard-{i}.tar" for i in range(4)]
    for path in paths:
        _write_tar(path)
    state = context.Array("i", [0, 0, 0])
    barrier = context.Barrier(4)
    results = context.Queue()
    processes = [
        context.Process(
            target=_coordinated_builder,
            args=(
                [paths[0] if shared_shard else path],
                tmp_path / "indexes",
                barrier,
                state,
                results,
            ),
        )
        for path in paths
    ]
    try:
        for process in processes:
            process.start()
        stats = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert state[0] == 0
        assert state[1] == (1 if shared_shard else 2)
        assert state[2] == (1 if shared_shard else 4)
        assert sum(item["built"] for item in stats) == (1 if shared_shard else 4)
        assert sum(item["cached"] for item in stats) == (3 if shared_shard else 0)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        results.close()
        results.join_thread()


def test_parallel_scan_error_reaps_workers_and_leaves_no_partial_indexes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    valid = tmp_path / "valid.tar"
    invalid = tmp_path / "invalid.tar"
    _write_tar(valid)
    invalid.write_bytes(b"not a tar")
    before = {child.pid for child in mp.active_children()}
    root = tmp_path / "indexes"
    with pytest.raises(tarfile.ReadError):
        IndexedTarDataset([invalid, valid], index_dir=root, index_workers=2)
    assert {child.pid for child in mp.active_children()} == before
    assert not list(root.glob("*.tmp"))
    # Completed indexes are reusable, and every build slot is released.
    dataset = IndexedTarDataset([valid], index_dir=root, index_workers=1)
    assert len(dataset) == 3
    dataset.close()


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX terminal SIGINT delivery")
def test_sigint_cancels_scans_reaps_processes_and_releases_locks(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"shard-{i}.tar" for i in range(2)]
    for path in paths:
        _write_tar(path)
    context = mp.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_interrupted_builder, args=(paths, tmp_path / "indexes", results)
    )
    process.start()
    try:
        deadline = time.monotonic() + 45
        while not all(path.with_suffix(".started").exists() for path in paths):
            assert process.is_alive()
            assert time.monotonic() < deadline, "Index workers never started"
            time.sleep(0.05)
        os.kill(process.pid, signal.SIGINT)
        outcome = results.get(timeout=20)
        process.join(timeout=10)
        assert process.exitcode == 0
        assert outcome == {"children": 0, "recovered": 6}
        assert not list((tmp_path / "indexes").glob("*.tmp"))
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        results.close()
        results.join_thread()
