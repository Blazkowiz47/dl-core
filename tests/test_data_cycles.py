"""Cycle refresh, tar selection, resource lifetime, and resume coverage."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from weakref import ref

import pytest
import torch

from dl_core.accelerators.cpu import CPUAccelerator
from dl_core.callbacks.dataset_refresh import DatasetRefreshCallback
from dl_core.core.base_callback import Callback, CallbackList
from dl_core.datasets import TarShardWrapper

from test_iteration_trainer import _build_trainer


class _SelectingTarWrapper(TarShardWrapper):
    """A concrete selector chooses a fixed or rotating subset of its own pool."""

    def build_shard_sources(self, split: str) -> list[dict[str, Any]]:
        selections = self.config["selections"]
        return [
            {
                "name": split,
                "shards": selections[self.current_epoch % len(selections)],
            }
        ]

    def build_dataset(self, data: list[dict], split: str) -> Any:
        if self.config["use_index"]:
            return self.build_indexed_dataset(data, split)
        return super().build_dataset(data, split)

    def transform(self, sample: dict[str, Any], split: str) -> dict[str, Any]:
        return {"image": torch.ones(1), "key": sample["key"]}


class _RestoringPoolWrapper(_SelectingTarWrapper):
    def get_data_cycle_state(self) -> dict[str, Any]:
        return {**super().get_data_cycle_state(), "pool": self.config["selections"]}

    def restore_data_cycle_state(self, state: dict[str, Any]) -> None:
        self.config["selections"] = state["pool"]


class _CycleRecorder(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.starts: list[tuple[int, dict[str, Any]]] = []
        self.ends: list[tuple[int, dict[str, Any]]] = []

    def on_data_cycle_start(self, cycle: int, logs: dict[str, Any]) -> None:
        assert self.trainer._data_iterator is None
        assert self.trainer.dataset_wrapper.current_epoch == cycle
        self.starts.append((cycle, logs))

    def on_data_cycle_end(self, cycle: int, logs: dict[str, Any]) -> None:
        self.ends.append((cycle, logs))


@pytest.fixture
def shard_paths(tmp_path: Path) -> list[str]:
    paths = []
    for shard in range(4):
        path = tmp_path / f"shard-{shard}.tar"
        with tarfile.open(path, "w") as archive:
            for sample in range(2):
                for extension, payload in {
                    "png": b"image" * (shard + sample + 1),
                    "json": ('{"id":' + str(shard * 10 + sample) + "}").encode(),
                }.items():
                    member = tarfile.TarInfo(f"{shard}-{sample}.{extension}")
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
        paths.append(str(path))
    return paths


def _tar_trainer(
    selections: list[list[str]],
    *,
    indexed: bool,
    frequency: int = 1,
    wrapper_type: type[_SelectingTarWrapper] = _SelectingTarWrapper,
) -> tuple[Any, _CycleRecorder, list[str]]:
    trainer = _build_trainer(9)
    trainer.accelerator = CPUAccelerator({})
    trainer.dataset_wrapper = wrapper_type(
        {
            "selections": selections,
            "use_index": indexed,
            "indexed_tar": {"index_dir": None},
            "required_extensions": ["png", "json"],
            "auto_split": False,
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
        }
    )
    recorder = _CycleRecorder()
    trainer.callbacks = CallbackList(
        [
            DatasetRefreshCallback(trigger="data_cycle", refresh_frequency=frequency),
            recorder,
        ]
    )
    trainer.callbacks.set_trainer(trainer)
    seen: list[str] = []

    def train_step(batch: dict[str, Any], index: int) -> dict[str, float]:
        seen.extend(batch["key"])
        return {"loss": float(index)}

    trainer.train_step = train_step
    trainer._save_checkpoint = lambda iteration, filename=None: None
    return trainer, recorder, seen


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_cycles_mix_a_fixed_selection_then_refresh_it(
    shard_paths: list[str], indexed: bool, frozen: bool
) -> None:
    selections = [shard_paths[:2]] if frozen else [shard_paths[:2], shard_paths[2:]]
    trainer, recorder, seen = _tar_trainer(selections, indexed=indexed)

    trainer.perform_training()

    first = ["0-0", "0-1", "1-0", "1-1"]
    second = first if frozen else ["2-0", "2-1", "3-0", "3-1"]
    assert seen == first + second + first[:1]
    assert [cycle for cycle, _ in recorder.starts] == [0, 1, 2]
    assert [logs["completed"] for _, logs in recorder.ends] == [True, True, False]
    assert [logs["iteration"] for _, logs in recorder.ends] == [4, 8, 9]
    assert trainer._data_iterator is None


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("frequency", [1, 2])
def test_resume_rebuilds_the_active_selection_once(
    shard_paths: list[str], indexed: bool, frequency: int
) -> None:
    selections = [shard_paths[:2], shard_paths[2:]]
    original, _, expected = _tar_trainer(
        selections, indexed=indexed, frequency=frequency
    )
    checkpoints: dict[int, dict[str, Any]] = {}
    original.checkpoint_frequency = 7
    original._save_checkpoint = lambda iteration, filename=None: checkpoints.update(
        {iteration: original._build_checkpoint_payload(iteration)}
    )
    original.perform_training()

    resumed, recorder, actual = _tar_trainer(
        selections, indexed=indexed, frequency=frequency
    )
    resumed.restore_progress_state(checkpoints[7])
    assert recorder.starts == []
    resumed.perform_training()

    assert actual == expected[7:]
    assert [cycle for cycle, _ in recorder.starts] == [1, 2]
    assert recorder.starts[0][1]["resuming"] is True
    assert recorder.starts[0][1]["iteration_in_cycle"] == 3


def test_stateful_wrapper_restores_its_own_bounded_pool(shard_paths: list[str]) -> None:
    original, _, _ = _tar_trainer(
        [shard_paths[:2]],
        indexed=True,
        wrapper_type=_RestoringPoolWrapper,
    )
    checkpoint = {}
    original.checkpoint_frequency = 3
    original._save_checkpoint = lambda iteration, filename=None: checkpoint.update(
        original._build_checkpoint_payload(iteration) if iteration == 3 else {}
    )
    original.perform_training()

    resumed, _, seen = _tar_trainer(
        [shard_paths[2:]],
        indexed=True,
        wrapper_type=_RestoringPoolWrapper,
    )
    resumed.restore_progress_state(checkpoint)
    resumed.perform_training()

    assert resumed.dataset_wrapper.config["selections"] == [shard_paths[:2]]
    assert set(seen) == {"0-0", "0-1", "1-0", "1-1"}


def test_resume_rejects_a_changed_selection(shard_paths: list[str]) -> None:
    trainer, _, _ = _tar_trainer([shard_paths[:2]], indexed=True)
    trainer._start_data_cycle(0, initial=True)
    checkpoint = trainer._build_checkpoint_payload(0)
    trainer._close_data_iterator()
    resumed, _, seen = _tar_trainer([shard_paths[2:]], indexed=True)
    resumed.restore_progress_state(checkpoint)

    with pytest.raises(RuntimeError, match="selection changed"):
        resumed.perform_training()
    assert seen == []
    assert resumed._data_iterator is None


def test_tar_cycle_state_uses_unsigned_logical_shard_paths() -> None:
    wrapper = _SelectingTarWrapper({"selections": [[]], "auto_split": False})
    wrapper.files_list["train"] = [
        {
            "name": "source",
            "shards": [
                {
                    "path": "https://example.test/shard.tar?sig=secret",
                    "etag": "v1",
                    "shard_id": "https://example.test/shard.tar?sig=secret",
                },
                {"path": "/cache/random-name.tar", "source_path": "nested/shard.tar"},
            ],
        }
    ]

    state = wrapper.get_data_cycle_state()

    assert state["train_sources"][0]["shards"] == [
        {
            "path": "https://example.test/shard.tar",
            "etag": "v1",
            "shard_id": "https://example.test/shard.tar",
        },
        {"path": "nested/shard.tar"},
    ]


def test_refresh_retires_workers_and_sampler_before_loading_new_files() -> None:
    events = []

    class Lease:
        pass

    lease = Lease()
    weak_lease = ref(lease)
    iterator = SimpleNamespace(
        _shutdown_workers=lambda: events.append("workers stopped")
    )
    loader = SimpleNamespace(dataset=lease, _iterator=iterator)
    accelerator = CPUAccelerator({})
    accelerator.samplers = {"train": SimpleNamespace(dataset=lease)}

    def get_split(split: str) -> list[Any]:
        assert weak_lease() is None
        events.append("new selection")
        return []

    trainer = SimpleNamespace(
        accelerator=accelerator,
        data_loader={"train": loader, "validation": "keep"},
        dataset_wrapper=SimpleNamespace(
            refresh_dataset=lambda split: None,
            get_split=get_split,
        ),
    )
    del lease, loader, iterator
    callback = DatasetRefreshCallback()
    callback.set_trainer(trainer)

    callback.on_epoch_start(1)

    assert events == ["workers stopped", "new selection"]
    assert trainer.data_loader["validation"] == "keep"


def test_cycle_trigger_does_not_also_refresh_at_epoch_start() -> None:
    callback = DatasetRefreshCallback(trigger="data_cycle")
    callback.on_epoch_start(1)
    with pytest.raises(ValueError, match="trigger"):
        DatasetRefreshCallback(trigger="batch")


@pytest.mark.parametrize("hook", ["on_data_cycle_start", "on_data_cycle_end"])
def test_cycle_callback_failures_stop_instead_of_disabling_refresh(hook: str) -> None:
    callback = Callback()

    def fail(*args: Any) -> None:
        raise ValueError("selection failed")

    setattr(callback, hook, fail)
    callbacks = CallbackList([callback])
    callbacks.set_trainer(SimpleNamespace(accelerator=CPUAccelerator({})))

    with pytest.raises(RuntimeError, match="selection failed"):
        getattr(callbacks, hook)(0)
    assert callback.enabled


def test_cycle_resume_rejects_a_different_world_size() -> None:
    trainer = _build_trainer(5)
    trainer.restore_progress_state({"data_cycle_world_size": 2})
    with pytest.raises(RuntimeError, match="same world size"):
        trainer.perform_training()


def test_filtered_batches_do_not_advance_the_cycle_cursor() -> None:
    trainer = _build_trainer(3)
    trainer.data_loader["train"] = [None, {}, {"image": torch.ones(1, 1)}, {}]
    trainer._save_checkpoint = lambda iteration, filename=None: None

    trainer.perform_training()

    assert trainer.current_iteration == 3
    assert trainer.data_cycle == 2
    assert trainer.iteration_in_cycle == 1
    trainer._close_data_iterator()  # Cleanup is safe after the final shutdown.


def test_completely_filtered_selection_fails_without_advancing() -> None:
    trainer = _build_trainer(3)
    trainer.data_loader["train"] = [None, {}]
    with pytest.raises(RuntimeError, match="empty train"):
        trainer.perform_training()
    assert trainer.current_iteration == 0
    assert trainer.data_cycle == 0


def test_resume_rejects_a_cursor_past_the_shared_cycle() -> None:
    trainer = _build_trainer(5)
    trainer.restore_progress_state({"iteration_in_cycle": 3})
    with pytest.raises(RuntimeError, match="past the shared cycle"):
        trainer.perform_training()
    assert trainer._data_iterator is None


@pytest.mark.parametrize("indexed", [False, True])
def test_cycle_refresh_joins_persistent_tar_workers(
    shard_paths: list[str], indexed: bool
) -> None:
    trainer, recorder, seen = _tar_trainer(
        [shard_paths[:2], shard_paths[2:]],
        indexed=indexed,
    )
    wrapper = trainer.dataset_wrapper
    wrapper.num_workers["train"] = 2
    wrapper.persistent_workers["train"] = True
    wrapper.prefetch_factor["train"] = 2
    workers = {}
    record_start = recorder.on_data_cycle_start

    def check_retired_workers(cycle: int, logs: dict[str, Any]) -> None:
        assert all(not worker.is_alive() for worker in workers.values())
        record_start(cycle, logs)

    def record_workers(index: int, split: str, batch: dict[str, Any]) -> None:
        for worker in trainer._data_iterator._workers:
            workers[worker.pid] = worker

    recorder.on_data_cycle_start = check_retired_workers
    recorder.on_batch_end = record_workers
    trainer.perform_training()

    assert set(seen[:4]) == {"0-0", "0-1", "1-0", "1-1"}
    assert set(seen[4:8]) == {"2-0", "2-1", "3-0", "3-1"}
    assert len(workers) == 6
    assert all(not worker.is_alive() for worker in workers.values())
    assert trainer.train_loader._iterator is None


def test_tar_refresh_only_invalidates_the_requested_split() -> None:
    wrapper = _SelectingTarWrapper({"selections": [[]], "auto_split": False})
    wrapper.files_list = {
        split: [{"name": split}] for split in ("train", "validation", "test")
    }
    wrapper.sampled_files_list = dict(wrapper.files_list)
    wrapper.reset_shard_progress({"train-shard": 2})
    wrapper.reset_shard_progress({"validation-shard": 1}, split="validation")

    wrapper.refresh_dataset("train")

    assert wrapper.files_list["train"] == []
    assert wrapper.sampled_files_list["train"] == []
    assert wrapper.files_list["validation"] == [{"name": "validation"}]
    assert (
        wrapper.get_shard_progress("validation-shard", split="validation")["total"] == 1
    )
    assert "train" not in wrapper._shard_progress
    wrapper.refresh_dataset()
    assert all(not sources for sources in wrapper.files_list.values())
    assert wrapper._shard_progress == {}
