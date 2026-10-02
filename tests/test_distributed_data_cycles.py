"""CPU DDP coverage for uneven cycle lengths, replay, and shared failures."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader, IterableDataset

from dl_core.callbacks.dataset_refresh import DatasetRefreshCallback
from dl_core.core.base_callback import Callback, CallbackList

from test_data_cycles import _CycleRecorder
from test_iteration_trainer import _build_trainer
from test_streaming_epoch_trainer import _GlooAccelerator


class _CycleStream(IterableDataset):
    def __init__(self, values: list[int], faulty: bool = False) -> None:
        self.values = values
        self.faulty = faulty

    def __len__(self) -> int:
        # Inventory estimates must not control a finite cycle boundary.
        return 100

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for index, value in enumerate(self.values):
            if self.faulty and index == 1:
                raise ValueError("malformed cycle sample")
            yield {"image": torch.tensor([float(value)])}


class _CycleWrapper:
    def __init__(self, rank: int, *, empty: bool = False, faulty: bool = False) -> None:
        self.rank = rank
        self.empty = empty
        self.faulty = faulty
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def refresh_dataset(self, split: str) -> None:
        assert split == "train"

    def get_split(self, split: str) -> DataLoader:
        assert split == "train"
        values = [1] if self.rank == 0 else [2, 3, 4, 5, 6]
        if self.empty and self.rank == 0:
            values = []
        elif self.faulty and self.rank == 0:
            values = [1, 2]
        return DataLoader(
            _CycleStream(
                [value + self.current_epoch * 10 for value in values],
                faulty=self.faulty and self.rank == 0,
            ),
            batch_size=1,
            num_workers=0,
        )

    def get_data_cycle_state(self) -> dict[str, Any]:
        return {"selection": self.current_epoch}

    def restore_data_cycle_state(self, state: dict[str, Any]) -> None:
        pass


def _distributed_trainer(
    rank: int,
    *,
    empty: bool = False,
    faulty: bool = False,
) -> tuple[Any, _CycleRecorder, list[int]]:
    trainer = _build_trainer(7)
    trainer.accelerator = _GlooAccelerator(rank)
    trainer.dataset_wrapper = _CycleWrapper(rank, empty=empty, faulty=faulty)
    recorder = _CycleRecorder()
    trainer.callbacks = CallbackList(
        [
            DatasetRefreshCallback(trigger="data_cycle"),
            recorder,
        ]
    )
    trainer.callbacks.set_trainer(trainer)
    model = DDP(torch.nn.Linear(1, 1, bias=False))
    trainer.models = {"main": model}
    trainer.optimizers = {"main": SGD(model.parameters(), lr=0.001)}
    seen = []

    def train_step(batch: dict[str, Any], index: int) -> dict[str, float]:
        seen.append(int(batch["image"].item()))
        loss = model(batch["image"]).square().mean()
        trainer.accelerator.backward(loss, model)
        trainer.accelerator.optimizer_step(trainer.optimizers["main"], model)
        return {"loss": float(loss.detach())}

    trainer.train_step = train_step
    trainer._save_checkpoint = lambda iteration, filename=None: None
    return trainer, recorder, seen


def _run_cycle_rank(rank: int, rendezvous: Path) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer, recorder, seen = _distributed_trainer(rank)
        checkpoints = {}
        trainer.checkpoint_frequency = 3
        trainer._save_checkpoint = lambda iteration, filename=None: checkpoints.update(
            {
                iteration: trainer._build_checkpoint_payload(iteration),
            }
        )
        trainer.perform_training()

        assert seen == ([1] * 5 + [11, 11] if rank == 0 else [2, 3, 4, 5, 6, 12, 13])
        assert [cycle for cycle, _ in recorder.starts] == [0, 1]
        assert [logs["iteration"] for _, logs in recorder.ends] == [5, 7]
        assert trainer.accelerator.accumulation_counter == 0
        assert checkpoints[4]["iteration_in_cycle"] == 4
        weights = [
            torch.empty_like(trainer.models["main"].module.weight) for _ in range(2)
        ]
        dist.all_gather(weights, trainer.models["main"].module.weight)
        torch.testing.assert_close(weights[0], weights[1])

        # The short rank has already replayed at this checkpoint. The resume
        # cursor must reconstruct that replay while preserving the shared cycle.
        resumed, resumed_events, remaining = _distributed_trainer(rank)
        resumed.restore_progress_state(checkpoints[4])
        resumed.perform_training()
        assert remaining == seen[4:]
        assert [cycle for cycle, _ in resumed_events.starts] == [0, 1]
        assert resumed_events.starts[0][1]["resuming"]

        empty, _, seen_empty = _distributed_trainer(rank, empty=True)
        with pytest.raises(RuntimeError, match="empty train|failed on another rank"):
            empty.perform_training()
        assert seen_empty == []
        assert empty._data_iterator is None

        faulty, _, seen_faulty = _distributed_trainer(rank, faulty=True)
        with pytest.raises(
            ValueError if rank == 0 else RuntimeError,
            match="malformed cycle sample" if rank == 0 else "failed on another rank",
        ):
            faulty.perform_training()
        assert len(seen_faulty) == 1
        assert faulty._data_iterator is None

        # Loader rebuilding is a required operation. One failed rank must stop
        # its peers rather than leave them training on another selection.
        broken, _, seen_broken = _distributed_trainer(rank)
        callback = Callback()

        def fail_selection(cycle: int, logs: dict[str, Any]) -> None:
            if rank == 0:
                raise ValueError("failed cycle selection")

        callback.on_data_cycle_start = fail_selection
        broken.callbacks = CallbackList([callback])
        broken.callbacks.set_trainer(broken)
        with pytest.raises(
            RuntimeError, match="failed cycle selection|another rank failed"
        ):
            broken.perform_training()
        assert callback.enabled
        assert seen_broken == []
    finally:
        dist.destroy_process_group()


def test_uneven_cycles_and_resume_with_cpu_ddp(tmp_path: Path) -> None:
    mp.spawn(
        _run_cycle_rank,
        args=(tmp_path / "cycle-ddp",),
        nprocs=2,
        join=True,
    )
