"""Dummy-data coverage for uneven streaming epochs and evaluation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader, IterableDataset

from dl_core.accelerators.cpu import CPUAccelerator
from dl_core.accelerators.multi_gpu import MultiGPUAccelerator
from dl_core.core.base_metric_manager import BaseMetricManager
from dl_core.core.base_trainer import EpochTrainer
from dl_core.core.registry import DATASET_REGISTRY


class _DummyStream(IterableDataset):
    """Finite, unsized stream with rank-specific valid samples."""

    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for value in self.values:
            yield {
                "image": torch.tensor([float(value)]),
                "label": torch.tensor(value),
            }


class _FaultyStream(_DummyStream):
    """Raise while fetching the lookahead batch on one rank."""

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield from super().__iter__()
        raise ValueError("malformed training sample")


class _EstimatedLengthStream(_DummyStream):
    """Iterable length can describe inventory rather than yielded samples."""

    def __len__(self) -> int:
        return 100


class _BufferedLinear(torch.nn.Module):
    """Simple model with a buffer that must be synced before local eval."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.register_buffer("offset", torch.zeros(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.linear(image) + self.offset


class _CountMetric:
    def compute(
        self, predictions: np.ndarray, labels: np.ndarray, **kwargs: Any
    ) -> dict[str, float]:
        del predictions, kwargs
        return {
            "global_count": float(len(labels)),
            "label_sum": float(labels.sum()),
        }


class _DummyMetricManager(BaseMetricManager):
    def setup_metrics(self) -> None:
        self.metrics = {
            split: {"count": _CountMetric()}
            for split in ("train", "validation", "test")
        }

    def get_best_metric_key(self) -> str:
        return "global_count"

    def print_logs(self, split: str) -> None:
        del split

    def get_logs(self, split: str) -> dict[str, float]:
        return self.results_cache.get(split, {})


class _Callbacks:
    def on_batch_start(
        self, batch_idx: int, split: str, batch_data: dict[str, Any]
    ) -> None:
        del batch_idx, split, batch_data

    def on_batch_end(
        self, batch_idx: int, split: str, batch_data: dict[str, Any]
    ) -> None:
        del batch_idx, split, batch_data


class _DummyTrainer(EpochTrainer):
    def setup_artifact_manager(self) -> None:
        pass

    def setup_model(self) -> None:
        pass

    def setup_criterion(self) -> None:
        pass

    def setup_optimizer(self) -> None:
        pass

    def setup_scheduler(self) -> None:
        pass

    def train_step(
        self, batch_data: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        del batch_idx
        model = self.models["main"]
        loss = model(batch_data["image"]).square().mean()
        self.accelerator.backward(loss, model)
        self.accelerator.optimizer_step(self.optimizers["main"], model)
        return {"loss": float(loss.detach())}

    def test_step(self, batch_data: dict[str, torch.Tensor]) -> dict[str, float]:
        self.metric_managers["dummy"].update(
            "test", self.models["main"](batch_data["image"]), batch_data
        )
        return {"loss": float(batch_data["image"].mean())}

    def validation_step(
        self, batch_data: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        self.metric_managers["dummy"].update(
            "validation", self.models["main"](batch_data["image"]), batch_data
        )
        return {"loss": float(batch_data["image"].mean())}


class _GlooAccelerator(CPUAccelerator):
    """Exercise the production DDP hooks with a CPU process group."""

    def __init__(self, rank: int) -> None:
        super().__init__({"gradient_accumulation_steps": 2})
        self.use_distributed = True
        self.global_rank = rank
        self.world_size = 2

    def is_main_process(self) -> bool:
        return self.global_rank == 0

    def wait_for_everyone(self, message: str = "") -> None:
        del message
        dist.barrier()

    def backward(
        self,
        loss: torch.Tensor,
        model: torch.nn.Module | None = None,
        finalize: bool = False,
    ) -> None:
        MultiGPUAccelerator.backward(self, loss, model, finalize)

    def prepare_eval_models(
        self, models: dict[str, torch.nn.Module]
    ) -> dict[str, torch.nn.Module]:
        return MultiGPUAccelerator.prepare_eval_models(self, models)


def test_setup_accepts_unsized_dummy_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyWrapper:
        auto_split = False

        def get_split(self, split: str) -> DataLoader:
            del split
            return DataLoader(_DummyStream([1]), batch_size=1)

        def get_stats(self, split: str) -> list[str]:
            del split
            return []

    monkeypatch.setattr(DATASET_REGISTRY, "get", lambda *args: _DummyWrapper())
    trainer = _DummyTrainer(
        {
            "trainer": {"dummy": {"epochs": 1}},
            "dataset": {"name": "dummy"},
        }
    )

    trainer._setup_data()

    assert all(loader is not None for loader in trainer.data_loader.values())


def test_epoch_trainer_rejects_resampled_stream() -> None:
    trainer = _DummyTrainer({"trainer": {"dummy": {"epochs": 1}}})
    trainer.accelerator = CPUAccelerator({})
    loader = DataLoader(_DummyStream([1]), batch_size=1)
    loader.dataset.is_resampled = True
    trainer.data_loader = {"train": loader, "validation": None, "test": None}

    with pytest.raises(ValueError, match="IterationTrainer for resampled data"):
        trainer.train_epoch()


def test_epoch_trainer_uses_stream_exhaustion_not_length_estimate() -> None:
    trainer = _DummyTrainer({"trainer": {"dummy": {"epochs": 1}}})
    trainer.accelerator = CPUAccelerator({"gradient_accumulation_steps": 2})
    trainer.callbacks = _Callbacks()
    model = _BufferedLinear()
    trainer.models = {"main": model}
    trainer.optimizers = {"main": SGD(model.parameters(), lr=0.01)}
    trainer.data_loader = {
        "train": DataLoader(_EstimatedLengthStream([1, 2, 3]), batch_size=1),
        "validation": None,
        "test": None,
    }

    trainer.train_epoch()

    assert trainer.global_step == 3
    assert trainer.accelerator.accumulation_counter == 0


def _run_dummy_rank(rank: int, rendezvous: Path) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = _DummyTrainer({"trainer": {"dummy": {"epochs": 1}}})
        trainer.accelerator = _GlooAccelerator(rank)
        trainer.callbacks = _Callbacks()
        model = DDP(_BufferedLinear())
        trainer.models = {"main": model}
        trainer.optimizers = {"main": SGD(model.parameters(), lr=0.01)}
        trainer.data_loader = {
            "train": DataLoader(
                _DummyStream([1, 2, 3] if rank == 0 else [1, 2, 3, 4, 5]),
                batch_size=1,
            ),
            "validation": DataLoader(
                _DummyStream([] if rank == 0 else [1, 2]), batch_size=1
            ),
            "test": DataLoader(
                _DummyStream([3] if rank == 0 else [4, 5, 6]), batch_size=1
            ),
        }

        trainer.train_epoch()
        assert trainer.global_step == 3
        assert trainer.accelerator.accumulation_counter == 0

        trainer.metric_managers["dummy"] = _DummyMetricManager(
            {"mode": "gather"}, trainer.accelerator, trainer
        )
        model.module.offset.fill_(10.0 * rank)
        validation_metrics = trainer.validation_epoch()
        assert isinstance(trainer.models["main"], DDP)
        assert float(model.module.offset) == 0.0
        assert validation_metrics["loss"] == pytest.approx(1.5)
        if rank == 0:
            assert validation_metrics["global_count"] == 2.0
            assert validation_metrics["label_sum"] == 3.0

        test_metrics = trainer.test_epoch()
        assert test_metrics["loss"] == pytest.approx(4.5)
        if rank == 0:
            assert test_metrics["global_count"] == 4.0
            assert test_metrics["label_sum"] == 18.0

        trainer.metric_managers["dummy"].mode = "average"
        with pytest.raises(RuntimeError, match="mode='gather'"):
            trainer.test_epoch()
        trainer.metric_managers["dummy"].mode = "gather"

        trainer.data_loader["validation"] = DataLoader(
            _DummyStream([]), batch_size=1
        )
        with pytest.raises(RuntimeError, match="No valid validation samples"):
            trainer.validation_epoch()

        trainer.data_loader["validation"] = DataLoader(
            _DummyStream([1, 2]) if rank == 0 else _FaultyStream([1]),
            batch_size=1,
        )
        with pytest.raises(
            ValueError if rank == 1 else RuntimeError,
            match="malformed training sample"
            if rank == 1
            else "validation evaluation failed on another rank",
        ):
            trainer.validation_epoch()

        original_test_loader = trainer.data_loader["test"]
        if rank == 0:
            trainer.data_loader["test"] = None
        with pytest.raises(RuntimeError, match="test loader is missing on some ranks"):
            trainer.test_epoch()
        trainer.data_loader["test"] = original_test_loader

        trainer.data_loader["train"] = DataLoader(_DummyStream([1]), batch_size=1)
        trainer.train_epoch()
        assert trainer.global_step == 4

        trainer.data_loader["train"] = DataLoader(
            _DummyStream([1, 2]) if rank == 0 else _FaultyStream([1]),
            batch_size=1,
        )
        with pytest.raises(
            ValueError if rank == 1 else RuntimeError,
            match="malformed training sample"
            if rank == 1
            else "failed on another rank",
        ):
            trainer.train_epoch()

        trainer.data_loader["train"] = DataLoader(
            _DummyStream([] if rank == 0 else [1, 2]), batch_size=1
        )
        with pytest.raises(RuntimeError, match="No shared training batches"):
            trainer.train_epoch()
    finally:
        dist.destroy_process_group()


def test_uneven_streaming_training_and_evaluation_on_dummy_data(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _run_dummy_rank,
        args=(tmp_path / "dummy-ddp",),
        nprocs=2,
        join=True,
    )
