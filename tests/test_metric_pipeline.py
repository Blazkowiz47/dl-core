"""Smoke tests for built-in metrics, metric managers, and trainer steps."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from dl_core.criterions.crossentropy import CrossEntropy
from dl_core.metric_managers.standard_manager import (
    StandardActMetricManager,
    StandardMetricManager,
)
from dl_core.metrics.accuracy import AccuracyMetric
from dl_core.metrics.auc import AUCMetric
from dl_core.metrics.f1 import F1Metric
from dl_core.trainers.standard_trainer import StandardTrainer


class _SingleProcessAccelerator:
    """Minimal accelerator stub for local metric and trainer smoke tests."""

    def __init__(self) -> None:
        self.use_distributed = False

    def is_main_process(self) -> bool:
        """Report single-process ownership."""

        return True

    def autocast_context(self) -> Any:
        """Return a no-op autocast context."""

        return nullcontext()

    def backward(
        self,
        loss: torch.Tensor,
        model: torch.nn.Module | None = None,
    ) -> None:
        """Run a plain local backward pass."""

        loss.backward()

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module | None = None,
    ) -> bool:
        """Run a plain local optimizer step."""

        optimizer.step()
        optimizer.zero_grad()
        return True

    def wait_for_everyone(self, context: str | None = None) -> None:
        """No-op synchronization helper."""


class _ToyClassifier(torch.nn.Module):
    """Small random-data classifier used for trainer smoke tests."""

    def __init__(self, in_features: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_classes)

    def forward(self, batch_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return logits and probabilities for one mock batch."""

        logits = self.linear(batch_data["features"])
        probabilities = torch.softmax(logits, dim=1)
        return {"logits": logits, "probabilities": probabilities}


def _random_probability_batch(
    generator: torch.Generator,
    batch_size: int = 8,
    num_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return seeded random probabilities and labels."""

    logits = torch.randn(batch_size, num_classes, generator=generator)
    probabilities = torch.softmax(logits, dim=1)
    labels = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size,),
        generator=generator,
    )
    return probabilities, labels


def _mock_trainer_batch(
    generator: torch.Generator,
    batch_size: int = 6,
    feature_dim: int = 4,
) -> dict[str, Any]:
    """Return a seeded random batch for the standard trainer."""

    return {
        "features": torch.randn(batch_size, feature_dim, generator=generator),
        "label": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
        "path": [f"sample-{index}" for index in range(batch_size)],
    }


def test_builtin_metrics_handle_random_probabilities() -> None:
    """Built-in metrics should return bounded scores for random mock data."""

    generator = torch.Generator().manual_seed(2026)
    probabilities, labels = _random_probability_batch(generator, batch_size=16)
    predictions = probabilities.numpy()
    targets = labels.numpy()

    metrics = [
        AccuracyMetric({"num_classes": 2}),
        AUCMetric({"num_classes": 2}),
        F1Metric({"num_classes": 2}),
    ]

    for metric in metrics:
        result = metric.compute(predictions, targets)
        assert len(result) == 1
        value = next(iter(result.values()))
        assert 0.0 <= value <= 100.0


def test_standard_metric_manager_computes_random_batches() -> None:
    """Standard metric manager should accumulate and compute mock batch metrics."""

    generator = torch.Generator().manual_seed(2027)
    accelerator = _SingleProcessAccelerator()
    manager = StandardMetricManager({"num_classes": 2}, accelerator)

    for _ in range(2):
        probabilities, labels = _random_probability_batch(generator, batch_size=10)
        manager.update("validation", probabilities, {"label": labels})

    metrics = manager.compute("validation")
    diagnostics = manager.compute_epoch_diagnostics("validation")
    logs = manager.get_logs("validation")

    assert {"accuracy", "auc", "f1"} <= metrics.keys()
    assert diagnostics["cm/num_samples"] == 20.0
    assert {"validation_accuracy", "validation_auc", "validation_f1"} <= logs.keys()


def test_standard_act_metric_manager_computes_random_halt_stats() -> None:
    """ACT metric manager should include halt-step summaries for halted samples."""

    generator = torch.Generator().manual_seed(2028)
    accelerator = _SingleProcessAccelerator()
    manager = StandardActMetricManager({"num_classes": 2}, accelerator)

    probabilities, _ = _random_probability_batch(generator, batch_size=6)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    steps = torch.randint(low=1, high=5, size=(6,), generator=generator)
    halted = torch.tensor([True, True, False, True, False, True], dtype=torch.bool)

    manager.update(
        "test",
        probabilities,
        {"label": labels, "steps": steps, "halted": halted},
    )

    metrics = manager.compute("test")

    assert {"accuracy", "auc", "f1"} <= metrics.keys()
    assert "halt_steps_label_0_mean" in metrics
    assert "halt_steps_label_1_mean" in metrics


def test_standard_trainer_step_methods_handle_random_mock_batches(
    tmp_path: Any,
) -> None:
    """Standard trainer step hooks should work with seeded random mock data."""

    generator = torch.Generator().manual_seed(2029)
    config = {
        "seed": 2029,
        "runtime": {"output_dir": str(tmp_path)},
        "trainer": {"standard": {"epochs": 1}},
    }
    trainer = StandardTrainer(config)
    trainer.accelerator = _SingleProcessAccelerator()
    trainer.models["main"] = _ToyClassifier()
    trainer.criterions["crossentropy"] = CrossEntropy({})
    trainer.optimizers["main"] = SGD(trainer.model.parameters(), lr=0.1)
    trainer.schedulers["main"] = StepLR(trainer.optimizers["main"], step_size=1)
    trainer.metric_managers["standard"] = StandardMetricManager(
        {"num_classes": 2},
        trainer.accelerator,
        trainer,
    )

    train_metrics = trainer.train_step(_mock_trainer_batch(generator), 0)
    test_metrics = trainer.test_step(_mock_trainer_batch(generator))
    validation_metrics = trainer.validation_step(_mock_trainer_batch(generator))

    assert "loss" in train_metrics
    assert "crossentropy_loss" in train_metrics
    assert "loss" in test_metrics
    assert "loss" in validation_metrics
    assert len(trainer.metric_managers["standard"].accumulated_data["train"]["labels"]) == 1
    assert len(trainer.metric_managers["standard"].accumulated_data["test"]["labels"]) == 1
    assert (
        len(trainer.metric_managers["standard"].accumulated_data["validation"]["labels"])
        == 1
    )
