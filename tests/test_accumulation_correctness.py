"""Regression tests for optimization-step correctness."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.optim import SGD

from dl_core.accelerators.cpu import CPUAccelerator
from dl_core.accelerators.multi_gpu import MultiGPUAccelerator
from dl_core.trainers.standard_trainer import StandardTrainer


def _loss(model: torch.nn.Module, value: float, target: float) -> torch.Tensor:
    """Return a scalar squared-error loss for one sample."""

    prediction = model(torch.tensor([[value]], dtype=torch.float32))
    return (prediction - target).square().mean()


def test_accumulation_matches_one_combined_batch() -> None:
    """Two microbatches should produce one averaged optimizer update."""

    accumulated_model = torch.nn.Linear(1, 1, bias=False)
    combined_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        accumulated_model.weight.fill_(0.25)
        combined_model.weight.copy_(accumulated_model.weight)

    accelerator = CPUAccelerator({"gradient_accumulation_steps": 2})
    accumulated_optimizer = SGD(accumulated_model.parameters(), lr=0.1)
    accumulated_optimizer.zero_grad()

    step_results = []
    for value, target in [(1.0, 2.0), (3.0, -1.0)]:
        accelerator.backward(_loss(accumulated_model, value, target))
        step_results.append(
            accelerator.optimizer_step(accumulated_optimizer, accumulated_model)
        )

    combined_optimizer = SGD(combined_model.parameters(), lr=0.1)
    values = torch.tensor([[1.0], [3.0]])
    targets = torch.tensor([[2.0], [-1.0]])
    (combined_model(values) - targets).square().mean().backward()
    combined_optimizer.step()

    assert step_results == [False, True]
    assert torch.allclose(accumulated_model.weight, combined_model.weight)


def test_final_partial_accumulation_window_is_rescaled_and_stepped() -> None:
    """A short final window should average its actual microbatches, not vanish."""

    partial_model = torch.nn.Linear(1, 1, bias=False)
    reference_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        partial_model.weight.fill_(0.25)
        reference_model.weight.copy_(partial_model.weight)

    accelerator = CPUAccelerator({"gradient_accumulation_steps": 4})
    partial_optimizer = SGD(partial_model.parameters(), lr=0.1)
    partial_optimizer.zero_grad()
    accelerator.backward(_loss(partial_model, 1.0, 2.0))
    assert not accelerator.optimizer_step(partial_optimizer, partial_model)
    accelerator.backward(_loss(partial_model, 3.0, -1.0), finalize=True)
    assert accelerator.optimizer_step(
        partial_optimizer,
        partial_model,
        finalize=True,
    )

    reference_optimizer = SGD(reference_model.parameters(), lr=0.1)
    values = torch.tensor([[1.0], [3.0]])
    targets = torch.tensor([[2.0], [-1.0]])
    (reference_model(values) - targets).square().mean().backward()
    reference_optimizer.step()

    assert torch.allclose(partial_model.weight, reference_model.weight)


def test_epoch_trainer_finalizes_custom_step_without_explicit_flag() -> None:
    """Custom epoch steps may use ordinary accelerator calls on the last batch."""
    trainer = StandardTrainer({"trainer": {"standard": {"epochs": 1}}})
    trainer.accelerator = CPUAccelerator({"gradient_accumulation_steps": 4})
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.25)
    trainer.models["main"] = model
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer.optimizers["main"] = optimizer
    trainer.callbacks = SimpleNamespace(
        on_batch_start=lambda *args: None,
        on_batch_end=lambda *args: None,
    )
    trainer.data_loader["train"] = [
        {"image": torch.tensor([[1.0]]), "target": torch.tensor([[2.0]])},
        {"image": torch.tensor([[3.0]]), "target": torch.tensor([[-1.0]])},
    ]
    trainer.current_epoch = 1

    def custom_train_step(
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        del batch_idx
        loss = (model(batch_data["image"]) - batch_data["target"]).square().mean()
        trainer.accelerator.backward(loss, model)
        trainer.accelerator.optimizer_step(optimizer, model)
        return {"loss": loss.item()}

    trainer.train_step = custom_train_step
    trainer.train_epoch()

    assert model.weight.item() != pytest.approx(0.25)
    assert trainer.accelerator.accumulation_counter == 0
    assert trainer.accelerator.finalize_accumulation is False


def test_distributed_final_window_does_not_skip_gradient_sync(
    monkeypatch: Any,
) -> None:
    """Trainer-owned finalization must bypass DDP no_sync on the last batch."""

    class _DDPStub:
        def __init__(self) -> None:
            self.no_sync_calls = 0

        def no_sync(self) -> Any:
            self.no_sync_calls += 1
            return nullcontext()

    monkeypatch.setattr("dl_core.accelerators.multi_gpu.DDP", _DDPStub)
    accelerator = object.__new__(MultiGPUAccelerator)
    accelerator.gradient_accumulation_steps = 4
    accelerator.accumulation_counter = 1
    accelerator.finalize_accumulation = True
    accelerator.scaler = None
    model = _DDPStub()
    loss = torch.tensor(1.0, requires_grad=True)

    accelerator.backward(loss, model=model)

    assert model.no_sync_calls == 0
    assert loss.grad == pytest.approx(0.25)


def test_scaled_gradients_are_unscaled_before_clipping(monkeypatch: Any) -> None:
    """FP16 clipping must inspect real gradients rather than scaled gradients."""

    events: list[str] = []
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.full_like(model.weight, 8.0)

    class _Scaler:
        def unscale_(self, current_optimizer: torch.optim.Optimizer) -> None:
            events.append("unscale")
            for group in current_optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(8.0)

        def step(self, current_optimizer: torch.optim.Optimizer) -> None:
            events.append("step")
            current_optimizer.step()

        def update(self) -> None:
            events.append("update")

    def _record_clip(parameters: Any, max_norm: float) -> None:
        del max_norm
        events.append("clip")
        assert next(iter(parameters)).grad.item() == pytest.approx(1.0)

    accelerator = CPUAccelerator({"max_grad_norm": 1.0})
    accelerator.scaler = _Scaler()
    monkeypatch.setattr(
        "dl_core.core.base_accelerator.clip_grad_norm_",
        _record_clip,
    )

    assert accelerator.optimizer_step(optimizer, model)
    assert events == ["unscale", "clip", "step", "update"]


def test_accumulation_steps_must_be_positive() -> None:
    """Invalid accumulation config should fail instead of silently never stepping."""

    with pytest.raises(ValueError, match="at least 1"):
        CPUAccelerator({"gradient_accumulation_steps": 0})
