"""Regression tests for optimization-step correctness."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.optim import SGD

from dl_core.accelerators.cpu import CPUAccelerator


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
