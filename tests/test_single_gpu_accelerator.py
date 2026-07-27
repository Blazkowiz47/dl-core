"""Tests for single-GPU accelerator model preparation."""

from __future__ import annotations

import pytest
import torch

from dl_core.accelerators.single_gpu import SingleGPUAccelerator


def test_single_gpu_accelerator_leaves_models_eager_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_compile(model: torch.nn.Module, *, mode: str) -> None:
        del model, mode
        pytest.fail("Model compilation must remain opt-in")

    monkeypatch.setattr(torch.nn.Module, "compile", fail_compile)
    accelerator = SingleGPUAccelerator({})
    model = torch.nn.Linear(2, 1)

    prepared_models, _, _, _, _ = accelerator.prepare(models={"main": model})

    assert prepared_models["main"] is model


def test_single_gpu_accelerator_compiles_models_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.nn.Module, str]] = []

    def record_compile(model: torch.nn.Module, *, mode: str) -> None:
        calls.append((model, mode))

    monkeypatch.setattr(torch.nn.Module, "compile", record_compile)
    accelerator = SingleGPUAccelerator(
        {
            "compile_models": True,
            "compile_mode": "reduce-overhead",
        }
    )
    model = torch.nn.Linear(2, 1)

    prepared_models, _, _, _, _ = accelerator.prepare(models={"online": model})

    assert prepared_models["online"] is model
    assert calls == [(model, "reduce-overhead")]


def test_single_gpu_accelerator_rejects_unknown_compile_mode() -> None:
    with pytest.raises(ValueError, match="compile_mode must be"):
        SingleGPUAccelerator(
            {
                "compile_models": True,
                "compile_mode": "fastest",
            }
        )
