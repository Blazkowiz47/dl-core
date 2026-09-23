"""Tests for class-neutral trainer probability diagnostics."""

from __future__ import annotations

import pytest
import torch

from dl_core.core.base_trainer import EpochTrainer


def test_probability_diagnostics_do_not_require_domain_class_names() -> None:
    """Generic classification diagnostics should work for arbitrary classes."""

    trainer = object()
    metrics = EpochTrainer.compute_probability_diagnostics(
        trainer,
        {
            "loss": 0.4,
            "probabilities_tensor": torch.tensor(
                [[0.8, 0.2], [0.1, 0.9]],
            ),
        },
        {"label": torch.tensor([0, 1])},
    )

    assert "probabilities_tensor" not in metrics
    assert metrics["prob_confidence_mean"] == pytest.approx(0.85)
    assert metrics["prob_margin_mean"] == pytest.approx(0.7)
    assert metrics["prob_true_class_mean"] == pytest.approx(0.85)
    assert metrics["prob_entropy_mean"] > 0.0


def test_probability_diagnostics_skip_invalid_labels_safely() -> None:
    """Invalid labels should retain label-independent diagnostics only."""

    trainer = object()
    metrics = EpochTrainer.compute_probability_diagnostics(
        trainer,
        {"probabilities": torch.tensor([[0.3, 0.7]])},
        {"label": torch.tensor([4])},
    )

    assert "prob_true_class_mean" not in metrics
    assert metrics["prob_confidence_mean"] == pytest.approx(0.7)
