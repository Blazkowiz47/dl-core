"""Tests for class-neutral trainer probability diagnostics."""

from __future__ import annotations

import math

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


def test_probability_diagnostics_do_not_truncate_soft_labels() -> None:
    """Float targets should not be silently coerced into class indices."""

    trainer = object()
    metrics = EpochTrainer.compute_probability_diagnostics(
        trainer,
        {
            "probabilities_tensor": torch.tensor(
                [[0.8, 0.2], [0.1, 0.9]],
            ),
        },
        {"label": torch.tensor([0.2, 0.8])},
    )

    assert "prob_true_class_mean" not in metrics
    assert metrics["prob_confidence_mean"] == pytest.approx(0.85)


def test_single_column_binary_probabilities_include_both_classes() -> None:
    """Sigmoid output should yield binary entropy and true-class confidence."""

    metrics = EpochTrainer.compute_probability_diagnostics(
        object(),
        {"probabilities_tensor": torch.tensor([[0.2], [0.9]])},
        {"label": torch.tensor([0, 1])},
    )

    expected_entropy = -(
        0.2 * math.log(0.2)
        + 0.8 * math.log(0.8)
        + 0.9 * math.log(0.9)
        + 0.1 * math.log(0.1)
    ) / 2
    assert metrics["prob_confidence_mean"] == pytest.approx(0.85)
    assert metrics["prob_margin_mean"] == pytest.approx(0.7)
    assert metrics["prob_true_class_mean"] == pytest.approx(0.85)
    assert metrics["prob_entropy_mean"] == pytest.approx(expected_entropy)
