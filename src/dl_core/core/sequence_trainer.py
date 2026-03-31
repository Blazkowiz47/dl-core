"""Sequence-oriented trainer foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .base_trainer import EpochTrainer


@dataclass
class SequenceStepOutput:
    """Structured output returned by sequence-specific step hooks."""

    metrics: dict[str, float] = field(default_factory=dict)
    loss: torch.Tensor | float | None = None
    probabilities: torch.Tensor | None = None
    batch_data_updates: dict[str, Any] = field(default_factory=dict)


class SequenceTrainer(EpochTrainer):
    """
    Epoch-based trainer foundation for sequence and NLP workloads.

    This subclass reuses the standard epoch loop while adding sequence-aware
    helpers for padded token batches plus dedicated sequence step hooks.
    Subclasses can either override the regular ``train_step`` /
    ``validation_step`` / ``test_step`` methods directly, or implement the
    ``sequence_*_step`` hooks and return ``SequenceStepOutput`` objects.
    """

    @property
    def sequence_config(self) -> dict[str, Any]:
        """Return sequence-specific trainer configuration."""

        return dict(self.trainer_config.get("sequence", {}))

    @property
    def teacher_forcing_enabled(self) -> bool:
        """Return whether teacher forcing is enabled for this trainer."""

        return bool(self.sequence_config.get("teacher_forcing", False))

    @property
    def max_sequence_length(self) -> int | None:
        """Return the optional maximum sequence length from config."""

        max_length = self.sequence_config.get("max_length")
        if max_length is None:
            return None
        return int(max_length)

    @property
    def sequence_input_keys(self) -> tuple[str, ...]:
        """Return batch keys forwarded to the underlying sequence model."""

        input_keys = self.sequence_config.get(
            "input_keys",
            ["input_ids", "attention_mask", "token_type_ids"],
        )
        if isinstance(input_keys, str):
            return (input_keys,)
        return tuple(str(key) for key in input_keys)

    def build_sequence_model_inputs(
        self,
        batch_data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Extract tensor model inputs for a token or sequence batch."""

        model_inputs: dict[str, torch.Tensor] = {}
        for key in self.sequence_input_keys:
            value = batch_data.get(key)
            if torch.is_tensor(value):
                model_inputs[key] = value
        return model_inputs

    def get_sequence_labels(
        self,
        batch_data: dict[str, Any],
    ) -> torch.Tensor | None:
        """Return the label tensor for the current sequence batch."""

        labels = batch_data.get("label")
        if not torch.is_tensor(labels):
            return None
        return labels

    def train_step(
        self,
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        """Dispatch one training step through the sequence step hook."""

        step_output = self.sequence_train_step(batch_data, batch_idx)
        return self._finalize_sequence_step_output(
            "train",
            batch_data,
            step_output,
        )

    def test_step(self, batch_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Dispatch one test step through the sequence step hook."""

        step_output = self.sequence_test_step(batch_data)
        return self._finalize_sequence_step_output("test", batch_data, step_output)

    def validation_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Dispatch one validation step through the sequence step hook."""

        step_output = self.sequence_validation_step(batch_data)
        return self._finalize_sequence_step_output(
            "validation",
            batch_data,
            step_output,
        )

    def sequence_train_step(
        self,
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> SequenceStepOutput:
        """
        Run one sequence-specific training step.

        Subclasses should override this hook when they want the
        ``SequenceTrainer`` base to handle metric-manager updates and scalar
        step metric extraction automatically.
        """

        raise NotImplementedError(
            "Override sequence_train_step() or train_step() in your "
            "SequenceTrainer subclass."
        )

    def sequence_test_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> SequenceStepOutput:
        """
        Run one sequence-specific test step.

        Subclasses should override this hook when they want the
        ``SequenceTrainer`` base to handle metric-manager updates and scalar
        step metric extraction automatically.
        """

        raise NotImplementedError(
            "Override sequence_test_step() or test_step() in your "
            "SequenceTrainer subclass."
        )

    def sequence_validation_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> SequenceStepOutput:
        """
        Run one sequence-specific validation step.

        Subclasses should override this hook when they want the
        ``SequenceTrainer`` base to handle metric-manager updates and scalar
        step metric extraction automatically.
        """

        raise NotImplementedError(
            "Override sequence_validation_step() or validation_step() in your "
            "SequenceTrainer subclass."
        )

    def _finalize_sequence_step_output(
        self,
        split: str,
        batch_data: dict[str, Any],
        step_output: SequenceStepOutput,
    ) -> dict[str, float]:
        """Convert a structured sequence step output into scalar metrics."""

        if step_output.batch_data_updates:
            batch_data.update(step_output.batch_data_updates)

        step_metrics = {
            key: float(value) for key, value in step_output.metrics.items()
        }

        if step_output.loss is not None and "loss" not in step_metrics:
            step_metrics["loss"] = self._loss_to_float(step_output.loss)

        probabilities = step_output.probabilities
        if probabilities is not None:
            if self.metric_managers:
                for manager in self.metric_managers.values():
                    manager.update(split, probabilities, batch_data)
            step_metrics["probabilities_tensor"] = probabilities

        return step_metrics

    def _loss_to_float(self, loss: torch.Tensor | float) -> float:
        """Convert a scalar tensor or float loss into a Python float."""

        if torch.is_tensor(loss):
            return float(loss.detach().item())
        return float(loss)
