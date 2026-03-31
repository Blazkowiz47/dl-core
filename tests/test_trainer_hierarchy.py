"""Tests for the specialized trainer base hierarchy."""

from __future__ import annotations

from typing import Any

import torch

from dl_core.core import (
    AdaptiveComputationStepOutput,
    AdaptiveComputationTrainer,
    BaseTrainer,
    CarryState,
    EpochTrainer,
    SequenceStepOutput,
    SequenceTrainer,
)
from dl_core.trainers import StandardTrainer


class _ConcreteSequenceTrainer(SequenceTrainer):
    """Concrete sequence trainer used for hierarchy tests."""

    def __init__(self) -> None:
        pass

    def setup_model(self) -> None:
        """No-op test implementation."""

    def setup_criterion(self) -> None:
        """No-op test implementation."""

    def setup_optimizer(self) -> None:
        """No-op test implementation."""

    def setup_scheduler(self) -> None:
        """No-op test implementation."""

    def sequence_train_step(
        self,
        batch_data: dict[str, Any],
        batch_idx: int,
    ) -> SequenceStepOutput:
        """No-op sequence-step implementation."""

        return SequenceStepOutput(
            loss=torch.tensor(1.5),
            metrics={"token_accuracy": 0.5},
            probabilities=torch.tensor(
                [[0.9, 0.1], [0.2, 0.8]],
                dtype=torch.float32,
            ),
        )

    def sequence_test_step(
        self,
        batch_data: dict[str, Any],
    ) -> SequenceStepOutput:
        """No-op sequence-step implementation."""

        return SequenceStepOutput(metrics={"token_accuracy": 0.75})

    def sequence_validation_step(
        self,
        batch_data: dict[str, Any],
    ) -> SequenceStepOutput:
        """No-op sequence-step implementation."""

        return SequenceStepOutput(metrics={"token_accuracy": 0.6})


class _ConcreteAdaptiveTrainer(AdaptiveComputationTrainer):
    """Concrete adaptive trainer used for hierarchy tests."""

    def __init__(self) -> None:
        pass

    def setup_model(self) -> None:
        """No-op test implementation."""

    def setup_criterion(self) -> None:
        """No-op test implementation."""

    def setup_optimizer(self) -> None:
        """No-op test implementation."""

    def setup_scheduler(self) -> None:
        """No-op test implementation."""

    def adaptive_train_step(
        self,
        carry_state: CarryState,
        batch_idx: int,
    ) -> AdaptiveComputationStepOutput:
        """No-op adaptive-step implementation."""

        next_carry_state = self.step_carry_state(
            carry_state,
            halted=torch.tensor([False, True]),
            step_increment=1,
        )
        return AdaptiveComputationStepOutput(
            loss=torch.tensor(2.0),
            metrics={"ponder_penalty": 0.1},
            probabilities=torch.tensor(
                [[0.8, 0.2], [0.3, 0.7]],
                dtype=torch.float32,
            ),
            carry_state=next_carry_state,
        )

    def adaptive_test_step(
        self,
        carry_state: CarryState,
    ) -> AdaptiveComputationStepOutput:
        """No-op adaptive-step implementation."""

        return AdaptiveComputationStepOutput(metrics={"ponder_penalty": 0.05})

    def adaptive_validation_step(
        self,
        carry_state: CarryState,
    ) -> AdaptiveComputationStepOutput:
        """No-op adaptive-step implementation."""

        return AdaptiveComputationStepOutput(metrics={"ponder_penalty": 0.08})


def test_base_trainer_is_epoch_trainer_alias() -> None:
    """`BaseTrainer` should remain a compatibility alias."""

    assert BaseTrainer is EpochTrainer


def test_standard_trainer_uses_epoch_trainer() -> None:
    """The built-in standard trainer should extend the epoch-based base."""

    assert issubclass(StandardTrainer, EpochTrainer)


def test_sequence_trainer_exposes_sequence_config() -> None:
    """Sequence trainers should expose their specialized config section."""

    trainer = _ConcreteSequenceTrainer()
    trainer.trainer_config = {
        "epochs": 2,
        "sequence": {"max_length": 128, "teacher_forcing": True},
    }

    assert trainer.sequence_config == {
        "max_length": 128,
        "teacher_forcing": True,
    }


def test_sequence_trainer_exposes_sequence_helpers() -> None:
    """Sequence trainers should expose NLP-oriented helper properties."""

    trainer = _ConcreteSequenceTrainer()
    trainer.trainer_config = {
        "epochs": 2,
        "sequence": {
            "max_length": 64,
            "teacher_forcing": True,
            "input_keys": ["input_ids", "attention_mask"],
        },
    }

    batch_data = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 4, dtype=torch.long),
        "label": torch.tensor([0, 1]),
    }

    assert trainer.teacher_forcing_enabled is True
    assert trainer.max_sequence_length == 64
    model_inputs = trainer.build_sequence_model_inputs(batch_data)

    assert set(model_inputs) == {"input_ids", "attention_mask"}
    assert torch.equal(model_inputs["input_ids"], batch_data["input_ids"])
    assert torch.equal(
        model_inputs["attention_mask"],
        batch_data["attention_mask"],
    )
    assert torch.equal(trainer.get_sequence_labels(batch_data), batch_data["label"])


def test_sequence_trainer_step_wrapper_returns_scalar_metrics() -> None:
    """Sequence trainer wrappers should convert structured outputs to scalars."""

    trainer = _ConcreteSequenceTrainer()
    trainer.metric_managers = {}

    batch_data = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "label": torch.tensor([0, 1]),
    }

    step_metrics = trainer.train_step(batch_data, 0)

    assert step_metrics["loss"] == 1.5
    assert step_metrics["token_accuracy"] == 0.5
    assert torch.is_tensor(step_metrics["probabilities_tensor"])


def test_adaptive_trainer_exposes_adaptive_config() -> None:
    """Adaptive trainers should expose their specialized config section."""

    trainer = _ConcreteAdaptiveTrainer()
    trainer.trainer_config = {
        "epochs": 2,
        "adaptive_computation": {"max_halt_steps": 8, "ponder_cost": 0.01},
    }

    assert trainer.adaptive_computation_config == {
        "max_halt_steps": 8,
        "ponder_cost": 0.01,
    }


def test_adaptive_trainer_initializes_carry_state() -> None:
    """Adaptive trainers should initialize carry state from batch data."""

    trainer = _ConcreteAdaptiveTrainer()
    trainer.trainer_config = {
        "epochs": 2,
        "adaptive_computation": {"max_halt_steps": 8, "ponder_cost": 0.01},
    }
    batch_data = {"label": torch.tensor([0, 1])}

    carry_state = trainer.initialize_carry_state(batch_data, "train")

    assert carry_state.batch_size == 2
    assert carry_state.step == 0
    assert torch.equal(carry_state.steps, torch.zeros(2, dtype=torch.long))
    assert torch.equal(carry_state.halted, torch.zeros(2, dtype=torch.bool))


def test_adaptive_trainer_step_wrapper_updates_batch_metadata() -> None:
    """Adaptive trainer wrappers should inject carry metadata into batch data."""

    trainer = _ConcreteAdaptiveTrainer()
    trainer.metric_managers = {}

    batch_data = {"label": torch.tensor([0, 1])}
    step_metrics = trainer.train_step(batch_data, 0)

    assert step_metrics["loss"] == 2.0
    assert step_metrics["ponder_penalty"] == 0.1
    assert torch.is_tensor(step_metrics["probabilities_tensor"])
    assert isinstance(batch_data["carry_state"], CarryState)
    assert torch.equal(batch_data["steps"], torch.tensor([1, 0]))
    assert torch.equal(batch_data["halted"], torch.tensor([False, True]))
