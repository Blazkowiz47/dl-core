"""Adaptive computation trainer foundation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .base_dataset import AdaptiveComputationDataset
from .base_trainer import EpochTrainer


@dataclass
class CarryState:
    """Carry state tracked across adaptive-computation reasoning steps."""

    current_batch: dict[str, Any]
    step: int = 0
    steps: torch.Tensor | None = None
    halted: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> torch.Tensor | None:
        """Return the label tensor for the current carry batch when present."""

        labels = self.current_batch.get("label")
        if not torch.is_tensor(labels):
            return None
        return labels

    @property
    def batch_size(self) -> int:
        """Return the current batch size stored inside the carry state."""

        if self.steps is not None:
            return int(self.steps.shape[0])
        if self.halted is not None:
            return int(self.halted.shape[0])
        if self.labels is not None:
            return int(self.labels.shape[0])
        return 0

    @property
    def active_mask(self) -> torch.Tensor | None:
        """Return a boolean mask of samples that are still active."""

        if self.halted is None:
            return None
        return ~self.halted

    def with_updates(
        self,
        *,
        current_batch: dict[str, Any] | None = None,
        step: int | None = None,
        steps: torch.Tensor | None = None,
        halted: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CarryState:
        """Return a copy of the carry state with selected fields updated."""

        return replace(
            self,
            current_batch=self.current_batch if current_batch is None else current_batch,
            step=self.step if step is None else step,
            steps=self.steps if steps is None else steps,
            halted=self.halted if halted is None else halted,
            metadata=self.metadata if metadata is None else metadata,
        )

    def as_batch_updates(self) -> dict[str, Any]:
        """Expose carry-state metadata in the format metric managers expect."""

        updates: dict[str, Any] = {"carry_state": self}
        if self.steps is not None:
            updates["steps"] = self.steps
        if self.halted is not None:
            updates["halted"] = self.halted
        return updates


@dataclass
class AdaptiveComputationStepOutput:
    """Structured output returned by adaptive-computation step hooks."""

    metrics: dict[str, float] = field(default_factory=dict)
    loss: torch.Tensor | float | None = None
    probabilities: torch.Tensor | None = None
    carry_state: CarryState | None = None
    batch_data_updates: dict[str, Any] = field(default_factory=dict)


class AdaptiveComputationTrainer(EpochTrainer):
    """
    Epoch-based trainer foundation for adaptive-time computation models.

    This subclass keeps the standard artifact, callback, accelerator, and
    distributed training flow, while adding a framework-level carry-state
    contract for recursive or halting-based training loops. Subclasses can
    still override ``train_step`` / ``validation_step`` / ``test_step``
    directly, or they can implement the ``adaptive_*_step`` hooks and return
    ``AdaptiveComputationStepOutput`` objects.
    """

    @property
    def adaptive_computation_config(self) -> dict[str, Any]:
        """Return adaptive-computation-specific trainer configuration."""

        return dict(self.trainer_config.get("adaptive_computation", {}))

    @property
    def max_halt_steps(self) -> int:
        """Return the configured maximum number of adaptive reasoning steps."""

        return int(self.adaptive_computation_config.get("max_halt_steps", 1))

    @property
    def ponder_cost(self) -> float:
        """Return the configured ACT ponder cost."""

        return float(self.adaptive_computation_config.get("ponder_cost", 0.0))

    @property
    def wrap_class_stream(self) -> bool:
        """Return whether class streams should wrap when exhausted."""

        return bool(self.adaptive_computation_config.get("wrap_class_stream", False))

    def adaptive_dataset(self) -> AdaptiveComputationDataset:
        """Return the adaptive dataset wrapper or raise a helpful error."""

        if not isinstance(self.dataset_wrapper, AdaptiveComputationDataset):
            raise TypeError(
                "AdaptiveComputationTrainer requires dataset_wrapper to be an "
                "AdaptiveComputationDataset."
            )
        return self.dataset_wrapper

    def initialize_carry_state(
        self,
        batch_data: dict[str, Any],
        split: str,
    ) -> CarryState:
        """Build the initial carry state for one adaptive-computation batch."""

        labels = batch_data.get("label")
        device: torch.device | None = None
        batch_size = 0

        if torch.is_tensor(labels):
            device = labels.device
            batch_size = int(labels.shape[0])
        else:
            try:
                batch_size = int(self._get_batch_size(batch_data))
            except Exception:
                batch_size = 0

        steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return CarryState(
            current_batch=batch_data,
            step=0,
            steps=steps,
            halted=halted,
        )

    def get_next_class_sample(
        self,
        label: Any,
        split: str,
        *,
        wrap_around: bool | None = None,
        transform_sample: bool = True,
    ) -> dict[str, Any] | None:
        """Return the next class-stream sample for one label."""

        if wrap_around is None:
            wrap_around = self.wrap_class_stream
        return self.adaptive_dataset().get_next_class_sample(
            label,
            split,
            wrap_around=wrap_around,
            transform_sample=transform_sample,
        )

    def get_next_class_samples(
        self,
        labels: torch.Tensor | list[Any],
        split: str,
        *,
        wrap_around: bool | None = None,
        transform_sample: bool = True,
    ) -> list[dict[str, Any] | None]:
        """Return the next class-stream sample for each label in a batch."""

        if torch.is_tensor(labels):
            label_values = labels.detach().cpu().tolist()
        else:
            label_values = list(labels)

        return [
            self.get_next_class_sample(
                label,
                split,
                wrap_around=wrap_around,
                transform_sample=transform_sample,
            )
            for label in label_values
        ]

    def step_carry_state(
        self,
        carry_state: CarryState,
        *,
        current_batch: dict[str, Any] | None = None,
        halted: torch.Tensor | None = None,
        steps: torch.Tensor | None = None,
        metadata_updates: dict[str, Any] | None = None,
        step_increment: int = 1,
    ) -> CarryState:
        """Return an updated carry state after one ACT reasoning step."""

        next_steps = steps
        if next_steps is None and carry_state.steps is not None:
            next_steps = carry_state.steps.clone()
            if next_steps.numel() > 0:
                if halted is None:
                    next_steps += step_increment
                else:
                    next_steps[~halted] += step_increment

        next_metadata = dict(carry_state.metadata)
        if metadata_updates:
            next_metadata.update(metadata_updates)

        next_halted = carry_state.halted if halted is None else halted
        return carry_state.with_updates(
            current_batch=(
                carry_state.current_batch
                if current_batch is None
                else current_batch
            ),
            step=carry_state.step + step_increment,
            steps=next_steps,
            halted=next_halted,
            metadata=next_metadata,
        )

    def train_step(
        self,
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        """Dispatch one training step through the adaptive step hook."""

        carry_state = self.initialize_carry_state(batch_data, "train")
        step_output = self.adaptive_train_step(carry_state, batch_idx)
        return self._finalize_adaptive_step_output(
            "train",
            batch_data,
            carry_state,
            step_output,
        )

    def test_step(self, batch_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Dispatch one test step through the adaptive step hook."""

        carry_state = self.initialize_carry_state(batch_data, "test")
        step_output = self.adaptive_test_step(carry_state)
        return self._finalize_adaptive_step_output(
            "test",
            batch_data,
            carry_state,
            step_output,
        )

    def validation_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Dispatch one validation step through the adaptive step hook."""

        carry_state = self.initialize_carry_state(batch_data, "validation")
        step_output = self.adaptive_validation_step(carry_state)
        return self._finalize_adaptive_step_output(
            "validation",
            batch_data,
            carry_state,
            step_output,
        )

    def adaptive_train_step(
        self,
        carry_state: CarryState,
        batch_idx: int,
    ) -> AdaptiveComputationStepOutput:
        """
        Run one adaptive-computation training step.

        Subclasses should override this hook when they want the
        ``AdaptiveComputationTrainer`` base to handle carry-state propagation,
        metric-manager updates, and scalar metric extraction automatically.
        """

        raise NotImplementedError(
            "Override adaptive_train_step() or train_step() in your "
            "AdaptiveComputationTrainer subclass."
        )

    def adaptive_test_step(
        self,
        carry_state: CarryState,
    ) -> AdaptiveComputationStepOutput:
        """
        Run one adaptive-computation test step.

        Subclasses should override this hook when they want the
        ``AdaptiveComputationTrainer`` base to handle carry-state propagation,
        metric-manager updates, and scalar metric extraction automatically.
        """

        raise NotImplementedError(
            "Override adaptive_test_step() or test_step() in your "
            "AdaptiveComputationTrainer subclass."
        )

    def adaptive_validation_step(
        self,
        carry_state: CarryState,
    ) -> AdaptiveComputationStepOutput:
        """
        Run one adaptive-computation validation step.

        Subclasses should override this hook when they want the
        ``AdaptiveComputationTrainer`` base to handle carry-state propagation,
        metric-manager updates, and scalar metric extraction automatically.
        """

        raise NotImplementedError(
            "Override adaptive_validation_step() or validation_step() in your "
            "AdaptiveComputationTrainer subclass."
        )

    def _finalize_adaptive_step_output(
        self,
        split: str,
        batch_data: dict[str, Any],
        carry_state: CarryState,
        step_output: AdaptiveComputationStepOutput,
    ) -> dict[str, float]:
        """Convert a structured ACT step output into scalar step metrics."""

        resolved_carry_state = step_output.carry_state or carry_state
        batch_updates = resolved_carry_state.as_batch_updates()
        batch_updates.update(step_output.batch_data_updates)
        if batch_updates:
            batch_data.update(batch_updates)

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
