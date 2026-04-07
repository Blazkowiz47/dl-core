"""Exponential moving average utilities for model parameters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Maintains EMA shadow weights for one or more models."""

    def __init__(
        self,
        models: Dict[str, nn.Module],
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        save_in_checkpoint: bool = True,
    ) -> None:
        """
        Initialize EMA manager.

        Args:
            models: Mapping of model keys to modules (unwrapped models preferred)
            decay: EMA decay in [0, 1)
            update_after_step: Delay EMA updates until this many optimizer steps
            update_every: Apply EMA every N optimizer steps
            save_in_checkpoint: Whether EMA state should be checkpointed
        """
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if update_every < 1:
            raise ValueError("EMA update_every must be >= 1")
        if update_after_step < 0:
            raise ValueError("EMA update_after_step must be >= 0")
        if not models:
            raise ValueError("EMA requires at least one model")

        self.models = models
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.save_in_checkpoint = save_in_checkpoint
        self.step_count = 0

        self.shadow_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.backup_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self._initialize_shadow_params()

    def _initialize_shadow_params(self) -> None:
        """Initialize EMA shadows from current model parameters."""
        for model_key, model in self.models.items():
            model_shadow: Dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if not param.is_floating_point():
                    continue
                model_shadow[name] = param.detach().clone().to(torch.float32)
            self.shadow_params[model_key] = model_shadow

    def update(self) -> bool:
        """
        Update EMA from current model parameters.

        Returns:
            True if an update was applied, otherwise False
        """
        self.step_count += 1

        if self.step_count <= self.update_after_step:
            return False
        if self.step_count % self.update_every != 0:
            return False

        with torch.no_grad():
            one_minus_decay = 1.0 - self.decay
            for model_key, model in self.models.items():
                shadow = self.shadow_params[model_key]
                for name, param in model.named_parameters():
                    if name not in shadow or not param.is_floating_point():
                        continue
                    param_data = param.detach().to(dtype=torch.float32)
                    shadow[name].lerp_(param_data, one_minus_decay)
        return True

    def copy_to_models(self) -> None:
        """Copy EMA weights into tracked models."""
        with torch.no_grad():
            for model_key, model in self.models.items():
                shadow = self.shadow_params[model_key]
                for name, param in model.named_parameters():
                    if name not in shadow or not param.is_floating_point():
                        continue
                    value = shadow[name].to(device=param.device, dtype=param.dtype)
                    param.copy_(value)

    def store(self) -> None:
        """Backup current model parameters for temporary EMA swapping."""
        self.backup_params = {}
        for model_key, model in self.models.items():
            backup: Dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if not param.is_floating_point():
                    continue
                backup[name] = param.detach().clone()
            self.backup_params[model_key] = backup

    def restore(self) -> None:
        """Restore model parameters from the latest backup."""
        if not self.backup_params:
            return

        with torch.no_grad():
            for model_key, model in self.models.items():
                backup = self.backup_params.get(model_key, {})
                for name, param in model.named_parameters():
                    if name not in backup or not param.is_floating_point():
                        continue
                    param.copy_(backup[name])
        self.backup_params = {}

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Temporarily swap model parameters to EMA values."""
        self.store()
        self.copy_to_models()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> Dict[str, Any]:
        """Serialize EMA resume state for checkpointing."""
        shadow_cpu: Dict[str, Dict[str, torch.Tensor]] = {}
        for model_key, model_shadow in self.shadow_params.items():
            shadow_cpu[model_key] = {
                name: tensor.detach().cpu() for name, tensor in model_shadow.items()
            }

        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "step_count": self.step_count,
            "shadow_params": shadow_cpu,
        }

    def model_state_dicts(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Build full drop-in model state dicts with EMA parameters applied.

        This keeps the original model buffers from ``model.state_dict()`` and
        only replaces matching parameter entries with EMA shadow values, so the
        result can be loaded directly with ``model.load_state_dict(...)``.
        """
        exported: Dict[str, Dict[str, torch.Tensor]] = {}
        for model_key, model in self.models.items():
            state_dict = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            }
            shadow = self.shadow_params.get(model_key, {})
            for name, tensor in shadow.items():
                if name not in state_dict:
                    continue
                state_dict[name] = tensor.detach().cpu().to(
                    dtype=state_dict[name].dtype
                )
            exported[model_key] = state_dict
        return exported

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore EMA state from checkpoint data."""
        self.decay = float(state_dict.get("decay", self.decay))
        self.update_after_step = int(
            state_dict.get("update_after_step", self.update_after_step)
        )
        self.update_every = int(state_dict.get("update_every", self.update_every))
        self.step_count = int(state_dict.get("step_count", self.step_count))

        shadow_params = state_dict.get("shadow_params")
        if not isinstance(shadow_params, dict):
            return

        for model_key, model in self.models.items():
            model_shadow = shadow_params.get(model_key)
            if not isinstance(model_shadow, dict):
                continue
            if model_key not in self.shadow_params:
                self.shadow_params[model_key] = {}

            device_by_name = {
                name: param.device for name, param in model.named_parameters()
            }
            for name, tensor in model_shadow.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                target_device = device_by_name.get(name, tensor.device)
                self.shadow_params[model_key][name] = tensor.detach().to(
                    device=target_device, dtype=torch.float32
                )
