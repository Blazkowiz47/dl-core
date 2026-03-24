"""CPU accelerator for debugging and testing."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_core.core.base_accelerator import BaseAccelerator
from dl_core.core.registry import register_accelerator


@register_accelerator("cpu")
class CPUAccelerator(BaseAccelerator):
    """
    CPU-only accelerator for debugging.

    Features:
    - No mixed precision (CPU doesn't support AMP)
    - Basic gradient accumulation
    - Always main process
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = torch.device("cpu")

    def prepare(
        self,
        models: Dict[str, Any] | None = None,
        optimizers: Dict[str, Optimizer] | None = None,
        criterions: Dict[str, Any] | None = None,
        schedulers: Dict[str, Any] | None = None,
        dataloaders: Dict[str, DataLoader | None] | None = None,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Move model and criterions to CPU, return others unchanged.
        """
        # Move model to device
        prepared_models = {}
        if models:
            for name, model in models.items():
                prepared_models[name] = model.to(self.device)

        # Move criterions to device
        prepared_criterions = {}
        if criterions:
            for name, criterion in criterions.items():
                if isinstance(criterion, nn.Module):
                    prepared_criterions[name] = criterion.to(self.device)
                else:
                    prepared_criterions[name] = criterion

        # Return dictionaries (empty dicts if None)
        return (
            prepared_models,
            optimizers or {},
            prepared_criterions,
            schedulers or {},
            dataloaders or {},
        )

    def backward(
        self, loss: torch.Tensor, model: Optional[torch.nn.Module] = None
    ) -> None:
        """
        Standard backward with gradient accumulation.

        Args:
            loss: Loss tensor to backpropagate
            model: Optional model (not used for CPU, included for API consistency)
        """
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

    def get_device(self) -> torch.device:
        """Return CPU device."""
        return self.device

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Return model as-is (no wrapping on CPU)."""
        return model

    def get_accelerator_state(self) -> Dict[str, Any]:
        """No accelerator state for CPU."""
        return {}

    def load_accelerator_state(self, state: Dict[str, Any]) -> None:
        """No-op for CPU."""
        pass

    def is_main_process(self) -> bool:
        """Always main process for CPU."""
        return True

    def wait_for_everyone(self, message: str = "") -> None:
        """No-op for single process."""
        self.logger.debug(f"wait_for_everyone called. {message} (no-op for single GPU)")
