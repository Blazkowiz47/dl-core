"""
Base accelerator class for compute abstraction.

Provides unified interface for different compute backends (CPU, single GPU, multi-GPU).
"""

import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_core.core.config_metadata import config_field


class BaseAccelerator(ABC):
    """
    Abstract base class for all accelerators.

    Accelerators handle compute-specific logic like:
    - Device placement
    - Distributed training (DDP)
    - Mixed precision training
    - Gradient accumulation
    - Checkpoint saving/loading
    """

    CONFIG_FIELDS = [
        config_field(
            "mixed_precision",
            "str | None",
            "Mixed-precision mode. Common values are 'fp16' and 'bf16'.",
            default=None,
        ),
        config_field(
            "seed",
            "int",
            "Accelerator seed used for distributed and worker reproducibility.",
            default=42,
        ),
        config_field(
            "gradient_accumulation_steps",
            "int",
            "Number of backward passes to accumulate before stepping.",
            default=1,
        ),
        config_field(
            "max_grad_norm",
            "float | None",
            "Clip gradients to this norm after accumulation.",
            default=None,
        ),
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize accelerator.

        Args:
            config: Accelerator configuration dict
        """
        self.config = config
        self.mixed_precision = config.get("mixed_precision")
        self.seed = config.get("seed", 42)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", None)
        if self.max_grad_norm is not None:
            self.max_grad_norm = float(self.max_grad_norm)
        self.use_distributed = False  # Override to True in MultiGPUAccelerator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Gradient accumulation counter
        self.accumulation_counter = 0

        # Mixed precision scaler (set to None by default, subclasses initialize if needed)
        self.scaler = None

        # Determine autocast dtype from mixed_precision config
        self._autocast_dtype: Optional[torch.dtype] = self._get_autocast_dtype()
        self._optimizer_step_handlers: list[
            Callable[[Optimizer, nn.Module | None], None]
        ] = []
        self.ema_manager: Any | None = None

    def _get_autocast_dtype(self) -> Optional[torch.dtype]:
        """
        Map mixed_precision config string to torch dtype.

        Returns:
            torch.float16 for "fp16", torch.bfloat16 for "bf16", None otherwise
        """
        if self.mixed_precision == "fp16":
            return torch.float16
        elif self.mixed_precision == "bf16":
            return torch.bfloat16
        return None

    def autocast_context(self) -> ContextManager:
        """
        Return autocast context manager for mixed precision forward/backward pass.

        Subclasses should override this to provide device-specific autocast.
        Base implementation returns nullcontext (no-op).

        Returns:
            Context manager for autocast (or nullcontext if mixed precision disabled)

        Example:
            with self.accelerator.autocast_context():
                output = model(input)
                loss = criterion(output, target)
                self.accelerator.backward(loss)
        """
        return nullcontext()

    @abstractmethod
    def prepare(
        self,
        models: Dict[str, Any] | None = None,
        optimizers: Dict[str, Optimizer] | None = None,
        criterions: Dict[str, Any] | None = None,
        schedulers: Dict[str, Any] | None = None,
        dataloaders: Dict[str, DataLoader | None] | None = None,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Prepare training components with device placement and wrapping.

        This method handles device placement for all training components.
        For multi-GPU, it wraps the model with DDP and adds DistributedSampler to dataloaders.

        Args:
            model: Main model (single model only)
            optimizers: Dictionary of optimizers (e.g., {'main': opt} or {'backbone': opt1, 'head': opt2})
            criterions: Dictionary of loss criterions (e.g., {'crossentropy': ce, 'arcface': af})
            schedulers: Dictionary of learning rate schedulers
            dataloaders: Dictionary of dataloaders (e.g., {'train': train_loader, 'test': test_loader})

        Returns:
            Tuple of (model, optimizers_dict, criterions_dict, schedulers_dict, dataloaders_dict)
            All dictionaries maintain the same keys as input.

        Example:
            model, optimizers, criterions, schedulers, dataloaders = accelerator.prepare(
                model=self.model,
                optimizers={'backbone': opt1, 'head': opt2},
                criterions={'crossentropy': ce_loss, 'arcface': af_loss},
                schedulers={'backbone': sched1, 'head': sched2},
                dataloaders={'train': train_loader, 'test': test_loader}
            )
        """
        pass

    @abstractmethod
    def backward(
        self, loss: torch.Tensor, model: Optional[torch.nn.Module] = None
    ) -> None:
        """
        Perform backward pass with mixed precision support.

        Args:
            loss: Loss tensor
            model: Optional model for gradient accumulation context (needed for DDP)
        """
        pass

    def optimizer_step(self, optimizer: Optimizer, model: nn.Module) -> bool:
        """
        Perform optimizer step with gradient accumulation and mixed precision.

        This method handles gradient accumulation by only stepping the optimizer
        every N backward passes (where N = gradient_accumulation_steps).

        NOTE: The accumulation counter is incremented in backward(), not here.
        This ensures that with multiple optimizers, all of them step at the same time.

        Args:
            optimizer: Optimizer to step
            model: Model to clip gradients for
        """
        self.accumulation_counter += 1
        if self.accumulation_counter == self.gradient_accumulation_steps:
            self.accumulation_counter = 0
            if self.max_grad_norm is not None:
                self.clip_gradients(model)
            if self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            self.on_optimizer_step(optimizer, model)
            optimizer.zero_grad()
            return True

        return False

    def set_ema_manager(self, ema_manager: Any | None) -> None:
        """
        Attach or clear an EMA manager.

        Args:
            ema_manager: Object exposing an ``update()`` method, or ``None``
        """
        self.ema_manager = ema_manager

    def register_optimizer_step_handler(
        self, handler: Callable[[Optimizer, nn.Module | None], None]
    ) -> None:
        """
        Register a hook called after successful optimizer steps.

        Args:
            handler: Callback invoked with ``(optimizer, model)``
        """
        self._optimizer_step_handlers.append(handler)

    def on_optimizer_step(
        self, optimizer: Optimizer, model: nn.Module | None = None
    ) -> None:
        """
        Hook called only when an optimizer step is actually executed.

        Args:
            optimizer: Optimizer that just stepped
            model: Model associated with the step
        """
        if self.ema_manager is not None and hasattr(self.ema_manager, "update"):
            self.ema_manager.update()

        for handler in self._optimizer_step_handlers:
            handler(optimizer, model)

    def clip_gradients(self, model: nn.Module) -> None:
        """
        Clip gradients by norm.

        Clips gradients of the model parameters to prevent exploding gradients.
        Only called when max_grad_norm is set in config.

        Args:
            model: Model to clip gradients for
        """
        if self.max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), self.max_grad_norm)

    @abstractmethod
    def get_device(self) -> torch.device:
        """
        Get device for tensor placement.

        Returns:
            Device to place tensors on
        """
        pass

    def to_device(self, batch: Dict[str, Any] | torch.Tensor) -> Dict[str, Any] | torch.Tensor:
        """
        Move batch data to device, handling nested structures.

        Args:
            batch: Dictionary with batch data

        Returns:
            Batch with tensors moved to device
        """
        device = self.get_device()
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)

        def move_to_device(obj):
            """Recursively move tensors to device, handling nested structures."""
            if isinstance(obj, torch.Tensor):
                return obj.to(device, non_blocking=True)
            elif isinstance(obj, list):
                return [move_to_device(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            else:
                # Return non-tensor objects as-is (strings, ints, etc.)
                return obj

        for key, value in batch.items():
            batch[key] = move_to_device(value)

        return batch

    @abstractmethod
    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Unwrap model for checkpoint saving.

        Returns raw model state (removes DDP wrapper if present).
        Needed because DDP adds 'module.' prefix to state dict keys.

        Args:
            model: Wrapped model

        Returns:
            Unwrapped model (raw model without DDP wrapper)
        """
        pass

    @abstractmethod
    def get_accelerator_state(self) -> Dict[str, Any]:
        """
        Get accelerator-specific state for checkpoint.

        Returns state that needs to be preserved across training runs,
        such as mixed precision scaler state.

        Returns:
            Dictionary of accelerator state (empty dict if none)
        """
        pass

    @abstractmethod
    def load_accelerator_state(self, state: Dict[str, Any]) -> None:
        """
        Load accelerator-specific state from checkpoint.

        Restores state like mixed precision scaler from checkpoint.

        Args:
            state: Full checkpoint dictionary (may contain accelerator state)
        """
        pass

    @abstractmethod
    def is_main_process(self) -> bool:
        """
        Check if current process is main process.

        Used for logging/saving to avoid duplication in distributed training.

        Returns:
            True if main process, False otherwise
        """
        pass

    @abstractmethod
    def wait_for_everyone(self, message: str = "") -> None:
        """
        Synchronize all processes.

        No-op for single process training.
        """
        pass

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather and average tensor across all processes for metrics.

        This is used to compute accurate metrics across all GPUs by averaging
        the values from each process. Default implementation returns tensor as-is
        for single GPU accelerators. Multi-GPU accelerators should override this.

        Args:
            tensor: Tensor to gather and average

        Returns:
            Averaged tensor (for multi-GPU) or original tensor (for single GPU)
        """
        return tensor

    def set_sampler_epoch(self, epoch: int) -> None:
        """
        Set epoch for distributed samplers.

        This is needed for DistributedSampler to ensure different shuffling
        for each epoch while maintaining reproducibility. Default implementation
        is a no-op for non-distributed accelerators (CPU, single GPU).

        Multi-GPU accelerators should override this to call set_epoch() on
        their distributed samplers.

        Args:
            epoch: Current epoch number
        """
        pass  # No-op for non-distributed accelerators

    def cleanup(self) -> None:
        """
        Cleanup accelerator resources.

        This is called at the end of training to properly release resources.
        Default implementation is a no-op. Multi-GPU accelerators should
        override this to destroy the process group.
        """
        pass  # No-op for non-distributed accelerators
