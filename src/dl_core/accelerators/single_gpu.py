"""Single GPU accelerator with mixed precision support."""

from contextlib import nullcontext
from typing import Any, ContextManager, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_core.core.base_accelerator import BaseAccelerator
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_accelerator


@register_accelerator("single_gpu")
class SingleGPUAccelerator(BaseAccelerator):
    """
    Single GPU accelerator.

    Features:
    - Automatic GPU selection (cuda if available, else cpu)
    - FP16 mixed precision support (torch.cuda.amp)
    - Gradient accumulation
    - GradScaler for mixed precision
    """

    CONFIG_FIELDS = BaseAccelerator.CONFIG_FIELDS + [
        config_field(
            "compile_models",
            "bool | list[str]",
            "Compile all models or only the listed model names in place.",
            default=False,
        ),
        config_field(
            "compile_mode",
            "str",
            "PyTorch compilation mode.",
            default="default",
        ),
    ]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compile_models = config.get("compile_models", False)
        if isinstance(compile_models, bool):
            self.compile_models: bool | set[str] = compile_models
        elif isinstance(compile_models, list) and all(
            isinstance(name, str) and name
            for name in compile_models
        ):
            self.compile_models = set(compile_models)
        else:
            raise TypeError(
                "compile_models must be a boolean or a list of model names"
            )
        self.compile_mode = str(config.get("compile_mode", "default"))
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(
                "compile_mode must be default, reduce-overhead, max-autotune, "
                "or max-autotune-no-cudagraphs"
            )

        # Log GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"Single GPU training on: {gpu_name}")

        # Mixed precision setup
        # Only FP16 requires GradScaler; BF16 does not need it
        if self.mixed_precision == "fp16" and torch.cuda.is_available():
            self.scaler = GradScaler()
            self.logger.info("Mixed precision: FP16 with GradScaler")
        elif self.mixed_precision == "bf16" and torch.cuda.is_available():
            self.logger.info("Mixed precision: BF16 (no GradScaler needed)")

    def autocast_context(self) -> ContextManager:
        """
        Return autocast context manager for mixed precision forward/backward pass.

        Supports both FP16 and BF16 mixed precision training.
        FP16 uses GradScaler for loss scaling, BF16 does not require it.

        Returns:
            torch.autocast context if mixed precision enabled, nullcontext otherwise
        """
        if self._autocast_dtype is not None and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=self._autocast_dtype)
        return nullcontext()

    def prepare(
        self,
        models: Dict[str, Any] | None = None,
        optimizers: Dict[str, Optimizer] | None = None,
        criterions: Dict[str, Any] | None = None,
        schedulers: Dict[str, Any] | None = None,
        dataloaders: Dict[str, DataLoader | None] | None = None,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Move model and criterions to GPU, return others unchanged.

        Optimizers, schedulers, and dataloaders don't need modification for single GPU.
        """
        if isinstance(self.compile_models, set):
            unknown_models = self.compile_models.difference(models or {})
            if unknown_models:
                unknown_names = ", ".join(sorted(unknown_models))
                raise ValueError(
                    f"compile_models contains unknown model names: {unknown_names}"
                )

        # Move model to device
        prepared_models = {}
        if models:
            for name, model in models.items():
                prepared_model = model.to(self.device)
                if self.compile_models is True or (
                    isinstance(self.compile_models, set)
                    and name in self.compile_models
                ):
                    prepared_model.compile(mode=self.compile_mode)
                    self.logger.info(
                        "Compiled model %s with mode=%s",
                        name,
                        self.compile_mode,
                    )
                prepared_models[name] = prepared_model

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
        self,
        loss: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
        finalize: bool = False,
    ) -> None:
        """
        Backward pass with mixed precision support.

        Args:
            loss: Loss tensor to backpropagate
            model: Optional model (not used for single GPU, included for API consistency)
            finalize: Unused on one GPU; accepted for API consistency
        """
        del finalize
        loss = loss / self.gradient_accumulation_steps

        if self.scaler is not None:
            # Mixed precision backward
            self.scaler.scale(loss).backward()
        else:
            # Standard backward
            loss.backward()

    def get_device(self) -> torch.device:
        """Return GPU device."""
        return self.device

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Return model as-is (no DDP on single GPU)."""
        return model

    def get_accelerator_state(self) -> Dict[str, Any]:
        """Return scaler state if using mixed precision."""
        if self.scaler is not None:
            return {"scaler_state_dict": self.scaler.state_dict()}
        return {}

    def load_accelerator_state(self, state: Dict[str, Any]) -> None:
        """Load scaler state if present."""
        if "scaler_state_dict" in state and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler_state_dict"])

    def is_main_process(self) -> bool:
        """Always main process for single GPU."""
        return True

    def wait_for_everyone(self, message: str = "") -> None:
        """No-op for single process."""
        self.logger.debug(f"wait_for_everyone called: {message} (no-op for single GPU)")
