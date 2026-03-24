"""Multi-GPU accelerator with DistributedDataParallel support."""

import os
from contextlib import nullcontext
from typing import Any, ContextManager, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler  # pyright: ignore[reportPrivateImportUsage]
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dl_core.core.base_accelerator import BaseAccelerator
from dl_core.core.registry import register_accelerator
from dl_core.utils import seed_worker


@register_accelerator("multi_gpu")
class MultiGPUAccelerator(BaseAccelerator):
    """
    Multi-GPU accelerator using DistributedDataParallel.

    Features:
    - DDP for multi-GPU training
    - Distributed sampler for dataloaders
    - FP16 mixed precision support
    - Gradient accumulation
    - Process synchronization
    - Rank-aware logging

    Usage:
        Launch with: torchrun --nproc_per_node=N train.py
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Initialize distributed (works for both single and multi-GPU)
        self._init_distributed()

        # Get local rank from environment
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Set device
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        self.use_distributed = True

        # Mixed precision setup
        # Only FP16 requires GradScaler; BF16 does not need it
        if self.mixed_precision == "fp16":
            self.scaler = GradScaler("cuda")
            self.logger.info("Mixed precision: FP16 with GradScaler")
        elif self.mixed_precision == "bf16":
            self.logger.info("Mixed precision: BF16 (no GradScaler needed)")

        # Store samplers for epoch setting
        self.samplers = {}

    def autocast_context(self) -> ContextManager:
        """
        Return autocast context manager for mixed precision forward/backward pass.

        Supports both FP16 and BF16 mixed precision training.
        FP16 uses GradScaler for loss scaling, BF16 does not require it.

        Returns:
            torch.autocast context if mixed precision enabled, nullcontext otherwise
        """
        if self._autocast_dtype is not None:
            return torch.autocast("cuda", dtype=self._autocast_dtype)
        return nullcontext()

    def _init_distributed(self) -> None:
        """Initialize distributed process group for single or multi-GPU."""
        if not dist.is_initialized():
            # Check if we're in a torchrun/distributed environment
            if "RANK" not in os.environ:
                # Not launched with torchrun - set up for single GPU mode
                os.environ["RANK"] = "0"
                os.environ["WORLD_SIZE"] = "1"
                os.environ["LOCAL_RANK"] = "0"
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "12355"

            # Get backend from config or default to nccl
            backend = self.config.get("backend", "nccl")

            # Try to initialize process group with specified backend
            try:
                dist.init_process_group(backend=backend)
            except Exception as e:
                # If NCCL fails, try gloo as fallback
                if backend == "nccl":
                    self.logger.warning(f"Failed to initialize NCCL backend: {e}")
                    self.logger.warning("Falling back to gloo backend")
                    dist.init_process_group(backend="gloo")
                else:
                    raise

    def prepare(
        self,
        models: Dict[str, Any] | None = None,
        optimizers: Dict[str, Optimizer] | None = None,
        criterions: Dict[str, Any] | None = None,
        schedulers: Dict[str, Any] | None = None,
        dataloaders: Dict[str, DataLoader | None] | None = None,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Prepare components for DDP training.

        - Wraps model with DDP for distributed training
        - Moves criterions to device
        - Adds DistributedSampler to all dataloaders
        - Optimizers and schedulers returned unchanged
        """
        # Move model to device and wrap with DDP
        prepared_models = {}
        if models:
            for name, model in models.items():
                model = model.to(self.device)
                try:
                    prepared_models[name] = DDP(
                        model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=self.config.get(
                            "find_unused_parameters", False
                        ),
                    )
                except (RuntimeError, Exception) as e:
                    # If DDP fails with NCCL, try to reinitialize with gloo
                    if (
                        "NCCL" in str(e)
                        or "ncclUnhandledCudaError" in str(e)
                        or "DistBackendError" in str(type(e))
                    ):
                        self.logger.warning(f"DDP initialization failed with NCCL: {e}")
                        self.logger.warning(
                            "Reinitializing distributed with gloo backend..."
                        )
                        if os.environ.get("RANK", "0") != "0":
                            self.logger.error(
                                "Reinitialization with gloo can only be done on rank 0 process."
                            )
                            exit(1)

                        # Destroy the existing process group
                        if dist.is_initialized():
                            dist.destroy_process_group()

                        # Reinitialize with gloo
                        dist.init_process_group(backend="gloo")

                        # Try DDP again
                        prepared_models[name] = DDP(
                            model,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=self.config.get(
                                "find_unused_parameters", False
                            ),
                        )
                        self.logger.info("Successfully reinitialized with gloo backend")
                    else:
                        raise

        # Move criterions to device (no DDP wrapping for criterions)
        prepared_criterions = {}
        if criterions:
            for name, criterion in criterions.items():
                if isinstance(criterion, nn.Module):
                    prepared_criterions[name] = criterion.to(self.device)
                else:
                    prepared_criterions[name] = criterion

        # Add DistributedSampler to dataloaders
        prepared_dataloaders = {}
        if dataloaders:
            for name, dataloader in dataloaders.items():
                if dataloader is not None:
                    dataset = dataloader.dataset
                    # Get seed from config, default to 42 for reproducibility
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=self.world_size,
                        rank=self.global_rank,
                        shuffle=(name == "train"),  # Only shuffle training data
                        seed=self.seed,
                    )
                    # Store sampler for epoch setting
                    self.samplers[name] = sampler

                    # Create new dataloader with distributed sampler
                    # Preserve all critical parameters to avoid worker initialization issues
                    prepared_dataloaders[name] = DataLoader(
                        dataset,
                        batch_size=dataloader.batch_size,
                        sampler=sampler,
                        num_workers=dataloader.num_workers,
                        pin_memory=dataloader.pin_memory,
                        drop_last=dataloader.drop_last,
                        collate_fn=dataloader.collate_fn,
                        prefetch_factor=dataloader.prefetch_factor,
                        persistent_workers=(dataloader.num_workers > 0),
                        worker_init_fn=lambda worker_id: seed_worker(
                            worker_id, base_seed=self.seed
                        ),
                        generator=None,
                    )
                else:
                    prepared_dataloaders[name] = None

        # Return all components
        return (
            prepared_models,
            optimizers or {},
            prepared_criterions,
            schedulers or {},
            prepared_dataloaders,
        )

    def backward(self, loss: torch.Tensor, model: nn.Module | None = None) -> None:
        """
        Backward pass with mixed precision and gradient accumulation.

        Args:
            loss: Loss tensor to backpropagate
            model: Model to use for no_sync context (for gradient accumulation with DDP)
        """

        loss = loss / self.gradient_accumulation_steps
        if not isinstance(model, DDP):
            raise ValueError("Model must be a DDP instance for MultiGPUAccelerator")

        if self.accumulation_counter < self.gradient_accumulation_steps - 1:
            # Use no_sync to prevent gradient synchronization
            with model.no_sync():
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
        else:
            # Last accumulation step - allow gradient sync
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

    def get_device(self) -> torch.device:
        """Return local GPU device."""
        return self.device

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Unwrap DDP model to get raw model.

        DDP wraps the model and adds 'module.' prefix to state dict keys.
        This method returns the underlying model without the wrapper.

        Args:
            model: Potentially DDP-wrapped model

        Returns:
            Raw model without DDP wrapper
        """
        if isinstance(model, DDP):
            return model.module
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
        """Check if current process is rank 0."""
        return self.global_rank == 0

    def wait_for_everyone(self, message: str = "") -> None:
        """Synchronize all processes."""
        if message:
            self.logger.debug(f"Waiting for all processes: {message}")
        dist.barrier()
        if message:
            self.logger.debug(f"All processes synchronized: {message}")

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather and average tensor across all processes for metrics.

        This is used to compute accurate metrics across all GPUs by averaging
        the values from each process. Typically used for loss and accuracy metrics.

        Args:
            tensor: Tensor to gather and average (should be scalar or will be averaged)

        Returns:
            Averaged tensor (only valid on rank 0, others return original)
        """
        if not dist.is_initialized():
            return tensor

        # Ensure tensor is on the correct device and is a float
        tensor = tensor.detach().clone().to(self.device)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

        # All-reduce to sum across processes
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

        # Return scalar if original was scalar
        return tensor.squeeze()

    def set_sampler_epoch(self, epoch: int) -> None:
        """
        Set epoch for all distributed samplers.

        This ensures different shuffling for each epoch while maintaining
        reproducibility across runs with the same seed.

        Args:
            epoch: Current epoch number
        """
        for name, sampler in self.samplers.items():
            sampler.set_epoch(epoch)

    def cleanup(self) -> None:
        """
        Cleanup distributed resources.

        Properly destroys the process group to avoid resource leaks.
        Should be called at the end of training.
        """
        if dist.is_initialized():
            dist.destroy_process_group()
