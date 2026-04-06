import math

import torch
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_scheduler
from torch.optim.lr_scheduler import LRScheduler


@register_scheduler("cosinewithwarmup")
class CosineWithWarmupLR(LRScheduler):
    """
    Cosine decay with linear warmup scheduler.

    Matches the behavior of cosine_schedule_with_warmup_lr_lambda in the repo.

    Args:
        optimizer: Torch optimizer.
        num_warmup_steps: Number of warmup steps (linear from 0 to base_lr).
        num_training_steps: Total number of training steps (after warmup decay to min_ratio).
        min_ratio: Minimum lr ratio (final lr = base_lr * min_ratio).
        num_cycles: Number of cosine cycles (default 0.5 -> single half-cycle).
        last_epoch: The index of last epoch (step). Default: -1 (initialization).
    Notes:
        - This scheduler expects you to call scheduler.step() every training step (not per epoch).
        - If you want different base LRs for multiple param groups, set those LRs on the optimizer
          before creating the scheduler (scheduler will read optimizer.param_groups to initialize base_lrs).
    """

    CONFIG_FIELDS = [
        config_field(
            "num_warmup_steps",
            "int",
            "Number of linear warmup steps before cosine decay begins.",
            required=True,
        ),
        config_field(
            "num_training_steps",
            "int",
            "Total number of scheduler steps across the full run.",
            required=True,
        ),
        config_field(
            "min_ratio",
            "float",
            "Final learning-rate ratio relative to the base LR.",
            default=0.0,
        ),
        config_field(
            "num_cycles",
            "float",
            "Number of cosine cycles after warmup.",
            default=0.5,
        ),
        config_field(
            "last_epoch",
            "int",
            "Initial scheduler step index when resuming.",
            default=-1,
        ),
    ]

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_ratio: float = 0.0,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        if num_training_steps < 1:
            raise ValueError("num_training_steps must be >= 1")
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        self.min_ratio = float(min_ratio)
        self.num_cycles = float(num_cycles)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate for each base_lr in self.base_lrs.
        _LRScheduler uses self.last_epoch as the current step index.
        """
        step = max(0, self.last_epoch)

        if step < self.num_warmup_steps:
            warmup_factor = float(step) / max(1, self.num_warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # After warmup: cosine decay
        progress = float(step - self.num_warmup_steps) / max(
            1, (self.num_training_steps - self.num_warmup_steps)
        )
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * 2.0 * self.num_cycles * progress))
        scale = self.min_ratio + max(0.0, (1.0 - self.min_ratio) * cos_decay)
        return [base_lr * scale for base_lr in self.base_lrs]
