"""Scheduler implementations."""

from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

from dl_core.core import SCHEDULER_REGISTRY
from .cosinewithwarmup import CosineWithWarmupLR

# Register PyTorch schedulers directly with multiple names
SCHEDULER_REGISTRY.register_class("onecycle", OneCycleLR)
SCHEDULER_REGISTRY.register_class("onecyclelr", OneCycleLR)
SCHEDULER_REGISTRY.register_class("step", StepLR)
SCHEDULER_REGISTRY.register_class("steplr", StepLR)
SCHEDULER_REGISTRY.register_class("cosine", CosineAnnealingLR)
SCHEDULER_REGISTRY.register_class("cosineannealing", CosineAnnealingLR)

__all__ = [
    "OneCycleLR",
    "StepLR",
    "CosineAnnealingLR",
    "CosineWithWarmupLR",
]
