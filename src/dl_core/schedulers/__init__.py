"""Scheduler implementations."""

from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

from dl_core.core import SCHEDULER_REGISTRY
from .cosinewithwarmup import CosineWithWarmupLR

# Register PyTorch schedulers directly with multiple names
SCHEDULER_REGISTRY.regiter_class("onecycle", OneCycleLR)
SCHEDULER_REGISTRY.regiter_class("onecyclelr", OneCycleLR)
SCHEDULER_REGISTRY.regiter_class("step", StepLR)
SCHEDULER_REGISTRY.regiter_class("steplr", StepLR)
SCHEDULER_REGISTRY.regiter_class("cosine", CosineAnnealingLR)
SCHEDULER_REGISTRY.regiter_class("cosineannealing", CosineAnnealingLR)

__all__ = [
    "OneCycleLR",
    "StepLR",
    "CosineAnnealingLR",
    "CosineWithWarmupLR",
]
