"""Optimizer implementations."""

from torch.optim import Adam, AdamW, SGD

from dl_core.core import OPTIMIZER_REGISTRY

# Register PyTorch optimizers directly
OPTIMIZER_REGISTRY.register_class("adam", Adam)
OPTIMIZER_REGISTRY.register_class("adamw", AdamW)
OPTIMIZER_REGISTRY.register_class("sgd", SGD)

__all__ = [
    "Adam",
    "AdamW",
    "SGD",
]
