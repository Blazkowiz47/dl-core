"""
Accelerator infrastructure for compute abstraction.

Provides unified interface for different compute backends:
- CPU: For debugging
- Single GPU: Main accelerator with mixed precision support
- Multi-GPU: DistributedDataParallel for distributed training
"""

# Import all accelerators to trigger registration
from .cpu import CPUAccelerator
from .single_gpu import SingleGPUAccelerator
from .multi_gpu import MultiGPUAccelerator


__all__ = [
    "CPUAccelerator",
    "SingleGPUAccelerator",
    "MultiGPUAccelerator",
]
