"""Distributed training utilities for multi-GPU operations."""

import logging
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.distributed as dist
from torch.distributed import all_gather_object, all_gather

log = logging.getLogger(__name__)


def gather_via_pickle(data: Any, accelerator: Any) -> Union[List[Any], None]:
    """
    Gather data from all ranks using pickle files (CPU-based, no size limits).

    Each rank saves its data to a temporary pickle file, then rank 0 loads all files
    and returns the list of data from all ranks. Non-rank-0 processes return None.

    This approach avoids NCCL limitations (2GB tensor size limit), GPU memory pressure,
    and CPU/GPU device compatibility issues.

    Args:
        data: Any pickle-able Python object (dict, list, numpy array, tensors, etc.)
        accelerator: Accelerator instance for multi-GPU operations

    Returns:
        List of data from all ranks (only on rank 0), None on other ranks

    Example:
        >>> # Each GPU has different data
        >>> my_data = {'scores': np.array([1, 2, 3]), 'labels': np.array([0, 1, 0])}
        >>> gathered = gather_via_pickle(my_data, accelerator)
        >>> # Rank 0: gathered = [data_rank0, data_rank1, data_rank2, ...]
        >>> # Rank 1,2,3: gathered = None
    """

    # Log entry on ALL ranks
    log.debug(
        f"gather_via_pickle ENTRY: "
        f"accelerator={accelerator}, "
        f"use_distributed={getattr(accelerator, 'use_distributed', 'NO_ATTR')}"
    )

    if accelerator is None:
        log.warning("gather_via_pickle EARLY RETURN: accelerator is None")
        return [data]  # No accelerator provided

    if not accelerator.use_distributed:
        log.debug(
            f"gather_via_pickle EARLY RETURN: "
            f"use_distributed={accelerator.use_distributed}"
        )
        return [data]  # Single GPU or distributed not active

    # Log that we're proceeding with gathering
    rank = accelerator.global_rank
    world_size = accelerator.world_size

    accelerator.wait_for_everyone()

    log.debug(
        f"gather_via_pickle PROCEEDING: "
        f"world_size={world_size}, "
        f"will write to rank_{rank}.pkl"
    )

    # Create temp directory for pickle files (shared across ranks)
    temp_dir = Path(tempfile.gettempdir()) / "torch_gather_pickle"
    temp_dir.mkdir(exist_ok=True)

    # Each rank saves its data to a unique pickle file
    pickle_file = temp_dir / f"rank_{rank}.pkl"

    # Save this rank's data
    with open(pickle_file, "wb") as f:
        pickle.dump(data, f)

    # Wait for all ranks to finish writing
    accelerator.wait_for_everyone()

    # Only rank 0 loads all files
    if accelerator.is_main_process():
        gathered_data = []

        # Load data from all ranks
        for i in range(world_size):
            rank_file = temp_dir / f"rank_{i}.pkl"

            # Wait for file to exist (with timeout)
            timeout = 6000  # seconds
            start_time = time.time()
            while not rank_file.exists():
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Rank 0 timeout waiting for {rank_file}")
                time.sleep(0.1)

            # Load the data
            with open(rank_file, "rb") as f:
                rank_data = pickle.load(f)
            gathered_data.append(rank_data)

        # Always cleanup after reading
        for i in range(world_size):
            rank_file = temp_dir / f"rank_{i}.pkl"
            if rank_file.exists():
                rank_file.unlink()

        accelerator.wait_for_everyone()
        return gathered_data
    else:
        accelerator.wait_for_everyone()

    return None


def gather_tensors(tensor: torch.Tensor, accelerator: Any) -> torch.Tensor:
    """
    Gather tensors across all processes for multi-GPU training.

    Handles variable-batch-size tensors automatically by:
    1. Gathering batch sizes from all ranks (preserves non-batch dimensions)
    2. Padding to max batch size along dim=0
    3. Gathering padded tensors
    4. Removing padding and concatenating along dim=0

    Preserves tensor shape: [batch, *other_dims] → [total_batch, *other_dims]

    Supports CPU tensors with NCCL backend by temporarily moving to GPU for gathering.

    Args:
        tensor: Tensor to gather (shape: [batch_size, *other_dims]), can be CPU or GPU
        accelerator: Accelerator instance for multi-GPU operations

    Returns:
        Gathered and concatenated tensor from all ranks (shape: [total_batch_size, *other_dims])
        Returns tensor on original device (CPU or GPU).
        Returns original tensor if not in distributed mode.

    Example:
        >>> # GPU 0: tensor shape [100, 2]
        >>> # GPU 1: tensor shape [80, 2]
        >>> # GPU 2: tensor shape [120, 2]
        >>> gathered = gather_tensors(tensor, accelerator)
        >>> # Result shape: [300, 2] (concatenated along batch dimension)
    """
    if accelerator is None:
        return tensor

    if not (hasattr(accelerator, "world_size") and accelerator.world_size > 1):
        return tensor

    if not accelerator.use_distributed:
        return tensor

    # Track original device to restore at the end
    original_device = tensor.device
    move_back_to_cpu = False

    # If tensor is on CPU but using NCCL backend, move to GPU temporarily
    # NCCL requires CUDA tensors, so we move CPU tensors to GPU for gathering
    if tensor.device.type == "cpu" and dist.get_backend() == "nccl":
        tensor = tensor.to(accelerator.device)
        move_back_to_cpu = True

    # Store original shape for reconstruction
    original_shape = tensor.shape
    batch_size = tensor.shape[0]
    other_dims = tensor.shape[1:]  # All dimensions after batch

    # Get local batch size
    local_batch_size = torch.tensor([batch_size], device=tensor.device)

    # Gather batch sizes from all ranks
    batch_size_list = [
        torch.zeros(1, dtype=torch.long, device=tensor.device)
        for _ in range(accelerator.world_size)
    ]
    all_gather(batch_size_list, local_batch_size)
    batch_sizes = [int(s.item()) for s in batch_size_list]

    # Get max batch size for padding
    max_batch_size = max(batch_sizes)

    # Pad local tensor to max batch size along dim=0 if needed
    if batch_size < max_batch_size:
        padding_shape = [max_batch_size - batch_size] + list(other_dims)
        padding = torch.zeros(
            padding_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded_tensor = torch.cat([tensor, padding], dim=0)
    else:
        padded_tensor = tensor

    # Create list to hold gathered tensors from all ranks
    # Each tensor will have shape [max_batch_size, *other_dims]
    gathered_shape = [max_batch_size] + list(other_dims)
    gathered_list = [
        torch.zeros(gathered_shape, dtype=tensor.dtype, device=tensor.device)
        for _ in range(accelerator.world_size)
    ]

    all_gather(gathered_list, padded_tensor)

    # Remove padding from each gathered tensor and concatenate along dim=0
    unpadded_list = [
        gathered_list[i][: batch_sizes[i]] for i in range(len(gathered_list))
    ]
    result = torch.cat(unpadded_list, dim=0)

    # Move result back to CPU if we moved to GPU temporarily
    if move_back_to_cpu:
        result = result.cpu()

    return result


def gather_list(scores: List[float], accelerator: Any) -> List[float]:
    """
    Gather a list of scores from all ranks.

    Generic utility for gathering any list of float values across GPUs.
    Converts list → tensor → gather → list.

    Args:
        scores: Local list of scores
        accelerator: Accelerator instance for multi-GPU operations

    Returns:
        Gathered list on main process, empty list on other processes

    Example:
        >>> # GPU 0: scores = [0.1, 0.2, 0.3]
        >>> # GPU 1: scores = [0.4, 0.5]
        >>> gathered = gather_list(scores, accelerator)
        >>> # On rank 0: [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> # On other ranks: []
    """
    if accelerator is None or not hasattr(accelerator, "world_size"):
        return scores

    if accelerator.world_size <= 1:
        return scores

    if not accelerator.use_distributed:
        return scores

    # Convert to tensor
    tensor = torch.tensor(scores, device=accelerator.device)

    # Gather across all processes
    gathered = gather_tensors(tensor, accelerator)

    # Convert back to list (only meaningful on main process)
    if accelerator.is_main_process():
        return gathered.cpu().numpy().tolist()
    else:
        return []


def deep_merge_score_dicts(gathered_list: List[Dict], nested: bool = False) -> Dict:
    """
    Deep merge score dictionaries from all ranks.

    For flat dicts: {key: [values]} → merged with all values concatenated
    For nested dicts: {dim: {attack: {dataset: [values]}}} → recursively merged

    Args:
        gathered_list: List of dictionaries from all ranks
        nested: If True, handles nested dict structure (for attack scores)

    Returns:
        Merged dictionary with all scores concatenated

    Examples:
        Flat dict merge:
        >>> rank0 = {"dataset1": [0.1, 0.2], "dataset2": [0.5]}
        >>> rank1 = {"dataset1": [0.3, 0.4], "dataset3": [0.6]}
        >>> deep_merge_score_dicts([rank0, rank1])
        {"dataset1": [0.1, 0.2, 0.3, 0.4], "dataset2": [0.5], "dataset3": [0.6]}

        Nested dict merge:
        >>> rank0 = {"2d": {"print": {"dataset1": [0.1, 0.2]}}}
        >>> rank1 = {"2d": {"print": {"dataset1": [0.3]}, "cut": {"dataset2": [0.4]}}}
        >>> deep_merge_score_dicts([rank0, rank1], nested=True)
        {"2d": {"print": {"dataset1": [0.1, 0.2, 0.3]}, "cut": {"dataset2": [0.4]}}}
    """
    if not gathered_list:
        return {}

    merged = {}

    if not nested:
        # Flat dictionary: {key: [values]}
        for rank_dict in gathered_list:
            for key, values in rank_dict.items():
                if key not in merged:
                    merged[key] = []
                merged[key].extend(values)
    else:
        # Nested dictionary: {dim: {attack: {dataset: [values]}}}
        for rank_dict in gathered_list:
            for dim, dim_dict in rank_dict.items():
                if dim not in merged:
                    merged[dim] = {}
                for attack, attack_dict in dim_dict.items():
                    if attack not in merged[dim]:
                        merged[dim][attack] = {}
                    for dataset, values in attack_dict.items():
                        if dataset not in merged[dim][attack]:
                            merged[dim][attack][dataset] = []
                        merged[dim][attack][dataset].extend(values)

    return merged


def gather_score_dicts(
    score_dict: Dict, accelerator: Any, nested: bool = False
) -> Dict:
    """
    Gather and merge score dictionaries from all ranks.

    High-level convenience function that combines all_gather_object() with
    deep_merge_score_dicts().

    Args:
        score_dict: Local score dictionary to gather
        accelerator: Accelerator instance for multi-GPU operations
        nested: If True, handles nested dict structure (for attack scores)

    Returns:
        Merged dictionary on main process, empty dict on other processes

    Example:
        >>> # Each GPU has partial scores
        >>> local_scores = {"dataset1": [0.1, 0.2, 0.3]}
        >>> merged_scores = gather_score_dicts(local_scores, accelerator)
        >>> # On rank 0: {"dataset1": [0.1, 0.2, 0.3, 0.4, 0.5, ...]}
        >>> # On other ranks: {}
    """
    # Check if distributed
    if accelerator is None or not hasattr(accelerator, "world_size"):
        return score_dict

    if accelerator.world_size <= 1:
        return score_dict

    if not accelerator.use_distributed:
        return score_dict

    gathered_list = [None] * accelerator.world_size
    all_gather_object(gathered_list, score_dict)

    # Only main process merges and returns data
    if accelerator.is_main_process():
        return deep_merge_score_dicts(gathered_list, nested=nested)
    else:
        return {}
