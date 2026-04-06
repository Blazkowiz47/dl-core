import logging
import random
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from dl_core.core.base_transform import BaseTransform
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import AUGMENTATION_REGISTRY, SAMPLER_REGISTRY
from dl_core.utils import (
    memory_usage,
    seed_worker,
    set_seeds_local,
    time_execution,
)

# Module-level logger
logger = logging.getLogger(__name__)

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.gif"]


class DatasetGenerator(Dataset):
    """
    PyTorch Dataset wrapper for BaseWrapper instances.

    Simple adapter class that wraps a BaseWrapper to create a torch.utils.data.Dataset
    compatible interface for DataLoader.

    Args:
        data: List of file dicts to iterate over
        transform: Transform function to apply to each file dict
        **kwargs: Additional keyword arguments (currently unused)

    Returns dataset[i] as the result of transform(data[i]).
    """

    def __init__(self, data: list[Any], transform: Callable, **kwargs) -> None:
        self.data = data
        self.transform = transform
        self.kwargs = kwargs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[Any]:
        datapoint = self.data[index]
        return self.transform(datapoint)


class BaseWrapper(ABC):
    """
    Task-agnostic dataset wrapper with automatic split handling and strategy injection.

    Handles file discovery, auto-splitting, caching, augmentation, and sampling.
    Subclasses implement transform() to load and preprocess files.

    Extension points:
    - get_train/validation/test_files() for custom file discovery
    - get_files(class_id, split) for non-directory sources (Azure, S3, etc.)
    - transform(file_dict, split) for loading and preprocessing

    File format: list[dict] with {"path": Path, "label": int, ...}
    Strategies: augmentation (required), sampler (optional)
    """

    CONFIG_FIELDS = [
        config_field(
            "rdir",
            "str",
            "Root dataset directory used by local filesystem datasets.",
        ),
        config_field(
            "classes",
            "list[str]",
            "Class ordering used to map labels to class indices.",
            default=[],
        ),
        config_field(
            "num_classes",
            "int",
            "Explicit number of classes when it cannot be inferred.",
        ),
        config_field(
            "batch_size",
            "int | dict[str, int]",
            "Batch size for each split.",
            default={"train": 32, "validation": 32, "test": 32},
        ),
        config_field(
            "num_workers",
            "int | dict[str, int]",
            "Number of DataLoader workers for each split.",
            default={"train": 1, "validation": 1, "test": 1},
        ),
        config_field(
            "shuffle",
            "bool | dict[str, bool]",
            "Whether each split should shuffle its data.",
            default={"train": True, "validation": True, "test": True},
        ),
        config_field(
            "pin_memory",
            "bool | dict[str, bool]",
            "Whether each split pins host memory for faster GPU transfer.",
            default={"train": False, "validation": False, "test": False},
        ),
        config_field(
            "drop_last",
            "bool | dict[str, bool]",
            "Whether each split drops incomplete final batches.",
            default={"train": False, "validation": False, "test": False},
        ),
        config_field(
            "prefetch_factor",
            "int | dict[str, int | None]",
            "DataLoader prefetch factor for worker-based loading.",
            default={"train": None, "validation": None, "test": None},
        ),
        config_field(
            "seed",
            "int",
            "Dataset-level seed used for splitting and worker seeding.",
            default=42,
        ),
        config_field(
            "auto_split",
            "bool",
            "Automatically derive validation/test partitions when split files "
            "are not provided.",
            default=True,
        ),
        config_field(
            "validation_partition",
            "float",
            "Validation split ratio used by automatic splitting.",
            default=0.05,
        ),
        config_field(
            "test_split",
            "float",
            "Test split ratio used by automatic splitting.",
            default=0.1,
        ),
        config_field(
            "stratify",
            "bool",
            "Preserve class balance when automatic splitting is used.",
            default=True,
        ),
        config_field(
            "sample_splits",
            "dict[str, bool]",
            "Control which splits apply the configured sampler.",
            default={"train": True, "validation": True, "test": False},
        ),
        config_field(
            "augmentation",
            "dict | dict[str, dict]",
            "Augmentation config or split-specific augmentation configs.",
        ),
        config_field(
            "sampler",
            "dict | dict[str, dict]",
            "Optional sampler config or split-specific sampler configs.",
        ),
    ]

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Initialize BaseWrapper with configuration.

        Args:
            config: Configuration dict with the following keys:

                Dataset identification:
                - name: Dataset name (str, default: "")
                - rdir: Root directory for dataset files (str, optional)
                - num_classes: Number of classes (int, optional)

                DataLoader parameters (int or dict[str, int] for split-specific):
                - batch_size: Batch size (default: 32 for all splits)
                - num_workers: Number of data loading workers (default: 1)
                - shuffle: Whether to shuffle data (default: True for train, True for val/test)
                - pin_memory: Pin memory for faster GPU transfer (default: False)
                - drop_last: Drop incomplete batches (default: False)
                - prefetch_factor: DataLoader prefetch factor (default: None)

                Split configuration:
                - validation_split: Validation split ratio (float, default: 0.05)
                - test_split: Test split ratio (float, default: 0.1)
                - stratify: Use stratified splitting (bool, default: True)
                - sample_splits: Which splits to apply sampling to (dict, default: train only)

                Strategy injection:
                - augmentation: Augmentation config (dict or split-specific dict)
                - sampler: Sampler config (dict or split-specific dict)

                Other:
                - seed: Random seed (int, default: 42)
                - classes: Class labels/names (optional, dataset-specific)
        """
        config = config.get("dataset") or config
        self.kwargs = kwargs
        self.config = config
        self.name: str = config.get("name", "")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # DataLoader configuration
        self.batch_size: dict[str, int] = config.get(
            "batch_size", {"train": 32, "validation": 32, "test": 32}
        )
        self.num_workers: dict[str, int] = config.get(
            "num_workers", {"train": 1, "validation": 1, "test": 1}
        )
        self.shuffle: dict[str, bool] = config.get(
            "shuffle", {"train": True, "validation": True, "test": True}
        )
        self.pin_memory: dict[str, bool] = config.get(
            "pin_memory", {"train": False, "validation": False, "test": False}
        )
        self.drop_last: dict[str, bool] = config.get(
            "drop_last", {"train": False, "validation": False, "test": False}
        )
        self.prefetch_factor: dict[str, int | None] = config.get(
            "prefetch_factor", {"train": None, "validation": None, "test": None}
        )

        # General configuration
        self.num_classes: int | None = config.get("num_classes")
        self.seed: int = config.get("seed", 42)
        self.rdir: str | None = config.get("rdir")
        self.sample_splits: dict[str, bool] = config.get(
            "sample_splits",
            {"train": True, "validation": True, "test": False},
        )
        self.auto_split: bool = config.get("auto_split", True)
        self.augmentation: BaseTransform | None = None

        # Partition configuration
        self.validation_partition = config.get("validation_partition", 0.05)
        self.test_partition = config.get("test_split", 0.1)
        self.stratify = config.get("stratify", True)
        # Classes config (optional - not all dataset types need it)
        self.classes = config.get("classes", [])

        # File storage (loaded once, cached)
        self.files_list: dict[str, list[dict]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        self.sampled_files_list: dict[str, list[dict]] = {
            "train": [],
            "validation": [],
            "test": [],
        }

        # State variables
        self.current_epoch = 0

        self.class_sample_pointers = {}
        for split in ["train", "validation", "test"]:
            self.class_sample_pointers[split] = {}

        # Setup internal values
        self._make_internal_values_consistent()
        self._setup_augmentations()
        self._setup_sampler()

    # ============================================================================
    # INTERNAL METHODS
    # ============================================================================
    def _ensure_reproducibility(self) -> None:
        """Set seeds for reproducibility."""
        set_seeds_local(self.seed)

    def _make_internal_values_consistent(self) -> None:
        """
        Normalize scalar config values to dictionary format.

        Converts single values (int, object) to dict[str, value] format for
        split-specific configuration. Ensures all splits have explicit values.

        Example:
            batch_size: 32 -> {"train": 32, "validation": 32, "test": 32}
            num_workers: 4 -> {"train": 4, "validation": 4, "test": 4}

        Modifies self attributes in-place by creating _batch_size, _num_workers,
        _shuffle, _pin_memory, _drop_last, and _prefetch_factor attributes.
        """
        if not isinstance(self.batch_size, dict):
            self.batch_size = {
                "train": self.batch_size,
                "validation": self.batch_size,
                "test": self.batch_size,
            }
        if not isinstance(self.num_workers, dict):
            self.num_workers = {
                "train": self.num_workers,
                "validation": self.num_workers,
                "test": self.num_workers,
            }
        if not isinstance(self.shuffle, dict):
            self.shuffle = {
                "train": self.shuffle,
                "validation": self.shuffle,
                "test": self.shuffle,
            }
        if not isinstance(self.pin_memory, dict):
            self.pin_memory = {
                "train": self.pin_memory,
                "validation": self.pin_memory,
                "test": self.pin_memory,
            }
        if not isinstance(self.drop_last, dict):
            self.drop_last = {
                "train": self.drop_last,
                "validation": self.drop_last,
                "test": self.drop_last,
            }
        if not isinstance(self.prefetch_factor, dict):
            self.prefetch_factor = {
                "train": self.prefetch_factor,
                "validation": self.prefetch_factor,
                "test": self.prefetch_factor,
            }

    def _setup_augmentations(self) -> None:
        """
        Setup augmentation strategy from config.

        Reads augmentation config and initializes a single augmentation object from registry.
        The same augmentation is applied across all splits (train, validation, test).

        Sets self.augmentation to a single augmentation object (or None if not configured).

        Config format:
            augmentation:
                augmentation_name: {param1: val1, param2: val2, ...}

        Example:
            augmentation:
                albumentations:
                    height: 224
                    width: 224
                    normalize: true

        Can be called to re-initialize augmentations if config changes.
        """
        # Initialize augmentation strategy
        augmentation_config = self.config.get("augmentation", {})
        if not augmentation_config:
            self.logger.info(
                "No augmentation config provided, skipping augmentation setup"
            )
            return

        augmentation_name, aug_params = next(iter(augmentation_config.items()))
        if not isinstance(aug_params, dict):
            aug_params = {}

        self.augmentation = AUGMENTATION_REGISTRY.get(augmentation_name, **aug_params)
        self.logger.info(
            f"Using augmentation: {augmentation_name} with params: {aug_params}"
        )

    def _setup_sampler(self) -> None:
        """
        Setup sampling strategies from config.

        Reads sampler config and initializes sampler objects from registry.
        Supports both split-specific and shared sampler configs.

        Sets self.sampler dict with keys: 'train', 'validation', 'test'
        Each value is a BaseSampler instance or None.

        Config format:
            # Split-specific
            sampler:
                train: {sampler_name: {param1: val1, ...}}
                validation: {sampler_name: {param1: val1, ...}}

            # Shared (applies to all splits)
            sampler:
                sampler_name: {param1: val1, ...}
        """
        # Initialize sampling strategy
        self.sampler: dict[str, Any] = {}
        sampler_config = self.config.get("sampler", {})
        if (
            "train" in sampler_config
            or "validation" in sampler_config
            or "test" in sampler_config
        ):
            pass
        else:
            sampler_config_cp = sampler_config.copy()
            sampler_config = {
                "train": sampler_config_cp,
                "validation": sampler_config_cp,
                "test": sampler_config_cp,
            }

        for split in ["train", "validation", "test"]:
            if split not in sampler_config:
                self.sampler[split] = None
                self.sample_splits[split] = False
                self.logger.info(f"No sampler configured for {split}")
                continue

            split_sampler_config = sampler_config[split]
            if not split_sampler_config:
                self.sampler[split] = None
                self.sample_splits[split] = False
                self.logger.info(f"No sampler configured for {split}")
                continue

            sampler_name = list(split_sampler_config.keys())[0]
            sampler_params = split_sampler_config[sampler_name]
            if not isinstance(sampler_params, dict):
                sampler_params = {}
            self.sampler[split] = SAMPLER_REGISTRY.get(
                sampler_name, **sampler_params, seed=self.seed
            )
            self.logger.info(
                f"Using sampler for {split}: {sampler_name} with params: {sampler_params}"
            )

    def _get_file_list(self, split: str) -> list[dict]:
        """
        Internal wrapper that manages file caching and applies sampling.

        DO NOT override this method. Override get_file_list() instead.

        This wrapper handles:
        1. Calling the abstract get_file_list(split) method
        2. Caching file lists to avoid repeated loading
        3. Applying configured samplers (e.g., attack-balanced sampling)
        4. Validation and logging

        Args:
            split: One of 'train', 'validation', 'test'

        Returns:
            List of file dicts with sampling applied: [{"path": Path, "label": int, ...}, ...]
        """
        self._ensure_reproducibility()
        # Validate split name
        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'validation', or 'test'"
            )
        self.logger.debug(f"Getting file list for {split} split")

        # Load files only if not already loaded for this split
        if not self.files_list[split]:
            self.files_list[split] = self.get_file_list(split)

        files_list = self.files_list[split]

        # Apply sampler if configured (instantiate per-split for split-aware config)
        if self.sample_splits[split]:
            # Apply sampling
            if not self.sampled_files_list[split]:
                files_list = self.sampler[split].sample(files_list, split)
                self.logger.info(
                    f"Applied sampler to {split}: resulted in {len(files_list)} files"
                )

        self.sampled_files_list[split] = files_list
        return files_list

    def _get_stats(self, split: str) -> list[str]:
        """Get dataset statistics for logging and analysis.

        Returns:
            List of formatted statistics strings
        """
        stats = []

        stats.append(f"Dataset: {self.name}")
        stats.append(f"Root directory: {self.rdir}")
        stats.append(f"Batch size: {self.batch_size}")
        stats.append(f"Num workers: {self.num_workers}")

        try:
            data = self.sampled_files_list[split]
            stats.append(f"{split.capitalize()} samples: {len(data)}")

            dist = self.get_class_distribution(data)
            stats.append(f"{split.capitalize()} class distribution: {dist}")

        except Exception as e:
            self.logger.debug(f"Could not get stats for {split}: {e}")
            stats.append(f"{split.capitalize()} samples: N/A")

        return stats

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"rdir='{self.rdir}', "
            f"batch_size={self.batch_size}, "
            f"num_classes={self.num_classes})"
        )

    def _auto_generate_partitions(self) -> None:
        """Internal method to auto-generate dataset partitions if missing."""
        self.logger.info("Pre-loading all dataset splits into memory")
        for split in ["train", "validation", "test"]:
            _ = self.get_split(split)

        self.logger.info("Auto-creating missing dataset splits if needed")
        need_validation = (
            len(self.files_list["validation"]) == 0
        ) and self.validation_partition > 0
        need_test = (len(self.files_list["test"]) == 0) and self.test_partition > 0
        if not need_validation and not need_test:
            self.logger.info("All splits already available, no auto-splitting needed")
            return

        all_train_files = self._get_file_list("train")
        if need_validation:
            all_train_files, validation_files = self.perform_split(
                all_train_files,
                ratio=self.validation_partition,
                stratify=self.stratify,
            )
            self.files_list["validation"] = validation_files

        if need_test:
            all_train_files, test_files = self.perform_split(
                all_train_files,
                ratio=self.test_partition,
                stratify=self.stratify,
            )
            self.files_list["test"] = test_files

        self.files_list["train"] = all_train_files
        self.sampled_files_list = {
            "train": [],
            "validation": [],
            "test": [],
        }

        self.logger.info(
            f"After auto-splitting - train: {len(self.files_list['train'])}, "
            f"validation: {len(self.files_list['validation'])}, "
            f"test: {len(self.files_list['test'])}"
        )

    def _perform_split(
        self,
        files: list[dict],
        ratio: float,
        stratify: bool,
    ) -> tuple[list[dict], list[dict]]:
        """Split files into two parts based on ratio with optional stratification."""
        if not files or ratio <= 0.0 or ratio >= 1.0:
            return files, []

        # Extract labels for stratification
        labels = [data["label"] for data in files] if stratify else None

        part1, part2 = train_test_split(
            files,
            test_size=ratio,
            stratify=labels,
        )

        self.logger.debug(f"Split: part1={len(part1)}, part2={len(part2)}")

        return part1, part2

    @memory_usage("Memory usage while creating DataLoader:")
    def _get_split(
        self,
        split: str,
        batch_size: int | None = None,
        num_workers: int | None = None,
        shuffle: bool | None = None,
        pin_memory: bool | None = None,
        drop_last: bool | None = None,
        prefetch_factor: int | None = None,
    ) -> DataLoader | None:
        """Internal method to create DataLoader for the specified split.

        Args:
            split: Dataset split name ('train', 'validation', 'test')
            batch_size: Override default batch size
            num_workers: Override default num_workers
            shuffle: Override default shuffle behavior

        Returns:
            DataLoader for the split
        """
        self.logger.debug(f"Creating DataLoader for {split} split")

        # Get file list with sampling applied
        data = self._get_file_list(split)

        if not data:
            self.logger.warning(f"No data for {split} split")
            return None

        shuffle = shuffle or self.shuffle[split]
        batch_size = batch_size or self.batch_size[split]
        num_workers = num_workers or self.num_workers[split]
        pin_memory = pin_memory or self.pin_memory[split]
        drop_last = drop_last or self.drop_last[split]
        prefetch_factor = prefetch_factor or self.prefetch_factor[split]

        # Create transform function with split bound using partial
        transform_fn = partial(self.transform, split=split)

        actual_batch_size = batch_size or self.batch_size
        actual_num_workers = num_workers or self.num_workers

        self.logger.info(
            f"[{split}] DataLoader config: batch_size={actual_batch_size}, "
            f"num_workers={actual_num_workers}, shuffle={shuffle}, "
            f"prefetch_factor={self.prefetch_factor}"
        )

        return DataLoader(
            DatasetGenerator(data, transform_fn),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
            prefetch_factor=prefetch_factor,
            worker_init_fn=partial(seed_worker, base_seed=self.seed),
            generator=None,
        )

    # ============================================================================
    # PUBLIC METHODS
    # ============================================================================
    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch for epoch-aware sampling strategies.

        Stores the epoch number which can be used by subclasses for
        deterministic but varying sampling across epochs.

        Args:
            epoch: Current training epoch number
        """
        self.current_epoch = epoch
        self.logger.debug(f"Dataset epoch set to {epoch}")

    def scan_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> list[Path]:
        """Scan directory for files with specified extensions.

        Args:
            directory: Directory to scan
            extensions: File extensions (e.g., ['*.jpg', '*.png'])
            recursive: Whether to scan subdirectories

        Returns:
            Sorted list of file paths
        """
        if extensions is None:
            extensions = self.file_extensions

        directory = Path(directory)
        files = []

        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return files

        for file in directory.rglob("*") if recursive else directory.glob("*"):
            if any(
                file.match(ext) if str(ext).startswith("*") else file.suffix == ext
                for ext in extensions
            ):
                files.append(file)

        return sorted(files)

    def load_splits(self) -> None:
        """Pre-load all dataset splits into memory."""
        self.auto_generate_partitions()
        self.logger.info("Loading all dataset splits into memory")
        for split in ["train", "validation", "test"]:
            _ = self.get_split(split)

    def clear_cache(self, split: str | None = None) -> None:
        """Clear file cache for specific split or all splits.

        Args:
            split: Which split to clear ('train', 'validation', 'test', or None for all)
        """
        if split is None:
            # Clear all
            self.files_list = {"train": [], "validation": [], "test": []}
            self.logger.debug("Cleared all file caches")
        else:
            # Clear specific split
            self.files_list[split] = []
            self.logger.debug(f"Cleared {split} file cache")

    def refresh_dataset(self, split: str | None = None) -> None:
        if split is None:
            self.sampled_files_list = {
                "train": [],
                "validation": [],
                "test": [],
            }
        else:
            self.sampled_files_list[split] = []

    # ============================================================================
    # OPTIONAL METHODS (Can be overridden by subclasses)
    # ============================================================================
    def post_init(self) -> None:
        """Post-initialization hook for subclasses.

        Can be used to set up additional attributes after base init.
        For example,
        - Initialize additional caches
        - Preprocess config parameters
        - Generate file-lists and store in attributes
        - call auto_generate_partitions() to create splits
        """
        pass

    @time_execution("Dataset split performed in:")
    def perform_split(
        self,
        files: list[dict],
        ratio: float,
        stratify: bool,
    ) -> tuple[list[dict], list[dict]]:
        """
        Split files into two parts based on ratio with optional stratification.
        By default, calls the internal method.
        Can be overridden by subclasses for custom behavior.
        Args:
            files: List of file dicts to split
            ratio: Ratio for the second split (e.g., 0.2 for 20% of the first set)
            stratify: Whether to stratify based on labels
        Returns:
            Tuple of two lists: (part1, part2)
        """

        return self._perform_split(
            files,
            ratio=ratio,
            stratify=stratify,
        )

    def auto_generate_partitions(self) -> None:
        """
        Auto-generate dataset partitions if missing.
        By default, calls the internal method.
        Can be overridden by subclasses for custom behavior.
        In order to decide which files belong to which split, you can use:
        - self.files_list
        self.files_list has the following structure:
        {
            'train': list of train file dicts,
            'validation': list of validation file dicts,
            'test': list of test file dicts
        }

        Ensure to set the following attributes:
        - self.files_list['train']
        - self.files_list['validation']
        - self.files_list['test']
        """
        """Internal method to auto-generate dataset partitions if missing."""
        self._auto_generate_partitions()

    def get_split(
        self,
        split: str,
        batch_size: int | None = None,
        num_workers: int | None = None,
        shuffle: bool | None = None,
        pin_memory: bool | None = None,
        drop_last: bool | None = None,
        prefetch_factor: int | None = None,
    ) -> DataLoader | None:
        """Create DataLoader for the specified split.

        Args:
            split: Dataset split name ('train', 'validation', 'test')
            batch_size: Override default batch size
            num_workers: Override default num_workers
            shuffle: Override default shuffle behavior
            pin_memory: Override default pin_memory behavior
            drop_last: Override default drop_last behavior
            prefetch_factor: Override default prefetch_factor behavior

        Returns:
            DataLoader for the split

        Note:
            By default, calls the internal method.
            User can override for custom behavior.
            Use self._get_file_list() to leverage the internal implementation.
            Or use custom file list but set the self.files_list accordingly.
            self.files_list should  have the following structure:
            {
                'train': list of train file dicts,
                'validation': list of validation file dicts,
                'test': list of test file dicts
            }

            Ensure to return a DataLoader object.

        """
        return self._get_split(
            split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
        )

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Custom collate function for batching dataset samples.

        Handles multiple tensor types and structures:
        - Single tensors: Stacks into batch dimension
        - Lists of tensors: Stacks each tensor separately (multi-view)
        - Numeric types: Converts to tensors and stacks
        - Non-tensors (strings, paths, etc.): Returns as list
        - None values: Filtered out per key

        Args:
            batch: List of sample dicts from __getitem__

        Returns:
            Batched data dict with same keys as input samples

        Example:
            # Single tensor per sample
            batch = [{"data": tensor1, "label": 0}, {"data": tensor2, "label": 1}]
            -> {"data": stacked([tensor1, tensor2]), "label": tensor([0, 1])}

            # Multi-view (list of tensors per sample)
            batch = [{"data": [view1, view2]}, {"data": [view3, view4]}]
            -> {"data": [stacked([view1, view3]), stacked([view2, view4])]}
        """
        if not batch:
            return {}

        keys = batch[0].keys()
        collated = {}

        for key in keys:
            # 1. Filter out None values for this specific key
            values = [sample[key] for sample in batch if sample.get(key) is not None]

            # If the entire batch for this key is None, skip or set to None
            if not values:
                collated[key] = None
                continue

            first_val = values[0]

            # 2. Tensors
            if isinstance(first_val, torch.Tensor):
                collated[key] = torch.stack(values)

            # 3. Multi-view (List of Tensors)
            elif (
                isinstance(first_val, (list, tuple))
                and len(first_val) > 0
                and isinstance(first_val[0], torch.Tensor)
            ):
                num_views = len(first_val)
                # Ensure all entries have the same number of views before stacking
                if all(len(v) == num_views for v in values):
                    collated[key] = [
                        torch.stack([v[i] for v in values]) for i in range(num_views)
                    ]
                else:
                    # Fallback if view counts vary (e.g., some samples have 2 views, some 3)
                    collated[key] = values

            # 4. Numeric types (int, float)
            elif isinstance(first_val, (int, float)):
                collated[key] = torch.tensor(values)

            # 5. Fallback (Strings, Paths, etc.)
            else:
                collated[key] = values

        return collated

    def get_next_sample(
        self,
        label: int | str,
        split: str,
        wrap_around: bool = False,
        custom_filter: Callable[[dict, Any], bool] = lambda batch, classid: (
            batch["label"] == classid
        ),
    ) -> dict | None:
        """Get the next sample for a specific class label from the sampled file list.

        This method can be used by sampling strategies to retrieve samples of a specific class.
        It iterates through the sampled file list for the given split and returns the next sample
        that matches the specified class label. It maintains an internal pointer to ensure that
        subsequent calls return different samples in a round-robin fashion.

        Args:
            label: Class label to retrieve
            split: Dataset split ('train', 'validation', 'test')
            wrap_around: Whether to wrap around to the beginning of the list when the end is reached
            custom_filter: Optional function to apply additional filtering logic.
                It should take a file dict and the class label as input and return True if the
                sample matches the criteria.

        Returns:
            A file dict matching the class label, or None if no more samples are available
        """
        if split in self.sampled_files_list:
            data = self.sampled_files_list[split]
        else:
            data = self._get_file_list(split)
        if label not in self.class_sample_pointers[split]:
            self.class_sample_pointers[split][label] = -1

        if wrap_around and self.class_sample_pointers[split][label] + 1 >= len(data):
            self.class_sample_pointers[split][label] = -1

        for idx, d in enumerate(
            data, start=self.class_sample_pointers[split][label] + 1
        ):
            if custom_filter(d, label):
                self.class_sample_pointers[split][label] = idx
                return self.transform(d, split)
        return

    def get_class_distribution(
        self, files: list[dict], key: str = "label"
    ) -> dict[int, int]:
        """Calculate class distribution from file list.

        Args:
            files: List of file dicts

        Returns:
            Dictionary mapping class labels to counts
        """
        distribution = {}
        for data in files:
            distribution[data[key]] = distribution.get(data[key], 0) + 1
        return distribution

    def get_stats(self, split: str) -> list[str]:
        """Get dataset statistics for logging and analysis.

        Returns:
            List of formatted statistics strings
        """
        return self._get_stats(split)

    def get_cache(self, cache_path: Path) -> Any | None:
        """Load data from cache file.

        Args:
            cache_path: Path to cache file

        Returns:
            Cached data or None if cache doesn't exist
        """
        raise NotImplementedError("Subclasses must implement get_cache")

    def set_cache(self, cache_path: Path, data: Any) -> None:
        """Save data to cache file.

        Args:
            cache_path: Path to cache file
            data: Data to cache
        """
        raise NotImplementedError("Subclasses must implement set_cache")

    # ============================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # ============================================================================

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """File extensions to scan (e.g., ['*.jpg', '*.png']).

        Override to specify which file types to load.
        """
        raise NotImplementedError("Subclasses must implement file_extensions")

    @abstractmethod
    def get_file_list(self, split: str) -> list[dict]:
        """Get file list for given split with caching, auto-splitting, and sampling.

        Args:
            split: One of 'train', 'validation', 'test'

        Returns:
            List of file dicts: [{"path": Path, "label": int, ...}, ...]
            The dict structure is flexible and can contain any metadata needed.
            The DatasetGenerator and transform() method should handle the dict accordingly.
            PyTorch DataLoader will batch the dicts as needed.
            Hard Requirement: each dict must have a "path" key with the file path.
        """
        raise NotImplementedError("Subclasses must implement get_file_list")

    @abstractmethod
    def transform(self, file_dict: dict, split: str) -> dict[str, Any]:
        """Transform file dict to model input.

        Subclasses should:
        1. Load file from file_dict["path"]
        2. Apply wrapper-specific preprocessing (e.g., face cropping)
        3. Call self.augmentation(data, split) for augmentation
        4. Return dict with required keys

        Args:
            file_dict: File metadata dict with keys:
                - "path": Path to file
                - "label": int class label
                - any other metadata
            split: Dataset split ('train', 'validation', 'test')

        Returns:
            Dictionary with keys:
            - "data": torch.Tensor (loaded and augmented data)
            - "label": int or torch.Tensor (class label)
            - "path": str (file path)
            - any other custom keys
        """
        raise NotImplementedError("Subclasses must implement transform")


class TextSequenceWrapper(BaseWrapper):
    """
    Base wrapper for tokenized text and sequence datasets.

    This wrapper keeps the standard `BaseWrapper` contract but adds sequence-
    aware batching so variable-length token tensors can be padded cleanly in
    local, single-GPU, and multi-GPU setups.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)

        sequence_keys = self.config.get(
            "sequence_keys",
            ["input_ids", "attention_mask", "token_type_ids"],
        )
        if isinstance(sequence_keys, str):
            sequence_keys = [sequence_keys]
        self.sequence_keys = set(sequence_keys)
        self.sequence_padding_values = {
            str(key): value
            for key, value in self.config.get("sequence_padding_values", {}).items()
        }

    @property
    def file_extensions(self) -> list[str]:
        """
        Return default text-oriented file extensions.

        Subclasses can override this when they scan a specific on-disk format.
        """

        return ["*.txt", "*.json", "*.jsonl", "*.csv", "*.tsv"]

    def _pad_sequence_values(
        self,
        values: list[torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        """Pad variable-length tensors for a sequence-oriented batch key."""

        if not values:
            raise ValueError("Cannot pad an empty list of sequence tensors")

        if all(value.shape == values[0].shape for value in values):
            return torch.stack(values)

        if values[0].ndim == 0:
            return torch.stack(values)

        padding_value = self.sequence_padding_values.get(key, 0)
        return pad_sequence(values, batch_first=True, padding_value=padding_value)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad configured sequence keys while preserving the base collate rules."""

        if not batch:
            return {}

        keys = batch[0].keys()
        collated: dict[str, Any] = {}

        for key in keys:
            values = [sample[key] for sample in batch if sample.get(key) is not None]

            if not values:
                collated[key] = None
                continue

            first_val = values[0]

            if key in self.sequence_keys and isinstance(first_val, torch.Tensor):
                collated[key] = self._pad_sequence_values(values, key)
            elif isinstance(first_val, torch.Tensor):
                collated[key] = torch.stack(values)
            elif (
                isinstance(first_val, (list, tuple))
                and len(first_val) > 0
                and isinstance(first_val[0], torch.Tensor)
            ):
                num_views = len(first_val)
                if all(len(v) == num_views for v in values):
                    collated[key] = [
                        torch.stack([v[i] for v in values]) for i in range(num_views)
                    ]
                else:
                    collated[key] = values
            elif isinstance(first_val, (int, float)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = values

        return collated


class AdaptiveComputationDataset(BaseWrapper):
    """
    Base wrapper for adaptive-time computation datasets with class sample streams.

    In addition to the standard dataset contract, this wrapper caches split data
    by class so ACT-style trainers can request the next sample for a label
    without rescanning the full split on every carry-state update.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.class_stream_key = self.config.get("class_stream_key", "label")
        class_stream_shuffle = self.config.get(
            "class_stream_shuffle",
            {"train": True, "validation": False, "test": False},
        )
        if not isinstance(class_stream_shuffle, dict):
            class_stream_shuffle = {
                "train": bool(class_stream_shuffle),
                "validation": bool(class_stream_shuffle),
                "test": bool(class_stream_shuffle),
            }
        self.class_stream_shuffle: dict[str, bool] = {
            split: bool(class_stream_shuffle.get(split, False))
            for split in ["train", "validation", "test"]
        }
        self.class_streams: dict[str, dict[Any, list[dict[str, Any]]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        self.class_stream_positions: dict[str, dict[Any, int]] = {
            "train": {},
            "validation": {},
            "test": {},
        }

    def _class_stream_seed(self, label: Any) -> int:
        """Build a stable per-class seed for epoch-aware class stream shuffling."""

        return self.seed + self.current_epoch + sum(ord(char) for char in str(label))

    def _build_class_streams(self, split: str) -> None:
        """Group the current split by class and reset stream positions."""

        records = self.sampled_files_list[split] or self._get_file_list(split)
        class_streams: dict[Any, list[dict[str, Any]]] = {}

        for record in records:
            label = record.get(self.class_stream_key, record.get("label"))
            class_streams.setdefault(label, []).append(record)

        if self.class_stream_shuffle[split]:
            for label, label_records in class_streams.items():
                rng = random.Random(self._class_stream_seed(label))
                rng.shuffle(label_records)

        self.class_streams[split] = class_streams
        self.class_stream_positions[split] = {
            label: 0 for label in class_streams
        }

    def _ensure_class_streams(self, split: str) -> None:
        """Build class streams lazily for the requested split."""

        if not self.class_streams[split]:
            self._build_class_streams(split)

    def set_epoch(self, epoch: int) -> None:
        """Refresh epoch-aware class stream order when the epoch changes."""

        super().set_epoch(epoch)
        for split in ["train", "validation", "test"]:
            self.class_streams[split] = {}
            self.class_stream_positions[split] = {}

    def clear_cache(self, split: str | None = None) -> None:
        """Clear cached files and adaptive class stream state."""

        super().clear_cache(split)

        if split is None:
            self.class_streams = {"train": {}, "validation": {}, "test": {}}
            self.class_stream_positions = {
                "train": {},
                "validation": {},
                "test": {},
            }
            return

        self.class_streams[split] = {}
        self.class_stream_positions[split] = {}

    def refresh_dataset(self, split: str | None = None) -> None:
        """Refresh sampled files and adaptive class stream state."""

        super().refresh_dataset(split)

        if split is None:
            self.class_streams = {"train": {}, "validation": {}, "test": {}}
            self.class_stream_positions = {
                "train": {},
                "validation": {},
                "test": {},
            }
            return

        self.class_streams[split] = {}
        self.class_stream_positions[split] = {}

    def reset_class_stream(self, split: str, label: Any | None = None) -> None:
        """Reset one class stream pointer or rebuild all streams for a split."""

        self._ensure_class_streams(split)

        if label is None:
            self._build_class_streams(split)
            return

        if label in self.class_stream_positions[split]:
            self.class_stream_positions[split][label] = 0

    def peek_next_class_sample(
        self,
        label: Any,
        split: str,
        *,
        wrap_around: bool = False,
        transform_sample: bool = True,
    ) -> dict[str, Any] | None:
        """Return the next class-specific sample without advancing the stream."""

        self._ensure_class_streams(split)
        label_records = self.class_streams[split].get(label, [])
        if not label_records:
            return None

        position = self.class_stream_positions[split].get(label, 0)
        if position >= len(label_records):
            if not wrap_around:
                return None
            position = 0

        record = label_records[position]
        if transform_sample:
            return self.transform(record, split)
        return record

    def get_next_class_sample(
        self,
        label: Any,
        split: str,
        *,
        wrap_around: bool = False,
        transform_sample: bool = True,
    ) -> dict[str, Any] | None:
        """Return the next class-specific sample and advance the stream pointer."""

        self._ensure_class_streams(split)
        label_records = self.class_streams[split].get(label, [])
        if not label_records:
            return None

        position = self.class_stream_positions[split].get(label, 0)
        if position >= len(label_records):
            if not wrap_around:
                return None
            position = 0

        self.class_stream_positions[split][label] = position + 1
        record = label_records[position]
        if transform_sample:
            return self.transform(record, split)
        return record


class FrameWrapper(BaseWrapper):
    """
    Wrapper for video datasets with efficient frame group caching.

    Caches expensive video group collection (e.g., downloading JSON metadata).
    Applies frames_per_video sampling before converting to file list.

    Extension points:
    - get_video_groups(split) → collect ALL frames grouped by video_id
    - convert_groups_to_files(groups, split) → convert to file list with labels

    Caching Options:
        | cache_video_groups | cache_sampled_video_groups | Behavior                                |
        |--------------------|----------------------------|-----------------------------------------|
        | ✅ True            | ✅ True                    | Cache both - fastest, most memory       |
        | ✅ True            | ❌ False                   | Cache raw, fresh sampling each epoch    |
        | ❌ False           | ✅ True                    | Fresh raw, cache sampled - saves memory |
        | ❌ False           | ❌ False                   | No caching - debugging/testing          |

    """

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)

        # Frame sampling configuration
        self.frames_per_video: dict[str, int] = {}
        self.cache_video_groups: bool = self.config.get("cache_video_groups", True)
        self.cache_sampled_video_groups: bool = self.config.get(
            "cache_sampled_video_groups", True
        )
        # Supports int (applies to all splits) or dict (split-specific)
        frames_per_video_raw = self.config.get("frames_per_video")

        if isinstance(frames_per_video_raw, dict):
            # Dict format: {train: 30, validation: 30, test: -1}
            train_default = frames_per_video_raw.get("train", 30)
            for split in ["train", "validation", "test"]:
                self.frames_per_video[split] = frames_per_video_raw.get(
                    split, train_default
                )
            self.logger.info(
                f"Frame sampling configured: {self.frames_per_video} frames per video"
            )
        elif isinstance(frames_per_video_raw, int):
            # Int format: applies to all splits
            self.frames_per_video = {
                "train": frames_per_video_raw,
                "validation": frames_per_video_raw,
                "test": frames_per_video_raw,
            }
            self.logger.info(
                f"Frame sampling configured: {frames_per_video_raw} frames per video for all splits"
            )
        else:
            self.frames_per_video = {
                "train": -1,
                "validation": -1,
                "test": -1,
            }
            self.logger.info("No frame sampling configured, loading all frames")

        self.dataset_frames_per_video = self.config.get("dataset_frames_per_video", {})
        if self.dataset_frames_per_video:
            self.logger.info(
                f"Per-dataset frame limits: {self.dataset_frames_per_video}"
            )
        else:
            self.logger.info("No per-dataset frame limits configured")

        self.logger.info("FrameWrapper initialized")

        self.video_groups: dict[str, dict[str, dict[str, list[str]]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        self.sampled_video_groups: dict[str, dict[str, dict[str, list[str]]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        self.rank_shard_sampled_files = config.get("rank_shard_sampled_files", False)

    def _get_video_groups(self, split: str) -> dict[str, dict[str, list[str]]]:
        """
        Get video groups with two-level caching.

        Implements configurable caching strategy:
        - cache_video_groups: Cache raw video groups (expensive to collect)
        - cache_sampled_video_groups: Cache sampled frames (after frame limiting)

        Args:
            split: Dataset split name ('train', 'validation', 'test')

        Returns:
            Sampled video groups: {dataset_name: {video_id: [frame_paths]}}
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Unknown split: {split}")

        if self.cache_video_groups:
            if not self.video_groups[split]:
                self.video_groups[split] = self.get_video_groups(split)
            raw_video_groups = self.video_groups[split]
        else:
            raw_video_groups = self.get_video_groups(split)

        self.logger.debug(
            f"Obtained raw video groups for {split}: {len(raw_video_groups)} datasets"
        )
        self.logger.debug(
            f"Obtained videos for {split}: {[(ds, len(videos)) for ds, videos in raw_video_groups.items()]}"
        )

        if self.cache_sampled_video_groups:
            if not self.sampled_video_groups[split]:
                self.sampled_video_groups[split] = self._sample_frames(
                    raw_video_groups, split
                )
            sampled_groups = self.sampled_video_groups[split]
        else:
            sampled_groups = self._sample_frames(raw_video_groups, split)

        self.logger.debug(
            f"Sampled video groups for {split}: {len(sampled_groups)} datasets"
        )

        return sampled_groups

    def _maybe_rank_shard_files(self, files: list[dict]) -> list[dict]:
        if not self.rank_shard_sampled_files:
            return files
        if not dist.is_initialized():
            return files
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return files[rank::world_size]

    @memory_usage("Memory usage while creating DataLoader:")
    def _get_split(
        self,
        split: str,
        batch_size: int | None = None,
        num_workers: int | None = None,
        shuffle: bool | None = None,
        pin_memory: bool | None = None,
        drop_last: bool | None = None,
        prefetch_factor: int | None = None,
    ) -> DataLoader | None:
        """Internal method to create DataLoader for the specified split.

        Args:
            split: Dataset split name ('train', 'validation', 'test')
            batch_size: Override default batch size
            num_workers: Override default num_workers
            shuffle: Override default shuffle behavior

        Returns:
            DataLoader for the split
        """
        self.logger.debug(f"Creating DataLoader for {split} split")

        # Get file list with sampling applied
        self._ensure_reproducibility()
        sampled_video_groups = self._get_video_groups(split)
        data = self.convert_groups_to_files(sampled_video_groups, split)
        self.files_list[split] = data  # For stats and external access
        self.logger.debug(
            f"Converted video groups to file list for {split}: {len(data)} files"
        )
        shuffle = shuffle or self.shuffle[split]

        # Apply sampler if configured (instantiate per-split for split-aware config)
        if self.sample_splits[split]:
            # Apply sampling
            if not self.sampled_files_list[split]:
                data = self.sampler[split].sample(data, split)
                if shuffle:
                    random.shuffle(data)
                data = self._maybe_rank_shard_files(data)
                self.logger.info(
                    f"Applied sampler to {split}: resulted in {len(data)} files"
                )
                self.sampled_files_list[split] = data
        else:
            if shuffle:
                random.shuffle(data)
            data = self._maybe_rank_shard_files(data)
            self.sampled_files_list[split] = data

        if not data:
            self.logger.warning(f"No data for {split} split")
            return None

        batch_size = batch_size or self.batch_size[split]
        num_workers = num_workers or self.num_workers[split]
        pin_memory = pin_memory or self.pin_memory[split]
        drop_last = drop_last or self.drop_last[split]
        prefetch_factor = prefetch_factor or self.prefetch_factor[split]

        # Create transform function with split bound using partial
        transform_fn = partial(self.transform, split=split)

        actual_batch_size = batch_size or self.batch_size
        actual_num_workers = num_workers or self.num_workers

        self.logger.info(
            f"[{split}] DataLoader config: batch_size={actual_batch_size}, "
            f"num_workers={actual_num_workers}, shuffle={shuffle}, "
            f"prefetch_factor={self.prefetch_factor}"
        )

        return DataLoader(
            DatasetGenerator(data, transform_fn),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
            prefetch_factor=prefetch_factor,
            worker_init_fn=partial(seed_worker, base_seed=self.seed),
            generator=None,
        )

    def _auto_generate_partitions(self) -> None:
        """
        Auto-generate dataset partitions if missing.
        By default, calls the internal method.
        Can be overridden by subclasses for custom behavior.
        In order to decide which files belong to which split, you can use:
        - self.files_list
        self.files_list has the following structure:
        {
            'train': list of train file dicts,
            'validation': list of validation file dicts,
            'test': list of test file dicts
        }

        Ensure to set the following attributes:
        - self.files_list['train']
        - self.files_list['validation']
        - self.files_list['test']
        """
        """Internal method to auto-generate dataset partitions if missing."""
        self._ensure_reproducibility()
        for split in ["train", "validation", "test"]:
            _ = self.get_split(split)

        self.logger.info("Auto-creating missing dataset splits if needed")
        need_validation = (
            len(self.video_groups["validation"]) == 0
        ) and self.validation_partition > 0

        need_test = (len(self.video_groups["test"]) == 0) and self.test_partition > 0
        if not need_validation and not need_test:
            self.logger.info("All splits already available, no auto-splitting needed")
            return

        # Split the video_groups
        old_video_groups = self.video_groups["train"]
        if len(old_video_groups) == 0:
            self.logger.warning("No training video groups available for auto-splitting")
            return

        videos = list(self.video_groups["train"].keys())

        if need_validation:
            train_videos, validation_videos = train_test_split(
                videos,
                test_size=self.validation_partition,
            )

            self.logger.debug(
                f"Split: part1={len(train_videos)}, part2={len(validation_videos)}"
            )
            self.video_groups["validation"] = {
                k: old_video_groups[k] for k in validation_videos
            }
            videos = train_videos

        if need_test:
            train_videos, test_videos = train_test_split(
                videos,
                test_size=self.test_partition,
            )
            self.logger.debug(
                f"Split: part1={len(train_videos)}, part2={len(test_videos)}"
            )
            self.video_groups["test"] = {k: old_video_groups[k] for k in test_videos}
            self.video_groups["train"] = {k: old_video_groups[k] for k in train_videos}

        self.sampled_video_groups = {
            "train": {},
            "validation": {},
            "test": {},
        }
        self.sampled_files_list = {
            "train": [],
            "validation": [],
            "test": [],
        }

    def _sample_frames(
        self, video_groups: dict[str, dict[str, list[str]]], split: str
    ) -> dict[str, dict[str, list[str]]]:
        """
        Sample limited frames from each video.

        Calls self.sample_frames() for actual sampling logic (can be overridden).
        Supports per-dataset frame limits via dataset_frames_per_video config.

        Args:
            video_groups: Nested dict of {dataset_name: {video_id: [frame_paths]}}
            split: Dataset split name ('train', 'validation', 'test')

        Returns:
            Sampled video groups with same structure, limited frames per video
        """
        # Get split-specific frame limit
        max_frames = self.frames_per_video.get(split, -1)

        # If no limit or negative limit, skip sampling (use all frames)
        if max_frames is None or max_frames < 0:
            total_frames = sum(
                len(frames)
                for videos in video_groups.values()
                for frames in videos.values()
            )
            self.logger.info(
                f"Frame sampling [{split}]: DISABLED - using all {total_frames:,} frames"
            )
            return video_groups

        sampled_groups = {}
        total_before = 0
        total_after = 0

        for dataset_name, videos in video_groups.items():
            # Get frame limit for this dataset (per-dataset override or split-specific default)
            dataset_max_frames = self.dataset_frames_per_video.get(
                dataset_name, max_frames
            )

            if (
                dataset_max_frames < 0
            ):  # No limit set or negative limit means keep all frames
                sampled_groups[dataset_name] = videos
                total_frames_dataset = sum(len(frames) for frames in videos.values())
                total_before += total_frames_dataset
                total_after += total_frames_dataset
                continue

            sampled_videos = {}
            for video_id, frames in videos.items():
                total_before += len(frames)

                if len(frames) <= dataset_max_frames:
                    sampled_videos[video_id] = frames
                    total_after += len(frames)
                else:
                    # Sample frames using the defined method
                    sampled_videos[video_id] = self.sample_frames(
                        frames, dataset_max_frames, split
                    )
                    total_after += len(sampled_videos[video_id])

            sampled_groups[dataset_name] = sampled_videos

        self.logger.info(
            f"Frame sampling [{split}]: {total_before:,} → {total_after:,} frames "
            f"({max_frames} per video)"
        )

        return sampled_groups

    def refresh_frames(self, split: str | None = None):
        if split is None:
            self.sampled_video_groups = {
                "train": {},
                "validation": {},
                "test": {},
            }
        else:
            self.sampled_video_groups[split] = {}

    # ============================================================================
    # PUBLIC METHODS (can be overridden by subclasses)
    # ============================================================================
    def sample_frames(
        self, frames: list[str], max_frames: int, split: str
    ) -> list[str]:
        """
        Sample limited frames from each video.
        Args:
            frames: List of frame file paths
            max_frames: Maximum number of frames to sample
            split: Dataset split name ('train', 'validation', 'test')
        Returns:
            Sampled list of frame file paths
        """
        # If no limit or negative limit, skip sampling (use all frames)
        if max_frames is None or max_frames < 0:
            return frames

        if len(frames) <= max_frames:
            return frames

        epoch_seed = self.seed + self.current_epoch
        rng = random.Random(epoch_seed)
        return sorted(rng.sample(sorted(frames), max_frames))

    def get_file_list(self, split: str) -> list[dict]:
        """
        Not needed since we override _get_split directly.
        """
        raise NotImplementedError(
            "FrameWrapper uses _get_split directly; get_file_list is not implemented."
        )

    def clear_cache(self, split: str | None = None) -> None:
        """Clear both video groups and sampled files.

        Override BaseWrapper to also clear frame collection cache.

        Args:
            split: Which split to clear (None for all)
        """
        # Call parent to clear file lists and Level 2 flags
        super().clear_cache(split)

        if split is None:
            # Clear all frame collections (Level 1)
            self.video_groups = {"train": {}, "validation": {}, "test": {}}
            self.sampled_video_groups = {
                "train": {},
                "validation": {},
                "test": {},
            }
            self.files_list = {"train": [], "validation": [], "test": []}
            self.logger.debug("Cleared all video groups and file caches")
        else:
            # Clear specific frame collection (Level 1)
            self.video_groups[split] = {}
            self.sampled_video_groups[split] = {}
            self.files_list[split] = []
            self.logger.debug(f"Cleared {split} video groups and file cache")

    # ============================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # ============================================================================
    @abstractmethod
    def get_video_groups(self, split: str) -> dict[str, dict[str, list[str]]]:
        """Collect ALL frames grouped by dataset and video_id for given split.

        Returns ALL available frames - no sampling at this level.
        Frame limiting should be done by FrameSampler strategy.

        Return empty dict if split doesn't exist - BaseWrapper will auto-split.

        Args:
            split: 'train', 'validation', or 'test'

        Returns:
            {
                "dataset1": {"video_id1": ["frame1.jpg", ...], ...},
                "dataset2": {...}
            }
        """
        raise NotImplementedError("Subclasses must implement get_video_groups")

    @abstractmethod
    def convert_groups_to_files(
        self,
        video_groups: dict[str, dict[str, list[str]]],
        split: str,
    ) -> list[dict]:
        """Convert video groups to file list with labels and metadata.

        Should include video_id in metadata for frame sampling.

        Args:
            video_groups: Output from get_video_groups() with ALL frames
            split: Split name

        Returns:
            List of file dicts with keys:
            - "path": Path to frame
            - "label": int class label
            - "video_id": str (for frame sampling)
            - any other metadata
        """
        raise NotImplementedError("Subclasses must implement convert_groups_to_files")
