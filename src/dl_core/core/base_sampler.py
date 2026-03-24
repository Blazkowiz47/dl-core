"""Base class for sampling strategies."""

import logging
import random
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Base class for all sampling strategies.

    Samplers control how data samples are selected from a dataset,
    enabling strategies like balanced sampling, attack-balanced sampling, etc.
    """

    def __init__(self, **kwargs):
        """Initialize sampling strategy.

        Args:
            **kwargs: Configuration parameters for the sampler
        """
        self.config = kwargs
        self.seed = kwargs.get("seed", 2024)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.current_epoch = 0  # Track epoch for epoch-aware sampling

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for epoch-aware sampling."""
        self.current_epoch = epoch
        self.logger.debug(f"Sampler epoch set to {epoch}")

    def _get_epoch_rng(self) -> random.Random:
        """Get epoch-aware random number generator.

        Creates isolated RNG for deterministic but epoch-varying sampling.
        Does NOT reset global random state - uses isolated Random instance.
        """
        epoch_seed = self.seed + self.current_epoch
        return random.Random(epoch_seed)

    def sample(self, files: list[dict], split: str) -> list[dict]:
        """
        Sample files for the given split.

        Args:
            files: List of file dicts with keys: path, label, video_id, dataset, attack_type, etc.
            split: Dataset split ('train', 'test', 'validation')

        Returns:
            Sampled list of file dicts
        """

        self.logger.info(f"Sampling data for split: {split}")

        # Apply sampling logic
        sampled_files = self.sample_data(files, split)

        self.logger.info(
            f"Sampled {len(sampled_files)} files from {len(files)} available files."
        )

        return sampled_files

    @abstractmethod
    def sample_data(self, files: list[dict], split: str) -> list[dict]:
        """Sample data from the DataFrame.

        Args:
            files: List of file dicts
            split: Dataset split ('train' or 'test')

        Returns:
            Sampled list of file dicts
        """
        pass
