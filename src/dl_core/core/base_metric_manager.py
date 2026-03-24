"""
Base abstract class for metric managers.

MetricManagers handle all data accumulation, distributed coordination,
and orchestration of individual metrics.

COMPUTATION MODES:
  - "gather": All ranks gather data to main rank, then compute
              Use for: EER, AUC, F1 (need global distributions)
  - "average": Each rank computes locally, then average results
               Use for: Accuracy, Loss (can be meaningfully averaged)
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist

from dl_core.core.base_accelerator import BaseAccelerator
from dl_core.utils.distributed_utils import gather_via_pickle


class BaseMetricManager(ABC):
    """
    Abstract base class for all metric managers.

    RESPONSIBILITIES:
    - Accumulate predictions/labels during training
    - Handle distributed data gathering
    - Orchestrate metric computation
    - Manage computation modes
    - Cache results for logging

    DOES NOT:
    - Implement metric-specific algorithms (that's in BaseMetric subclasses)
    - Perform metric calculations directly

    === ARCHITECTURE ===

    Data Flow:
        1. Training: update() called for each batch → accumulate data locally
        2. Epoch End: compute() called → gather data → compute metrics
        3. Logging: get_logs() called → return cached results

    Distributed Coordination:
        - Mode "gather":
          * ALL ranks call gather_via_pickle()
          * ALL ranks participate in barriers
          * Main rank computes, others return defaults
          * Used for metrics needing global data

        - Mode "average":
          * ALL ranks compute locally
          * ALL ranks call all_reduce()
          * All ranks get same averaged result
          * Used for simple metrics

    === USAGE ===

    class MyMetricManager(BaseMetricManager):
        def __init__(self, config, accelerator, trainer=None):
            # Set mode in config: config['mode'] = 'gather' or 'average'
            # Default is 'gather' if not specified
            super().__init__(config, accelerator, trainer)

        def setup_metrics(self):
            # Initialize pure function metrics
            for split in ["train", "validation", "test"]:
                self.metrics[split] = {
                    "accuracy": AccuracyMetric(),
                    "auc": AUCMetric(),
                }

    === DEADLOCK PREVENTION ===

    This architecture prevents NCCL deadlocks by ensuring:
    1. ALL ranks participate in collective operations
    2. Rank divergence happens AFTER collectives, not before
    3. Distributed logic centralized in manager, not scattered in metrics
    4. Clear mode definitions prevent accidental rank divergence

    Args:
        config: Configuration dictionary
        accelerator: Accelerator for distributed coordination
        trainer: Optional trainer reference for artifact management
    """

    def __init__(
        self,
        config: Dict[str, Any],
        accelerator: BaseAccelerator,
        trainer: Any = None,
    ):
        """
        Initialize the metric manager.

        Args:
            config: Configuration dictionary
                   - mode: Computation mode ("gather" or "average", default: "gather")
            accelerator: Accelerator instance for multi-GPU operations
            trainer: Trainer instance (optional, for plot generation)
        """
        self.config = config
        self.accelerator = accelerator
        self.trainer = trainer
        self.log = getLogger(f"{__name__}.{self.__class__.__name__}")
        self.current_epoch = 0
        self.name = "base"

        # Computation mode: "gather" or "average" (default: "gather")
        self.mode = config.get("mode", "gather")

        # Data storage (per split)
        # Structure: {split: {"predictions": [...], "labels": [...], "metadata": [...]}}
        self.accumulated_data: Dict[str, Dict[str, list]] = {
            "train": {"predictions": [], "labels": [], "metadata": []},
            "validation": {"predictions": [], "labels": [], "metadata": []},
            "test": {"predictions": [], "labels": [], "metadata": []},
        }

        # Metric instances (per split)
        # Structure: {split: {metric_name: BaseMetric instance}}
        self.metrics: Dict[str, Dict[str, Any]] = {}

        # Computed results cache (per split)
        # Structure: {split: {metric_name: float}}
        self.results_cache: Dict[str, Dict[str, float]] = {}
        self.diagnostics_cache: Dict[str, Dict[str, float]] = {
            "train": {},
            "validation": {},
            "test": {},
        }

        # Initialize metrics
        self.setup_metrics()

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch number.

        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch

    @abstractmethod
    def setup_metrics(self) -> None:
        """
        Setup the metrics that this manager will track.

        Should initialize metric objects and populate self.metrics dict.

        Example:
            for split in ["train", "validation", "test"]:
                self.metrics[split] = {
                    "accuracy": AccuracyMetric({"num_classes": 2}),
                    "auc": AUCMetric({"num_classes": 2}),
                }
        """
        pass

    def reset_metrics(self, split: str) -> None:
        """
        Clear accumulated data for a split.

        Args:
            split: Dataset split ('train', 'validation', or 'test')
        """
        if split in self.accumulated_data:
            self.accumulated_data[split] = {
                "predictions": [],
                "labels": [],
                "metadata": [],
            }
        if split in self.results_cache:
            self.results_cache[split] = {}
        if split in self.diagnostics_cache:
            self.diagnostics_cache[split] = {}

    def update(
        self,
        split: str,
        probabilities: torch.Tensor,
        batch_data: Dict[str, Any],
    ) -> None:
        """
        Accumulate predictions and labels during training.

        Called after each batch to store data for later metric computation.

        Args:
            split: Dataset split ('train', 'validation', or 'test')
            probabilities: Model output probabilities, shape (batch_size, num_classes)
            batch_data: Batch containing labels and metadata
        """
        labels = batch_data.get("label")
        if labels is None:
            return

        # Move to CPU and convert to numpy
        preds_np = probabilities.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Accumulate
        self.accumulated_data[split]["predictions"].append(preds_np)
        self.accumulated_data[split]["labels"].append(labels_np)

        # Optional: accumulate metadata for per-attack metrics
        self._accumulate_metadata(batch_data, split)

    def _accumulate_metadata(self, batch_data: Dict[str, Any], split: str) -> None:
        """
        Accumulate additional metadata (e.g., attack types, datasets).

        Subclasses can override to store domain-specific metadata.

        Args:
            batch_data: Batch data dictionary
            split: Dataset split
        """
        pass

    def compute(self, split: str) -> Dict[str, float]:
        """
        Compute metrics using the configured mode.

        This method:
        1. Prepares local data (concatenates accumulated batches)
        2. Uses self.mode to determine computation strategy
        3. Executes computation in appropriate mode
        4. Caches and returns results

        ALL RANKS must call this method to avoid deadlocks.

        Args:
            split: Dataset split ('train', 'validation', or 'test')

        Returns:
            Dictionary of computed metrics {metric_name: value}
        """
        if split not in self.metrics:
            return {}

        # Get accumulated data as numpy arrays
        local_data = self._prepare_local_data(split)

        # Compute metrics according to mode
        results = {}

        for metric_name, metric in self.metrics[split].items():
            if self.mode == "gather":
                metric_result = self._compute_gather_mode(
                    metric, local_data, metric_name
                )
            elif self.mode == "average":
                metric_result = self._compute_average_mode(
                    metric, local_data, metric_name
                )
            else:
                self.log.warning(
                    f"Unknown mode '{self.mode}' for {metric_name}, defaulting to gather"
                )
                metric_result = self._compute_gather_mode(
                    metric, local_data, metric_name
                )

            results.update(metric_result)

        # Cache results
        self.results_cache[split] = results
        return results

    def _prepare_local_data(self, split: str) -> Dict[str, Any]:
        """
        Convert accumulated lists to numpy arrays.

        Args:
            split: Dataset split

        Returns:
            Dictionary with concatenated predictions, labels, and metadata
        """
        data = self.accumulated_data[split]

        if not data["predictions"]:
            return {
                "predictions": np.array([]),
                "labels": np.array([]),
                "metadata": {},
            }

        return {
            "predictions": np.concatenate(data["predictions"], axis=0),
            "labels": np.concatenate(data["labels"], axis=0),
            "metadata": data["metadata"],  # May be empty or structured
        }

    def _compute_gather_mode(
        self, metric: Any, local_data: Dict[str, Any], metric_name: str
    ) -> Dict[str, float]:
        """
        Mode 1: Gather all data to main rank, compute there.

        ALL ranks participate in gathering. Non-main ranks return
        default values AFTER the collective operation completes.

        Args:
            metric: Metric instance to compute
            local_data: Local data from this rank
            metric_name: Name of metric (for logging)

        Returns:
            Computed metric dict (only meaningful on main rank)
        """
        # ALL ranks participate in gathering
        gathered_data = self._gather_data(local_data)

        # Early exit for non-main ranks AFTER collective op
        if not self.accelerator.is_main_process():
            # Return default value - this will be ignored by trainer
            return self._get_default_value(metric_name)

        # Main rank computes on gathered data
        try:
            metadata_dict = gathered_data.get("metadata", {})
            if isinstance(metadata_dict, dict):
                result = metric.compute(
                    predictions=gathered_data["predictions"],
                    labels=gathered_data["labels"],
                    **metadata_dict,
                )
            else:
                result = metric.compute(
                    predictions=gathered_data["predictions"],
                    labels=gathered_data["labels"],
                )
            return result
        except Exception as e:
            self.log.error(
                f"Error computing {metric_name} in gather mode: {e}", exc_info=True
            )
            return self._get_default_value(metric_name)

    def _compute_average_mode(
        self, metric: Any, local_data: Dict[str, Any], metric_name: str
    ) -> Dict[str, float]:
        """
        Mode 2: Compute locally on each rank, average results.

        ALL ranks compute and participate in averaging.

        Args:
            metric: Metric instance to compute
            local_data: Local data from this rank
            metric_name: Name of metric (for logging)

        Returns:
            Averaged metric dict (same value on all ranks)
        """
        # Each rank computes locally
        try:
            local_result = metric.compute(
                predictions=local_data["predictions"], labels=local_data["labels"]
            )
        except Exception as e:
            self.log.error(f"Error computing {metric_name} locally: {e}", exc_info=True)
            local_result = self._get_default_value(metric_name)

        # Average across ranks if distributed
        if self.accelerator.use_distributed:
            averaged_result = {}
            # Get device - default to cuda if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for key, value in local_result.items():
                value_tensor = torch.tensor([value], device=device, dtype=torch.float32)
                dist.all_reduce(value_tensor, op=dist.ReduceOp.AVG)
                averaged_result[key] = value_tensor.item()
            return averaged_result
        else:
            return local_result

    def _gather_data(self, local_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather data from all ranks using pickle-based gathering.

        ALL ranks must call this method.

        Args:
            local_data: Local data dict with predictions, labels, metadata

        Returns:
            Gathered data (only meaningful on main rank)
        """
        if not self.accelerator.use_distributed:
            return local_data

        # Use existing gather_via_pickle - ALL ranks participate
        gathered_list = gather_via_pickle(local_data, self.accelerator)

        # Only main rank merges
        if not self.accelerator.is_main_process():
            return {
                "predictions": np.array([]),
                "labels": np.array([]),
                "metadata": {},
            }

        # Merge gathered data
        all_predictions = []
        all_labels = []
        all_metadata: Dict[str, Any] = {}

        if gathered_list is not None:
            for rank_data in gathered_list:
                if rank_data["predictions"].size > 0:
                    all_predictions.append(rank_data["predictions"])
                    all_labels.append(rank_data["labels"])

                # Merge metadata if present
                if "metadata" in rank_data and rank_data["metadata"]:
                    all_metadata = self._merge_metadata(
                        all_metadata, rank_data["metadata"]
                    )

        return {
            "predictions": (
                np.concatenate(all_predictions, axis=0)
                if all_predictions
                else np.array([])
            ),
            "labels": (
                np.concatenate(all_labels, axis=0) if all_labels else np.array([])
            ),
            "metadata": all_metadata,
        }

    def _merge_metadata(self, existing: Dict, new: Dict) -> Dict:
        """
        Merge metadata from different ranks.

        Subclasses can override to implement domain-specific merging logic
        (e.g., concatenating per-attack lists).

        Args:
            existing: Existing merged metadata
            new: New metadata to merge

        Returns:
            Merged metadata dictionary
        """
        return existing

    def _get_default_value(self, metric_name: str) -> Dict[str, float]:
        """
        Get default value for a metric when computation fails or on non-main ranks.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with default value
        """
        return {metric_name: 0.0}

    @abstractmethod
    def get_best_metric_key(self) -> str:
        """
        Get the metric key that should be used for best model selection.

        Returns:
            Metric key string (e.g., 'accuracy', 'eer', '[global] eer')
        """
        pass

    def get_metric_info(self) -> Dict[str, Any]:
        """
        Get information about this metric manager.

        Returns:
            Dictionary containing manager metadata
        """
        return {
            "name": self.__class__.__name__,
            "best_metric_key": self.get_best_metric_key(),
            "current_epoch": self.current_epoch,
        }

    @abstractmethod
    def print_logs(self, split: str) -> None:
        """
        Print or log human-readable metrics summary.

        Args:
            split: Dataset split ('train', 'validation', or 'test')
        """
        pass

    @abstractmethod
    def get_logs(self, split: str) -> Dict[str, float]:
        """
        Get metrics dictionary ready for logging.

        Args:
            split: Dataset split ('train', 'validation', or 'test')

        Returns:
            Dictionary with metric names as keys and numeric values
        """
        pass

    def generate_plots(self, epoch: int, split: str) -> None:
        """
        Generate visualizations for the metrics.

        Default implementation does nothing. Subclasses can override to
        generate plots (e.g., score distributions, error curves, etc.).

        Args:
            epoch: Current epoch number
            split: Dataset split ('train', 'validation', or 'test')
        """
        pass  # Default: no plots

    def compute_epoch_diagnostics(self, split: str) -> Dict[str, float]:
        """
        Compute optional epoch-level diagnostics.

        Subclasses can override to return extra scalar diagnostics that are not
        part of the primary metric set (for example confusion-matrix counts).

        Args:
            split: Dataset split ('train', 'validation', or 'test')

        Returns:
            Dictionary of diagnostic scalar metrics.
        """
        self.diagnostics_cache[split] = {}
        return {}
