"""
Standard metric manager for basic classification tasks.

Handles standard classification metrics like accuracy, AUC, and F1 score
without any domain-specific logic.
"""

from typing import Any, Dict

import numpy as np
import torch

from dl_core.core import (
    METRIC_REGISTRY,
    BaseAccelerator,
    BaseMetricManager,
    config_field,
    register_metric_manager,
)


@register_metric_manager("standard")
class StandardMetricManager(BaseMetricManager):
    """
    Standard metric manager for general classification tasks.

    NEW ARCHITECTURE:
    - Metrics are pure functions (no state)
    - Manager handles data accumulation and distributed gathering
    - No rank divergence bugs possible

    Provides basic classification metrics:
    - Accuracy (can be averaged across ranks)
    - AUC (requires global data)
    - F1 score (requires global data)

    Suitable for general purpose classification without domain-specific requirements.
    """

    CONFIG_FIELDS = BaseMetricManager.CONFIG_FIELDS + [
        config_field(
            "num_classes",
            "int",
            "Number of classes expected by the managed metrics.",
            default=2,
        ),
        config_field(
            "best_metric_key",
            "str",
            "Metric key used for best-checkpoint selection.",
            default="accuracy",
        ),
    ]

    def __init__(
        self,
        config: Dict[str, Any],
        accelerator: BaseAccelerator,
        trainer: Any = None,
    ):
        """
        Initialize standard metric manager.

        Args:
            config: Configuration dictionary
            accelerator: Accelerator instance for multi-GPU operations
            trainer: Trainer instance (optional, for plot generation)
        """
        self.num_classes = config.get("num_classes", 2)
        self.best_metric_key_config = config.get("best_metric_key", "accuracy")
        super().__init__(config, accelerator, trainer)
        self.name = "standard"

    def setup_metrics(self) -> None:
        """
        Setup standard classification metrics.

        Metrics are pure functions - no accelerator needed!
        """
        # Get metric classes from registry
        accuracy_cls = METRIC_REGISTRY.get_class("accuracy")
        auc_cls = METRIC_REGISTRY.get_class("auc")
        f1_cls = METRIC_REGISTRY.get_class("f1")

        # Setup metrics for all splits
        for split in ["train", "validation", "test"]:
            self.metrics[split] = {
                "accuracy": accuracy_cls({"num_classes": self.num_classes}),
                "auc": auc_cls({"num_classes": self.num_classes}),
                "f1": f1_cls(
                    {
                        "num_classes": self.num_classes,
                        "average": "macro" if self.num_classes > 2 else "binary",
                    }
                ),
            }

        self.log.debug(f"Setup standard metrics for {self.num_classes} classes")

    def get_best_metric_key(self) -> str:
        """
        Get the metric key used for best model selection.

        Returns:
            Metric key string
        """
        return self.best_metric_key_config

    def print_logs(self, split: str) -> None:
        """
        Print human-readable metrics summary.

        Args:
            split: Dataset split ('train', 'validation', or 'test')
        """
        # Only print on main process
        if not self.accelerator.is_main_process():
            return

        # Get cached metrics
        metrics = self.results_cache.get(split, {})
        diagnostics = self.diagnostics_cache.get(split, {})

        if metrics or diagnostics:
            self.log.info(f"{split.upper()} Metrics - Epoch {self.current_epoch}")
            for key, value in metrics.items():
                self.log.info(f"  {key}: {value:.4f}")
            for key, value in diagnostics.items():
                self.log.info(f"  {key}: {value:.4f}")

    def get_logs(self, split: str) -> Dict[str, float]:
        """
        Get metrics dictionary for logging.

        Args:
            split: Dataset split ('train', 'validation', or 'test')

        Returns:
            Dictionary with metric names as keys and numeric values
        """
        # Get cached metrics
        metrics = self.results_cache.get(split, {})

        # Add split prefix to keys
        return {f"{split}_{k}": v for k, v in metrics.items()}

    def compute_epoch_diagnostics(self, split: str) -> Dict[str, float]:
        """
        Compute confusion-matrix diagnostics from gathered predictions.

        Args:
            split: Dataset split ('train', 'validation', or 'test')

        Returns:
            Dictionary with confusion-matrix scalar diagnostics.
        """
        local_data = self._prepare_local_data(split)
        gathered_data = self._gather_data(local_data)
        if not self.accelerator.is_main_process():
            return {}

        predictions = gathered_data.get("predictions")
        labels = gathered_data.get("labels")
        if not isinstance(predictions, np.ndarray) or not isinstance(
            labels, np.ndarray
        ):
            return {}
        if predictions.size == 0 or labels.size == 0:
            return {}

        labels = labels.reshape(-1)
        if predictions.ndim == 1:
            pred_labels = (predictions >= 0.5).astype(np.int64)
        elif predictions.ndim == 2 and predictions.shape[1] == 1:
            pred_labels = (predictions[:, 0] >= 0.5).astype(np.int64)
        else:
            pred_labels = np.argmax(predictions, axis=1).astype(np.int64)

        if labels.shape[0] != pred_labels.shape[0]:
            return {}

        prefix = "cm"
        diagnostics: Dict[str, float] = {
            f"{prefix}/num_samples": float(labels.shape[0]),
        }

        # Binary confusion-matrix details (class 1 as "positive")
        if self.num_classes == 2:
            pos_label = 1
            neg_label = 0
            tn = np.sum((labels == neg_label) & (pred_labels == neg_label))
            fp = np.sum((labels == neg_label) & (pred_labels == pos_label))
            fn = np.sum((labels == pos_label) & (pred_labels == neg_label))
            tp = np.sum((labels == pos_label) & (pred_labels == pos_label))
            diagnostics.update(
                {
                    f"{prefix}/tn": float(tn),
                    f"{prefix}/fp": float(fp),
                    f"{prefix}/fn": float(fn),
                    f"{prefix}/tp": float(tp),
                    f"{prefix}/pred_positive_rate": float(pred_labels.mean()),
                }
            )

        self.diagnostics_cache[split] = diagnostics
        return diagnostics


@register_metric_manager("standard_act")
class StandardActMetricManager(StandardMetricManager):
    """
    Standard metric manager for general classification tasks with ACT.

    NEW ARCHITECTURE:
    - Metrics are pure functions (no state)
    - Manager handles data accumulation and distributed gathering
    - No rank divergence bugs possible

    Provides basic classification metrics:
    - Accuracy (can be averaged across ranks)
    - AUC (requires global data)
    - F1 score (requires global data)

    Suitable for general purpose classification without domain-specific requirements.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for split in ["train", "validation", "test"]:
            self.accumulated_data[split]["steps"] = []

    def setup_metrics(self) -> None:
        """
        Setup standard classification metrics for ACT.

        Metrics are pure functions - no accelerator needed!
        """
        super().setup_metrics()
        halt_steps_cls = METRIC_REGISTRY.get_class("halt_steps")
        for split in ["train", "validation", "test"]:
            self.metrics[split]["halt_steps"] = halt_steps_cls({})

    def _prepare_local_data(self, split: str) -> Dict[str, Any]:
        local_data = super()._prepare_local_data(split)
        step_batches = self.accumulated_data[split].get("steps", [])
        local_data["metadata"] = {
            "steps": np.concatenate(step_batches, axis=0)
            if step_batches
            else np.array([])
        }
        return local_data

    def _merge_metadata(self, existing: Dict, new: Dict) -> Dict:
        merged = dict(existing)
        existing_steps = merged.get("steps")
        new_steps = new.get("steps")

        if isinstance(existing_steps, np.ndarray) and existing_steps.size > 0:
            if isinstance(new_steps, np.ndarray) and new_steps.size > 0:
                merged["steps"] = np.concatenate([existing_steps, new_steps], axis=0)
        elif isinstance(new_steps, np.ndarray):
            merged["steps"] = new_steps

        return merged

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
        halted = batch_data.get("halted")
        steps = batch_data.get("steps")
        if labels is None:
            return
        if halted is None:
            raise ValueError(
                "Batch data must contain 'halted' key for ACT metric manager"
            )
        if steps is None:
            raise ValueError(
                "Batch data must contain 'steps' key for ACT metric manager"
            )

        # Move to CPU and convert to numpy
        halted = halted.detach().cpu()
        preds_np = probabilities.detach().cpu()[halted].numpy()
        labels_np = labels.detach().cpu()[halted].numpy()
        steps_np = steps.detach().cpu()[halted].numpy()

        # Accumulate
        self.accumulated_data[split]["predictions"].append(preds_np)
        self.accumulated_data[split]["labels"].append(labels_np)
        self.accumulated_data[split]["steps"].append(steps_np)

        # Optional: accumulate metadata for per-attack metrics
        self._accumulate_metadata(batch_data, split)

    def reset_metrics(self, split: str) -> None:
        super().reset_metrics(split)
        if split in self.accumulated_data:
            self.accumulated_data[split]["steps"] = []
