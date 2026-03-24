"""
Base abstract class for individual metrics.

Metrics are pure computation functions with no state or distributed logic.
All data gathering and distributed coordination is handled by MetricManagers.
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Dict

import numpy as np


class BaseMetric(ABC):
    """
    Abstract base class for all metrics - pure computation functions.

    DESIGN PHILOSOPHY:
    - Metrics are STATELESS - no data accumulation
    - Metrics are PURE FUNCTIONS - data in, metric out
    - Metrics have NO distributed logic - managers handle gathering
    - Metrics are TESTABLE - can be tested in isolation without GPU/distributed setup

    === ARCHITECTURE ===

    MetricManagers handle:
    - Data accumulation during training
    - Distributed gathering (gather_via_pickle, all_reduce)
    - Rank divergence management
    - Mode selection (gather-then-compute vs compute-then-average)

    Metrics handle:
    - Pure computation from provided data
    - Domain-specific algorithms (EER, AUC, F1, etc.)
    - Validation and error handling

    === STANDARD PATTERN ===

    class MyMetric(BaseMetric):
        def __init__(self, config: Dict[str, Any] | None = None):
            super().__init__(config)
            self.some_param = self.config.get("some_param", default)

        def compute(
            self,
            predictions: np.ndarray,  # Shape: (N, num_classes) or (N,)
            labels: np.ndarray,       # Shape: (N,)
            **kwargs                  # Additional metric-specific args
        ) -> Dict[str, float]:
            # Validate inputs
            if len(predictions) == 0:
                return {"my_metric": 0.0}

            # Perform computation
            result = my_algorithm(predictions, labels)

            return {"my_metric": float(result)}

    === EXAMPLE ===

    # Old way (stateful, distributed in metric):
    class OldMetric(BaseMetric):
        def __init__(self, accelerator):
            self.accelerator = accelerator
            self.all_preds = []  # State!

        def update(self, preds, labels):
            self.all_preds.append(preds)  # Accumulate state

        def compute(self):
            gathered = gather_via_pickle(...)  # Distributed logic in metric!
            if not self.is_main_process():
                return {}
            return {"metric": calculate(gathered)}

    # New way (stateless, pure function):
    class NewMetric(BaseMetric):
        def compute(self, predictions, labels, **kwargs):
            # No state, no distributed logic
            return {"metric": calculate(predictions, labels)}

    === BENEFITS ===

    1. Testability: Can unit test metrics without distributed setup
    2. Maintainability: Single source of truth for distributed logic
    3. Debuggability: Pure functions are easier to debug
    4. Reusability: Can use metrics outside of training context
    5. Safety: No rank divergence bugs possible

    Args:
        config: Optional configuration dictionary
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize the metric.

        Args:
            config: Optional configuration dictionary with metric-specific parameters
        """
        self.config = config or {}
        self.log = getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def compute(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute metric from predictions and labels.

        This is a PURE FUNCTION:
        - No side effects
        - No state modification
        - Same inputs always produce same outputs
        - No distributed operations

        Args:
            predictions: Model predictions as numpy array
                        - Binary: shape (N,) or (N, 2) with probabilities
                        - Multi-class: shape (N, num_classes) with probabilities
            labels: Ground truth labels as numpy array, shape (N,)
            **kwargs: Additional metric-specific arguments passed by manager

        Returns:
            Dictionary with metric name(s) as keys and computed values as floats

        Example:
            >>> metric = MyMetric()
            >>> result = metric.compute(
            ...     predictions=np.array([[0.1, 0.9], [0.8, 0.2]]),
            ...     labels=np.array([1, 0])
            ... )
            >>> print(result)
            {"my_metric": 0.95}
        """
        pass
