"""
Base criterion class for standardized loss computation with dictionary returns.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch.nn import Module

# Module-level logger
logger = logging.getLogger(__name__)


class BaseCriterion(Module, ABC):
    """
    Abstract base class for all loss functions in the training module.

    All criterion implementations must inherit from this class and return
    standardized dictionary outputs for maximum flexibility.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize the base criterion.

        Args:
            config: Configuration dictionary containing loss parameters
            **kwargs: Additional keyword arguments
        """
        super(BaseCriterion, self).__init__()
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.kwargs: Dict[str, Any] = kwargs
        self.name = self.__class__.__name__

        self.logger.debug(f"Initialized {self.name} criterion with config: {config}")

    @abstractmethod
    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss components.

        Must return dict with 'loss' key containing the total loss
        for backpropagation. Other keys are optional (for logging).

        Returns:
            {
              'loss': torch.Tensor,  # Required - for backward
              'comp1': torch.Tensor, # Optional - for logging
              'comp2': torch.Tensor, # Optional - for logging
            }
        """
        raise NotImplementedError("Subclasses must implement compute_loss method")

    def validate_output(self, loss_dict):
        """Validate the output of compute_loss."""

        assert isinstance(loss_dict, dict), (
            f"{self.__class__.__name__}.compute_loss() must return dict, "
            f"got {type(loss_dict)}"
        )

        assert "loss" in loss_dict, (
            f"{self.__class__.__name__}.compute_loss() must return "
            f"dict with 'loss' key. Got keys: {list(loss_dict.keys())}"
        )

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with standardized dictionary output.

        Args:
            predictions: Model predictions tensor
            targets: Ground truth targets tensor
            **kwargs: Additional arguments (features, auxiliary outputs, etc.)

        Returns:
            Standardized loss dictionary:
            {
                "loss": torch.Tensor,  # Total weighted loss for backprop
                "components": {        # Individual loss components
                    "component_name": torch.Tensor,
                    ...
                },
                "weights": {           # Weights used for each component
                    "component_name": float,
                    ...
                }
            }
        """
        self.validate_inputs(predictions, targets)
        loss_components = self.compute_loss(predictions, targets, **kwargs)
        self.validate_output(loss_components)

        # # Log loss components for debugging
        # if self.training:
        #     component_str = ", ".join(
        #         [f"{k}={v.item():.4f}" for k, v in loss_components.items()]
        #     )
        #     self.logger.debug(f"{self.name} components: {component_str}")

        return loss_components

    def get_loss_info(self) -> Dict[str, Any]:
        """
        Get information about this criterion.

        Returns:
            Dictionary containing criterion metadata
        """
        return {
            "name": self.name,
            "config": self.config,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    def validate_inputs(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Validate input tensors.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Raises:
            ValueError: If inputs are invalid
        """
        assert isinstance(predictions, torch.Tensor), (
            f"predictions must be torch.Tensor, got {type(predictions)}"
        )

        assert isinstance(targets, torch.Tensor), (
            f"targets must be torch.Tensor, got {type(targets)}"
        )

        assert predictions.device == targets.device, (
            f"predictions and targets must be on same device, got {predictions.device} vs {targets.device}"
        )

        # Check batch size consistency
        assert predictions.shape[0] == targets.shape[0], (
            f"Batch size mismatch: predictions {predictions.shape[0]} vs targets {targets.shape[0]}"
        )
