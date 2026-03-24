"""
Base model class for standardized model outputs with dictionary returns.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn import Module, Parameter

# Module-level logger
logger = logging.getLogger(__name__)


class BaseModel(Module, ABC):
    """
    Abstract base class for all models in the training module.

    All model implementations must inherit from this class and return
    standardized dictionary outputs for maximum flexibility.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize the base model.

        Args:
            name: Model name
            config: Configuration dictionary containing model parameters
            **kwargs: Additional keyword arguments (may include real_class_index)
        """
        super(BaseModel, self).__init__()
        self.kwargs = kwargs
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.kwargs: Dict[str, Any] = kwargs
        self.name = self.__class__.__name__

        # Extract common parameters

        self.device_name = config.get("device", "cuda")
        self.num_classes = config["num_classes"]

        self.logger.debug(f"Initialized {self.name} model with config: {config}")

    @abstractmethod
    def compute_forward(self, batch_data: dict, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute the actual forward pass.

        This method must be implemented by subclasses to define the specific
        model architecture forward pass.

        Args:
            batch_data: Dictionary containing batch inputs. Must contain 'data' key.
                       May contain additional keys like 'metadata', 'path', etc.
            **kwargs: Additional arguments for forward pass

        Returns:
            Dictionary containing model outputs. At minimum must contain:
            - "probabilities": tensor (class probabilities)
            - "logits": tensor (raw logits)
            - "features": tensor (feature representations, or None)
        Note:
            - "probabilities" and "logits" must always be present.
            - May also contain other auxiliary outputs.
        """
        raise NotImplementedError("Subclasses must implement compute_forward method")

    def forward(self, batch_data: dict, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with standardized dictionary output.

        Args:
            batch_data: Dictionary containing batch inputs. Must contain 'data' key.
                       May contain additional keys like 'metadata', 'path', etc.
            **kwargs: Additional arguments for forward pass

        Returns:
            Standardized model output dictionary:
            {
                "probabilities": torch.Tensor,  # Class probabilities
                "logits": torch.Tensor,         # Raw logits
                "features": torch.Tensor        # Feature representations (or None)
            }
        """

        batch_size = self.validate_input(batch_data)
        outputs = self.compute_forward(batch_data, **kwargs)
        self.validate_outputs(outputs, batch_size)

        return outputs

    def validate_outputs(self, outputs, batch_size: int) -> None:
        """
        Validate the outputs from compute_forward.
        Args:
            outputs: Output dictionary from compute_forward
            batch_size: Expected batch size
        Raises:
            AssertionError: If outputs do not conform to expected format
        """
        if batch_size == -1:
            # Skip validation if batch size could not be inferred
            return

        # Validate that compute_forward returns a dictionary
        assert isinstance(outputs, dict), (
            f"{self.name}.compute_forward() must return a dictionary, got {type(outputs)}"
        )

        assert "probabilities" in outputs, (
            f"{self.name}.compute_forward() must return 'probabilities' key"
        )

        # Ensure probabilities are present
        assert "logits" in outputs, (
            f"{self.name}.compute_forward() must return 'logits' key"
        )

        assert outputs["probabilities"].shape[0] == batch_size, (
            f"Batch size mismatch: expected {batch_size}, got {outputs['probabilities'].shape[0]}"
        )
        assert outputs["logits"].shape[0] == batch_size, (
            f"Batch size mismatch: expected {batch_size}, got {outputs['logits'].shape[0]}"
        )

    def _log_standardized_output_shapes(self, output: Dict[str, torch.Tensor]) -> None:
        """
        Log output shapes for debugging.

        Args:
            output: Standardized output dictionary
        """
        shapes = []

        for key, value in output.items():
            if output[key] is None:
                shapes.append(f"{key}=None")
            elif isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                shapes.append(f"{key}={list(value.shape)}")
            else:
                shapes.append(f"{key}={len(value)}")

        self.logger.debug(f"{self.name} output shapes: {', '.join(shapes)}")

    def validate_input(self, batch_data: dict) -> int:
        """
        Validate input batch dictionary.

        Args:
            batch_data: Input batch dictionary

        Raises:
            AssertionError: If input is not valid

        Returns:
            Batch size inferred from input
        """
        assert isinstance(batch_data, dict), (
            f"Input must be dict, got {type(batch_data)}"
        )

        # Support both 'data' and 'image' keys for flexibility
        if "data" in batch_data:
            x = batch_data["data"]
        elif "image" in batch_data:
            x = batch_data["image"]
        elif "images" in batch_data:
            x = batch_data["images"]
        else:
            self.logger.warning("batch_data doesn't contain 'data' or 'image' key")
            x = None  # Dummy tensor to trigger assertion

        if x is not None and not isinstance(x, torch.Tensor):
            raise AssertionError(
                f"batch_data['data'/'image'/'images'] must be torch.Tensor, got {type(x)}"
            )

        if x is None:
            return -1

        return x.shape[0]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about this model.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "name": self.name,
            "config": self.config,
            "num_classes": self.num_classes,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "device": next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else "cpu",
        }

    def get_feature_shapes(self, input_shape: List[int]) -> Dict[str, List[int]]:
        """
        Get expected output shapes for given input shape.

        Args:
            input_shape: Expected input shape [batch_size, ...]

        Returns:
            Dictionary of expected output shapes
        """
        # Create dummy input
        dummy_input = torch.zeros(input_shape)
        if next(self.parameters(), None) is not None:
            dummy_input = dummy_input.to(next(self.parameters()).device)

        # Forward pass in eval mode
        was_training = self.training
        self.eval()

        with torch.no_grad():
            output = self(dummy_input)

        # Restore training mode
        if was_training:
            self.train()

        # Extract shapes from standardized output
        shapes = {}

        if output["probabilities"] is not None:
            shapes["probabilities"] = list(output["probabilities"].shape)

        if output["logits"] is not None:
            shapes["logits"] = list(output["logits"].shape)

        if output["features"] is not None:
            shapes["features"] = list(output["features"].shape)

        return shapes

    def freeze_backbone(self) -> None:
        """
        Freeze backbone parameters (if model has a backbone attribute).
        """
        backbone = getattr(self, "backbone", None)
        if isinstance(backbone, Module):
            for param in backbone.parameters():
                param.requires_grad = False
            self.logger.info(f"Froze backbone parameters for {self.name}")
        else:
            self.logger.warning(f"No backbone attribute found in {self.name}")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze backbone parameters (if model has a backbone attribute).
        """
        backbone = getattr(self, "backbone", None)
        if isinstance(backbone, Module):
            for param in backbone.parameters():
                param.requires_grad = True
            self.logger.info(f"Unfroze backbone parameters for {self.name}")
        else:
            self.logger.warning(f"No backbone attribute found in {self.name}")

    def get_backbone_parameters(self) -> Optional[List[Parameter]]:
        """
        Get backbone parameters if available.

        Returns:
            List of backbone parameters or None if no backbone
        """
        backbone = getattr(self, "backbone", None)
        if isinstance(backbone, Module):
            return list(backbone.parameters())
        return None

    def get_head_parameters(self) -> Optional[List[Parameter]]:
        """
        Get head parameters if available.

        Returns:
            List of head parameters or None if no head
        """
        head = getattr(self, "head", None)
        if isinstance(head, Module):
            return list(head.parameters())
        return None

    def get_trainable_parameter_groups(self) -> dict[str, list[Parameter]]:
        """Return trainable parameter groups with a sensible default."""

        trainable_params = [param for param in self.parameters() if param.requires_grad]
        return {"main": trainable_params}

    def init_weights(self) -> None:
        """
        Initialize model weights.

        This method can be overridden by subclasses for custom initialization.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        self.logger.debug(f"Initialized weights for {self.name}")
