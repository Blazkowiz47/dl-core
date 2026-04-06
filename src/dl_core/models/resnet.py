from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import Linear
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)

from dl_core.core.base_model import BaseModel
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_model


@register_model("resnet")
class ResNet(BaseModel):
    """
    ResNet model with dictionary output.

    Supports ResNet18, ResNet34, ResNet50, and ResNet101 architectures
    with pretrained weights and custom classification heads.
    """

    CONFIG_FIELDS = BaseModel.CONFIG_FIELDS + [
        config_field(
            "variant",
            "str",
            "ResNet backbone variant: resnet18, resnet34, resnet50, or "
            "resnet101.",
            default="resnet18",
        ),
        config_field(
            "pretrained",
            "bool",
            "Load torchvision pretrained weights for the selected variant.",
            default=False,
        ),
        config_field(
            "use_pretrained_custom",
            "bool",
            "Load custom checkpoint weights after building the backbone.",
            default=False,
        ),
        config_field(
            "custom_weights_path",
            "str",
            "Path to custom pretrained weights when "
            "use_pretrained_custom=true.",
            default="weights/competition_pretrained_resnet50.pth",
        ),
    ]

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

        # Model-specific parameters
        # Check kwargs for variant first, then fall back to name
        self.variant = config.get("variant", "resnet18")  # resnet18, resnet34, etc.
        self.name = self.variant  # For logging consistency
        self.pretrained = bool(config.get("pretrained", False))
        self.use_pretrained_custom = config.get("use_pretrained_custom", False)
        self.custom_weights_path = config.get(
            "custom_weights_path", "weights/competition_pretrained_resnet50.pth"
        )

        # Initialize backbone and head
        self._build_architecture()

        # Load custom pretrained weights if specified
        if self.use_pretrained_custom and self.variant == "resnet50":
            self._load_custom_weights()

        self.logger.debug(
            f"Initialized ResNet model: {self.variant} with {self.num_classes} classes"
        )

    def _build_architecture(self):
        """Build ResNet architecture based on variant."""
        if self.variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if self.pretrained else None
            self.module = resnet18(weights=weights)
            self.feature_dim = 512
        elif self.variant == "resnet34":
            weights = ResNet34_Weights.DEFAULT if self.pretrained else None
            self.module = resnet34(weights=weights)
            self.feature_dim = 512
        elif self.variant == "resnet50":
            weights = ResNet50_Weights.DEFAULT if self.pretrained else None
            self.module = resnet50(weights=weights)
            self.feature_dim = 2048
        elif self.variant == "resnet101":
            weights = ResNet101_Weights.DEFAULT if self.pretrained else None
            self.module = resnet101(weights=weights)
            self.feature_dim = 2048
        else:
            raise NotImplementedError(f"ResNet variant: {self.variant} not supported")

        # Replace the final fully connected layer
        self.module.fc = Linear(self.feature_dim, self.num_classes)

    def _load_custom_weights(self):
        """Load custom pretrained weights."""
        try:
            checkpoint = torch.load(
                self.custom_weights_path, weights_only=False, map_location="cpu"
            )
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            self.load_state_dict(state_dict)
            self.logger.info(
                f"Loaded custom pretrained weights from {self.custom_weights_path}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load custom weights: {e}")

    def compute_forward(self, batch_data: dict, **kwargs) -> Dict[str, Tensor]:
        """
        Forward pass with dictionary output.

        Args:
            batch_data: Dictionary containing batch inputs. Must contain 'image' key.
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary containing:
            - probabilities: Softmax probabilities [batch_size, num_classes]
            - logits: Classification logits [batch_size, num_classes]
            - features: Feature representations [batch_size, feature_dim]
        """
        # Extract image tensor (already on correct device from trainer)
        x = batch_data["image"]

        # Forward through backbone layers
        x = self.module.conv1(x)
        x = self.module.bn1(x)
        x = self.module.relu(x)
        x = self.module.maxpool(x)

        x = self.module.layer1(x)
        x = self.module.layer2(x)
        x = self.module.layer3(x)
        x = self.module.layer4(x)

        # Global average pooling and feature extraction
        x = self.module.avgpool(x)
        features = torch.flatten(x, 1)

        # Classification predictions (logits)
        logits = self.module.fc(features)

        # Standardized outputs (like dino): probabilities, logits, features
        if self.num_classes == 1 or logits.shape[-1] == 1:
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=1)

        return {
            "probabilities": probabilities,
            "logits": logits,
            "features": features,
        }

    def get_backbone_parameters(self):
        """Get backbone parameters (everything except fc layer)."""
        backbone_params = []
        for name, param in self.module.named_parameters():
            if not name.startswith("fc"):
                backbone_params.append(param)
        return backbone_params

    def get_head_parameters(self):
        """Get classification head parameters."""
        return list(self.module.fc.parameters())

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for name, param in self.module.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
        self.logger.info(f"Froze backbone parameters for {self.variant}")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for name, param in self.module.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = True
        self.logger.info(f"Unfroze backbone parameters for {self.variant}")

    def get_layer_features(self, x: Tensor, layer_name: str) -> Tensor:
        """
        Extract features from intermediate layers.

        Args:
            x: Input tensor
            layer_name: Layer to extract features from ('layer1', 'layer2', 'layer3', 'layer4')

        Returns:
            Feature tensor from specified layer
        """
        x = self.module.conv1(x)
        x = self.module.bn1(x)
        x = self.module.relu(x)
        x = self.module.maxpool(x)

        x = self.module.layer1(x)
        if layer_name == "layer1":
            return x

        x = self.module.layer2(x)
        if layer_name == "layer2":
            return x

        x = self.module.layer3(x)
        if layer_name == "layer3":
            return x

        x = self.module.layer4(x)
        if layer_name == "layer4":
            return x

        raise ValueError(f"Invalid layer_name: {layer_name}")

    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        return self.feature_dim
