"""Cross-entropy loss criterion."""

from typing import Any, Dict

import torch
from torch.nn import CrossEntropyLoss

from dl_core.core.base_criterion import BaseCriterion
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_criterion


@register_criterion("crossentropy")
class CrossEntropy(BaseCriterion):
    """
    Cross-entropy loss criterion with optional label smoothing and class weighting.

    Config format:
        loss:
          crossentropy:
            class_weights: [1.0, 2.0]  # Optional
            ignore_index: -100         # Optional
            label_smoothing: 0.1       # Optional
    """

    CONFIG_FIELDS = [
        config_field(
            "class_weights",
            "list[float] | None",
            "Optional per-class weights passed to CrossEntropyLoss.",
            default=None,
        ),
        config_field(
            "ignore_index",
            "int",
            "Target index that should be ignored by the loss.",
            default=-100,
        ),
        config_field(
            "label_smoothing",
            "float",
            "Amount of label smoothing applied by the criterion.",
            default=0.0,
        ),
    ]

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

        # config is the criterion-specific config dict
        # Initialize cross-entropy loss with optional parameters
        weight = None
        if "class_weights" in config and config["class_weights"] is not None:
            weight = torch.tensor(config["class_weights"], dtype=torch.float32)

        ignore_index = config.get("ignore_index", -100)
        label_smoothing = config.get("label_smoothing", 0.0)

        self.criterion = CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing
        )

        # self.logger.debug(
        #     f"Initialized CrossEntropy with weight={weight}, "
        #     f"ignore_index={ignore_index}, label_smoothing={label_smoothing}"
        # )

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-entropy loss.

        Args:
            predictions: Model logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size] (long tensor)
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary with 'loss' key containing the cross-entropy loss
        """
        # Ensure targets are long type for cross-entropy
        if targets.dtype != torch.long:
            targets = targets.long()

        # Compute cross-entropy loss
        ce_loss = self.criterion(predictions, targets)

        return {"loss": ce_loss}
