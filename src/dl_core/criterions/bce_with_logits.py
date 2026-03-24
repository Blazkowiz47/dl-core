"""Binary cross-entropy with logits criterion."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from torch.nn import BCEWithLogitsLoss

from dl_core.core.base_criterion import BaseCriterion
from dl_core.core.registry import register_criterion


def _to_tensor(value: Any) -> torch.Tensor | None:
    """Convert config value to float tensor if present."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, (list, tuple, Iterable)) and not isinstance(
        value, (str, bytes)
    ):
        return torch.tensor(list(value), dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


@register_criterion("bce_with_logits")
class BCEWithLogits(BaseCriterion):
    """Wrapper around ``torch.nn.BCEWithLogitsLoss`` with registry integration."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

        weight = _to_tensor(config.get("weight"))
        pos_weight = _to_tensor(config.get("pos_weight"))
        reduction = config.get("reduction", "mean")

        self.criterion = BCEWithLogitsLoss(
            weight=weight,
            pos_weight=pos_weight,
            reduction=reduction,
        )

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Compute BCE-with-logits loss."""

        # Ensure shape/ dtype alignment
        target_tensor = targets.float()
        if target_tensor.shape != predictions.shape:
            target_tensor = target_tensor.view_as(predictions)

        loss = self.criterion(predictions, target_tensor)
        return {"loss": loss}
