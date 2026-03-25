"""Base model helpers for biometric classification tasks."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .base_model import BaseModel


class BaseBiometricModel(BaseModel):
    """Base class for biometric models with genuine-score helpers."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.real_class_index = config.get("real_class_index", 0)
        self.logger.debug(f"Real class index set to: {self.real_class_index}")

    def get_genuine_scores(self, batch_data: dict[str, Any]) -> Tensor:
        """Return the probability assigned to the configured genuine class."""
        outputs = self.forward(batch_data)
        probabilities = outputs["probabilities"]
        return probabilities[:, self.real_class_index]
