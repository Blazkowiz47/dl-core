from typing import Any, Dict

from torch import Tensor

from .base_model import BaseModel


class BasePadModel(BaseModel):
    """Base class for models that use padding."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        # Real/genuine class index for PAD tasks (default: 0)
        # This is the class index representing genuine/bonafide samples
        self.real_class_index = config.get("real_class_index", 0)

        self.logger.debug(f"Real class index set to: {self.real_class_index}")

    def get_genuine_scores(self, batch_data: Dict) -> Tensor:
        """
        Get genuine/bonafide scores directly from input batch.

        This method performs a forward pass and extracts genuine scores
        in one step. Models can override this method to provide domain-specific
        score extraction logic. For PAD models, this extracts the probability
        that each sample is genuine/bonafide.

        Args:
            batch_data (Dict): Input batch dictionary containing data for
        Returns:
            Tensor of genuine scores (batch_size,) representing probability
            that each sample is genuine/bonafide

        Note:
            This default implementation performs a forward pass and extracts
            probabilities for the real/genuine class (using self.real_class_index).
            Models can override this method for custom behavior.
        Example:
        """
        outputs = self.forward(batch_data)
        # Extract genuine/real class probabilities
        probabilities = outputs["probabilities"]
        genuine_scores = probabilities[:, self.real_class_index]

        return genuine_scores
