"""Minimal transform strategy with basic preprocessing only."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dl_core.core.registry import AUGMENTATION_REGISTRY
from dl_core.core.base_transform import BaseTransform


@AUGMENTATION_REGISTRY.register("minimal")
class MinimalTransform(BaseTransform):
    """Minimal transforms with only resize, crop, normalize.

    No augmentation - useful for testing or when augmentation is handled elsewhere.

    Config parameters:
        height: Target height (default: 224)
        width: Target width (default: 224)
    """

    def _create_train_transforms(self) -> A.Compose:
        """Create minimal training transforms."""
        return A.Compose(
            [
                A.Resize(height=self.height, width=self.width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _create_test_transforms(self) -> A.Compose:
        """Create minimal test transforms."""
        return A.Compose(
            [
                A.Resize(256,256),
                A.CenterCrop(height=self.height, width=self.width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
