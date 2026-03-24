"""Standard transform strategy with moderate augmentation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dl_core.core.registry import AUGMENTATION_REGISTRY
from dl_core.core.base_transform import BaseTransform


@AUGMENTATION_REGISTRY.register("standard")
class StandardTransform(BaseTransform):
    """Standard transforms with moderate augmentation pipeline.

    Suitable for most PAD/MAD/Deepfake tasks with balanced augmentation.

    Config parameters:
        height: Target height (default: 224)
        width: Target width (default: 224)
    """

    def _create_train_transforms(self) -> A.Compose:
        """Create standard training transforms."""
        return A.Compose(
            [
                A.Resize(self.height, self.width),
                A.HorizontalFlip(p=0.1),
                A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=0.1, p=0.1),
                A.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.1
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.1
                ),
                A.CoarseDropout(
                    (1, 4),
                    hole_height_range=(self.height * 24 // 224, self.height*80//224),
                    hole_width_range=(self.width*24//224, self.width*80//224),
                    fill=0,
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.ImageCompression("webp", (95, 100), p=1),
                        A.ImageCompression("jpeg", (95, 100), p=1),
                    ],
                    p=0.01,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _create_test_transforms(self) -> A.Compose:
        """Create standard test transforms."""
        return A.Compose(
            [
                A.Resize(height=self.height, width=self.width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
