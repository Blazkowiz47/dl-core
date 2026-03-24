"""Base class for image transform strategies."""

from abc import ABC, abstractmethod

import albumentations as A


class BaseTransform(ABC):
    """Base class for all transform strategies.

    Transforms control data augmentation pipelines for training and testing.
    """

    def __init__(self, **kwargs):
        """Initialize transform strategy.

        Args:
            **kwargs: Configuration parameters for the transform
        """
        self.config = kwargs
        self.height = kwargs.get("height", 224)
        self.width = kwargs.get("width", 224)

        # Setup transforms
        self.train_transforms = self._create_train_transforms()
        self.test_transforms = self._create_test_transforms()

        if self._create_validation_transforms() is None:
            self.validation_transforms = self._create_test_transforms()
        else:
            self.validation_transforms = self._create_validation_transforms()

    @abstractmethod
    def _create_train_transforms(self) -> A.Compose:
        """Create training transforms.

        Returns:
            Albumentations compose pipeline for training
        """
        pass

    @abstractmethod
    def _create_test_transforms(self) -> A.Compose:
        """Create test transforms.

        Returns:
            Albumentations compose pipeline for testing
        """
        pass

    def _create_validation_transforms(self) -> A.Compose:
        """Create test transforms.

        Returns:
            Albumentations compose pipeline for testing
        """
        return None

    def get_transforms(self, split: str) -> A.Compose:
        """Get transforms for the specified split.

        Args:
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Appropriate transform pipeline
        """
        if split == "train":
            return self.train_transforms
        elif split == "test":
            return self.test_transforms
        elif split == "validation":
            return self.validation_transforms
        else:
            raise ValueError(f"Unknown split: {split}")

    def apply(self, image, split: str = "train"):
        """Apply transforms to image.

        Args:
            image: Input image
            split: Dataset split

        Returns:
            Transformed image
        """
        transforms = self.get_transforms(split)
        return transforms(image=image)["image"]

    def __call__(self, image, split: str = "train"):
        """Apply transforms to image.

        Args:
            image: Input image
            split: Dataset split

        Returns:
            Transformed image
        """
        return self.apply(image, split)
