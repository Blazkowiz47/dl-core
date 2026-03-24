"""
Standard local filesystem dataset wrapper.

This dataset loads data from local filesystem with:
- Support for simple directory structures
- Multiple directories per class
- Configurable image extensions
- Built-in augmentation strategies
"""

from typing import Any

import cv2
import numpy as np
import os
import torch

from dl_core.core.base_dataset import BaseWrapper
from dl_core.core.registry import register_dataset

# Image file extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]


@register_dataset("standard")
class StandardWrapper(BaseWrapper):
    """
    Standard local filesystem dataset loader.

    Task-agnostic dataset that loads images from directories.
    Each class can have one or more directories.

    Config structure:
        classes:
            0:  # Class label
                train: ['/path/to/class0/train']
                test: ['/path/to/class0/test']
            1:
                train: ['/path/to/class1/train']
                test: ['/path/to/class1/test']
    """

    def __init__(self, config: dict[str, Any], **kwargs):
        """
        Initialize standard dataset wrapper.

        Args:
            config: Dataset configuration
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__(config, **kwargs)
        self.height = self.config.get("height", 224)
        self.width = self.config.get("width", 224)
        synthetic_config = self.config.get("synthetic", {})
        self.synthetic_enabled = bool(synthetic_config.get("enabled", False))
        self.synthetic_num_samples = int(synthetic_config.get("num_samples", 1000))

        # Initialize parent (handles file loading from classes config)

    @property
    def file_extensions(self) -> list[str]:
        """File extensions to scan for images."""
        return list(IMAGE_EXTENSIONS)

    def get_file_list(self, split: str) -> list[dict]:
        """
        Get list of files for a given split.

        Args:
            split: Split name ('train', 'validation', 'test')
        Returns:
            List of file metadata dictionaries with keys:
            - path: str (file path)
            - label: str (class label)
        """
        if self.synthetic_enabled:
            return self._get_synthetic_file_list(split)

        if self.rdir is None:
            raise ValueError("Root directory (rdir) is not set.")
        split_dirs_first = False
        class_labels = [
            x
            for x in os.listdir(self.rdir)
            if os.path.isdir(os.path.join(self.rdir, x))
        ]
        if split in class_labels:
            class_labels = [
                x
                for x in os.listdir(os.path.join(self.rdir, split))
                if os.path.isdir(os.path.join(self.rdir, split, x))
            ]
            split_dirs_first = True

        data = []
        for class_label in class_labels:
            if split_dirs_first:
                dir_path = os.path.join(self.rdir, split, class_label)
            else:
                dir_path = os.path.join(self.rdir, class_label, split)

            if not os.path.exists(dir_path):
                self.logger.info(f"Directory does not exist: {dir_path}, skipping.")
                continue

            files = self.scan_directory(dir_path, IMAGE_EXTENSIONS)
            for file_path in files:
                data.append({"path": file_path, "label": class_label})

        return data

    def _get_synthetic_file_list(self, split: str) -> list[dict]:
        """Generate synthetic sample metadata for smoke tests and scaffolds."""
        if split != "train":
            return []

        classes = self.classes or [str(idx) for idx in range(self.num_classes or 2)]
        num_classes = len(classes)
        if num_classes == 0:
            raise ValueError("Synthetic dataset requires at least one class label")

        return [
            {
                "index": idx,
                "label": classes[idx % num_classes],
                "path": f"synthetic://{split}/{classes[idx % num_classes]}/{idx}",
                "synthetic": True,
            }
            for idx in range(self.synthetic_num_samples)
        ]

    def transform(self, file_dict: dict, split: str) -> dict[str, Any]:
        """
        Load and preprocess a single image from filesystem.

        Args:
            file_dict: Dictionary with file metadata (path, label)
            split: Split name ('train', 'validation', 'test')

        Returns:
            Dictionary with keys:
                - image: torch.Tensor (preprocessed image)
                - label: int (class label)
                - path: str (file path)
        """
        path = file_dict["path"]
        label = file_dict["label"]
        class_label = self.classes.index(label)

        if file_dict.get("synthetic", False):
            image = self._generate_synthetic_image(file_dict["index"])
            if self.augmentation:
                image_tensor = self.augmentation.apply(image, split)
            else:
                image_tensor = (
                    torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                )

            return {
                "image": image_tensor.float(),
                "label": class_label,
                "class": label,
                "path": str(path),
            }

        # Load image
        image = cv2.imread(str(path))

        if image is None:
            self.logger.warning(f"Failed to load image: {path}, returning dummy tensor")
            return {
                "image": torch.zeros(3, self.height, self.width),
                "label": label,
                "path": str(path),
            }

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation strategy (from BaseWrapper)
        if self.augmentation:
            image_tensor = self.augmentation.apply(image, split)
        else:
            # Fallback: convert to tensor without augmentation
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Explicitly delete numpy array to free memory
        del image

        return {
            "image": image_tensor.float(),
            "label": class_label,
            "class": label,
            "path": str(path),
        }

    def _generate_synthetic_image(self, index: int) -> np.ndarray:
        """Create a deterministic synthetic RGB image for a given sample index."""
        rng = np.random.default_rng(self.seed + index)
        return rng.integers(
            0,
            256,
            size=(self.height, self.width, 3),
            dtype=np.uint8,
        )
