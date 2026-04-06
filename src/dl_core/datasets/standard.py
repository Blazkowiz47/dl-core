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
import os
import torch

from dl_core.core.base_dataset import BaseWrapper
from dl_core.core.config_metadata import config_field
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

    CONFIG_FIELDS = [
        config_field(
            "height",
            "int",
            "Fallback image height used when a file fails to load.",
            default=224,
        ),
        config_field(
            "width",
            "int",
            "Fallback image width used when a file fails to load.",
            default=224,
        ),
    ]

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
