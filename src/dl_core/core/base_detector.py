"""
Base class for face detectors.

All face detector implementations should inherit from BaseFaceDetector
and implement the required methods.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FaceDetection:
    """Face detection result from a specific detection method."""

    bbox: list[int]  # [x, y, w, h] format
    confidence: float

    @staticmethod
    def validate_bbox(bbox: list[int]) -> bool:
        """
        Validate bounding box format.

        Args:
            bbox: Bounding box in [x, y, w, h] format

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(bbox, list):
            return False
        if len(bbox) != 4:
            return False
        if not all(isinstance(v, (int, float)) for v in bbox):
            return False
        # Width and height should be positive (x,y can be negative if partially outside)
        if bbox[2] <= 0 or bbox[3] <= 0:
            return False
        return True


@dataclass
class LandmarkDetection:
    """Landmark detection result from a specific detection method."""

    landmarks: list[list[int]]  # List of [x, y] coordinates
    confidence: float | None = None

    @staticmethod
    def validate_landmarks(landmarks: list[list[int]]) -> bool:
        """
        Validate landmarks format.

        Args:
            landmarks: List of [x, y] coordinates

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(landmarks, list):
            return False
        if len(landmarks) == 0:
            return False
        # Check each landmark is [x, y] with non-negative values
        for landmark in landmarks:
            if not isinstance(landmark, list):
                return False
            if len(landmark) != 2:
                return False
            if not all(isinstance(v, (int, float)) for v in landmark):
                return False
            if any(v < 0 for v in landmark):
                return False
        return True


class BaseFaceDetector(ABC):
    """
    Abstract base class for face detectors.

    All face detector implementations must inherit from this class
    and implement the detect() and detect_all() methods.
    """

    name: str = "base"  # Override in subclass
    logger = logging.getLogger(f"{__name__}.{name}")

    def detect(
        self, image: np.ndarray
    ) -> tuple[Optional[FaceDetection], Optional[LandmarkDetection]]:
        """
        Detect the largest face in an image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            FaceDetection with largest face, or None if no face found
        """
        face_dict, landmark_dict = self.detect_all(image)
        if not face_dict:
            face_results = None
        else:
            face_results = list(face_dict.values())[0]
        if not landmark_dict:
            landmark_results = None
        else:
            landmark_results = list(landmark_dict.values())[0]
        return (face_results, landmark_results)

    def detect_many(
        self, images: list[np.ndarray]
    ) -> list[tuple[Optional[FaceDetection], Optional[LandmarkDetection]]]:
        """
        Detect the largest face for multiple images.

        Detectors can override this for true batch inference. The default
        implementation loops over images and calls ``detect``.

        Args:
            images: List of input images as numpy arrays (BGR format)

        Returns:
            List of per-image detection tuples aligned with input order
        """
        return [self.detect(image) for image in images]

    @abstractmethod
    def detect_all(
        self, image: np.ndarray
    ) -> tuple[dict[str, FaceDetection], dict[str, LandmarkDetection]]:
        """
        Detect all faces in an image with optional landmarks.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Tuple of:
                - dict mapping detector name to FaceDetection (keyed by self.name)
                - dict mapping detector name to LandmarkDetection (empty if no landmarks)
            Both dicts sorted by face area (largest first)
        """
        pass
