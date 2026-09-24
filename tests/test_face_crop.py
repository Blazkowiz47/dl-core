"""Tests for bounding-box crop behavior used by Azure datasets."""

import numpy as np

from dl_core.utils.common import crop_face_with_bbox


def test_crop_face_applies_configured_margin() -> None:
    """Nonzero margins should expand the crop around the face box."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    crop, bbox = crop_face_with_bbox(image, [10, 10, 20, 20], margin=(50, 50))

    assert bbox == [0, 0, 40, 40]
    assert crop.shape == (40, 40, 3)
