import os
import random
from datetime import datetime
import logging

import numpy as np
import torch

image_extensions: list[str] = [".jpg", ".png", ".jpeg"]
video_extensions: list[str] = [".mov", ".mp4"]

log = logging.getLogger(__name__)


def get_run_name(model: str) -> str:
    """
    Generates a unique run name based on model name, timestamp, and process ID.

    Args:
        model: Name of the model being used

    Returns:
        Formatted run name string
    """
    return f"{model}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}_pid_{os.getpid()}"


def deep_flatten_dict(d: dict) -> dict:
    """
    Smarter deep flattens a nested dictionary, handling lists of
    dictionaries by using a 'name' key to create the flattened key.
    All final values are converted to strings, with list items
    joined by a comma.

    This function recursively traverses a dictionary. If it encounters a
    nested dictionary, it combines the keys with a dot '.' to create a
    single, flattened key. If it encounters a list, it checks for
    dictionaries within the list and uses their 'name' key to further
    flatten the structure. All values are then converted to strings.

    Args:
        d: The input dictionary to be flattened.

    Returns:
        A new dictionary with all nested keys flattened into a single level,
        and all values as strings.
    """
    flattened = {}

    def _flatten(current_item, prefix: str = ""):
        """
        Helper function for recursive flattening.
        """
        if isinstance(current_item, dict):
            for key, value in current_item.items():
                new_key = f"{prefix}.{key}" if prefix else key
                _flatten(value, f"{new_key}")
        elif isinstance(current_item, list):
            # Check if this is a list of dictionaries with 'name' keys
            is_named_list = any(
                isinstance(item, dict) and "name" in item for item in current_item
            )
            if is_named_list:
                for item in current_item:
                    if isinstance(item, dict) and "name" in item:
                        # Use the 'name' key from the dictionary within the list
                        name = item["name"]
                        # Recurse on the item, using the name as a sub-key
                        _flatten(item, f"{prefix}{name}.")
            else:
                # This case handles simple lists like "metric_managers"
                value_to_log = ", ".join(map(str, current_item))
                flattened[prefix[:-1]] = value_to_log
        else:
            # Base case: add the key-value pair to the flattened dictionary
            value_to_log = str(current_item)
            if prefix.endswith("."):
                flattened[prefix[:-1]] = value_to_log
            else:
                flattened[prefix] = value_to_log

    _flatten(d)
    return flattened


def set_seeds(seed: int = 2025, deterministic: bool = True) -> None:
    """
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 2024.
    """
    log.debug(f"Setting seed to: {seed}")
    # Set the seed for general torch operations
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.use_deterministic_algorithms(deterministic)
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                ":16:8"  # In order to prevent errors with cuBLAS
            )

    log.info(f"Seeds set: seed={seed}, deterministic={deterministic}")


def set_seeds_local(seed: int = 2025, deterministic: bool = True) -> None:
    """
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 2024.
    """
    log.debug(f"Setting seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.use_deterministic_algorithms(deterministic)
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                ":16:8"  # In order to prevent errors with cuBLAS
            )

    log.info(f"Seeds set: seed={seed}, deterministic={deterministic}")


def seed_worker(worker_id: int, base_seed: int | None = None) -> None:
    """Initialize random state for DataLoader worker processes.

    Each worker gets deterministic seed based on PyTorch's initial_seed
    (includes base seed + worker_id + epoch from generator).
    """
    worker_seed = (base_seed or torch.initial_seed()) + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_epoch_generator(epoch: int, base_seed: int | None = None) -> torch.Generator:
    """Create deterministic generator for epoch-aware shuffling.

    Enables reproducible shuffling that varies by epoch.
    Same as frame sampling pattern: seed + epoch.
    """
    if base_seed is None:
        base_seed = torch.initial_seed()
    generator = torch.Generator()
    generator.manual_seed(base_seed + epoch)
    return generator


def initialise_dirs(dirs: list[str]):
    """
    Initialises all the required directories.
    """
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def crop_face_with_bbox(
    image: np.ndarray,
    bbox: list[int],
    margin: tuple = (0, 0),
) -> tuple[np.ndarray, list[int]]:
    """
    Crop face from image using provided bbox with margin support.

    Args:
        image: Input image as numpy array
        bbox: Bounding box as [x, y, w, h] format

    Returns:
        Tuple of (cropped_face_image, adjusted_bboxes)
    """
    # Convert to [x1, y1, x2, y2] format
    x, y, w, h = bbox
    original_bbox = [x, y, x + w, y + h]

    # Apply margin if configured
    if margin != (0, 0):
        adjusted_bbox = apply_margin_to_bbox(
            original_bbox, image.shape[0], image.shape[1]
        )
    else:
        adjusted_bbox = original_bbox

    # Ensure bbox is within image bounds
    adjusted_bbox = [
        max(0, adjusted_bbox[0]),
        max(0, adjusted_bbox[1]),
        min(adjusted_bbox[2], image.shape[1]),
        min(adjusted_bbox[3], image.shape[0]),
    ]

    # Crop the face
    face_image = image[
        adjusted_bbox[1] : adjusted_bbox[3], adjusted_bbox[0] : adjusted_bbox[2]
    ]

    return face_image, adjusted_bbox


def apply_margin_to_bbox(
    bbox: list[int],
    image_height: int,
    image_width: int,
    margin: tuple[int, int] = (0, 0),
) -> list[int]:
    """
    Apply margin to bounding box coordinates based on face crop dimensions.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        image_height: Original image height
        image_width: Original image width

    Returns:
        Adjusted bounding box with margin applied: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox

    # Calculate face crop dimensions
    face_height = y2 - y1
    face_width = x2 - x1

    # Get margin percentages
    mh, mw = margin

    # Calculate margin amounts based on face dimensions
    height_margin = int(face_height * mh / 100)
    width_margin = int(face_width * mw / 100)

    # Apply margins and clamp to image boundaries
    new_x1 = max(0, x1 - width_margin)
    new_y1 = max(0, y1 - height_margin)
    new_x2 = min(image_width, x2 + width_margin)
    new_y2 = min(image_height, y2 + height_margin)

    return [new_x1, new_y1, new_x2, new_y2]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{val" + self.fmt + "}({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters

    def display(self, batch):
        entries = [self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return " ".join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class MeterTracker:
    """
    Automatically tracks multiple AverageMeters for step metrics.

    Features:
    - Auto-creates meters for any metric returned by step functions
    - Updates all meters automatically
    - Configurable progress bar display
    - No manual meter management needed

    Usage:
        tracker = MeterTracker()
        step_metrics = {"loss": 0.5, "accuracy": 0.95}
        tracker.update(step_metrics, batch_size=32)
        pbar.set_postfix(tracker.get_postfix(["loss", "accuracy"]))
        final_metrics = tracker.get_averages()
    """

    def __init__(self):
        self.meters: dict[str, AverageMeter] = {}

    @staticmethod
    def _to_float_scalar(value: object) -> float | None:
        """
        Convert common numeric containers to Python float.

        Supports scalar tensors/arrays and falls back to mean for multi-element
        tensors/arrays to prevent formatting/runtime failures in logging.
        """
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            if value.numel() == 1:
                return float(value.detach().item())
            return float(value.detach().mean().item())

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            if value.size == 1:
                return float(value.item())
            return float(value.mean())

        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def update(self, metrics: dict[str, object], batch_size: int) -> None:
        """
        Update all meters with step metrics.
        Auto-creates meters on first encounter.

        Args:
            metrics: Dictionary of metric_name -> value
            batch_size: Batch size for weighted averaging
        """
        for key, value in metrics.items():
            scalar = self._to_float_scalar(value)
            if scalar is None:
                continue
            if key not in self.meters:
                # Auto-create meter on first encounter
                self.meters[key] = AverageMeter(key, ":.4f")

            self.meters[key].update(scalar, batch_size)

    def get_averages(self) -> dict[str, float]:
        """
        Get average values for all tracked meters.

        Returns:
            Dictionary of metric_name -> average_value
        """
        return {key: meter.avg for key, meter in self.meters.items()}

    def get_postfix(self, keys: list[str] | None = None) -> dict[str, str]:
        """
        Get formatted postfix dict for progress bar.

        Args:
            keys: List of metric names to include. If None, includes all.

        Returns:
            Dictionary formatted for tqdm.set_postfix()
        """
        if keys is None:
            keys = list(self.meters.keys())

        postfix = {}
        for key in keys:
            if key in self.meters:
                postfix[key] = f"{self.meters[key].avg:.5f}"

        return postfix

    def reset(self) -> None:
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()
