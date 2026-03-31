"""
Core framework components.
"""

# Base classes
from .base_accelerator import BaseAccelerator
from .base_criterion import BaseCriterion
from .base_dataset import (
    AdaptiveComputationDataset,
    BaseWrapper,
    FrameWrapper,
    TextSequenceWrapper,
)
from .base_detector import BaseFaceDetector, FaceDetection, LandmarkDetection
from .base_executor import BaseExecutor
from .base_metrics_source import BaseMetricsSource
from .base_metric_manager import BaseMetricManager
from .base_biometric_model import BaseBiometricModel
from .base_model import BaseModel
from .base_sampler import BaseSampler
from .base_tracker import BaseTracker
from .adaptive_computation_trainer import (
    AdaptiveComputationStepOutput,
    AdaptiveComputationTrainer,
    CarryState,
)
from .base_callback import Callback
from .base_trainer import BaseTrainer
from .epoch_trainer import EpochTrainer
from .sequence_trainer import SequenceStepOutput, SequenceTrainer
from .base_transform import BaseTransform

# Registry system
from .registry import (
    ACCELERATOR_REGISTRY,
    AUGMENTATION_REGISTRY,
    CALLBACK_REGISTRY,
    CRITERION_REGISTRY,
    DATASET_REGISTRY,
    EXECUTOR_REGISTRY,
    FACE_DETECTOR_REGISTRY,
    METRIC_MANAGER_REGISTRY,
    METRICS_SOURCE_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    BIOMETRIC_PREPROCESSOR_REGISTRY,
    SAMPLER_REGISTRY,
    SCHEDULER_REGISTRY,
    TRACKER_REGISTRY,
    TRAINER_REGISTRY,
    ComponentRegistry,
    get_all_registered_components,
    print_registry_info,
    register_accelerator,
    register_augmentation,
    register_callback,
    register_criterion,
    register_dataset,
    register_executor,
    register_face_detector,
    register_metric,
    register_metric_manager,
    register_metrics_source,
    register_biometric_preprocessor,
    register_model,
    register_optimizer,
    register_sampler,
    register_scheduler,
    register_tracker,
    register_trainer,
)

__all__ = [
    # Base classes
    "BaseAccelerator",
    "BaseModel",
    "BaseBiometricModel",
    "BaseCriterion",
    "BaseWrapper",
    "FrameWrapper",
    "TextSequenceWrapper",
    "AdaptiveComputationDataset",
    "BaseExecutor",
    "BaseTracker",
    "BaseMetricsSource",
    "BaseMetricManager",
    "BaseSampler",
    "Callback",
    "BaseTrainer",
    "EpochTrainer",
    "SequenceTrainer",
    "SequenceStepOutput",
    "AdaptiveComputationTrainer",
    "AdaptiveComputationStepOutput",
    "CarryState",
    "BaseTransform",
    "BaseFaceDetector",
    "FaceDetection",
    "LandmarkDetection",
    # Registry system
    "ComponentRegistry",
    "MODEL_REGISTRY",
    "TRAINER_REGISTRY",
    "DATASET_REGISTRY",
    "CRITERION_REGISTRY",
    "METRIC_REGISTRY",
    "METRIC_MANAGER_REGISTRY",
    "CALLBACK_REGISTRY",
    "FACE_DETECTOR_REGISTRY",
    "ACCELERATOR_REGISTRY",
    "AUGMENTATION_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "BIOMETRIC_PREPROCESSOR_REGISTRY",
    "SAMPLER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "EXECUTOR_REGISTRY",
    "TRACKER_REGISTRY",
    "METRICS_SOURCE_REGISTRY",
    "register_model",
    "register_trainer",
    "register_dataset",
    "register_face_detector",
    "register_criterion",
    "register_metric",
    "register_callback",
    "register_accelerator",
    "register_augmentation",
    "register_optimizer",
    "register_sampler",
    "register_scheduler",
    "register_executor",
    "register_tracker",
    "register_metrics_source",
    "register_metric_manager",
    "register_biometric_preprocessor",
    "get_all_registered_components",
    "print_registry_info",
]
