"""
Component registry system for automatic registration and discovery.

This module provides a decorator-based system for registering models, trainers,
datasets, and loss functions, eliminating the need for manual registration in
__init__.py files.

Example usage:
    @MODEL_REGISTRY.register("resnet")
    class Resnet(BaseModel):
        # ... implementation ...

    # Later, get the model:
    model = MODEL_REGISTRY.get("resnet18", model_name, config, log)
"""

from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Type, Union

logger = getLogger(__name__)


class ComponentRegistry:
    """
    Registry for automatic component discovery and instantiation.

    Supports both exact matching and prefix matching for flexible component naming.
    """

    def __init__(self, component_type: str = "Component"):
        """
        Initialize the registry.

        Args:
            component_type: Name of the component type for error messages
        """
        self._components: Dict[str, Type] = {}
        self.component_type = component_type

    def regiter_class(self, name: str, cls: Type):
        """
        Register a component class directly.
        Args:
            name: Name to register the class under
            cls: Class to register
        """
        if name in self._components:
            existing_cls = self._components[name]
            if existing_cls != cls:
                raise ValueError(
                    f"{self.component_type} '{name}' is already registered "
                    "with class "
                    f"{existing_cls.__name__}, cannot register {cls.__name__}"
                )
        else:
            self._components[name] = cls

    def register(self, names: Union[str, List[str]]):
        """
        Decorator to register a component class.

        Args:
            names: Single name or list of names to register the class under

        Returns:
            Decorator function
        """

        def decorator(cls: Type) -> Type:
            # Convert single name to list
            names_list = [names] if isinstance(names, str) else names

            # Register class under all provided names
            for name in names_list:
                if name in self._components:
                    existing_cls = self._components[name]
                    if existing_cls != cls:
                        raise ValueError(
                            f"{self.component_type} '{name}' is already registered "
                            "with class "
                            f"{existing_cls.__name__}, cannot register {cls.__name__}"
                        )
                else:
                    self._components[name] = cls

            return cls

        return decorator

    def get(self, name: str, *args, **kwargs) -> Any:
        """
        Get and instantiate a registered component.

        Args:
            name: Name of the component to get (supports prefix matching)
            *args: Arguments to pass to component constructor
            **kwargs: Keyword arguments to pass to component constructor

        Returns:
            Instantiated component

        Raises:
            NotImplementedError: If component is not found
        """

        # First try exact match
        if name in self._components:
            cls = self._components[name]
            return cls(*args, **kwargs)

        # Then try prefix matching for flexible naming (e.g., resnet18 -> resnet)
        for registered_name, cls in self._components.items():
            if name.startswith(registered_name):
                return cls(*args, **kwargs)

        # If no match found, provide helpful error message
        available_names = list(self._components.keys())
        raise NotImplementedError(
            f"{self.component_type} '{name}' not found. "
            f"Available {self.component_type.lower()}s: {available_names}"
        )

    def list_registered(self) -> List[str]:
        """
        Get list of all registered component names.

        Returns:
            List of registered component names
        """
        return list(self._components.keys())

    def is_registered(self, name: str) -> bool:
        """
        Check if a component name is registered.

        Args:
            name: Component name to check

        Returns:
            True if registered, False otherwise
        """
        # Check exact match
        if name in self._components:
            return True

        # Check prefix match
        for registered_name in self._components.keys():
            if name.startswith(registered_name):
                return True

        return False

    def get_class(self, name: str) -> Type:
        """
        Get the class without instantiating it.

        Args:
            name: Component name

        Returns:
            Component class

        Raises:
            NotImplementedError: If component is not found
        """

        if not name:
            raise ValueError(f"{self.component_type} name cannot be empty")

        # Debug logging
        logger.debug(f"Looking for {self.component_type} '{name}'")
        logger.debug(
            f"Available {self.component_type.lower()}s: {list(self._components.keys())}"
        )

        # First try exact match
        if name in self._components:
            return self._components[name]

        # Then try prefix matching
        for registered_name, cls in self._components.items():
            if name.startswith(registered_name):
                return cls

        # If no match found, provide helpful error message
        available_names = list(self._components.keys())
        raise NotImplementedError(
            f"{self.component_type} '{name}' not found. "
            f"Available {self.component_type.lower()}s: {available_names}"
        )

    def get_registered_names_for_class(self, cls: Type) -> List[str]:
        """
        Get all registered names that resolve to a specific class.

        Args:
            cls: Registered component class

        Returns:
            Registered names for the class, in registration order
        """
        return [
            name
            for name, registered_cls in self._components.items()
            if registered_cls == cls
        ]


# Global registries for different component types
MODEL_REGISTRY = ComponentRegistry("Model")
TRAINER_REGISTRY = ComponentRegistry("Trainer")
DATASET_REGISTRY = ComponentRegistry("Dataset")
CRITERION_REGISTRY = ComponentRegistry("Criterion")
METRIC_MANAGER_REGISTRY = ComponentRegistry("MetricManager")
CALLBACK_REGISTRY = ComponentRegistry("Callback")
ACCELERATOR_REGISTRY = ComponentRegistry("Accelerator")
METRIC_REGISTRY = ComponentRegistry("Metric")
AUGMENTATION_REGISTRY = ComponentRegistry("Augmentation")
SAMPLER_REGISTRY = ComponentRegistry("Sampler")
OPTIMIZER_REGISTRY = ComponentRegistry("Optimizer")
SCHEDULER_REGISTRY = ComponentRegistry("Scheduler")
EXECUTOR_REGISTRY = ComponentRegistry("Executor")
TRACKER_REGISTRY = ComponentRegistry("Tracker")
METRICS_SOURCE_REGISTRY = ComponentRegistry("MetricsSource")
BIOMETRIC_PREPROCESSOR_REGISTRY = ComponentRegistry("BiometricPreprocessor")
FACE_DETECTOR_REGISTRY = ComponentRegistry("FaceDetector")


def register_model(names: Union[str, List[str]]):
    """Convenience decorator for registering models."""
    return MODEL_REGISTRY.register(names)


def register_trainer(names: Union[str, List[str]]):
    """Convenience decorator for registering trainers."""
    return TRAINER_REGISTRY.register(names)


def register_dataset(names: Union[str, List[str]]):
    """Convenience decorator for registering datasets."""
    return DATASET_REGISTRY.register(names)


def register_criterion(names: Union[str, List[str]]):
    """Convenience decorator for registering criterions."""
    return CRITERION_REGISTRY.register(names)


def register_metric_manager(names: Union[str, List[str]]):
    """Convenience decorator for registering metric managers."""
    return METRIC_MANAGER_REGISTRY.register(names)


def register_callback(names: Union[str, List[str]]):
    """Convenience decorator for registering callbacks."""
    return CALLBACK_REGISTRY.register(names)


def register_accelerator(names: Union[str, List[str]]):
    """Convenience decorator for registering accelerators."""
    return ACCELERATOR_REGISTRY.register(names)


def register_metric(names: Union[str, List[str]]):
    """Convenience decorator for registering metrics."""
    return METRIC_REGISTRY.register(names)


def register_augmentation(names: Union[str, List[str]]):
    """Convenience decorator for registering augmentations."""
    return AUGMENTATION_REGISTRY.register(names)


def register_sampler(names: Union[str, List[str]]):
    """Convenience decorator for registering samplers."""
    return SAMPLER_REGISTRY.register(names)


def register_optimizer(names: Union[str, List[str]]):
    """Convenience decorator for registering optimizers."""
    return OPTIMIZER_REGISTRY.register(names)


def register_scheduler(names: Union[str, List[str]]):
    """Convenience decorator for registering schedulers."""
    return SCHEDULER_REGISTRY.register(names)


def register_executor(names: Union[str, List[str]]):
    """Convenience decorator for registering executors."""
    return EXECUTOR_REGISTRY.register(names)


def register_tracker(names: Union[str, List[str]]):
    """Convenience decorator for registering trackers."""
    return TRACKER_REGISTRY.register(names)


def register_metrics_source(names: Union[str, List[str]]):
    """Convenience decorator for registering metrics sources."""
    return METRICS_SOURCE_REGISTRY.register(names)



def register_biometric_preprocessor(names: Union[str, List[str]]):
    """Convenience decorator for registering biometric preprocessors."""
    return BIOMETRIC_PREPROCESSOR_REGISTRY.register(names)


def register_face_detector(names: Union[str, List[str]]):
    """Convenience decorator for registering face detectors."""
    return FACE_DETECTOR_REGISTRY.register(names)


# Utility functions for registry information
def get_all_registered_components() -> Dict[str, List[str]]:
    """
    Get all registered components across all registries.

    Returns:
        Dictionary mapping component type to list of registered names
    """
    return {
        "models": MODEL_REGISTRY.list_registered(),
        "trainers": TRAINER_REGISTRY.list_registered(),
        "datasets": DATASET_REGISTRY.list_registered(),
        "criterions": CRITERION_REGISTRY.list_registered(),
        "metric_managers": METRIC_MANAGER_REGISTRY.list_registered(),
        "callbacks": CALLBACK_REGISTRY.list_registered(),
        "accelerator": ACCELERATOR_REGISTRY.list_registered(),
        "metric": METRIC_REGISTRY.list_registered(),
        "augmentations": AUGMENTATION_REGISTRY.list_registered(),
        "samplers": SAMPLER_REGISTRY.list_registered(),
        "optimizers": OPTIMIZER_REGISTRY.list_registered(),
        "schedulers": SCHEDULER_REGISTRY.list_registered(),
        "executors": EXECUTOR_REGISTRY.list_registered(),
        "trackers": TRACKER_REGISTRY.list_registered(),
        "metrics_sources": METRICS_SOURCE_REGISTRY.list_registered(),
        "biometric_preprocessors": BIOMETRIC_PREPROCESSOR_REGISTRY.list_registered(),
        "face_detectors": FACE_DETECTOR_REGISTRY.list_registered(),
    }


def print_registry_info(log: Optional[Logger] = None):
    """Print information about all registered components."""
    components = get_all_registered_components()

    if log is None:
        print("🔧 DL Training Framework - Registered Components")
        print("=" * 50 + "\n")
    else:
        log.info("🔧 DL Training Framework - Registered Components")
        log.info("=" * 50 + "\n")

    for component_type, names in components.items():
        if log is None:
            print(f"\n📦 {component_type.title()}:")
        else:
            log.info(f"📦 {component_type.title()}:")
        if names:
            for name in sorted(names):
                if log is None:
                    print(f"   • {name}")
                else:
                    log.info(f"   • {name}")
        else:
            if log is None:
                print("   (none registered)")
            else:
                log.info("   (none registered)")

    if log is None:
        print(f"\nTotal components: {sum(len(names) for names in components.values())}")
    else:
        log.info(
            f"\nTotal components: {sum(len(names) for names in components.values())}"
        )
