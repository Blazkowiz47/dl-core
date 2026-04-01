"""
Epoch-based trainer foundation with pluggable metric managers.

This module keeps the shared training loop, accelerator integration, callback
hooks, and checkpoint flow used by the built-in trainers. `BaseTrainer`
remains as a compatibility alias for the epoch-based implementation so
existing imports continue to work.
"""

import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_core.core import BaseAccelerator, BaseCriterion, BaseModel, BaseWrapper
from dl_core.core.base_callback import Callback, CallbackList
from dl_core.core.base_metric_manager import BaseMetricManager
from dl_core.utils import (
    MeterTracker,
    set_seeds,
    ArtifactManager,
    ExponentialMovingAverage,
)
from dl_core.utils.config_names import (
    resolve_config_experiment_name,
    resolve_config_run_name,
)

from .registry import (
    ACCELERATOR_REGISTRY,
    CALLBACK_REGISTRY,
    DATASET_REGISTRY,
    METRIC_MANAGER_REGISTRY,
)

# Module-level logger
logger = logging.getLogger(__name__)


class EpochTrainer(ABC):
    """
    Epoch-based trainer base class for supervised training workloads.

    Provides standardized training pipeline with:
    - Dictionary-based interfaces for models, criterions, optimizers, schedulers
    - Pluggable metric management for different domains
    - Multi-GPU training support via accelerators
    - Checkpoint management with full state preservation
    - Callback system for extensibility
    - Automatic meter tracking for step metrics

    Architecture:
        - Public methods: High-level interface (train_epoch, save_checkpoint, etc.)
        - Internal methods: Implementation details (prefixed with _)
        - Abstract methods: Must implement in subclasses (train_step, setup_model, etc.)
        - Hook methods: Optional overrides (preprocess_batch, generate_epoch_logs, etc.)

    Subclass Requirements:
        Must implement:
        - setup_model() - Initialize models in self.models dict
        - setup_criterion() - Initialize loss functions in self.criterions dict
        - setup_optimizer() - Initialize optimizers in self.optimizers dict
        - setup_scheduler() - Initialize LR schedulers in self.schedulers dict
        - train_step() - Per-batch training logic
        - test_step() - Per-batch test/evaluation logic
        - validation_step() - Per-batch validation logic

    Optional Overrides:
        - preprocess_batch() - Custom batch preprocessing
        - generate_epoch_logs() - Custom epoch-level metrics
        - register_model_states() - Custom checkpoint state handling
        - load_model_states() - Custom checkpoint loading

    Usage:
        class MyTrainer(BaseTrainer):
            def setup_model(self):
                self.models['main'] = MyModel(self.config)

            def train_step(self, batch_data, batch_idx):
                output = self.models['backbone'](batch_data['image'])
                loss = self.criterions['ce'](output, batch_data['label'])
                self.accelerator.backward(loss)
                self.accelerator.optimizer_step(self.optimizers['main'], self.model)
                return {'loss': loss.item()}

        trainer = MyTrainer(config)
        trainer.run()
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the base trainer.

        Args:
            config: Configuration dictionary
        """

        self.model_not_initialised_error = ValueError(
            "Model is not initialized. Call setup_model() first."
        )
        self.optimizer_not_initialised_error = ValueError(
            "Optimizer is not initialized. Call setup_optimizer() first."
        )
        self.criterion_not_initialised_error = ValueError(
            "Criterion is not initialized. Call setup_criterion() first."
        )
        self.dataset_not_initialised_error = ValueError(
            "Dataset is not initialized. Call setup_dataset() first."
        )

        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.seed = config.get("seed", 42)

        self.accelerator: BaseAccelerator
        self.dataset_wrapper: BaseWrapper
        self.artifact_manager: ArtifactManager
        self.callbacks: CallbackList

        self.models: dict[str, BaseModel] = {}
        self.criterions: dict[str, BaseCriterion] = {}
        self.optimizers: dict[str, Optimizer] = {}
        self.schedulers: dict[str, LRScheduler] = {}
        self.data_loader: dict[str, DataLoader | None] = {}
        self.metric_managers: dict[str, BaseMetricManager] = {}
        self.ema: ExponentialMovingAverage | None = None
        self.use_ema_for_eval = False

        self.metrics_history: dict[str, dict[int, dict[str, float]]] = {
            "train": {},
            "validation": {},
            "test": {},
            "general": {},
        }

        # Initialize training parameters
        trainer_config = config.get("trainer", {})
        try:
            trainer_name, trainer_config = next(iter(trainer_config.items()))
        except Exception:
            trainer_name = self.__class__.__name__
            trainer_config = {"epochs": 10, "pbar_metrics": ["loss"]}
            self.logger.warning(
                f"Trainer configuration missing, using default settings for {trainer_name}",
            )

        self.trainer_name = trainer_name
        self.trainer_config = trainer_config
        self.logger.info(f"Using trainer: {trainer_name}")
        self.logger.info(f"Trainer configuration: {trainer_config}")
        self.epochs = trainer_config["epochs"]
        self.show_progress = trainer_config.get("show_progress", False)
        self.print_freq = trainer_config.get("print_freq", 100)
        self.continue_model = trainer_config.get("continue_model")
        self.skip_baseline_eval = trainer_config.get("skip_baseline_eval", False)
        self.test_frequency = trainer_config.get("test_frequency", 1)
        self.validation_frequency = trainer_config.get("validation_frequency", 1)
        self.log_weights = trainer_config.get(
            "log_weights",
            trainer_config.get("log-weights", False),
        )
        self.overfit_single_batch_enabled = trainer_config.get(
            "overfit_single_batch", False
        )
        self.overfit_iterations = trainer_config.get("overfit_iterations", 1000)
        self.overfit_num_batches = trainer_config.get("overfit_num_batches", 8)
        self.pbar_metrics: dict[str, list[str]] = trainer_config.get("pbar_metrics")
        if self.pbar_metrics is None:
            self.pbar_metrics = {
                "train": ["loss"],
                "validation": ["loss"],
                "test": ["loss"],
            }
        elif isinstance(self.pbar_metrics, list):
            self.pbar_metrics = {
                "train": self.pbar_metrics,
                "validation": self.pbar_metrics,
                "test": self.pbar_metrics,
            }

        # Training state
        self.current_epoch: int
        self.best_metric: float | None = None
        self.epochs_no_improvement: int = 0
        self.global_step: int = 0

        self.stop_training = False  # For early stopping
        self.current_checkpoint: dict[str, Any] | None = None
        self.current_checkpoint_epoch: int | None = None

        self.logger.info(f"Initialized {self.__class__.__name__}")

        self.setup_artifact_manager()
        self.logger.info("Artifact manager is set up")

    # =======================================================================
    # Internal methods
    # ======================================================================
    def _checkpoint_dir_cleanup(self) -> None:
        """
        Clean up empty checkpoint directories.

        Called after training to remove artifact directories that contain no files.
        Only runs on main process (rank 0).
        """
        if not self.accelerator.is_main_process():
            return

        if os.path.isdir(self.checkpoint_dir) and not os.listdir(self.checkpoint_dir):
            os.system(f"rm -rf {self.artifact_manager.output_dir}")

    def _load_continue_model(self) -> None:
        # Load checkpoint if specified in trainer config (resume training)
        # This is different from loading pretrained weights - it restores full training state
        if not self.continue_model:
            return

        self.logger.info(f"Resuming training from checkpoint: {self.continue_model}")
        self.load_checkpoint(self.continue_model)
        self.accelerator.wait_for_everyone("after loading checkpoint")

    def _run(self) -> None:
        """
        Main training loop entry point.
        """
        run_status = "completed"
        error_message: str | None = None
        try:
            self.setup()
            self.logger.info("Setup complete, starting training")
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            traceback.print_exc()
            exit(1)

        self.setup_current_epoch(0)

        try:
            self.load_continue_model()
        except Exception as e:
            self.logger.error(f"Failed to load continue model: {e}")
            traceback.print_exc()

        self.accelerator.wait_for_everyone("before training start")

        try:
            # Dispatch to appropriate experiment type
            if self.overfit_single_batch_enabled:
                self.logger.info(
                    f"=== OVERFIT SINGLE BATCH MODE === "
                    f"(iterations={self.overfit_iterations})"
                )
                self.overfit_single_batch()
            else:
                self.perform_training()
        except Exception as e:
            run_status = "failed"
            error_message = str(e)
            self.logger.error(f"Training failed: {e}")
            traceback.print_exc()
            self._checkpoint_dir_cleanup()
            raise

        finally:
            self.accelerator.wait_for_everyone("before on_training_end callbacks")
            try:
                self.persist_run_analysis(
                    status=run_status,
                    error_message=error_message,
                )
            except Exception as analysis_error:
                self.logger.warning(
                    f"Failed to persist run analysis artifacts: {analysis_error}"
                )
            try:
                final_logs = {
                    "final_epoch": self.current_epoch,
                    "total_epochs": self.epochs,
                    "status": run_status,
                }
                if error_message is not None:
                    final_logs["error_message"] = error_message
                self.callbacks.on_training_end(final_logs)
            except Exception as upload_error:
                self.logger.error(
                    f"Failed to upload artifacts after training failure: {upload_error}"
                )
            self.finalize_training()
            try:
                self.callbacks.on_training_finalized(final_logs)
            except Exception as upload_error:
                self.logger.error(
                    "Failed to upload finalized artifacts after training "
                    f"failure: {upload_error}"
                )

    def _inject_seed_into_configs(self) -> None:
        # Trainer configs
        self.config["trainer"]["seed"] = self.seed
        # Dataset config - check if single dataset (has 'name' key) or multiple datasets
        self.config["dataset"]["seed"] = self.seed
        # Model configs
        for model_name in self.config["models"]:
            self.config["models"][model_name]["seed"] = self.seed

        # Accelerator configs
        self.config["accelerator"]["seed"] = self.seed

    def _setup_accelerator(self) -> None:
        # Create accelerator FIRST (before other components)
        accelerator_config = self.config.get("accelerator", {})
        if not accelerator_config:
            # Fallback for legacy configs that have accelerator in runtime
            accelerator_config = self.config.get("runtime", {}).get("accelerator", {})

        accelerator_config = accelerator_config.copy()
        self.accelerator = ACCELERATOR_REGISTRY.get(
            accelerator_config.get("type", "cpu"),
            accelerator_config,
        )
        self.logger.info(f"Using accelerator: {self.accelerator.__class__.__name__}")
        self.accelerator.wait_for_everyone()
        self.logger.info("All ranks synchronized after accelerator setup")

    def _setup(self) -> None:
        """
        Setup training components.
        """
        self.logger.info("Setting up training components...")
        # Set random seeds
        set_seeds(self.seed)
        self._inject_seed_into_configs()

        self.setup_accelerator()
        self.logger.info("Accelerator is set up")

        self.setup_data()
        self.logger.info("Data loaders are set up")

        self.setup_model()
        self.logger.info("Model is set up")

        self.setup_criterion()
        self.logger.info("Criterions are set up")

        self.setup_optimizer()
        self.logger.info("Optimizers are set up")

        self.setup_scheduler()
        self.logger.info("Schedulers are set up")

        self.setup_metrics()
        self.logger.info("Metric managers are set up")

        self.accelerator.wait_for_everyone("before preparing components")
        # Prepare model, optimizers, criterions, schedulers, and dataloaders with accelerator
        # This wraps them for the target device (CPU/single GPU/multi-GPU)
        # Accelerator handles device placement, DDP wrapping, and distributed sampler

        (
            self.models,
            self.optimizers,
            self.criterions,
            self.schedulers,
            self.data_loader,
        ) = self.accelerator.prepare(
            models=self.models,
            optimizers=self.optimizers,
            criterions=self.criterions,
            schedulers=self.schedulers,
            dataloaders=self.data_loader,
        )
        self.logger.info("All components prepared with accelerator")

        self.logger.info(
            f"Prepared components with accelerator: "
            f"models={list(self.models.keys())}, "
            f"optimizers={list(self.optimizers.keys())}, "
            f"criterions={list(self.criterions.keys())}, "
            f"schedulers={list(self.schedulers.keys())}, "
            f"dataloaders={list(self.data_loader.keys())}"
        )

        self.accelerator.wait_for_everyone("after preparing components")
        self.setup_ema()
        self.logger.info("EMA setup is complete")

        self.accelerator.wait_for_everyone("after ema setup")
        # Setup callbacks
        self.setup_callbacks()
        self.logger.info("Callbacks are set up")
        self.logger.info("Training setup completed")

    def _setup_ema(self) -> None:
        """Setup optional EMA manager and attach it to accelerator."""
        ema_config = self.config.get("ema", {})
        if not isinstance(ema_config, dict):
            self.logger.warning("EMA config must be a dict; disabling EMA")
            self.accelerator.set_ema_manager(None)
            return

        if not ema_config.get("enabled", False):
            self.accelerator.set_ema_manager(None)
            return

        model_keys = ema_config.get("models")
        if model_keys is None:
            selected_keys = list(self.models.keys())
        elif isinstance(model_keys, list) and all(
            isinstance(item, str) for item in model_keys
        ):
            selected_keys = model_keys
        else:
            self.logger.warning(
                "EMA config key 'models' must be a list[str]; using all models"
            )
            selected_keys = list(self.models.keys())

        missing_keys = [key for key in selected_keys if key not in self.models]
        if missing_keys:
            raise ValueError(f"EMA references unknown model keys: {missing_keys}")

        unwrapped_models = {
            key: self.accelerator.unwrap_model(self.models[key])
            for key in selected_keys
        }
        self.ema = ExponentialMovingAverage(
            models=unwrapped_models,
            decay=float(ema_config.get("decay", 0.9999)),
            update_after_step=int(ema_config.get("update_after_step", 0)),
            update_every=int(ema_config.get("update_every", 1)),
            save_in_checkpoint=bool(ema_config.get("save_in_checkpoint", True)),
        )
        self.use_ema_for_eval = bool(ema_config.get("eval_with_ema", False))
        self.accelerator.set_ema_manager(self.ema)
        self.logger.info(
            "EMA enabled for models=%s, decay=%s, update_after_step=%s, "
            "update_every=%s, eval_with_ema=%s",
            selected_keys,
            ema_config.get("decay", 0.9999),
            ema_config.get("update_after_step", 0),
            ema_config.get("update_every", 1),
            self.use_ema_for_eval,
        )

    def _get_eval_param_context(self) -> ContextManager[Any]:
        """Return EMA parameter swap context for evaluation when configured."""
        if self.ema is None or not self.use_ema_for_eval:
            return nullcontext()
        return self.ema.average_parameters()

    def _setup_data(self) -> None:
        """Setup data loaders."""
        self.logger.debug("Setting data")
        self.dataset_wrapper = DATASET_REGISTRY.get(
            self.config["dataset"]["name"], self.config["dataset"]
        )
        if self.dataset_wrapper.auto_split:
            self.dataset_wrapper.auto_generate_partitions()

        loader_info = ""
        for split in ["train", "validation", "test"]:
            self.data_loader[split] = self.dataset_wrapper.get_split(split)
            if self.data_loader[split] is None:
                self.logger.warning(f"No data loader for split: {split}")
                continue

            capitalized_split = split.capitalize()
            stats = self.dataset_wrapper.get_stats(split)
            self.logger.info(f"{capitalized_split} dataset statistics:")
            for stat in stats:
                self.logger.info(f"  {stat}")
            dataloader = self.data_loader[split]
            if dataloader is not None:
                loader_info = f"{capitalized_split}: {len(dataloader)} batches"

        self.logger.info(f"Setup data loaders - {loader_info}")

    def _setup_artifact_manager(self) -> None:
        runtime_config = self.config.get("runtime", {})
        output_dir = runtime_config.get("output_dir", "artifacts")
        config_path = self.config.get("_config_path")
        experiment_name = resolve_config_experiment_name(
            self.config,
            config_path=config_path,
        )
        run_name = resolve_config_run_name(
            self.config,
            config_path=config_path,
            fallback=self.__class__.__name__,
        )
        sweep_file = self.config.get("sweep_file")
        if sweep_file:
            sweep_file = Path(sweep_file).name.replace(".yaml", "")

        # Initialize artifact manager with the flattened local artifact layout.
        self.artifact_manager = ArtifactManager(
            run_name=run_name,
            output_dir=output_dir,
            experiment_name=experiment_name,
            sweep_name=sweep_file,
        )
        self.checkpoint_dir = str(self.artifact_manager.get_checkpoints_dir())
        self.visualization_dir = str(self.artifact_manager.get_plots_dir())

        # Persist the full, merged configuration into artifacts for reproducibility
        try:
            self.artifact_manager.save_config(self.config)
        except Exception as e:
            self.logger.warning(f"Failed to save config into artifacts: {e}")

    def _setup_metrics(self) -> None:
        """
        Setup metric managers.
        """
        for manager_name in self.config["metric_managers"]:
            self.metric_managers[manager_name] = METRIC_MANAGER_REGISTRY.get(
                manager_name, self.config, self.accelerator, self
            )
        self.logger.info(
            f"Using metric managers: {', '.join(self.metric_managers.keys())}"
        )

    def _setup_callbacks(self) -> None:
        """
        Setup callbacks from configuration using CALLBACK_REGISTRY directly.

        Expects config format:
        {
            "callbacks": {
                "checkpoint": {"monitor": "eer", "mode": "min"},
                "early_stopping": {"patience": 10},
                ...
            }
        }
        """
        callback_instances = []
        callbacks_config = self.config.get("callbacks", {})

        if not isinstance(callbacks_config, dict):
            self.logger.warning(
                "Callbacks config must be a dictionary, got: %s", type(callbacks_config)
            )
            callbacks_config = {}

        for callback_name, callback_params in callbacks_config.items():
            if not callback_params:
                callback_params = {}
            try:
                callback = CALLBACK_REGISTRY.get(callback_name, **callback_params)
                callback_instances.append(callback)
            except Exception as e:
                self.logger.warning(f"Failed to create callback '{callback_name}': {e}")
                continue

        self.callbacks = CallbackList(callback_instances)
        self.callbacks.set_trainer(self)

        if callback_instances:
            callback_names = [cb.__class__.__name__ for cb in callback_instances]
            self.logger.info(f"Setup callbacks: {', '.join(callback_names)}")
        else:
            self.logger.debug("No callbacks configured")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint with support for multiple optimizers and schedulers.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not self.models:
            raise RuntimeError("Model must be setup before loading checkpoint")

        try:
            # Load checkpoint on appropriate device
            if self.accelerator.is_main_process():
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.accelerator.get_device(),
                    weights_only=False,
                )
            else:
                checkpoint = None

            if dist.is_initialized():
                checkpoint_list = [checkpoint]
                dist.broadcast_object_list(checkpoint_list, src=0)
                checkpoint = checkpoint_list[0]

            if checkpoint is None:
                raise RuntimeError("Failed to load checkpoint: checkpoint is None")

            # Load model states (allows subclasses to handle multiple models)
            self.load_model_states(checkpoint)

            # Load optimizer states
            for name, optimizer in self.optimizers.items():
                state_key = f"optimizer_{name}_state_dict"
                if state_key in checkpoint and checkpoint[state_key] is not None:
                    optimizer.load_state_dict(checkpoint[state_key])
                    self.logger.info(f"Loaded optimizer state for: {name}")

            # Load scheduler states
            for name, scheduler in self.schedulers.items():
                state_key = f"scheduler_{name}_state_dict"
                if state_key in checkpoint and checkpoint[state_key] is not None:
                    scheduler.load_state_dict(checkpoint[state_key])
                    self.logger.info(f"Loaded scheduler state for: {name}")

            # Restore accelerator state (scaler, etc.)
            self.accelerator.load_accelerator_state(checkpoint)

            # Load other training state for resuming training
            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
            if "callback_states" in checkpoint:
                for i, callback in enumerate(self.callbacks.callbacks):
                    key = f"callback_{i}_{callback.__class__.__name__}"
                    if key in checkpoint["callback_states"]:
                        callback.set_state(checkpoint["callback_states"][key])
                        self.logger.info(
                            f"Restored state for {callback.__class__.__name__}"
                        )
            if "epoch" in checkpoint:
                # Resume from next epoch
                self.setup_current_epoch(checkpoint["epoch"])
                self.logger.info(f"Resuming from epoch {self.current_epoch}")

            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            self.logger.info(
                f"Training state restored: epoch={self.current_epoch}, "
                f"best_metric={self.best_metric}, global_step={self.global_step}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _set_models_mode(self, mode: str) -> None:
        """
        Set all models in self.models to train or eval mode.

        Args:
            mode: Either 'train' or 'eval'

        Raises:
            ValueError: If mode is not 'train' or 'eval'
        """
        self.logger.info(f"Setting models to {mode} mode")
        for model in self.models.values():
            if mode == "train":
                model.train()
            elif mode == "eval":
                model.eval()
            else:
                raise ValueError(f"Invalid mode: {mode}")

    def _build_checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        """
        Build the checkpoint payload for one epoch.

        Args:
            epoch: Current epoch number
        """
        # Prepare criterion state dicts
        criterion_states = {}
        for name, criterion in self.criterions.items():
            if hasattr(criterion, "state_dict"):
                criterion_states[f"criterion_{name}_state_dict"] = (
                    criterion.state_dict()
                )

        # Prepare optimizer state dicts
        optimizer_states = {}
        for name, optimizer in self.optimizers.items():
            optimizer_states[f"optimizer_{name}_state_dict"] = optimizer.state_dict()

        # Prepare scheduler state dicts
        scheduler_states = {}
        for name, scheduler in self.schedulers.items():
            scheduler_states[f"scheduler_{name}_state_dict"] = scheduler.state_dict()

        # Create comprehensive checkpoint dict
        checkpoint_dict = {
            "epoch": epoch,
            "train_metrics": self.train_metrics.get(epoch, {}),
            "test_metrics": self.test_metrics.get(epoch, {}),
            "config": self.config,
            "best_metric": self.best_metric,
            "epochs_no_improvement": self.epochs_no_improvement,
            "global_step": self.global_step,
        }

        # Add criterion, optimizer and scheduler states
        checkpoint_dict.update(criterion_states)
        checkpoint_dict.update(optimizer_states)
        checkpoint_dict.update(scheduler_states)

        # Save callback states
        callback_states = {}
        for i, callback in enumerate(self.callbacks.callbacks):
            state = callback.get_state()
            if state:  # Only save non-empty states
                callback_states[f"callback_{i}_{callback.__class__.__name__}"] = state
        if callback_states:
            checkpoint_dict["callback_states"] = callback_states

        # Allow trainers to register their model states
        self.register_model_states(checkpoint_dict)

        # Add accelerator state (scaler, etc.)
        accelerator_state = self.accelerator.get_accelerator_state()
        if accelerator_state:
            checkpoint_dict.update(accelerator_state)

        return checkpoint_dict

    def _get_current_checkpoint(self, epoch: int) -> dict[str, Any]:
        """Return the cached checkpoint payload for the active epoch."""

        if (
            self.current_checkpoint is None
            or self.current_checkpoint_epoch != epoch
        ):
            self.current_checkpoint = self._build_checkpoint_payload(epoch)
            self.current_checkpoint_epoch = epoch
        return self.current_checkpoint

    def _save_checkpoint(self, epoch: int, filename: str | None = None) -> None:
        """
        Save model checkpoint with simplified approach.

        Args:
            epoch: Current epoch number
            filename: Optional checkpoint filename override
        """
        # Only save on main process (for DDP)
        if not self.accelerator.is_main_process():
            return

        checkpoint_dict = self._get_current_checkpoint(epoch)
        if filename is None:
            checkpoint_path = self.artifact_manager.get_epoch_checkpoint_path(epoch)
        else:
            checkpoint_path = self.artifact_manager.get_final_checkpoint_path(
                filename
            )

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_dict, checkpoint_path)

        self.logger.debug(f"Saved checkpoint for epoch {epoch}: {checkpoint_path}")

    def _finalize_training(self) -> None:
        """
        Finalize training (cleanup, final logging, etc.).
        """
        # CRITICAL: Synchronize after model cleanup before accelerator cleanup
        # Do NOT catch exceptions here - if NCCL is broken, we want to know immediately

        self.accelerator.wait_for_everyone("before accelerator cleanup")
        try:
            self.accelerator.cleanup()
            self.logger.info("Cleaned up accelerator resources")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup accelerator: {e}")

    def _broadcast_stop_training(self) -> None:
        if dist.is_initialized():
            # All ranks create tensor, but broadcast uses rank 0's value (src=0)
            stop_tensor = torch.tensor(
                self.stop_training,
                dtype=torch.bool,
                device=self.accelerator.get_device(),
            )
            dist.broadcast(stop_tensor, src=0)
            self.stop_training = stop_tensor.item()

    def _perform_training(self) -> None:
        """
        Main training loop.
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")

        # CRITICAL: Synchronize all ranks before starting training
        # to ensure all processes are ready together
        self.accelerator.wait_for_everyone("before on_train_start callbacks")

        # Trigger training start callbacks with logs
        self.callbacks.on_training_start()
        # CRITICAL: Synchronize after callbacks because tracker setup can take
        # different amounts of time across ranks.
        self.accelerator.wait_for_everyone("after on_train_start callbacks")

        # Run baseline evaluation only for fresh runs (not when resuming)
        # Extract skip_baseline_eval from trainer-specific config
        self.perform_baseline_evaluation()

        # Determine start epoch (resume from checkpoint or start from 1)
        start_epoch = self.current_epoch + 1 if self.current_epoch > 0 else 1

        for epoch in range(start_epoch, self.epochs + 1):
            self.setup_current_epoch(epoch)

            self.accelerator.wait_for_everyone("before on_epoch_start")
            self.callbacks.on_epoch_start(epoch)
            self.accelerator.wait_for_everyone("after on_epoch_start")

            # Training phase
            self.callbacks.on_train_start(epoch, {})
            self.accelerator.wait_for_everyone("after on_train_start")

            self.logger.debug(f"Starting training for epoch {epoch}")
            train_metrics = self.train_epoch()
            self.set_metrics("train", train_metrics)
            self.accelerator.wait_for_everyone(f"after training for epoch {epoch}")

            self.callbacks.on_train_end(epoch, train_metrics)
            self.accelerator.wait_for_everyone("after on_train_end")

            epoch_level_logs = self.generate_epoch_logs(
                epoch
            )  # Generate epoch-level logs (weight norms, etc.)
            self.set_metrics("general", epoch_level_logs)
            self.accelerator.wait_for_everyone("after setting general metrics")

            # Validation phase
            if (
                (epoch % self.validation_frequency == 0 or epoch == self.epochs)
                and self.validation_loader is not None
                and len(self.validation_loader)
            ):
                self.callbacks.on_validation_start(self.current_epoch)
                self.accelerator.wait_for_everyone("after on_validation_start")

                validation_metrics = self.validation_epoch()
                self.set_metrics("validation", validation_metrics)
                self.accelerator.wait_for_everyone("after validation epoch")

                self.callbacks.on_validation_end(self.current_epoch, validation_metrics)
                self.accelerator.wait_for_everyone("after on_validation_end")

            # Testing phase (periodic based on config)
            if (
                (epoch % self.test_frequency == 0 or epoch == self.epochs)
                and self.test_loader is not None
                and len(self.test_loader)
            ):
                self.callbacks.on_test_start(self.current_epoch)
                self.accelerator.wait_for_everyone("after on_test_start")

                test_metrics = self.test_epoch()
                self.set_metrics("test", test_metrics)
                self.accelerator.wait_for_everyone("after test epoch")

                self.callbacks.on_test_end(self.current_epoch, test_metrics)
                self.accelerator.wait_for_everyone("after on_test_end")

            self.accelerator.wait_for_everyone("after train, validation and test epoch")

            epoch_logs_dict = (
                self.compile_epoch_logs()
            )  # Compile epoch logs for logging and callbacks
            self.callbacks.on_epoch_end(epoch, epoch_logs_dict)
            self.log_metrics(epoch)
            self.accelerator.wait_for_everyone(f"after log_metrics for epoch {epoch}")

            self.save_checkpoint(epoch, filename="latest.pth")
            self.callbacks.on_checkpoint(epoch, self.current_metrics)
            self.accelerator.wait_for_everyone("after checkpoint save")

            # Broadcast early stopping decision to all ranks
            self.broadcast_stop_training()

            # Check for early stopping after sync - all ranks have same state now
            if self.stop_training:
                self.logger.info(f"Training stopped early at epoch {epoch + 1}")
                break

        # CRITICAL: Synchronize before training end callbacks
        self.accelerator.wait_for_everyone("Training complete")
        # Trigger training end callbacks
        self.logger.info("Training completed")

    # =======================================================================
    # Overfit single batch mode
    # =======================================================================
    def _overfit_single_batch(self) -> dict[str, float]:
        """
        Overfit multiple batches to verify architecture can learn.

        This is a diagnostic tool to verify the model architecture and
        training loop are sound before full training. The model should
        be able to drive loss to near zero on the batches.

        Returns:
            Dictionary with final loss and other metrics

        Raises:
            RuntimeError: If overfitting fails (indicates
                fundamental architecture or gradient issues)
        """
        self.logger.info(
            f"Starting overfit test on {self.overfit_num_batches} batches "
            f"({self.overfit_iterations} iterations)"
        )

        # Get train loader
        train_loader = self.data_loader.get("train")
        if train_loader is None:
            raise RuntimeError("No train loader available for overfit test")

        # Get test loader
        test_loader = self.data_loader.get("test")
        if test_loader is None:
            raise RuntimeError("No test loader available for overfit test")

        # Extract N batches from train loader
        train_batches = []
        for i, batch in enumerate(train_loader):
            if i >= self.overfit_num_batches:
                break
            batch = self.preprocess_batch(batch, "train")
            batch = self.accelerator.to_device(batch)
            train_batches.append(batch)

        # Extract N batches from test loader
        test_batches = []
        for i, batch in enumerate(test_loader):
            if i >= self.overfit_num_batches:
                break
            batch = self.preprocess_batch(batch, "test")
            batch = self.accelerator.to_device(batch)
            test_batches.append(batch)

        if len(train_batches) < self.overfit_num_batches:
            self.logger.warning(
                f"Only found {len(train_batches)} train batches, "
                f"requested {self.overfit_num_batches}"
            )
        if len(test_batches) < self.overfit_num_batches:
            self.logger.warning(
                f"Only found {len(test_batches)} test batches, "
                f"requested {self.overfit_num_batches}"
            )

        total_train_samples = sum(self._get_batch_size(b) for b in train_batches)
        total_test_samples = sum(self._get_batch_size(b) for b in test_batches)
        self.logger.info(
            f"Using {len(train_batches)} train batches with {total_train_samples} total samples"
        )
        self.logger.info(
            f"Using {len(test_batches)} test batches with {total_test_samples} total samples"
        )

        # Helper function to average metrics across batches
        def average_dicts(dicts: list[dict]) -> dict:
            """Average numeric values across list of dicts."""
            if not dicts:
                return {}
            result = {}
            for key in dicts[0].keys():
                if key == "stop":  # Skip stop signal
                    continue
                values = [
                    d[key]
                    for d in dicts
                    if key in d and isinstance(d[key], (int, float))
                ]
                if values:
                    result[key] = sum(values) / len(values)
            return result

        # Tracking
        meters = MeterTracker()
        initial_metrics: dict[str, float] = {}
        final_metrics: dict[str, float] = {}
        log_freq = max(1, self.overfit_iterations // 10)

        # Training loop on multiple batches
        show_progress = self.show_progress and self.accelerator.is_main_process()
        pbar = tqdm(
            range(self.overfit_iterations),
            desc=f"Overfitting {len(train_batches)} batches",
            disable=not show_progress,
        )

        for iteration in pbar:
            # Process all batches for this iteration
            batch_metrics = []
            for batch_idx, (train_batch, test_batch) in enumerate(
                zip(train_batches, test_batches)
            ):
                metrics = self.overfit_step(
                    train_batch, test_batch, iteration, batch_idx
                )
                # Remove stop signal from individual batch (we compute it from average)
                metrics.pop("stop", None)
                batch_metrics.append(metrics)

            # Average metrics across all batches
            averaged_metrics = average_dicts(batch_metrics)

            # Track averaged metrics
            avg_batch_size = (
                total_train_samples // len(train_batches) if train_batches else 1
            )
            meters.update(averaged_metrics, avg_batch_size)

            # Store initial and final metrics
            if not initial_metrics:
                initial_metrics = averaged_metrics.copy()
                # Log initial state
                metric_str = ", ".join(
                    [f"{k}={v:.6f}" for k, v in initial_metrics.items()]
                )
                self.logger.info(f"Initial averaged metrics: {metric_str}")

            final_metrics = averaged_metrics.copy()

            # Early stopping: both train_acc AND test_acc >= 1.0
            train_acc = averaged_metrics.get("train_acc", 0.0)
            test_acc = averaged_metrics.get("test_acc", 0.0)
            should_stop = (train_acc >= 1.0) and (test_acc >= 1.0)

            # Log progress
            if iteration % log_freq == 0 or iteration == self.overfit_iterations - 1:
                metric_str = ", ".join(
                    [f"{k}={v:.6f}" for k, v in averaged_metrics.items()]
                )
                self.logger.info(
                    f"Iteration {iteration}/{self.overfit_iterations}: {metric_str}"
                )
                # Update progress bar with all metrics
                pbar_dict = {k: f"{v:.6f}" for k, v in averaged_metrics.items()}
                pbar.set_postfix(pbar_dict)

            # Early stopping
            if should_stop:
                self.logger.info(
                    f"Early stopping at iteration {iteration}: "
                    f"train_acc={train_acc:.4f} and test_acc={test_acc:.4f} "
                    f"(both >= 1.0 averaged across {len(train_batches)} batches)"
                )
                break

            self.global_step += 1

        # Compute final metrics averages
        averaged_final_metrics: dict[str, float] = meters.get_averages()

        # Log final summary
        self.logger.info(f"Overfit test complete on {len(train_batches)} batches:")
        self.logger.info(f"  Initial: {initial_metrics}")
        self.logger.info(f"  Final: {final_metrics}")
        self.logger.info(f"  Averaged: {averaged_final_metrics}")

        self.set_metrics("train", averaged_final_metrics)
        self.logger.info("Overfit test completed")

        return averaged_final_metrics

    def overfit_single_batch(self) -> dict[str, float]:
        """
        Public wrapper for single batch overfit test.

        This diagnostic test verifies the model can learn by repeatedly
        training on a single batch. The loss should decrease to near zero.

        If this test fails (loss doesn't approach zero), it indicates:
        - Gradient flow issues (vanishing/exploding gradients)
        - Architecture bugs (disconnected computation graph)
        - Learning rate problems (too high or too low)
        - Implementation errors in train_step()

        Returns:
            Dictionary with final training metrics

        Example:
            Config usage:
            ```yaml
            trainer:
              my_trainer:
                overfit_single_batch: true
                overfit_iterations: 1000  # default: 1000
            ```
        """
        # Trigger training start callbacks
        self.callbacks.on_training_start()
        self.accelerator.wait_for_everyone("after on_train_start for overfit")

        # Run the overfit test
        metrics = self._overfit_single_batch()

        # Trigger training end callbacks
        final_logs = {
            "final_epoch": 1,
            "total_epochs": 1,
            "overfit_iterations": self.overfit_iterations,
            "test_type": "single_batch_overfit",
        }
        self.callbacks.on_training_end(final_logs)

        return metrics

    def _perform_baseline_evaluation(self) -> None:
        if self.skip_baseline_eval:
            self.logger.info("Skipping baseline evaluation as per configuration")
            return

        # Run baseline test evaluation first
        self.logger.info("=== BASELINE EVALUATION ===")

        self.accelerator.wait_for_everyone("before on_test_start for baseline")
        self.callbacks.on_test_start(self.current_epoch)
        self.accelerator.wait_for_everyone("after on_test_start for baseline")

        baseline_metrics = self.test_epoch()

        self.accelerator.wait_for_everyone("before on_test_end for baseline")
        self.callbacks.on_test_end(self.current_epoch, baseline_metrics)
        self.accelerator.wait_for_everyone("after on_test_end for baseline")

        self.set_metrics("test", baseline_metrics)
        epoch_level_logs = self.generate_epoch_logs(
            0
        )  # Generate epoch-level logs (weight norms, etc.)
        self.set_metrics("general", epoch_level_logs)
        self.log_metrics(self.current_epoch)

    def _setup_current_epoch(self, epoch: int) -> None:
        """
        Set current epoch and update epoch-dependent components.

        Updates self.current_epoch and propagates the epoch number to:
        - Metric managers (for logging)
        - Distributed samplers (for data shuffling)
        - Dataset wrapper (for epoch-specific sampling)

        Args:
            epoch: Epoch number to set
        """
        self.current_epoch = epoch
        self.logger.debug(f"Setting metric_managers epoch to {epoch}")
        for manager in self.metric_managers.values():
            manager.set_epoch(epoch)

        self.logger.debug(f"Setting distributed sampler epoch to {epoch}")
        self.accelerator.set_sampler_epoch(epoch)
        self.logger.debug(f"Setting dataset epoch to {epoch}")
        self.dataset_wrapper.set_epoch(epoch)

    def _set_metrics(self, split, metrics: dict[str, float]) -> None:
        """
        Set metrics for the given split and current epoch.

        Args:
            split: Data split ('train', 'validation', 'test')
            metrics: Dictionary of metrics to set
        """
        if not self.accelerator.is_main_process():
            return

        if split not in self.metrics_history:
            raise ValueError(f"Invalid split: {split}")

        if self.current_epoch not in self.metrics_history[split]:
            self.metrics_history[split][self.current_epoch] = {}

        self.metrics_history[split][self.current_epoch].update(metrics)

    def _generate_epoch_logs(self, epoch: int) -> dict[str, float]:
        """
        Generate epoch-level logs (model stats, weight norms, etc.).

        Called once per epoch to compute metrics that aren't tied to a specific
        split (train/validation/test). Subclasses can override to add custom metrics.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of epoch-level metrics

        Example:
            {
                "model_name/weights/total_norm": 45.3,
                "model_name/gradients/layer1": 2.1,
                "model_name/layers/encoder": 12.5
            }
        """
        if not self.accelerator.is_main_process():
            return {}

        epoch_logs: dict[str, float] = {}

        # Training-state signals useful for debugging convergence and scheduling.
        epoch_logs["state/global_step"] = float(self.global_step)
        epoch_logs["state/epochs_no_improvement"] = float(self.epochs_no_improvement)
        if self.best_metric is not None:
            epoch_logs["state/best_metric"] = float(self.best_metric)

        # Optimizer-level and param-group-level hyperparameters.
        for optimizer_name, optimizer in self.optimizers.items():
            for group_idx, group in enumerate(optimizer.param_groups):
                group_name = str(group.get("name", f"group_{group_idx}"))
                safe_group_name = group_name.replace(".", "_").replace("/", "_")
                prefix = f"optimizers/{optimizer_name}/{safe_group_name}"
                lr_value = group.get("lr")
                weight_decay_value = group.get("weight_decay")
                momentum_value = group.get("momentum")

                if lr_value is not None:
                    epoch_logs[f"{prefix}/lr"] = float(lr_value)
                if weight_decay_value is not None:
                    epoch_logs[f"{prefix}/weight_decay"] = float(weight_decay_value)
                if momentum_value is not None:
                    epoch_logs[f"{prefix}/momentum"] = float(momentum_value)

        # Scheduler state and per-group current LR values.
        for scheduler_name, scheduler in self.schedulers.items():
            last_epoch = getattr(scheduler, "last_epoch", None)
            if last_epoch is not None:
                epoch_logs[f"schedulers/{scheduler_name}/last_epoch"] = float(
                    last_epoch
                )

            get_last_lr = getattr(scheduler, "get_last_lr", None)
            if callable(get_last_lr):
                try:
                    lrs = get_last_lr()
                except Exception:
                    lrs = []
                for group_idx, lr_value in enumerate(lrs):
                    epoch_logs[f"schedulers/{scheduler_name}/group_{group_idx}/lr"] = (
                        float(lr_value)
                    )

        # Model weight and layer norms are opt-in because they add a lot of log noise.
        if self.models:
            for model_name, model in self.models.items():
                trainable_params = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                frozen_params = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if not parameter.requires_grad
                )
                total_params = trainable_params + frozen_params
                epoch_logs[f"{model_name}/params/trainable"] = float(trainable_params)
                epoch_logs[f"{model_name}/params/frozen"] = float(frozen_params)
                epoch_logs[f"{model_name}/params/total"] = float(total_params)
                if total_params > 0:
                    epoch_logs[f"{model_name}/params/trainable_ratio"] = float(
                        trainable_params / total_params
                    )

                if self.log_weights:
                    weight_norms = self.compute_model_weight_norms(model)
                    epoch_logs.update(
                        {f"{model_name}/{k}": v for k, v in weight_norms.items()}
                    )

        return epoch_logs

    def _compile_epoch_logs(self) -> dict:
        """
        Compile epoch logs into flattened dictionary for callbacks.

        Converts nested metrics_history structure into flat dict with
        "split/metric_name" keys for easy logging to external trackers.

        Returns:
            Flattened dict like {"train/loss": 0.5, "validation/acc": 0.9, "epoch": 10}
            Only returns data on rank 0; empty dict on other processes.
        """
        return self._compile_epoch_logs_for_epoch(self.current_epoch)

    def _compile_epoch_logs_for_epoch(self, epoch: int) -> dict[str, float]:
        """
        Compile flattened metrics for a specific epoch.

        Args:
            epoch: Epoch to flatten

        Returns:
            Flattened metrics for the requested epoch.
        """
        if not self.accelerator.is_main_process():
            return {}

        checkpoint_logs: dict[str, float] = {"epoch": float(epoch)}
        for split, metrics in self.metrics_history.items():
            epoch_metrics = metrics.get(epoch, {})
            for k, v in epoch_metrics.items():
                checkpoint_logs[f"{split}/{k}"] = float(v)

        return checkpoint_logs

    def _get_selection_metric_config(self) -> tuple[str | None, str | None]:
        """
        Resolve the default metric used to rank runs.

        Returns:
            Tuple of metric key and optimization mode.
        """
        callbacks_config = self.config.get("callbacks", {})
        if not isinstance(callbacks_config, dict):
            return None, None

        checkpoint_config = callbacks_config.get("checkpoint")
        if not isinstance(checkpoint_config, dict):
            return None, None

        monitor = checkpoint_config.get("monitor")
        mode = checkpoint_config.get("mode", "min")
        if not isinstance(monitor, str):
            return None, None
        if mode not in {"min", "max"}:
            mode = "min"

        return monitor, mode

    def _get_recorded_epochs(self) -> list[int]:
        """Return sorted epochs that contain any recorded metrics."""
        recorded_epochs: set[int] = set()
        for split_metrics in self.metrics_history.values():
            recorded_epochs.update(split_metrics.keys())
        return sorted(recorded_epochs)

    def _select_best_epoch(
        self,
        selection_metric: str | None,
        selection_mode: str | None,
        recorded_epochs: list[int],
    ) -> tuple[int | None, float | None]:
        """
        Resolve the best epoch for the configured selection metric.

        Args:
            selection_metric: Metric key to evaluate
            selection_mode: Optimization direction, either ``min`` or ``max``
            recorded_epochs: Epochs with recorded metrics

        Returns:
            Tuple of best epoch and best metric value.
        """
        if not recorded_epochs:
            return None, None

        if selection_metric is None or selection_mode is None:
            return recorded_epochs[-1], None

        best_epoch: int | None = None
        best_value: float | None = None
        for epoch in recorded_epochs:
            epoch_logs = self._compile_epoch_logs_for_epoch(epoch)
            resolved_metric = Callback.resolve_log_key(epoch_logs, selection_metric)
            if resolved_metric is None:
                continue

            metric_value = float(epoch_logs[resolved_metric])
            if best_value is None:
                best_epoch = epoch
                best_value = metric_value
                continue

            is_better = (
                metric_value < best_value
                if selection_mode == "min"
                else metric_value > best_value
            )
            if is_better:
                best_epoch = epoch
                best_value = metric_value

        if best_epoch is None:
            return recorded_epochs[-1], None

        return best_epoch, best_value

    def _serialize_metrics_history(
        self,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        Convert metrics history into a JSON-serializable structure.

        Returns:
            Metrics history with stringified epoch keys.
        """
        history: dict[str, dict[str, dict[str, float]]] = {}
        for split, split_metrics in self.metrics_history.items():
            history[split] = {
                str(epoch): {
                    metric_name: float(metric_value)
                    for metric_name, metric_value in epoch_metrics.items()
                }
                for epoch, epoch_metrics in sorted(split_metrics.items())
            }
        return history

    def _build_run_summary(
        self,
        status: str,
        error_message: str | None,
    ) -> dict[str, Any]:
        """
        Build the normalized run summary used by local sweep analysis.

        Args:
            status: Final run status
            error_message: Optional failure message

        Returns:
            Run summary dictionary.
        """
        selection_metric, selection_mode = self._get_selection_metric_config()
        recorded_epochs = self._get_recorded_epochs()
        best_epoch, selection_value = self._select_best_epoch(
            selection_metric,
            selection_mode,
            recorded_epochs,
        )

        final_epoch = recorded_epochs[-1] if recorded_epochs else self.current_epoch
        final_metrics = self._compile_epoch_logs_for_epoch(final_epoch)
        best_metrics = (
            self._compile_epoch_logs_for_epoch(best_epoch)
            if best_epoch is not None
            else {}
        )

        return {
            "status": status,
            "error_message": error_message,
            "run_name": self.artifact_manager.run_name,
            "experiment_name": self.artifact_manager.experiment_name,
            "sweep_name": self.artifact_manager.sweep_name,
            "artifact_dir": str(self.artifact_manager.run_dir),
            "recorded_epochs": recorded_epochs,
            "final_epoch": final_epoch,
            "total_epochs": self.epochs,
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "selection_value": selection_value,
            "final_metrics": final_metrics,
            "best_metrics": best_metrics,
        }

    def _persist_run_analysis(
        self,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """
        Persist normalized run artifacts used by local analysis tools.

        Args:
            status: Final run status
            error_message: Optional failure message
        """
        if not self.accelerator.is_main_process():
            return

        summary = self._build_run_summary(status, error_message)
        history = self._serialize_metrics_history()
        run_info = {
            "status": status,
            "error_message": error_message,
            "run_name": self.artifact_manager.run_name,
            "experiment_name": self.artifact_manager.experiment_name,
            "sweep_name": self.artifact_manager.sweep_name,
            "artifact_dir": str(self.artifact_manager.run_dir),
            "config_path": str(self.artifact_manager.run_dir / "config.yaml"),
            "metrics_summary_path": str(self.artifact_manager.get_metrics_summary_path()),
            "metrics_history_path": str(self.artifact_manager.get_metrics_history_path()),
            "current_epoch": self.current_epoch,
            "total_epochs": self.epochs,
        }

        self.artifact_manager.save_metrics(summary, filename="summary.json")
        self.artifact_manager.save_metrics(history, filename="history.json")
        self.artifact_manager.save_run_info(run_info)

    def persist_run_analysis(
        self,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """
        Persist normalized run artifacts used by analysis tools.

        Args:
            status: Final run status
            error_message: Optional failure message
        """
        self._persist_run_analysis(status=status, error_message=error_message)

    def _log_metrics(self, epoch: int) -> None:
        """
        Log metrics for an epoch to console.

        Logs metrics via metric managers for train/validation/test splits,
        and logs general epoch-level metrics (weight norms, etc.) directly.

        Args:
            epoch: Epoch number
        """
        # Console logging via metric managers
        for split in ["train", "validation", "test"]:
            for manager in self.metric_managers.values():
                manager.print_logs(split)

        # Log general-level metrics to console
        self.logger.info(f"Epoch {epoch} - General Stats:")
        for key, value in self.metrics_history["general"].get(epoch, {}).items():
            if isinstance(value, (float, int)):
                self.logger.info(f"  {key}: {float(value):.6e}")
            else:
                self.logger.info(f"  {key}: {value}")

    def _get_batch_size(self, batch_data: dict) -> int:
        """
        Get batch size from batch data, handling multi-crop cases.

        Args:
            batch_data: Batch dictionary with 'image' key

        Returns:
            Batch size
        """
        data_key = None
        for key in ["image", "images", "data"]:
            if key in batch_data:
                data_key = key
                break
        if isinstance(batch_data[data_key], list):
            return len(batch_data[data_key])
        else:
            return batch_data[data_key].size(0)

    def compute_probability_diagnostics(
        self,
        step_metrics: dict[str, Any],
        batch_data: dict[str, Any],
    ) -> dict[str, float]:
        """
        Add probability-distribution diagnostics when step outputs include probabilities.

        This is model-agnostic and only activates if the step metrics dictionary
        contains a probability tensor under ``probabilities_tensor`` or
        ``probabilities``.

        Added metrics:
            - ``prob_entropy_mean``
            - ``score_bonafide_mean``
            - ``score_attack_mean``
            - ``score_delta``

        Args:
            step_metrics: Step metrics returned by trainer step function.
            batch_data: Original batch containing labels.

        Returns:
            Step metrics with optional diagnostics added and non-scalar tensors removed.
        """
        prob_key = None
        if "probabilities_tensor" in step_metrics:
            prob_key = "probabilities_tensor"
        elif "probabilities" in step_metrics and torch.is_tensor(
            step_metrics["probabilities"]
        ):
            prob_key = "probabilities"

        if prob_key is None:
            return step_metrics

        bonafide_index: int = self.dataset_wrapper.classes.index("real")
        probabilities = step_metrics.pop(prob_key)
        if not torch.is_tensor(probabilities):
            return step_metrics

        probabilities = probabilities.detach().float()
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)

        if probabilities.ndim != 2 or probabilities.shape[0] == 0:
            return step_metrics

        probs = probabilities.clamp(1e-8, 1.0)
        entropy = -(probs * probs.log()).sum(dim=1)
        pos_scores = probs[:, bonafide_index]

        step_metrics["prob_entropy_mean"] = entropy.mean().item()

        labels = batch_data.get("label")
        if not torch.is_tensor(labels):
            return step_metrics
        labels = labels.detach().view(-1)
        if labels.numel() != pos_scores.numel():
            return step_metrics
        labels = labels.to(pos_scores.device)
        bonafide_mask = labels == bonafide_index
        attack_mask = labels != bonafide_index

        if bonafide_mask.any():
            step_metrics["score_bonafide_mean"] = (
                pos_scores[bonafide_mask].mean().item()
            )
        if attack_mask.any():
            step_metrics["score_attack_mean"] = pos_scores[attack_mask].mean().item()
        if bonafide_mask.any() and attack_mask.any():
            step_metrics["score_delta"] = (
                step_metrics["score_bonafide_mean"] - step_metrics["score_attack_mean"]
            )

        return step_metrics

    def _train_epoch(self) -> dict[str, float]:
        """
        Training loop for one epoch.

        Returns:
            Dictionary of training metrics for this epoch
        """
        # Note: Metric managers are optional for self-supervised learning
        # Only warn if none are configured, don't fail
        if not self.metric_managers:
            self.logger.warning(
                "No metric managers configured - metrics will not be tracked"
            )

        self.set_models_mode("train")
        split_text = "train"
        if self.train_loader is None:
            self.logger.warning(
                f"No validation data loader available - skipping {split_text} epoch"
            )
            return {}

        # Metrics tracking
        meters = MeterTracker()

        # Reset metrics for new epoch
        for manager in self.metric_managers.values():
            manager.reset_metrics(split_text)

        # Create progress bar for training (only on rank 0)
        show_progress = self.show_progress and self.accelerator.is_main_process()
        data_loader = self.data_loader[split_text]
        if data_loader is None:
            self.logger.warning(
                f"No data loader for split: {split_text} - skipping epoch"
            )
            return {}

        pbar = tqdm(
            data_loader,
            desc=f"Epoch {self.current_epoch} [{split_text.capitalize()}]",
            leave=False,
            disable=not show_progress,
        )

        self.logger.debug(f"Starting training epoch {self.current_epoch}")

        # Synchronize batch counts across all ranks to ensure consistent training
        local_batch_count = len(data_loader)
        min_batch_count = local_batch_count

        if self.accelerator.use_distributed:
            # Only synchronize for training, not validation/test
            batch_count_tensor = torch.tensor(
                local_batch_count,
                dtype=torch.int64,
                device=self.accelerator.get_device(),
            )
            batch_counts = [
                torch.zeros_like(batch_count_tensor)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(batch_counts, batch_count_tensor)
            min_batch_count = min(bc.item() for bc in batch_counts)

            if local_batch_count != min_batch_count:
                self.logger.info(
                    f"Adjusting batch count from {local_batch_count} to {min_batch_count} for consistency across ranks"
                )

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(pbar):
            # Stop at minimum batch count to ensure all ranks process same number of batches
            if batch_idx >= min_batch_count:
                self.logger.debug(
                    f"Stopping at batch {batch_idx} (min_batch_count={min_batch_count})"
                )
                break
            # Move batch to device
            start = time.time()
            batch_data = self.preprocess_batch(batch_data, split_text)
            batch_data = self.accelerator.to_device(batch_data)
            assert isinstance(batch_data, dict), "preprocess_batch must return a dict"
            self.callbacks.on_batch_start(
                batch_idx,
                split_text,
                batch_data,
            )
            # Perform training step
            step_metrics = self.train_step(batch_data, batch_idx)
            step_metrics = self.compute_probability_diagnostics(
                step_metrics, batch_data
            )
            step_metrics["batch_time"] = time.time() - start
            self.callbacks.on_batch_end(batch_idx, split_text, batch_data)

            batch_size = self._get_batch_size(batch_data)
            meters.update(step_metrics, batch_size)
            pbar.set_postfix(meters.get_postfix(self.pbar_metrics[split_text]))

            # Logging (reduced frequency since we have progress bar)
            if batch_idx % self.print_freq == 0 and batch_idx > 0:
                text = f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                for key, meter in meters.get_averages().items():
                    text += f"{key.capitalize()}: {meter:.4f}  "
                self.logger.info(text)

            self.global_step += 1

            # # DIAGNOSTIC: Early exit after 100 batches
            # if batch_idx >= 99:
            #     self.logger.info("[DIAGNOSTIC] Early exit after 100 training batches")
            #     break

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        # CRITICAL: Synchronize all processes after training epoch
        self.accelerator.wait_for_everyone(f"after {split_text} epoch completion")

        # Compute epoch metrics using all metric managers
        epoch_metrics: dict[str, float] = meters.get_averages()
        for manager_name, manager in self.metric_managers.items():
            manager_metrics = manager.compute(split_text)
            self.accelerator.wait_for_everyone(
                f"after computing {split_text} metrics for {manager_name}"
            )
            epoch_metrics.update(manager_metrics)
            epoch_metrics.update(manager.compute_epoch_diagnostics(split_text))
            # Generate plots
            manager.generate_plots(self.current_epoch, split_text)
            self.accelerator.wait_for_everyone(
                f"after generating {split_text} plots for {manager_name}"
            )

        return epoch_metrics

    def _test_epoch(self) -> dict[str, float]:
        """
        Test loop for one epoch.

        Returns:
            Dictionary of validation metrics for this epoch
        """
        # Note: Metric managers are optional for self-supervised learning
        # Only warn if none are configured, don't fail
        if not self.metric_managers:
            self.logger.warning(
                "No metric managers configured - metrics will not be tracked"
            )

        self.set_models_mode("eval")
        split_text = "test"

        if self.test_loader is None:
            self.logger.warning(
                f"No {split_text} data loader available - skipping {split_text} epoch"
            )
            return {}

        # Metrics tracking
        meters = MeterTracker()

        # Reset metrics for new epoch
        for manager in self.metric_managers.values():
            manager.reset_metrics(split_text)

        with torch.no_grad():
            # Create progress bar for testing (only on rank 0)
            show_progress = self.show_progress and self.accelerator.is_main_process()
            pbar = tqdm(
                self.data_loader[split_text],
                desc=f"Epoch {self.current_epoch} [{split_text.capitalize()}]",
                leave=False,
                disable=not show_progress,
            )

            for batch_idx, batch_data in enumerate(pbar):
                # Move batch to device
                batch_data = self.preprocess_batch(batch_data, split_text)
                batch_data = self.accelerator.to_device(batch_data)
                self.callbacks.on_batch_start(batch_idx, split_text, batch_data)
                # Perform test step
                start = time.time()
                step_metrics = self.test_step(batch_data)
                step_metrics = self.compute_probability_diagnostics(
                    step_metrics, batch_data
                )
                step_metrics["batch_time"] = time.time() - start
                self.callbacks.on_batch_end(batch_idx, split_text, batch_data)
                batch_size = self._get_batch_size(batch_data)
                meters.update(step_metrics, batch_size)
                pbar.set_postfix(meters.get_postfix(self.pbar_metrics[split_text]))

                # # DIAGNOSTIC: Early exit after 100 batches
                # if batch_idx >= 99:
                #     self.logger.info("[DIAGNOSTIC] Early exit after 100 test batches")
                #     break

        # Log before barrier on ALL ranks (using print to bypass logger filtering)
        self.accelerator.wait_for_everyone(f"after {split_text} loop completion")

        # Compute epoch metrics using all metric managers
        epoch_metrics: dict[str, float] = meters.get_averages()
        for manager_name, manager in self.metric_managers.items():
            manager_metrics = manager.compute(split_text)
            self.accelerator.wait_for_everyone(
                f"after computing {split_text} metrics for {manager_name}"
            )
            epoch_metrics.update(manager_metrics)
            epoch_metrics.update(manager.compute_epoch_diagnostics(split_text))
            # Generate plots
            manager.generate_plots(self.current_epoch, split_text)
            self.accelerator.wait_for_everyone(
                f"after generating {split_text} plots for {manager_name}"
            )

        return epoch_metrics

    def _validation_epoch(self) -> dict[str, float]:
        """
        Validation loop for one epoch.

        Returns:
            Dictionary of validation metrics for this epoch
        """
        # Note: Metric managers are optional for self-supervised learning
        # Only warn if none are configured, don't fail
        if not self.metric_managers:
            self.logger.warning(
                "No metric managers configured - metrics will not be tracked"
            )

        self.set_models_mode("eval")
        split_text = "validation"

        if self.validation_loader is None:
            self.logger.warning(
                f"No {split_text} data loader available - skipping {split_text} epoch"
            )
            return {}

        # Metrics tracking
        meters = MeterTracker()

        # Reset metrics for new epoch
        for manager in self.metric_managers.values():
            manager.reset_metrics(split_text)

        with torch.no_grad():
            # Create progress bar for testing (only on rank 0)
            show_progress = self.show_progress and self.accelerator.is_main_process()
            pbar = tqdm(
                self.data_loader[split_text],
                desc=f"Epoch {self.current_epoch} [{split_text.capitalize()}]",
                leave=False,
                disable=not show_progress,
            )

            for batch_idx, batch_data in enumerate(pbar):
                # Move batch to device
                batch_data = self.preprocess_batch(batch_data, split_text)
                batch_data = self.accelerator.to_device(batch_data)
                assert isinstance(batch_data, dict), (
                    "preprocess_batch must return a dict"
                )
                self.callbacks.on_batch_start(batch_idx, split_text, batch_data)
                # Perform test step
                start = time.time()
                step_metrics = self.validation_step(batch_data)
                step_metrics = self.compute_probability_diagnostics(
                    step_metrics, batch_data
                )
                step_metrics["batch_time"] = time.time() - start
                self.callbacks.on_batch_end(batch_idx, split_text, batch_data)
                batch_size = self._get_batch_size(batch_data)
                meters.update(step_metrics, batch_size)
                pbar.set_postfix(meters.get_postfix(self.pbar_metrics[split_text]))

                # DIAGNOSTIC: Early exit after 100 batches
                # if batch_idx >= 99:
                #     self.logger.info(
                #         f"[DIAGNOSTIC] Early exit after 100 {split_text} batches"
                #     )
                #     break

        # Log before barrier on ALL ranks (using print to bypass logger filtering)
        self.accelerator.wait_for_everyone(f"after {split_text} loop completion")

        # Compute epoch metrics using all metric managers
        epoch_metrics: dict[str, float] = meters.get_averages()
        for manager_name, manager in self.metric_managers.items():
            manager_metrics = manager.compute(split_text)
            self.accelerator.wait_for_everyone(
                f"after computing {split_text} metrics for {manager_name}"
            )
            epoch_metrics.update(manager_metrics)
            epoch_metrics.update(manager.compute_epoch_diagnostics(split_text))
            # Generate plots
            manager.generate_plots(self.current_epoch, split_text)
            self.accelerator.wait_for_everyone(
                f"after generating {split_text} plots for {manager_name}"
            )

        return epoch_metrics

    # =======================================================================
    # Methods that can be overridden by subclasses
    # ======================================================================
    def run(self) -> None:
        """
        Main training loop entry point.
        Calls the internal _run method.
        Steps:
        1. Setup training components
        2. Execute training loop
        3. Handle exceptions and cleanup
        4. Upload artifacts
        """
        self._run()

    def setup(self) -> None:
        """
        Setup training components.
        Calls the internal _setup method.
        Steps:
        1. Set random seeds
        2. Initialize artifact manager
        3. Create accelerator
        4. Setup data loaders
        5. Setup model
        6. Setup criterions
        7. Setup optimizers
        8. Setup schedulers
        9. Setup metric managers
        10. Prepare components with accelerator
        11. Setup callbacks
        12. Load checkpoint if resuming
        """
        self._setup()

    def setup_accelerator(self) -> None:
        """
        Setup accelerator.
        Calls the internal _setup_accelerator method.
        """
        self._setup_accelerator()

    def setup_artifact_manager(self) -> None:
        """
        Setup artifact manager.
        Calls the internal _setup_artifact_manager method.
        """
        self._setup_artifact_manager()

    def setup_data(self) -> None:
        """
        Setup data loaders.
        """
        self._setup_data()

    def setup_metrics(self) -> None:
        """
        Setup metrics manager.
        """
        self._setup_metrics()

    def setup_ema(self) -> None:
        """Setup optional EMA manager."""
        self._setup_ema()

    def setup_callbacks(self) -> None:
        """
        Setup callbacks.
        """
        self._setup_callbacks()

    def load_continue_model(self) -> None:
        """
        Load checkpoint if resuming training.
        Calls the internal _load_continue_model method.
        """
        self._load_continue_model()

    def set_models_mode(self, mode: str) -> None:
        """
        Set all models to train or eval mode.
        Calls the internal _set_models_mode method.

        Args:
            mode: 'train' or 'eval'
        """
        self._set_models_mode(mode)

    def preprocess_batch(self, batch_data: dict, split: str) -> dict:
        """
        Preprocess batch data before feeding into the model.

        Args:
            batch_data: Input batch data
            split: Data split ('train', 'validation', 'test')

        Returns:
            Preprocessed batch data
        """
        return batch_data

    def setup_current_epoch(self, epoch: int) -> None:
        """
        Setup operations at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        self._setup_current_epoch(epoch)

    def set_metrics(self, split, metrics: dict[str, float]) -> None:
        """
        Set metrics for the given split and current epoch.

        Args:
            split: Data split ('train', 'validation', 'test')
            metrics: Dictionary of metrics to set
        """
        self._set_metrics(split, metrics)

    def compile_epoch_logs(self) -> dict:
        """
        Compile epoch-level logs for checkpointing and callbacks.
        Called at the end of each epoch to gather all metrics into a single
        dictionary for easy access.

        Args:
            None
        Returns:
            Dictionary of compiled epoch-level logs
        """
        return self._compile_epoch_logs()

    def broadcast_stop_training(self) -> None:
        """
        Broadcast early stopping decision to all ranks.
        Calls the internal _broadcast_stop_training method.
        """
        self._broadcast_stop_training()

    # =======================================================================
    # Main training loop
    # =======================================================================
    def perform_training(self) -> None:
        """
        Main training loop.
        """
        self._perform_training()

    def perform_baseline_evaluation(self) -> None:
        """
        Perform baseline evaluation before training.
        """
        self._perform_baseline_evaluation()

    def test(self) -> dict[str, float]:
        """
        Standalone testing/evaluation method.

        Runs model evaluation on test set without training.

        Returns:
            Dictionary of test metrics
        """
        self.logger.info("Starting standalone testing/evaluation")

        if not self.metric_managers:
            raise RuntimeError(
                "At least one metric manager must be setup before testing"
            )

        # Update metric managers epoch (use current_epoch or 0)
        test_epoch = max(self.current_epoch, 1)
        for manager in self.metric_managers.values():
            manager.set_epoch(test_epoch)

        # Run test evaluation
        test_metrics = self.test_epoch()

        # Log final test results
        self.logger.info("=== FINAL TEST RESULTS ===")

        # Use metric managers for detailed logging
        for name, manager in self.metric_managers.items():
            manager.print_logs("test")

        self.logger.info("Testing completed")
        return test_metrics

    def train_epoch(self) -> dict[str, float]:
        """
        Execute one training epoch.

        Runs training loop for all batches in train_loader, computes metrics,
        and returns aggregated results. Calls _train_epoch() internally.

        The internal method handles:
        - Setting models to train mode
        - Batch preprocessing and device placement
        - Calling train_step() for each batch
        - Metric aggregation via MeterTracker
        - Synchronization across distributed processes

        Returns:
            Dictionary of epoch-level training metrics (loss, accuracy, etc.)

        Note:
            Override train_step() to customize per-batch training logic,
            not this method.
        """
        return self._train_epoch()

    def test_epoch(self) -> dict[str, float]:
        """
        Test loop for one epoch.

        Returns:
            Dictionary of test metrics for this epoch
        """
        with self._get_eval_param_context():
            return self._test_epoch()

    def validation_epoch(self) -> dict[str, float]:
        """
        Validation loop for one epoch.

        Returns:
            Dictionary of validation metrics for this epoch
        """
        with self._get_eval_param_context():
            return self._validation_epoch()

    # =======================================================================
    # Abstract setup methods to be implemented by subclasses
    # =======================================================================
    @abstractmethod
    def setup_model(self) -> None:
        """
        Setup model in the models dict. Device placement is handled by accelerator.prepare() in setup().

        Example:
            self.models['main'] = MODEL_REGISTRY.get(model_name, self.config)
            self.logger.info(f"Initialized model: {self.config['model']}")
        """
        raise self.model_not_initialised_error

    @abstractmethod
    def setup_criterion(self) -> None:
        """
        Setup loss criterion(s). Device placement is handled by accelerator in train/test loops.

        Example:
            for criterion_config in self.config["loss"]:
                criterion_name = criterion_config["name"]
                criterion = CRITERION_REGISTRY.get(criterion_name, self.config)
                self.criterions[criterion_name] = criterion
            self.logger.info(f"Initialized {', '.join(self.criterions.keys())} criterion")
        """
        raise self.criterion_not_initialised_error

    @abstractmethod
    def setup_optimizer(self) -> None:
        """
        Setup optimizer(s) for training. Must be implemented by subclasses.

        Optimizers should be stored in self.optimizers dict with descriptive keys.
        Multiple optimizers are supported for complex training scenarios (e.g.,
        separate optimizers for different model components).

        The trainer will automatically handle:
        - Optimizer state saving/loading in checkpoints
        - Gradient accumulation via accelerator.optimizer_step()

        Example:
            def setup_optimizer(self):
                self.optimizers['main'] = torch.optim.Adam(
                    self.models['backbone'].parameters(),
                    lr=self.config['optimizer']['lr']
                )

        For multi-optimizer setups:
            def setup_optimizer(self):
                self.optimizers['backbone'] = torch.optim.SGD(...)
                self.optimizers['head'] = torch.optim.Adam(...)
        """
        raise self.optimizer_not_initialised_error

    @abstractmethod
    def setup_scheduler(self) -> None:
        """
        Setup learning rate scheduler(s). Must be implemented by subclasses.

        Schedulers should be stored in self.schedulers dict with keys matching
        the corresponding optimizer keys in self.optimizers.

        The trainer will automatically handle:
        - Scheduler state saving/loading in checkpoints
        - Scheduler stepping (via callbacks or manual calls)

        Example:
            def setup_scheduler(self):
                self.schedulers['main'] = torch.optim.lr_scheduler.StepLR(
                    self.optimizers['main'],
                    step_size=30,
                    gamma=0.1
                )

        Note: If no scheduler is needed, leave self.schedulers empty.
        """
        raise NotImplementedError("Subclasses must implement setup_scheduler")

    @abstractmethod
    def train_step(
        self, batch_data: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """
        Single training step. Must be implemented by subclasses.

        Args:
            batch_data: Batch of training data
            batch_idx: Index of the current batch

        Returns:
            Dictionary of step metrics
        """
        raise NotImplementedError("Subclasses must implement train_step")

    @abstractmethod
    def test_step(self, batch_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Single test step. Must be implemented by subclasses.

        Args:
            batch_data: Batch of test data

        Returns:
            Dictionary of step metrics
        """
        raise NotImplementedError("Subclasses must implement test_step")

    @abstractmethod
    def validation_step(self, batch_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Single validation step. Must be implemented by subclasses.

        Args:
            batch_data: Batch of validation data

        Returns:
            Dictionary of step metrics
        """
        raise NotImplementedError("Subclasses must implement validation_step")

    def overfit_step(
        self,
        train_batch: dict[str, torch.Tensor],
        test_batch: dict[str, torch.Tensor],
        iteration: int,
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        """
        Perform one overfitting iteration with train and test evaluation.

        This method is called by _overfit_single_batch for each iteration.
        Trainers should override this to implement trainer-specific logic
        for computing losses and accuracies on both train and test batches.

        The trainer is responsible for:
        - Setting model mode (train/eval) as needed
        - Running forward/backward on train batch
        - Running forward (no gradients) on test batch
        - Computing all relevant metrics
        - Determining early stopping condition

        Args:
            train_batch: Single batch from train loader (already preprocessed)
            test_batch: Single batch from test loader (already preprocessed)
            iteration: Current iteration number
            batch_idx: Index of the batch (0 to overfit_num_batches-1)

        Returns:
            Dictionary with metrics. All keys are logged and displayed in progress bar
            except for 'stop' which controls early stopping:

            Required keys:
                - train_loss (float): Loss on train batch
                - test_loss (float): Loss on test batch
                - train_accuracy (float): Accuracy on train batch
                - test_accuracy (float): Accuracy on test batch
                - stop (bool): Ignored in multi-batch mode (base trainer computes stop)

            Optional keys:
                Any additional metrics (e.g., spatial_loss, temporal_loss) will be
                logged and displayed in the progress bar.

        Example:
            return {
                "train_loss": 0.4,
                "test_loss": 0.5,
                "train_accuracy": 0.95,
                "test_accuracy": 1.0,
                "spatial_loss": 0.3,  # Optional trainer-specific metric
                "temporal_loss": 0.5,  # Optional trainer-specific metric
                "stop": True  # Ignored - stop computed from averaged metrics
            }
        """
        # Default implementation for standard trainers
        # Subclasses with special logic (e.g., video models) should override this

        # ===== TRAIN STEP =====
        self.set_models_mode("train")
        train_metrics = self.train_step(train_batch, iteration)
        train_loss = train_metrics.get("loss", 0.0)

        # Calculate train accuracy
        with torch.no_grad():
            model_output = self.model(train_batch)
            probs = model_output.get("probabilities")
            if probs is None:
                logits = model_output.get("logits")
                if logits is not None:
                    probs = torch.nn.functional.softmax(logits, dim=-1)

            if probs is not None:
                preds = probs.argmax(dim=-1)
                labels = train_batch["label"]
                train_accuracy = (preds == labels).float().mean().item()
            else:
                train_accuracy = 0.0

        # ===== TEST STEP (EVAL MODE) =====
        self.set_models_mode("eval")

        with torch.no_grad():
            # Forward on test batch
            test_output = self.model(test_batch)

            # Calculate test loss
            test_loss_value = 0.0
            labels = test_batch["label"]
            for name, criterion in self.criterions.items():
                logits = test_output.get("logits")
                loss_output = criterion(logits, labels)

                if isinstance(loss_output, dict):
                    test_loss_value = loss_output["loss"].item()
                else:
                    test_loss_value = loss_output.item()
                break  # Use first criterion

            # Calculate test accuracy
            probs = test_output.get("probabilities")
            if probs is None:
                logits = test_output.get("logits")
                if logits is not None:
                    probs = torch.nn.functional.softmax(logits, dim=-1)

            if probs is not None:
                preds = probs.argmax(dim=-1)
                test_accuracy = (preds == labels).float().mean().item()
            else:
                test_accuracy = 0.0

        # Early stopping condition: test accuracy reaches 1.0
        should_stop = test_accuracy >= 1.0

        return {
            "train_loss": train_loss,
            "test_loss": test_loss_value,
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
            "stop": should_stop,
        }

    def log_metrics(self, epoch: int) -> None:
        """
        Log metrics for an epoch.
        Calls the internal _log_metrics method.

        Args:
            epoch: Epoch number
            epoch_logs: Nested dict with structure:
                {
                    "train": {"loss": 0.5, "accuracy": 0.95},
                    "validation": {"loss": 0.6, "accuracy": 0.93},
                    "test": {"eer": 0.03, "accuracy": 0.97},
                    "epoch": {"weights/total_norm": 45.3, "gradients/layer1": 2.1}
                }
        """
        self._log_metrics(epoch)

    def save_checkpoint(self, epoch: int, filename: str | None = None) -> None:
        """
        Save model checkpoint with simplified approach.
        It saves models states, optimizers, schedulers, criterions, callback states,
        config, best_metric, epoch, train/test metrics, global_step.

        Args:
            epoch: Current epoch number
            filename: Optional checkpoint filename override
        """
        self._save_checkpoint(epoch, filename=filename)

    def register_model_states(self, checkpoint_dict: dict) -> None:
        """
        Register model states in checkpoint dict. Override in subclasses if needed.

        Default implementation handles single model case.
        Uses accelerator.unwrap_model() to handle DDP unwrapping.

        Args:
            checkpoint_dict: Dictionary to add model states to

        Example for multi-model trainers:
        ```python
        def register_model_states(self, checkpoint_dict: dict) -> None:
            # Unwrap models before saving (handles DDP)
            teacher_unwrapped = self.accelerator.unwrap_model(self.teacher)
            student_unwrapped = self.accelerator.unwrap_model(self.student)
            model_unwrapped = self.accelerator.unwrap_model(self.model)

            checkpoint_dict["teacher_state_dict"] = teacher_unwrapped.state_dict()
            checkpoint_dict["student_state_dict"] = student_unwrapped.state_dict()
            checkpoint_dict["model_state_dict"] = model_unwrapped.state_dict()
        ```
        """
        # Unwrap model (handles DDP)
        checkpoint_dict["models_state_dict"] = {}
        for name, model in self.models.items():
            unwrapped_model = self.accelerator.unwrap_model(model)
            checkpoint_dict["models_state_dict"][name] = unwrapped_model.state_dict()
        if self.ema is not None and self.ema.save_in_checkpoint:
            checkpoint_dict["ema_state_dict"] = self.ema.state_dict()

    def load_model_states(self, checkpoint: dict) -> None:
        """
        Load model states from checkpoint. Override in subclasses if needed.

        Args:
            checkpoint: Checkpoint dictionary containing model states

        Example for multi-model trainers:
        ```python
        def load_model_states(self, checkpoint: dict) -> None:
            if "teacher_state_dict" in checkpoint:
                self.teacher.load_state_dict(checkpoint["teacher_state_dict"])
            if "student_state_dict" in checkpoint:
                self.student.load_state_dict(checkpoint["student_state_dict"])
            if "model_state_dict" in checkpoint and self.model is not None:
                self.model.load_state_dict(checkpoint["model_state_dict"])
        ```
        """
        # Default implementation loads single model if it exists
        if "models_state_dict" not in checkpoint:
            self.logger.warning("No model states found in checkpoint to load")
            return

        # Unwrap model to handle DDP wrapping (removes 'module.' prefix)
        for name, model in self.models.items():
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(checkpoint["models_state_dict"][name])
            self.logger.info(f"Loaded {name} model state")

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            self.logger.info("Loaded EMA state")
        elif self.ema is not None:
            self.logger.warning(
                "EMA is enabled but checkpoint has no ema_state_dict; starting fresh EMA"
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint with support for multiple optimizers and schedulers.
        Calls internal method to load model states.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self._load_checkpoint(checkpoint_path)

    def finalize_training(self) -> None:
        self._finalize_training()

    # =======================================================================
    # Additional utility methods
    # =======================================================================

    def get_trainer_info(self) -> dict[str, Any]:
        """
        Get information about this trainer.

        Returns:
            Dictionary containing trainer metadata
        """
        info = {
            "name": self.__class__.__name__,
            "models": [x.name for x in self.models.values()],
            "config": self.config,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "callbacks": [x.__class__.__name__ for x in self.callbacks.callbacks],
            "dataloaders": {
                "train": len(self.train_loader) if self.train_loader else 0,
                "validation": len(self.validation_loader)
                if self.validation_loader
                else 0,
                "test": len(self.test_loader) if self.test_loader else 0,
            },
        }

        # Add metric managers info
        if self.metric_managers:
            info["metric_managers"] = {}
            for name, manager in self.metric_managers.items():
                info["metric_managers"][name] = manager.get_metric_info()

        return info

    @classmethod
    def compute_model_weight_norms(cls, model: nn.Module) -> dict[str, float]:
        """
        Compute norms of model weights for monitoring gradient flow and weight changes.

        Returns:
            Dictionary of weight norms by layer/parameter type
        """

        weight_norms = {}

        try:
            # Overall model weight norm
            total_norm = 0.0
            total_params = 0

            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Parameter weight norm
                    param_norm = param.data.norm().item()
                    weight_norms[f"weights/{name.replace('.', '_')}"] = param_norm

                    if param.grad is not None:
                        # Gradient norm
                        grad_norm = param.grad.data.norm().item()
                        weight_norms[f"gradients/{name.replace('.', '_')}"] = grad_norm

                    total_norm += param_norm**2
                    total_params += param.numel()

            # Overall norms
            weight_norms["weights/total_norm"] = total_norm**0.5
            weight_norms["weights/mean_norm"] = (
                total_norm / max(total_params, 1)
            ) ** 0.5

            weight_norms["weights/total_paras"] = total_params
            # Layer-wise statistics
            layer_norms = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                    layer_weights = []
                    for param_name, param in module.named_parameters():
                        if param.requires_grad:
                            layer_weights.append(param.data.norm().item())

                    if layer_weights:
                        layer_key = (
                            f"layers/{name.replace('.', '_')}"
                            if name
                            else "layers/root"
                        )
                        layer_norms[layer_key] = np.mean(layer_weights)

            weight_norms.update(layer_norms)

        except Exception as e:
            logger.warning(f"Failed to compute weight norms: {e}")

        return weight_norms

    def generate_epoch_logs(self, epoch: int) -> dict[str, float]:
        """
        Generate epoch-level logs (model stats, weight norms, etc.).

        Called once per epoch to compute metrics not tied to a specific split.
        Default implementation computes weight norms for all models.

        Override _generate_epoch_logs() to add custom epoch-level metrics like:
        - Learning rate tracking
        - Gradient statistics
        - Custom model diagnostics
        - Resource usage metrics

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of epoch-level metrics

        Example:
            {
                "model_name/weights/total_norm": 45.3,
                "model_name/gradients/layer1": 2.1,
                "model_name/layers/encoder": 12.5
            }

        Example Override:
            def _generate_epoch_logs(self, epoch: int) -> dict[str, float]:
                logs = super()._generate_epoch_logs(epoch)  # Get weight norms
                logs['lr'] = self.optimizers['main'].param_groups[0]['lr']
                logs['gpu_memory'] = torch.cuda.max_memory_allocated() / 1e9
                return logs
        """
        return self._generate_epoch_logs(epoch)

    # =======================================================================
    # Properties for easy access
    # ======================================================================

    @property
    def train_loader(self) -> DataLoader | None:
        return self.data_loader["train"]

    @property
    def validation_loader(self) -> DataLoader | None:
        return self.data_loader["validation"]

    @property
    def test_loader(self) -> DataLoader | None:
        return self.data_loader["test"]

    @property
    def model(self) -> BaseModel:
        if len(self.models) != 1:
            raise AttributeError(
                "Multiple models found. Please access the specific model from the 'models' dict."
            )
        return next(iter(self.models.values()))

    @property
    def criterion(self) -> BaseCriterion:
        if len(self.criterions) != 1:
            raise AttributeError(
                "Multiple models found. Please access the specific model from the 'models' dict."
            )
        return next(iter(self.criterions.values()))

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if len(self.optimizers) != 1:
            raise AttributeError(
                "Multiple optimizers found. Please access the specific optimizer from the 'optimizers' dict."
            )
        return next(iter(self.optimizers.values()))

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        if len(self.schedulers) != 1:
            raise AttributeError(
                "Multiple schedulers found. Please access the specific scheduler from the 'schedulers' dict."
            )
        return next(iter(self.schedulers.values()))

    @property
    def train_metrics(self) -> dict[int, dict[str, float]]:
        """Get training metrics history."""
        if not self.accelerator.is_main_process():
            self.logger.warning(
                "Accessing train_metrics on non-main process may return incomplete data"
                "Metrics history is only fully populated on main process."
            )
        return self.metrics_history["train"]

    @property
    def validation_metrics(self) -> dict[int, dict[str, float]]:
        """Get validation metrics history."""
        if not self.accelerator.is_main_process():
            self.logger.warning(
                "Accessing validation_metrics on non-main process may return incomplete data"
                "Metrics history is only fully populated on main process."
            )
        return self.metrics_history["validation"]

    @property
    def test_metrics(self) -> dict[int, dict[str, float]]:
        """Get test metrics history."""
        if not self.accelerator.is_main_process():
            self.logger.warning(
                "Accessing test_metrics on non-main process may return incomplete data"
                "Metrics history is only fully populated on main process."
            )
        return self.metrics_history["test"]

    @property
    def current_metrics(self) -> dict[str, dict[str, float]]:
        """
        Get metrics for current epoch across all splits.

        Returns:
            {
                "train": {"loss": 0.5, "acc": 0.9},
                "validation": {"loss": 0.6, "acc": 0.88},
                "test": {"loss": 0.55, "acc": 0.89}
            }
        """
        epoch = self.current_epoch
        return {
            split: self.metrics_history[split].get(epoch, {})
            for split in ["train", "validation", "test"]
        }


# Compatibility alias for existing imports.
BaseTrainer = EpochTrainer
