"""
Standard trainer implementation that loads everything from config.
"""

from typing import Any

import torch

from dl_core.core import (
    CRITERION_REGISTRY,
    METRIC_MANAGER_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    BaseTrainer,
    register_trainer,
)


@register_trainer("standard")
class StandardTrainer(BaseTrainer):
    """
    Standard trainer that loads all components from config.

    Supports single model, multiple criterions, single optimizer,
    and optional scheduler.
    """

    def _get_single_component_config(
        self,
        section_name: str,
        default_name: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Normalize a single-component config section.

        The standard trainer accepts the simple shape:

        section_name:
          name: adamw
          lr: 1e-4

        It also accepts the legacy nested single-entry shape for compatibility:

        section_name:
          main:
            name: adamw
            lr: 1e-4
        """
        section_cfg = self.config.get(section_name, {})
        if not section_cfg:
            raise ValueError(f"No {section_name} configured")
        if not isinstance(section_cfg, dict):
            raise ValueError(f"'{section_name}' must be a dict")

        if "name" in section_cfg:
            component_name = str(section_cfg.get("name") or default_name or "main")
            return component_name, {
                key: value for key, value in section_cfg.items() if key != "name"
            }

        nested_items = list(section_cfg.items())
        if len(nested_items) != 1:
            raise ValueError(
                f"Standard trainer supports exactly one {section_name[:-1]} "
                f"configuration. Use a specialized trainer for multiple "
                f"{section_name}."
            )

        component_key, component_cfg = nested_items[0]
        if not isinstance(component_cfg, dict):
            raise ValueError(
                f"'{section_name}.{component_key}' must be a dict of parameters"
            )

        component_cfg = dict(component_cfg)
        component_name = str(component_cfg.get("name") or component_key)
        return component_name, {
            key: value for key, value in component_cfg.items() if key != "name"
        }

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return only trainable model parameters."""
        parameters = [param for param in self.model.parameters() if param.requires_grad]
        if not parameters:
            raise ValueError("Model has no trainable parameters")
        return parameters

    def setup_model(self) -> None:
        """Setup model from config."""
        models_cfg = self.config.get("models")
        if not isinstance(models_cfg, dict) or not models_cfg:
            raise ValueError("No models configured in training config")

        model_name, model_cfg = next(iter(models_cfg.items()))
        model_cfg = dict(model_cfg or {})
        self.models["main"] = MODEL_REGISTRY.get(model_name, model_cfg)

        # Log model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(str(self.model))
        self.logger.info(f"Initialized model: {model_cfg}")
        self.logger.info(f"Model parameters: {total_params:,}")

    def setup_criterion(self) -> None:
        """Setup criterions from config."""
        criterions_cfg = self.config.get("criterions", {})
        if not isinstance(criterions_cfg, dict):
            raise ValueError("Criterions config must be a mapping of names to configs")

        for criterion_name, criterion_cfg in criterions_cfg.items():
            criterion_cfg = dict(criterion_cfg or {})
            criterion_cfg.setdefault("name", criterion_name)

            # Pass only the criterion config
            criterion = CRITERION_REGISTRY.get(
                criterion_name,
                criterion_cfg,
            )
            self.criterions[criterion_name] = criterion

        self.logger.info(f"Initialized criterions: {', '.join(self.criterions.keys())}")

    def setup_metrics(self) -> None:
        """Setup metrics from config."""
        metric_manager_configs = self.config.get("metric_managers", {})
        if not isinstance(metric_manager_configs, dict):
            raise ValueError(
                "Metric managers config must be a mapping of names to configs"
            )

        for manager_name, manager_cfg in metric_manager_configs.items():
            # Remove 'name' to avoid conflict, pass rest as kwargs
            manager_kwargs = {
                k: v for k, v in (manager_cfg or {}).items() if k != "name"
            }
            self.metric_managers[manager_name] = METRIC_MANAGER_REGISTRY.get(
                manager_name,
                manager_kwargs,
                self.accelerator,
                self,
            )
        self.logger.info(
            f"Using metric managers: {', '.join(self.metric_managers.keys())}"
        )

    def setup_optimizer(self) -> None:
        """Setup optimizer from config."""
        if not self.models:
            raise ValueError("Model is not initialized. Call setup_model() first.")

        optimizer_name, optimizer_kwargs = self._get_single_component_config(
            "optimizers",
            default_name="adam",
        )
        self.optimizers["main"] = OPTIMIZER_REGISTRY.get(
            optimizer_name,
            self._trainable_parameters(),
            **optimizer_kwargs,
        )
        self.logger.info(f"Initialized optimizer: {optimizer_name}")

    def setup_scheduler(self) -> None:
        """Setup scheduler from config (optional)."""
        schedulers_cfg = self.config.get("schedulers", {})

        if not schedulers_cfg:
            self.logger.info("No scheduler configured")
            return

        if "main" not in self.optimizers:
            raise ValueError("Optimizer must be setup before scheduler")

        scheduler_name, scheduler_kwargs = self._get_single_component_config(
            "schedulers"
        )
        self.schedulers["main"] = SCHEDULER_REGISTRY.get(
            scheduler_name,
            self.optimizers["main"],
            **scheduler_kwargs,
        )

        self.logger.info(f"Initialized scheduler: {scheduler_name}")

    def train_step(
        self, batch_data: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """Single training step."""
        labels = batch_data["label"]
        paths = batch_data["path"]

        # Zero gradients
        self.optimizers["main"].zero_grad()

        # Forward and backward pass with mixed precision autocast
        with self.accelerator.autocast_context():
            # Forward pass
            model_output = self.model(batch_data)

            # Compute loss from all criterions
            total_loss = torch.tensor(0.0, device=labels.device)
            loss_components = {}

            for name, criterion in self.criterions.items():
                loss_output = criterion(model_output["logits"], labels)
                criterion_loss = loss_output["loss"]
                total_loss += criterion_loss
                loss_components[f"{name}_loss"] = criterion_loss.item()

            # Backward pass (inside autocast context)
            self.accelerator.backward(total_loss, self.model)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        # Optimizer step
        self.accelerator.optimizer_step(self.optimizers["main"], self.model)

        # Scheduler step
        if "main" in self.schedulers:
            self.schedulers["main"].step()

        # Update metrics
        if self.metric_managers:
            with torch.no_grad():
                attack_types = batch_data.get("attack_type")
                for manager in self.metric_managers.values():
                    manager.update("train", model_output["probabilities"], batch_data)

        return {"loss": total_loss.item(), **loss_components}

    def test_step(self, batch_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single test step."""
        labels = batch_data["label"]
        paths = batch_data["path"]

        # Forward pass with mixed precision autocast
        with self.accelerator.autocast_context():
            model_output = self.model(batch_data)

            # Compute loss from all criterions
            total_loss = torch.tensor(0.0, device=labels.device)
            loss_components = {}

            for name, criterion in self.criterions.items():
                loss_output = criterion(model_output["logits"], labels)
                criterion_loss = loss_output["loss"]
                total_loss += criterion_loss
                loss_components[f"{name}_loss"] = criterion_loss.item()

        # Update metrics
        if self.metric_managers:
            attack_types = batch_data.get("attack_type")
            for manager in self.metric_managers.values():
                manager.update("test", model_output["probabilities"], batch_data)

        return {"loss": total_loss.item(), **loss_components}

    def validation_step(self, batch_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single validation step."""
        labels = batch_data["label"]
        paths = batch_data["path"]

        # Forward pass with mixed precision autocast
        with self.accelerator.autocast_context():
            model_output = self.model(batch_data)

            # Compute loss from all criterions
            total_loss = torch.tensor(0.0, device=labels.device)
            loss_components = {}

            for name, criterion in self.criterions.items():
                loss_output = criterion(model_output["logits"], labels)
                criterion_loss = loss_output["loss"]
                total_loss += criterion_loss
                loss_components[f"{name}_loss"] = criterion_loss.item()

        # Update metrics
        if self.metric_managers:
            attack_types = batch_data.get("attack_type")
            for manager in self.metric_managers.values():
                manager.update("validation", model_output["probabilities"], batch_data)

        return {"loss": total_loss.item(), **loss_components}
