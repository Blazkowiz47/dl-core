"""Callback that refreshes dataset-backed dataloaders between epochs."""

from __future__ import annotations

from typing import Any

from dl_core.core.base_callback import Callback
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_callback


@register_callback("dataset_refresh")
class DatasetRefreshCallback(Callback):
    """Refresh selected dataset splits and rebuild their dataloaders."""

    CONFIG_FIELDS = Callback.CONFIG_FIELDS + [
        config_field(
            "refresh_frequency",
            "int",
            "Refresh the selected splits every N epochs.",
            default=1,
        ),
        config_field(
            "splits",
            "list[str]",
            "Dataset splits to refresh on matching epochs.",
            default=["train"],
        ),
    ]

    def __init__(
        self,
        refresh_frequency: int = 1,
        splits: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset refresh callback."""
        super().__init__(
            refresh_frequency=refresh_frequency,
            splits=splits,
            **kwargs,
        )
        self.refresh_frequency = max(int(refresh_frequency), 1)
        self.splits = list(splits or ["train"])
        invalid_splits = sorted(set(self.splits) - {"train", "validation", "test"})
        if invalid_splits:
            raise ValueError(
                "DatasetRefreshCallback splits must be drawn from "
                f"train/validation/test, got: {invalid_splits}"
            )

    def on_epoch_start(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Refresh selected dataset splits and rebuild their dataloaders."""
        if epoch % self.refresh_frequency != 0:
            return

        dataset_wrapper = getattr(self.trainer, "dataset_wrapper", None)
        if dataset_wrapper is None:
            self.logger.warning("No dataset wrapper available for dataset refresh")
            return

        refreshed_loaders: dict[str, Any] = {}
        for split in self.splits:
            dataset_wrapper.refresh_dataset(split)
            refreshed_loaders[split] = dataset_wrapper.get_split(split)

        _, _, _, _, prepared_loaders = self.trainer.accelerator.prepare(
            dataloaders=refreshed_loaders
        )
        self.trainer.data_loader.update(prepared_loaders)
        self.logger.info(
            "Refreshed dataset splits at epoch %s: %s",
            epoch,
            ", ".join(self.splits),
        )
