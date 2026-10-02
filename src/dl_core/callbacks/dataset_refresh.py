"""Rebuild selected dataset loaders at epoch or data-cycle boundaries."""

from __future__ import annotations

from typing import Any, Literal

from dl_core.core.base_callback import Callback
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_callback


@register_callback("dataset_refresh")
class DatasetRefreshCallback(Callback):
    """Refresh selected dataset splits and rebuild their dataloaders."""

    CONFIG_FIELDS = Callback.CONFIG_FIELDS + [
        config_field(
            "trigger",
            "str",
            "Refresh at 'epoch' or 'data_cycle' boundaries.",
            default="epoch",
        ),
        config_field(
            "refresh_frequency",
            "int",
            "Refresh the selected splits every N epochs or data cycles.",
            default=1,
        ),
        config_field(
            "splits",
            "list[str]",
            "Dataset splits to refresh at the selected boundary.",
            default=["train"],
        ),
    ]

    def __init__(
        self,
        refresh_frequency: int = 1,
        splits: list[str] | None = None,
        trigger: Literal["epoch", "data_cycle"] = "epoch",
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset refresh callback."""
        super().__init__(
            refresh_frequency=refresh_frequency,
            splits=splits,
            trigger=trigger,
            **kwargs,
        )
        self.refresh_frequency = max(int(refresh_frequency), 1)
        if trigger not in {"epoch", "data_cycle"}:
            raise ValueError(
                "DatasetRefreshCallback trigger must be epoch or data_cycle"
            )
        self.trigger = trigger
        self.splits = list(splits or ["train"])
        invalid_splits = sorted(set(self.splits) - {"train", "validation", "test"})
        if invalid_splits:
            raise ValueError(
                "DatasetRefreshCallback splits must be drawn from "
                f"train/validation/test, got: {invalid_splits}"
            )

    def on_epoch_start(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Preserve epoch refresh unless a data-cycle trigger was configured."""
        if self.trigger == "epoch" and epoch % self.refresh_frequency == 0:
            self._refresh(epoch)

    def on_data_cycle_start(
        self, cycle: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Refresh cycle sources, including the active selection on resume."""
        if self.trigger != "data_cycle":
            return
        if cycle % self.refresh_frequency != 0 and not (logs or {}).get("initial"):
            return
        # A resumed intermediate cycle still uses the last refreshed selection.
        selection_cycle = cycle - cycle % self.refresh_frequency
        self._refresh(selection_cycle, current_cycle=cycle)

    def _refresh(self, index: int, *, current_cycle: int | None = None) -> None:
        """Refresh selected dataset splits and rebuild their dataloaders."""
        dataset_wrapper = getattr(self.trainer, "dataset_wrapper", None)
        if dataset_wrapper is None:
            raise RuntimeError("No dataset wrapper available for dataset refresh")

        refreshed_loaders: dict[str, Any] = {}
        if current_cycle is not None and index != current_cycle:
            dataset_wrapper.set_epoch(index)
        try:
            for split in self.splits:
                loader = self.trainer.data_loader.pop(split, None)
                iterator = getattr(loader, "_iterator", None)
                if iterator is not None:
                    iterator._shutdown_workers()
                    loader._iterator = None
                # A distributed sampler can retain the retired indexed dataset.
                samplers = getattr(self.trainer.accelerator, "samplers", {})
                samplers.pop(split, None)
                del iterator, loader
                dataset_wrapper.refresh_dataset(split)
                refreshed_loaders[split] = dataset_wrapper.get_split(split)
                if split == "train" and refreshed_loaders[split] is None:
                    raise RuntimeError("Dataset refresh rebuilt an empty train loader")
        finally:
            if current_cycle is not None and index != current_cycle:
                dataset_wrapper.set_epoch(current_cycle)

        _, _, _, _, prepared_loaders = self.trainer.accelerator.prepare(
            dataloaders=refreshed_loaders
        )
        self.trainer.data_loader.update(prepared_loaders)
        self.logger.info(
            "Refreshed dataset splits at %s %s: %s",
            self.trigger,
            index,
            ", ".join(self.splits),
        )
