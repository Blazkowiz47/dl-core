"""CLI smoke coverage for dataset inspection and one-batch smoke helpers."""

from __future__ import annotations

from pathlib import Path

from dl_core.dataset_inspect import main as inspect_main
from dl_core.smoke import main as smoke_main


def _write_local_cli_project(root_dir: Path) -> tuple[Path, str, str]:
    """Create a tiny local project that exposes a dataset and model."""
    dataset_name = f"toy_dataset_{root_dir.name.replace('-', '_')}"
    model_name = f"toy_model_{root_dir.name.replace('-', '_')}"

    (root_dir / "pyproject.toml").write_text(
        "[project]\nname = 'cli-demo'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    (root_dir / "configs").mkdir(parents=True, exist_ok=True)
    (root_dir / "src" / "datasets").mkdir(parents=True, exist_ok=True)
    (root_dir / "src" / "models").mkdir(parents=True, exist_ok=True)
    (root_dir / "src" / "datasets" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (root_dir / "src" / "models" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (root_dir / "src" / "datasets" / "toy_dataset.py").write_text(
        f"""
\"\"\"Local dataset used by dataset CLI tests.\"\"\"

from __future__ import annotations

from typing import Any

import torch

from dl_core.core import register_dataset
from dl_core.core.base_dataset import BaseWrapper


@register_dataset("{dataset_name}")
class ToyDataset(BaseWrapper):
    \"\"\"Small in-memory dataset for CLI tests.\"\"\"

    @property
    def file_extensions(self) -> list[str]:
        return []

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        counts = {{"train": 4, "validation": 2, "test": 1}}
        return [
            {{"label": idx % 2, "index": idx, "path": f"{{split}}://{{idx}}"}}
            for idx in range(counts.get(split, 0))
        ]

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        value = float(file_dict["index"] + 1)
        return {{
            "image": torch.full((3, 8, 8), value),
            "label": file_dict["label"],
            "path": file_dict["path"],
        }}
""",
        encoding="utf-8",
    )
    (root_dir / "src" / "models" / "toy_model.py").write_text(
        f"""
\"\"\"Local model used by dataset CLI tests.\"\"\"

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dl_core.core.base_model import BaseModel
from dl_core.core.registry import register_model


@register_model("{model_name}")
class ToyModel(BaseModel):
    \"\"\"Small linear model for CLI tests.\"\"\"

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.proj = nn.Linear(3 * 8 * 8, self.num_classes)

    def compute_forward(
        self,
        batch_data: dict,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        features = batch_data["image"].reshape(batch_data["image"].shape[0], -1)
        logits = self.proj(features)
        probabilities = torch.softmax(logits, dim=1)
        return {{
            "probabilities": probabilities,
            "logits": logits,
            "features": features,
        }}
""",
        encoding="utf-8",
    )

    config_path = root_dir / "configs" / "base.yaml"
    config_path.write_text(
        "dataset:\n"
        f"  name: {dataset_name}\n"
        "  batch_size: {train: 2, validation: 2, test: 1}\n"
        "  num_workers: {train: 0, validation: 0, test: 0}\n"
        "  shuffle: {train: false, validation: false, test: false}\n"
        "  auto_split: false\n"
        "models:\n"
        "  main:\n"
        f"    name: {model_name}\n"
        "    num_classes: 2\n",
        encoding="utf-8",
    )
    return config_path, dataset_name, model_name


def test_dataset_cli_reports_split_sizes_and_batch_shapes(
    tmp_path: Path,
    capsys,
) -> None:
    """Dataset inspection should print split sizes and a preview batch."""
    config_path, dataset_name, _ = _write_local_cli_project(tmp_path)

    assert inspect_main(["--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert f"Dataset: {dataset_name}" in output
    assert "Split sizes" in output
    assert "train: 4 samples" in output
    assert "validation: 2 samples" in output
    assert "test: 1 samples" in output
    assert "train batch" in output
    assert "image: tensor(shape=[2, 3, 8, 8]" in output


def test_smoke_cli_resolves_nested_model_name(
    tmp_path: Path,
    capsys,
) -> None:
    """The smoke helper should respect models.<key>.name registry targets."""
    config_path, dataset_name, model_name = _write_local_cli_project(tmp_path)

    assert smoke_main(["--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert f"Dataset: {dataset_name}" in output
    assert f"Model: main -> {model_name}" in output
    assert "Forward output" in output
