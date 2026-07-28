"""End-to-end integration tests for public Dreamer workflows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import random
from typing import Iterator
import warnings

import numpy as np
import pytest
import torch
import yaml

from dl_core import load_builtin_components
from dl_core.single_run import main as run_main
from dl_core.trainers import DreamerTrainer
from dl_core.utils.config_validator import ConfigValidator
from dreamer_test_models import (  # noqa: F401
    ProjectDreamerActor,
    ProjectDreamerCritic,
    ProjectDreamerWorldModel,
)


@pytest.fixture(autouse=True)
def _restore_process_state() -> Iterator[None]:
    """Keep integration tests from leaking runtime process configuration."""
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    cuda_random_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    deterministic_algorithms = (
        torch.are_deterministic_algorithms_enabled()
    )
    deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    environment_state = {
        name: os.environ.get(name)
        for name in (
            "PYTHONHASHSEED",
            "CUBLAS_WORKSPACE_CONFIG",
            "NCCL_DEBUG",
        )
    }
    root_logger = logging.getLogger()
    root_handlers = list(root_logger.handlers)
    root_level = root_logger.level
    logger_names = (
        "azure.storage",
        "azure",
        "azure.core",
        "azure.identity",
        "azure.identity._internal.decorators",
        "azure.core.pipeline.policies.http_logging_policy",
        "matplotlib",
        "PIL",
        "torch",
        "torchvision",
        "onnxruntime",
        "azureml",
        "urllib3.connectionpool",
        "urllib3",
        "requests",
    )
    logger_levels = {
        name: logging.getLogger(name).level for name in logger_names
    }
    warning_filters = list(warnings.filters)

    yield

    random.setstate(python_random_state)
    np.random.set_state(numpy_random_state)
    torch.random.set_rng_state(torch_random_state)
    if cuda_random_state is not None:
        torch.cuda.set_rng_state_all(cuda_random_state)
    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=deterministic_warn_only,
    )
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    for name, value in environment_state.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    for handler in root_logger.handlers:
        if handler not in root_handlers:
            handler.close()
    root_logger.handlers[:] = root_handlers
    root_logger.setLevel(root_level)
    for name, level in logger_levels.items():
        logging.getLogger(name).setLevel(level)
    warnings.filters[:] = warning_filters


def _config(tmp_path: Path, *, total_timesteps: int = 8) -> dict:
    return {
        "seed": 23,
        "deterministic": True,
        "environment": {
            "name": "gymnasium_vector",
            "id": "FrozenLake-v1",
            "num_envs": 2,
            "vectorization_mode": "sync",
            "kwargs": {"is_slippery": False},
        },
        "evaluation_environment": {
            "name": "gymnasium",
            "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": False},
        },
        "models": {
            "world_model": {
                "name": "test_dreamer_world_model",
                "embedding_size": 8,
                "deterministic_size": 8,
                "stochastic_size": 2,
                "classes": 3,
                "hidden_size": 16,
            },
            "actor": {
                "name": "test_dreamer_actor",
                "hidden_size": 16,
            },
            "critic": {
                "name": "test_dreamer_critic",
                "hidden_size": 16,
            },
        },
        "optimizers": {
            "world_model": {"name": "adam", "lr": 1e-3},
            "actor": {"name": "adam", "lr": 1e-3},
            "critic": {"name": "adam", "lr": 1e-3},
        },
        "trainer": {
            "dreamer": {
                "total_timesteps": total_timesteps,
                "max_episode_steps": 4,
                "evaluation_frequency": 0,
                "evaluation_episodes": 1,
                "checkpoint_frequency": 0,
                "buffer_size": 16,
                "batch_size": 1,
                "sequence_length": 2,
                "burn_in": 0,
                "learning_starts": 0,
                "train_frequency": 2,
                "gradient_steps": 1,
                "imagination_horizon": 2,
            }
        },
        "accelerator": {"type": "cpu"},
        "runtime": {"output_dir": str(tmp_path / "artifacts")},
        "experiment": {
            "name": "dreamer-integration",
            "run_name": "tiny",
        },
    }


def test_dreamer_config_validates_and_preflights(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    load_builtin_components()
    config_path = tmp_path / "dreamer.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(tmp_path)),
        encoding="utf-8",
    )

    validator = ConfigValidator(str(config_path))
    assert validator.validate()
    monkeypatch.setattr(
        "sys.argv",
        ["dl-run", "--config", str(config_path), "--validate-only"],
    )

    assert run_main() == 0

    output = capsys.readouterr().out
    assert "RL preflight complete" in output
    assert "Trainer: dreamer" in output
    assert "ProjectDreamerWorldModel" in output
    assert "ProjectDreamerActor" in output
    assert "ProjectDreamerCritic" in output
    assert "No environment steps or training updates were run." in output
    assert not (tmp_path / "artifacts").exists()


def test_dreamer_runs_vector_training_and_independent_evaluation(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = DreamerTrainer(_config(tmp_path))

    trainer.run()

    assert trainer.global_step == 8
    assert trainer.collector_step == 4
    assert trainer.update_step == 3
    assert len(trainer.evaluation_metrics) == 1
    assert trainer.evaluation_metrics[0]["global_step"] == 8.0
    checkpoint_path = trainer.artifact_manager.get_final_checkpoint_path(
        "latest.pth"
    )
    history_path = (
        Path(trainer.artifact_manager.get_run_artifact_dir())
        / "final/metrics/history.json"
    )
    assert checkpoint_path.exists()
    assert history_path.exists()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["global_step"] == 8
    assert checkpoint["update_step"] == 3
    assert checkpoint["algorithm_state"]["replay_buffer"] is not None


def test_dreamer_resume_continues_vector_training(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = DreamerTrainer(_config(tmp_path))
    trainer.run()
    checkpoint_path = trainer.artifact_manager.get_final_checkpoint_path(
        "latest.pth"
    )

    resumed_config = _config(tmp_path, total_timesteps=10)
    resumed_config["trainer"]["dreamer"]["continue_model"] = str(
        checkpoint_path
    )
    resumed = DreamerTrainer(resumed_config)
    resumed.run()

    assert resumed.global_step == 10
    assert resumed.collector_step == 5
    assert resumed.update_step == 4
    resumed_checkpoint = torch.load(
        resumed.artifact_manager.get_final_checkpoint_path("latest.pth"),
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_checkpoint["global_step"] == 10
    assert resumed_checkpoint["update_step"] == 4
