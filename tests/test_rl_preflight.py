"""Tests for reinforcement-learning configuration preflight."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box
from pytest import MonkeyPatch
import yaml

from dl_core import load_builtin_components
from dl_core.core import register_model
from dl_core.single_run import main as run_main


@register_model("preflight_sac_actor")
class _PreflightSACActor(torch.nn.Module):
    """Project actor used to prove preflight resolves local registrations."""

    def __init__(self, config: dict[str, object]):
        super().__init__()
        self.linear = torch.nn.Linear(
            int(config["input_dim"]),
            2 * int(config["action_dim"]),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        statistics = self.linear(observations)
        mean, log_std = statistics.chunk(2, dim=1)
        return {"mean": mean, "log_std": log_std}


@register_model("preflight_sac_critics")
class _PreflightSACCritics(torch.nn.Module):
    """Project critics used to prove preflight resolves local registrations."""

    def __init__(self, config: dict[str, object]):
        super().__init__()
        input_dim = int(config["input_dim"]) + int(config["action_dim"])
        self.q1 = torch.nn.Linear(input_dim, 1)
        self.q2 = torch.nn.Linear(input_dim, 1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = torch.cat((observations, actions), dim=1)
        return {
            "q1": self.q1(inputs).squeeze(1),
            "q2": self.q2(inputs).squeeze(1),
        }


def test_rl_preflight_resolves_algorithm_components_without_stepping(
    tmp_path: Path,
    capsys,
    monkeypatch: MonkeyPatch,
) -> None:
    load_builtin_components()

    class _NoStepEnvironment:
        observation_space = Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        action_space = Box(-2.0, 2.0, shape=(1,), dtype=np.float32)

        def reset(self, **_kwargs):
            raise AssertionError("RL preflight must not reset the environment")

        def step(self, _action):
            raise AssertionError("RL preflight must not step the environment")

        def close(self) -> None:
            self.closed = True

    environments = []
    cleanup_calls = []

    def _make_environment(_config):
        environment = _NoStepEnvironment()
        environment.closed = False
        environments.append(environment)
        return environment

    monkeypatch.setattr(
        "dl_core.environments.make_environment",
        _make_environment,
    )
    monkeypatch.setattr(
        "dl_core.accelerators.cpu.CPUAccelerator.cleanup",
        lambda _self: cleanup_calls.append(True),
    )
    output_dir = tmp_path / "artifacts"
    config_path = tmp_path / "sac.yaml"
    config = {
        "seed": 31,
        "environment": {"name": "gymnasium", "id": "Pendulum-v1"},
        "models": {
            "actor": {"name": "preflight_sac_actor"},
            "critics": {"name": "preflight_sac_critics"},
        },
        "optimizers": {"name": "adam", "lr": 1e-3},
        "trainer": {
            "sac": {
                "total_timesteps": 10,
                "evaluation_episodes": 0,
                "checkpoint_frequency": 0,
            }
        },
        "accelerator": {"type": "cpu"},
        "runtime": {"output_dir": str(output_dir)},
        "experiment": {"name": "preflight", "run_name": "sac"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["dl-run", "--config", str(config_path), "--validate-only"],
    )

    assert run_main() == 0

    output = capsys.readouterr().out
    assert "RL preflight complete" in output
    assert "Environment: gymnasium" in output
    assert "actor -> test_rl_preflight._PreflightSACActor" in output
    assert "critics -> test_rl_preflight._PreflightSACCritics" in output
    assert "No environment steps or training updates were run." in output
    assert len(environments) == 2
    assert all(environment.closed for environment in environments)
    assert cleanup_calls == [True]
    assert not output_dir.exists()
