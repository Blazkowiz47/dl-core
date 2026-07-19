"""Tests for reinforcement-learning configuration preflight."""

from __future__ import annotations

from pathlib import Path

from dl_core import load_builtin_components
from dl_core.single_run import _run_preflight


def test_rl_preflight_resolves_algorithm_components_without_stepping(
    tmp_path: Path,
    capsys,
) -> None:
    load_builtin_components()
    config = {
        "seed": 31,
        "environment": {"name": "gymnasium", "id": "Pendulum-v1"},
        "models": {
            "actor": {"name": "sac_gaussian_actor", "hidden_sizes": [8]},
            "critics": {"name": "sac_twin_q_network", "hidden_sizes": [8]},
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
        "runtime": {"output_dir": str(tmp_path / "artifacts")},
        "experiment": {"name": "preflight", "run_name": "sac"},
    }

    _run_preflight(
        config,
        config_path=tmp_path / "sac.yaml",
        mode="local",
        run_name="sac",
    )

    output = capsys.readouterr().out
    assert "RL preflight complete" in output
    assert "Environment: gymnasium" in output
    assert "actor -> dl_core.models.sac.SACGaussianActor" in output
    assert "critics -> dl_core.models.sac.SACTwinQNetwork" in output
    assert "No environment steps or training updates were run." in output
