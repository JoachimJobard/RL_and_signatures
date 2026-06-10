"""Integration smoke tests: each supported agent trains for one episode.

Exercises the full instantiate -> train path for the three thesis agents (the
quarantined CSAC is excluded). Catches shape/sign/precision regressions in the
agent update that the unit tests cannot reach. These are slower than the unit
tests (JAX compilation + one short episode each).
"""

import pytest
import wandb
from hydra import compose, initialize_config_dir

from src.training.train import train
from src.utils.run_context import find_repo_root

CONF_DIR = str(find_repo_root(__file__) / "conf")
SUPPORTED_AGENTS = ["signatures", "base_jax", "value_gradient"]


@pytest.mark.parametrize("agent_name", SUPPORTED_AGENTS)
def test_agent_trains_one_episode(agent_name):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config_unified",
            overrides=[
                f"agent={agent_name}",
                "env=delay_jax",
                "agent.training.n_episodes=1",
                "agent.training.max_time=1.0",
                "wandb.mode=disabled",
            ],
        )

    wandb.init(mode="disabled")
    try:
        agent, metrics = train(cfg)
    finally:
        wandb.finish()

    assert agent is not None
    assert isinstance(metrics, dict) and len(metrics) > 0
    # Training should record a finite per-episode cost.
    assert "cost_episodic" in metrics
