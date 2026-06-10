"""Every environment config must instantiate.

This pins code_review F-D6: four env configs referenced a non-existent module
(`src.env_rk_jax` instead of `src.envs.env_rk_jax`), so instantiating them raised
ModuleNotFoundError. Instantiating every `conf/env/*.yaml` here would catch that
class of regression.
"""

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from src.utils.run_context import find_repo_root

ENV_CONFIG_DIR = find_repo_root(__file__) / "conf" / "env"
ENV_CONFIGS = sorted(ENV_CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", ENV_CONFIGS, ids=lambda p: p.name)
def test_env_config_instantiates(config_path):
    cfg = OmegaConf.load(config_path)
    assert "environment_params" in cfg, f"{config_path.name} has no environment_params block"
    env = hydra.utils.instantiate(cfg.environment_params)
    # All environments expose the state dimension N.
    assert hasattr(env, "N") and env.N >= 1
