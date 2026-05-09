"""
Unified training module for all CTAC agent variants.

Uses hydra.utils.instantiate with _target_ to dynamically create the correct agent.
This replaces: train_signatures.py, train_base_jax.py, train_CSAC.py

Usage:
    from src.training.train import train
    agent, metrics = train(cfg)
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import jax.numpy as jnp
from typing import Any, Protocol, runtime_checkable

from src.envs.env_rk_jax import JAXDDEEnv
from src.configs import (
    TrainingConfig, DiscountConfig, NoiseConfig,
    SignatureConfig, NetworkConfig, AlgorithmConfig,
    ReplayBufferConfig,
)


# =============================================================================
# Agent Protocol (interface commune)
# =============================================================================

@runtime_checkable
class TrainableAgent(Protocol):
    """Protocol defining the interface all agents must implement."""
    env: Any
    x0: Any
    wrapper: Any
    training: Any
    algorithm: Any
    sliding_signature: Any
    signature_conf: Any
    
    def train(self) -> dict:
        """Train the agent and return metrics dictionary."""
        ...
    def eval_callback(self, episode: int) -> None:
        """Optional callback for periodic evaluation during training."""
        ...
    def _fill_buffer_initial(self) -> None:
        ...
    def update_buffer(self, x: Any) -> None:
        ...
    def get_eval_action(self, x_scaled: Any) -> Any:
        ...

# =============================================================================
# Environment Building
# =============================================================================

def build_environment(cfg: DictConfig) -> JAXDDEEnv:
    """
    Build environment from Hydra config using _target_.
    
    Parameters
    ----------
    cfg : DictConfig
        Full configuration (uses cfg.env.environment_params)
    
    Returns
    -------
    JAXDDEEnv or subclass
        Initialized environment
    """
    env_params = cfg.env.environment_params
    env = hydra.utils.instantiate(env_params)
    return env


# =============================================================================
# Agent Building
# =============================================================================

def build_agent(cfg: DictConfig, env: JAXDDEEnv) -> TrainableAgent:
    """
    Build agent from Hydra config using _target_.
    
    Parameters
    ----------
    cfg : DictConfig
        Full configuration (uses cfg.agent)
    env : JAXDDEEnv
        Environment instance
    
    Returns
    -------
    TrainableAgent
        Initialized agent (CTACSignatureJAX, CTACJAX, CSAC, etc.)
    """
    agent_cfg = cfg.agent

    def _make_config(cls, section):
        if section is None:
            return cls()
        return cls(**OmegaConf.to_container(section, resolve=True))  # type: ignore

    agent_kwargs = {
        'env': env,
        'rng_key': agent_cfg.get('rng_key', cfg.seed),
        'training': _make_config(TrainingConfig, agent_cfg.get('training')),
        'discount': _make_config(DiscountConfig, agent_cfg.get('discount')),
        'noise': _make_config(NoiseConfig, agent_cfg.get('noise')),
        'signature_conf': _make_config(SignatureConfig, agent_cfg.get('signature')),
        'network': _make_config(NetworkConfig, agent_cfg.get('network')),
        'algorithm': _make_config(AlgorithmConfig, agent_cfg.get('algorithm')),
    }

    # CSAC-specific: replay buffer config
    if 'replay_buffer' in agent_cfg:
        agent_kwargs['replay_buffer'] = _make_config(
            ReplayBufferConfig, agent_cfg.get('replay_buffer'))

    # Forward remaining agent-level keys (x0, eval_callback, ...)
    config_keys = {
        '_target_', 'training', 'discount', 'noise',
        'signature', 'network', 'algorithm', 'rng_key',
        'replay_buffer', 'env',
    }
    agent_cfg_dict = OmegaConf.to_container(agent_cfg, resolve=True)
    if isinstance(agent_cfg_dict, dict):
        for k, v in agent_cfg_dict.items(): 
            if k not in config_keys:
                agent_kwargs[str(k)] = v

    agent = hydra.utils.instantiate(
        {'_target_': agent_cfg._target_},
        **agent_kwargs
    )
    return agent


# =============================================================================
# Training Function
# =============================================================================

def train(cfg: DictConfig, eval_callback=None) -> tuple[TrainableAgent, dict]:
    """
    Train an agent and return it with metrics.
    
    This is the main entry point for training any agent type.
    The agent class is determined by cfg.agent._target_.
    
    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration
    eval_callback : callable, optional
        Callback ``(agent, episode) -> None`` invoked inside the training loop.
        Created via ``make_eval_callback`` for periodic trajectory snapshots.
    
    Returns
    -------
    tuple[TrainableAgent, dict]
        (trained_agent, metrics_dict)
    
    Examples
    --------
    >>> agent, metrics = train(cfg)  # Agent type from cfg.agent._target_
    """
    # Build environment
    print("=" * 60)
    print("Building Environment")
    print("=" * 60)
    env = build_environment(cfg)
    print(f"  Type: {env.__class__.__name__}")
    print(f"  State dim: {env.N}")
    print(f"  Action dim: {env.B.shape[1]}")
    if hasattr(env, 'delay') and env.delay is not None:
        print(f"  Delay: {np.array(env.delay)}")
    print(f"  Step size: {env.step_size}")
    
    # Build agent
    print("\n" + "=" * 60)
    print("Building Agent")
    print("=" * 60)
    agent = build_agent(cfg, env)
    agent_name = agent.__class__.__name__
    print(f"  Type: {agent_name}")
    
    # Print agent-specific info
    if (depth := getattr(agent, "depth", None)) is not None:
        print(f"  Signature depth: {depth}")
    if (window_size := getattr(agent, "window_size", None)) is not None:
        print(f"  Window size: {window_size}")
    if (semi := getattr(agent, "semi_gradient", None)) is not None:
        print(f"  Semi-gradient: {semi}")
    if (discounted := getattr(agent, "discounted", None)) is not None:
        print(f"  Discounted: {discounted}")
    n_episodes = cfg.agent.get('training', cfg.agent.get('training_params', {})).get('n_episodes', '?')
    print(f"  Episodes: {n_episodes}")
    
    # Set evaluation / fixed initial state from config
    if hasattr(cfg, 'eval') and 'x0_test' in cfg.eval:
        agent.x0 = jnp.array(np.array(cfg.eval.x0_test))
        print(f"  Eval x0: {np.array(cfg.eval.x0_test)}")

    # Attach periodic evaluation callback if provided
    if eval_callback is not None:
        agent.eval_callback = eval_callback 
    
    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    metrics = agent.train()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    if 'cost_episodic' in metrics and len(metrics['cost_episodic']) > 0:
        print(f"  Final episodic cost: {metrics['cost_episodic'][-1]:.4f}")
    if 'loss_episodic' in metrics and len(metrics['loss_episodic']) > 0:
        print(f"  Final episodic loss: {metrics['loss_episodic'][-1]:.4f}")
    
    return agent, metrics


# =============================================================================
# Standalone Entry Point
# =============================================================================

@hydra.main(config_path="../../conf", config_name="config_unified", version_base=None)
def main(cfg: DictConfig):
    """Standalone training entry point."""
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    agent, metrics = train(cfg)
    return agent, metrics


if __name__ == "__main__":
    main()
