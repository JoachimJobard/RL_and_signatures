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
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from src.envs.env_rk_jax import JAXDDEEnv
from src.training.evaluate import conform_initial_state
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
    representation_buffer: Any
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

def apply_discount_truncation_horizon(cfg: DictConfig, env: JAXDDEEnv) -> None:
    """Opt-in discount-truncation horizon (results-CHANGING; OFF by default).

    When ``cfg.eval.discount_tolerance`` is set to a tolerance ``eps`` in (0, 1), the explicit
    rollout horizon is REPLACED, IN PLACE, by the discount-truncation horizon

        T = (1/gamma) ln(1/eps),   N = ceil(T/dt),

    the time after which the discounted tail of the infinite-horizon objective
    J = int_0^inf e^{-gamma s} c(s) ds is bounded by (c_max/gamma) e^{-gamma T} = (c_max/gamma) eps,
    with c_max the supremum of the instantaneous cost rate c(s) = x'Qx + u'Ru over the tail. Both
    ``agent.training.max_time`` and ``eval.T_sim`` are set to T so training and evaluation use the
    same truncated horizon. The truncation is announced once (the repository forbids silent
    truncations). Deterministic per cell: it yields a fewer-step FIXED lax.scan, fully compatible
    with the Part 1 fast path.

    When ``discount_tolerance`` is null/absent this function is a NO-OP -- the explicit,
    pre-registered ``T_sim``/``max_time`` are left untouched and the run is bit-identical to one
    without the knob.
    """
    import math

    eval_cfg = cfg.get("eval", None) if hasattr(cfg, "get") else None
    eps = eval_cfg.get("discount_tolerance", None) if eval_cfg is not None else None
    if eps is None:
        return  # knob unset: explicit horizon unchanged, bit-identical to pre-knob behaviour

    discount = cfg.agent.get("discount", None)
    discounted = bool(discount.get("discounted", False)) if discount is not None else False
    gamma = float(discount.get("gamma", 0.0)) if discount is not None else 0.0
    if not discounted or gamma <= 0.0:
        raise ValueError(
            "eval.discount_tolerance requires a discounted objective with gamma>0 "
            f"(got discounted={discounted}, gamma={gamma}); the truncation horizon "
            "T=(1/gamma)ln(1/eps) is undefined otherwise.")
    eps = float(eps)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eval.discount_tolerance must lie in (0, 1); got {eps}.")

    dt = float(env.step_size)
    T = (1.0 / gamma) * math.log(1.0 / eps)
    N = math.ceil(T / dt)
    tail_factor = eps / gamma  # tail <= c_max * tail_factor  (= (c_max/gamma) e^{-gamma T})

    training = cfg.agent.get("training", cfg.agent.get("training_params", None))
    old_max_time = float(training.get("max_time")) if training is not None else None
    old_T_sim = float(cfg.eval.get("T_sim")) if "T_sim" in cfg.eval else None
    training.max_time = T
    cfg.eval.T_sim = T

    print("=" * 60)
    print("[discount-truncation horizon] OPT-IN horizon ENABLED (results-changing)")
    print(f"  discount tolerance eps = {eps:.3e},  gamma = {gamma:g},  dt = {dt:g}")
    print(f"  horizon  T = (1/gamma) ln(1/eps) = {T:.6f}  (was max_time={old_max_time}, T_sim={old_T_sim})")
    print(f"  steps    N = ceil(T/dt) = {N}")
    print(f"  discounted-tail bound: J_tail <= (c_max/gamma) e^(-gamma T) = (c_max/gamma) eps "
          f"= c_max * {tail_factor:.6e}")
    print(f"           (c_max = sup of the cost rate x'Qx + u'Ru over the truncated tail)")
    print("=" * 60)


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
        Initialized agent (ContinuousTimeActorCritic, CTACJAX, CSAC, etc.)
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

def train(
    cfg: DictConfig,
    eval_callback: Optional[Callable[..., None]] = None,
) -> tuple[TrainableAgent, dict]:
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

    # Opt-in discount-truncation horizon (results-changing; OFF by default). Must run BEFORE
    # build_agent, which reads the (possibly overridden) max_time. A no-op when the knob is unset.
    apply_discount_truncation_horizon(cfg, env)

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
    
    # Set evaluation / fixed initial state from config. Conform to the env's state
    # dimension so a 2D default like [1, 1] does not break training on a 1D env
    # (e.g. Mackey-Glass), which uses agent.x0 as the fixed initial condition.
    if hasattr(cfg, 'eval') and 'x0_test' in cfg.eval:
        agent.x0 = jnp.array(conform_initial_state(np.array(cfg.eval.x0_test), agent.env.N))
        print(f"  Eval x0: {np.array(agent.x0)}")

    # Attach periodic evaluation callback if provided
    if eval_callback is not None:
        # ``eval_callback`` is declared as a method on the ``TrainableAgent``
        # Protocol; attaching it as an instance attribute is intentional and
        # supported at runtime, so the method-assign check is suppressed here.
        agent.eval_callback = eval_callback  # type: ignore[method-assign]
    
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
def main(cfg: DictConfig) -> tuple[TrainableAgent, dict]:
    """Standalone training entry point."""
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    agent, metrics = train(cfg)
    return agent, metrics


if __name__ == "__main__":
    main()
