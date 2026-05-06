"""
Agent Registry - Simplified factory for CTAC.

All variants are now handled via flags in the single CTAC class.
No more subclasses or registry mapping needed.

Usage:
    from src.agents.registry import get_agent_class
    
    agent_cls = get_agent_class(cfg.agent)
    agent = agent_cls(
        env=env,
        discounted=cfg.agent.discounted,
        semi_gradient=cfg.agent.semi_gradient,
        actor_oracle=cfg.agent.actor_oracle,
        ...
    )
"""

from typing import Type
from omegaconf import DictConfig

from legacy_code.RL_algos_legacy.base import CTAC


def get_agent_class(agent_cfg: DictConfig) -> Type[CTAC]:
    """
    Get the agent class based on config.
    
    With the modular CTAC implementation, this always returns CTAC.
    Variants are controlled via flags passed to __init__.
    
    If _target_ is specified, use that directly (for legacy/custom agents).
    
    Args:
        agent_cfg: The agent config from Hydra
        
    Returns:
        The CTAC class (or custom class if _target_ specified)
    """
    if '_target_' in agent_cfg:
        import hydra
        return hydra.utils.get_class(agent_cfg._target_)
    
    return CTAC


def get_agent_flags(agent_cfg: DictConfig) -> dict:
    """
    Extract variant flags from config for CTAC instantiation.
    
    Args:
        agent_cfg: The agent config from Hydra
        
    Returns:
        Dict of flags to pass to CTAC.__init__
    """
    return {
        'discounted': agent_cfg.get('discounted', False),
        'semi_gradient': agent_cfg.get('semi_gradient', False),
        'integral_td': agent_cfg.get('integral_td', False),
        'actor_oracle': agent_cfg.get('actor_oracle', False),
        'critic_oracle': agent_cfg.get('critic_oracle', False),
    }


def list_available_variants() -> list[dict]:
    """List all available CTAC flag combinations."""
    flags = ['discounted', 'semi_gradient', 'integral_td', 'actor_oracle', 'critic_oracle']
    return [{'class': 'CTAC', 'flags': flags, 'note': 'All combinations supported via flags'}]
