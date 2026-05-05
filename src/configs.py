"""
Configuration dataclasses for CTAC agents.

Groups the ~30 scattered parameters into logical, documented units.
Used by CTACSignatureJAX and (eventually) all JAX-based agents.

The old interface (training_params dict + individual kwargs) is supported
via `from_legacy_params()` for backward compatibility with CTACJAX etc.

Hydra YAML layout (agent config):
    training:   { n_episodes, max_time, actor_lr, ... }
    discount:   { discounted, tau, V_target, V_bad }
    noise:      { sigma, schedule, decay, smooth, length_scale }
    signature:  { depth, window_size, time_augmentation, ... }
    network:    { std_init, normalize_entries }
    algorithm:  { semi_gradient, integral_td, preheat, ... }
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class TrainingConfig:
    """Optimization, episode, and logging parameters."""
    n_episodes: int = 1000
    max_time: float = 20.0
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    scale: float = 1.0
    clip_gradient: float = 10.0
    clip_action: float = 10.0
    divergence_threshold: float = 50.0
    eval_interval: int = 50
    eval_start_episode: int = 0
    patience: int = 0               # 0 = no early stopping
    log_interval: int = 50
    init_log_interval: int = 50
    memory_clear_interval: int = 20
    discretization_state: float = 0.01
    tau_polyak: float = 0.0          # Polyak averaging rate (0 = disabled, used by VG & CSAC)


@dataclass
class DiscountConfig:
    """Doya continuous-time discounting: δ = r + V̇ − V/τ."""
    discounted: bool = False
    tau: float = 1.0                 # discount time constant
    V_target: float = 0.0
    V_bad: float = -1.0


@dataclass
class NoiseConfig:
    """Exploration noise configuration."""
    sigma: float = 0.1
    schedule: str = "adaptive"       # 'constant', 'linear_decay', 'adaptive'
    decay: bool = True               # legacy flag for sigma decay
    smooth: bool = False             # use GP-sampled smooth noise
    length_scale: float = 0.2       # GP kernel length scale


@dataclass
class SignatureConfig:
    """Path signature architecture parameters."""
    depth: int = 2
    window_size: int = 10
    time_augmentation: bool = True
    origin_augmentation: bool = True
    state_augmentation: bool = False
    time_origin: float = 1.0
    bias: bool = False
    force_signature_window: bool = False  # if True, override window_size to match max delay (VG-specific)


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    std_init: float = 0.01          # initialization std (was std_network)
    normalize_entries: bool = False
    hidden_dims: tuple[int, ...] = ()  # hidden layer sizes (used by VG & CSAC critics)
    normalize_layers: bool = False     # use LayerNorm in network
    normalize_sigs: bool = False       # use LayerNorm on signature input


@dataclass
class ReplayBufferConfig:
    """Experience replay parameters (CSAC-specific)."""
    capacity: int = 100_000
    batch_size: int = 256
    n_updates_per_epoch: int = 128


@dataclass
class AlgorithmConfig:
    """Algorithm variant flags (Doya 2000 options + practical additions)."""
    semi_gradient: bool = False
    integral_td: bool = False
    actor_oracle: bool = False
    critic_oracle: bool = False
    actor_update_frequency: int = 1
    preheat: bool = True
    burning_steps: int = 0
    fix_initial_state: bool = False
    # CTACJAX-specific (Markov baseline with delayed state)
    delayed_state: bool = False
    whole_state_delay: bool = False


# =============================================================================
# Legacy conversion
# =============================================================================

def from_legacy_params(
    training_params: dict,
    *,
    depth: int = 2,
    discounted: bool = False,
    semi_gradient: bool = False,
    integral_td: bool = False,
    fix_initial_state: bool = False,
    decay_noise: bool = True,
    time_augmentation: bool = True,
    state_augmentation: bool = False,
    origin_augmentation: bool = True,
    time_origin: float = 1.0,
    bias: bool = True,
    actor_oracle: bool = False,
    critic_oracle: bool = False,
    preheat: bool = True,
    actor_update_frequency: int = 1,
    window_size: int = 10,
    smooth_noise: bool = False,
    noise_length_scale: float = 0.2,
    burning_steps: int = 0,
    delayed_state: bool = False,
    whole_state_delay: bool = False,
) -> tuple[TrainingConfig, DiscountConfig, NoiseConfig, SignatureConfig, NetworkConfig, AlgorithmConfig]:
    """Convert old-style ``training_params`` dict + kwargs → config dataclasses.

    This allows :class:`CTACJAX` (and other subclasses that still pass a dict)
    to work without modification.
    """
    tp = training_params  # alias for brevity

    training = TrainingConfig(
        n_episodes=tp.get('n_episodes', 1000),
        max_time=tp.get('max_time', 20.0),
        actor_lr=tp.get('actor_lr', 1e-3),
        critic_lr=tp.get('critic_lr', 1e-3),
        scale=tp.get('scale', 1.0),
        clip_gradient=tp.get('clip_gradient', 10.0),
        clip_action=tp.get('clip_action', 10.0),
        divergence_threshold=tp.get('divergence_threshold', 50.0),
        eval_interval=tp.get('eval_interval', 50),
        eval_start_episode=tp.get('eval_start_episode', 0),
        patience=tp.get('patience', 0),
        log_interval=tp.get('log_interval', 50),
        init_log_interval=tp.get('init_log_interval', 50),
        memory_clear_interval=tp.get('memory_clear_interval', 20),
        discretization_state=tp.get('discretization_state', 0.01),
        tau_polyak=tp.get('tau_polyak', 0.0),
    )

    discount = DiscountConfig(
        discounted=discounted,
        tau=tp.get('tau', 1.0),
        V_target=tp.get('V_target', 0.0),
        V_bad=tp.get('V_bad', -1.0),
    )

    noise = NoiseConfig(
        sigma=tp.get('sigma', 0.1),
        schedule=tp.get('noise_schedule', 'adaptive'),
        decay=decay_noise,
        smooth=smooth_noise,
        length_scale=noise_length_scale,
    )

    sig = SignatureConfig(
        depth=depth,
        window_size=window_size,
        time_augmentation=time_augmentation,
        origin_augmentation=origin_augmentation,
        state_augmentation=state_augmentation,
        time_origin=time_origin,
        bias=bias,
    )

    net = NetworkConfig(
        std_init=tp.get('std_network', 0.01),
        normalize_entries=tp.get('normalize_entries', False),
        hidden_dims=tuple(tp.get('hidden_dims', ())),
        normalize_layers=tp.get('normalize_layers', False),
        normalize_sigs=tp.get('normalize_sigs', False),
    )

    algo = AlgorithmConfig(
        semi_gradient=semi_gradient,
        integral_td=integral_td,
        actor_oracle=actor_oracle,
        critic_oracle=critic_oracle,
        actor_update_frequency=actor_update_frequency,
        preheat=preheat,
        burning_steps=burning_steps,
        fix_initial_state=fix_initial_state,
        delayed_state=delayed_state,
        whole_state_delay=whole_state_delay,
    )

    return training, discount, noise, sig, net, algo


def configs_to_flat_dict(*configs) -> dict:
    """Merge all config objects into a single flat dict (for save/checkpoint compat)."""
    d: dict = {}
    for cfg in configs:
        d.update(asdict(cfg))
    return d
