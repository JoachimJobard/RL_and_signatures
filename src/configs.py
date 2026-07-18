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

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, cast


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
    clip_gradient: float | None = None  # None/<=0: no gradient clipping (default); else global-norm bound
    clip_action: float | None = None  # None/<=0: no action clipping (default); else bound |u|
    # Episode termination on a diverging state, as an OPT-IN bound on ||x||.
    # None/<=0 means NO trimming: the episode runs to its horizon whatever the state does.
    #
    # Default OFF, by the project owner's decision: "I don't want to trim trajectories I prefer
    # the algorithm to fail ... we rather need to understand why it diverges initially." A cut
    # episode's accumulated cost is not the cost of a completed one, so the bound edits the very
    # objective whose value is being reported; and a run that is rescued from divergence cannot be
    # diagnosed, because the divergence is what wants studying.
    #
    # Divergence is NOT silent when this is off. It surfaces two ways, both loud: a non-finite
    # state trips the NaN guard in _is_episode_done (which reports through
    # src/utils/clamp_reporting.py and is NOT a trim -- a trajectory containing NaN is already
    # mathematically over, and there is nothing left to integrate), and the reported closed-loop
    # cost is then the honest full-horizon value, however large, rather than a number truncated at
    # the moment the plant misbehaved.
    #
    # Runs predating this change carry `divergence_threshold: 100.0` (or 50.0) in their own saved
    # config.yaml and remain reproducible by re-running at that value.
    divergence_threshold: float | None = None
    eval_interval: int = 50
    eval_start_episode: int = 0
    patience: int = 0               # 0 = no early stopping
    log_interval: int = 50
    init_log_interval: int = 50
    memory_clear_interval: int = 20
    discretization_state: float = 0.01
    tau_polyak: float = 0.0          # Polyak averaging rate (0 = disabled, used by VG & CSAC)
    # Robbins-Monro learning-rate decay for the ACTOR optimiser: alpha_k = actor_lr / (1+k)^p.
    # 0 (default) is a constant rate, which fails the RM square-summability condition and converges
    # only to a noise ball. Admissible RM range is p in (1/2, 1]; the largest-step choice is
    # p -> 1/2^+. For the averaged actor / policy gradient the actor steps once per episode, so k is
    # the episode index (a per-episode schedule).
    lr_decay_power: float = 0.0
    # Robbins-Monro decay for the CRITIC optimiser: alpha_k = critic_lr / (1+k)^critic_lr_decay_power,
    # k the critic's own (per-step) update count. Robbins-Monro is classically the value-function
    # scheme (TD learning is stochastic approximation; Tsitsiklis, Sutton-Barto), so the critic is
    # the canonical object.
    #   - value gradient: the control IS the critic gradient, so the critic is the policy and this
    #     is single-timescale value RM.
    #   - actor-critic: TWO-TIMESCALE stochastic approximation (Borkar) -- BOTH critic and actor
    #     carry RM step sizes with the actor on the SLOWER timescale (alpha_actor/alpha_critic->0).
    #     Set lr_decay_power (actor) > critic_lr_decay_power (critic); combined with the actor's
    #     once-per-episode versus the critic's per-step cadence this makes the actor asymptotically
    #     slower. Caveat to watch empirically (via the SNR monitor): a decaying critic rate can
    #     stall tracking of the non-stationary value target if the critic is not fast enough.
    critic_lr_decay_power: float = 0.0


@dataclass
class DiscountConfig:
    """Doya continuous-time discounting: δ = r + V̇ − V/τ."""
    discounted: bool = False
    tau: float = 1.0                 # discount time constant
    V_target: float = 0.0
    V_bad: float = -1.0


@dataclass
class NoiseConfig:
    """Exploration noise configuration.

    Three temporal structures, selected by (ou, smooth):
      - ou=True             -> Ornstein-Uhlenbeck (Doya 2000): correlation exp(-dt/tau_n) over a
                               step, a function of physical time, dt-INDEPENDENT. The correct
                               continuous-time exploration; overrides smooth.
      - ou=False, smooth=True  -> squared-exponential Gaussian process, correlation
                               exp(-dt^2/2 length_scale^2) (dt-dependent, and white at small
                               length_scale). Retained for reproducing older runs.
      - ou=False, smooth=False -> i.i.d. white noise (no continuous-time limit; exploration
                               vanishes as dt -> 0).
    """
    sigma: float = 0.1
    schedule: str = "adaptive"       # 'constant', 'linear_decay', 'adaptive'
    decay: bool = True               # legacy flag for sigma decay
    smooth: bool = False             # use GP-sampled smooth noise
    length_scale: float = 0.2        # GP kernel length scale (only if ou=False, smooth=True)
    ou: bool = False                 # use Ornstein-Uhlenbeck noise (Doya 2000); overrides smooth
    tau_n: float = 1.0               # OU correlation time (Doya uses 1.0)


@dataclass
class SignatureConfig:
    """History-representation architecture parameters.

    ``kind`` selects the representation for the value-gradient agent:
    ``"signature"`` (depth-``depth`` signature), ``"raw_history"`` (degree-``degree``
    polynomial of the discretised window), or ``"markovian"`` (degree-``degree``
    polynomial of the current state). ``depth``/``degree`` are the capacity knobs
    swept in the H2 fairness sweep.
    """
    kind: str = "signature"
    degree: int = 2
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
    # Learning signal the ACTOR regresses the score function against.
    #   "td"          -- Doya (2000) Equation 20 / the Gullapalli stochastic real-valued rule:
    #                    the actor is updated every step against the instantaneous temporal-
    #                    difference error, wdot^A = eta^A delta(t) n(t) dA/dw. This is the
    #                    continuous actor-critic and is the default.
    #   "monte_carlo" -- REINFORCE with NO baseline: the actor is updated ONCE per episode
    #                    against the realised return-to-go R_t = int_t^T r ds, estimated by
    #                    R_t = sum_{k >= t} r_k dt, and A_t = R_t. The actor does NOT read the
    #                    critic at all, so its learning signal contains no bootstrap and no value
    #                    estimate. That is the point of the learner: an H1 verdict measured under
    #                    it is a statement about the REPRESENTATION rather than about the value
    #                    machinery. The cost is variance: R_t is not centred.
    # The two are estimators of the SAME policy gradient and differ in their bias/variance
    # trade-off: the temporal-difference signal is bootstrapped (biased whilst the critic is
    # wrong, low variance), the Monte-Carlo signal is unbiased and higher variance.
    actor_target: str = "td"
    # Actor update cadence for the TD target. False (default): Doya's ONLINE form -- the actor is
    # updated every actor_update_frequency steps from that step's single-sample gradient. True: the
    # per-step actor gradients delta_t (n_t/sigma^2) dA are ACCUMULATED over the episode and applied
    # as ONE averaged step, matching the Monte-Carlo policy gradient's once-per-episode cadence.
    #   Why the averaged form: with a constant learning rate the per-update noise-ball radius is
    #   proportional to the gradient variance; averaging over the T steps of an episode reduces that
    #   variance by up to 1/T, so the averaged step is far more stable than the single-sample online
    #   step (the online form at frequency=10 is the worst of both -- it neither averages nor sees
    #   every step). It also makes the actor-critic-versus-policy-gradient comparison a clean
    #   one-variable contrast: identical once-per-episode cadence, differing ONLY in the target
    #   (TD error delta_t versus realised return R_t). Doya (2000) is cited for the actor-gradient
    #   FORM delta*n*dA; batching it per episode is a variance/comparability choice, stated as such.
    # Ignored when actor_target is monte_carlo (which is inherently once-per-episode).
    actor_averaged: bool = False
    preheat: bool = True
    burning_steps: int = 0
    fix_initial_state: bool = False
    # CTACJAX-specific (Markov baseline with delayed state)
    delayed_state: bool = False
    whole_state_delay: bool = False
    # LSTD critic solver (value-gradient agent): direct least-squares solve of the
    # continuous-time TD fixed point instead of semi-gradient SGD (stable for high-dim
    # signature features; see conf/agent/value_gradient.yaml and value_gradient_jax.py).
    lstd: bool = False
    lstd_reg: float = 1.0e-3
    lstd_forget: float = 0.7
    lstd_rank: int = 0        # >0: project onto top-k PCs of the centred feature covariance
                              # before solving (truncated-SVD LSTD; cures rank-deficiency by
                              # discarding the null/noise directions). 0 = no truncation.


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
        clip_gradient=tp.get('clip_gradient', None),
        clip_action=tp.get('clip_action', None),
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
        ou=tp.get('ou', False),
        tau_n=tp.get('tau_n', 1.0),
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


def configs_to_flat_dict(*configs: Any) -> dict:
    """Merge all config objects into a single flat dict (for save/checkpoint compat)."""
    d: dict = {}
    for cfg in configs:
        if cfg is None:
            continue
        if is_dataclass(cfg):
            d.update(asdict(cast(Any, cfg)))
        elif isinstance(cfg, dict):
            d.update(cfg)
        else:
            d.update(vars(cfg))
    return d
