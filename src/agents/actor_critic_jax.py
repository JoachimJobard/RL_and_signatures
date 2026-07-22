"""
Continuous-Time Actor-Critic (CTAC) - Modular Implementation

"""

import jax
import numpy as np
import tqdm
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable
from src.utils.dynamic_signature import SlidingSignatureJAX, DequeBuffer
from src.representations.factory import make_representation, RepresentationBuffer
from src.utils.optim import build_optimizer
from src.networks.LQR_actor_critics import (
    ActorFlax, ActorFlaxLayerNorm, CriticFlax, CriticFlaxLayerNorm, CriticFlaxQuadratic,
)
from src.utils.step_metrics import StepMetrics
from src.utils.step_context import StepContextSignature
from src.utils.state_counter import StateCounter
from src.utils.clamp_reporting import report_clamp_activation
from src.utils.exploration_noise import sample_ou_trajectory
from src.configs import (
    TrainingConfig, DiscountConfig, NoiseConfig,
    SignatureConfig, NetworkConfig, AlgorithmConfig,
    configs_to_flat_dict,
)
import jax.numpy as jnp
import optax
import scipy


from src.envs.env_rk_jax import JAXDDEEnv, JAXEnvWrapper


class ContinuousTimeActorCritic:
  
    def __init__(
        self,
        env: JAXDDEEnv,
        training: TrainingConfig,
        discount: DiscountConfig,
        noise: NoiseConfig,
        signature_conf: SignatureConfig,
        network: NetworkConfig,
        algorithm: AlgorithmConfig,
        rng_key: int = 42,
        x0: jnp.ndarray | None = None,
        eval_callback: Callable[..., Any] | None = None,
    ):
        self.training = training
        self.discount = discount
        self.noise = noise
        self.signature_conf = signature_conf
        self.network = network
        self.algorithm = algorithm

        # =================================================================
        # Flat aliases — keeps the rest of the class unchanged
        # =================================================================

        # =================================================================
        # Environment
        # =================================================================
        self.env_params = {
            'A': np.array(env.A),
            'B': np.array(env.B),
            'Q': np.array(env.Q),
            'R': np.array(env.R),
            'A1': np.array(env.A1),
            'delay': np.array(env.delay) if env.delay is not None else None,
            'step_size': env.step_size,
            'resolution': env.resolution,
        }

        self.key = jax.random.PRNGKey(rng_key)
        self._init_env(env, rng_key)   # sets self.env / self.wrapper (needed by _init_oracle)
        self._init_oracle()
        self._init_episode_state(x0, eval_callback=eval_callback)
        self._init_checkpoints()
        self._init_signatures()
        self._init_networks()
        
        # JIT-compiled update functions (created once)
        self._jit_critic_update = self._make_critic_update_fn()
        self._jit_actor_update = self._make_actor_update_fn()
        self._jit_monte_carlo_actor_update = self._make_monte_carlo_actor_update_fn()
        self._jit_actor_snr = self._make_actor_snr_fn() if getattr(self.training, "monitor_snr", False) else None
        self._last_actor_snr = float("nan")
        # Rich per-episode actor diagnostics (built alongside the SNR when monitor_snr is on). The
        # keys are logged every episode so the BEGINNING of training is fully resolved -- an actor
        # poisoned by a cold critic in the first episodes never recovers, and a converged-state
        # glance would not reveal it.
        self._jit_actor_diag = self._make_actor_diag_fn() if getattr(self.training, "monitor_snr", False) else None
        self._last_actor_diag: dict[str, float] = {}
        self._jit_select_action = self._make_select_action_fn()
        self._jit_compute_values = self._make_compute_values_fn()
        # Validate algorithm.actor_target once, at construction, rather than on the hot path: an
        # unknown value must fail loudly at start-up, not silently fall through to the default.
        self._reset_monte_carlo_episode_buffer()
        # Cross-episode accumulators for rollouts_per_update > 1: the per-episode records are appended
        # here and one averaged actor step is taken per K episodes (see _batched_actor_step).
        self._batch_sig, self._batch_noise, self._batch_adv = [], [], []
        if self._actor_target_is_monte_carlo:
            print(
                "algorithm.actor_target='monte_carlo': the actor is REINFORCE with NO baseline "
                "(A_t = R_t, the realised return-to-go), updated ONCE per episode. The actor does "
                "not read the critic at all, so its learning signal contains no bootstrap and no "
                "value estimate."
            )

    def _init_env(self, env: JAXDDEEnv, rng_key: int) -> None:
        self.env = env
        self.wrapper = JAXEnvWrapper(env, rng_key=jax.random.PRNGKey(rng_key))
        self.episode = 0
        if self.network.normalize_entries:
            self.training.scale = self.training.divergence_threshold
            print("normalize_entries is True: scaling states by ", self.training.scale)
            print("normalize_sigs is ", self.network.normalize_sigs)

    def _init_oracle(self)-> None:
        """Build the analytic oracle for the actor_oracle / critic_oracle rungs.

        For a LINEAR DELAYED plant (JAXDDEEnv, tau>0) the oracle is the DELAYED-LQR:
        a history functional. The optimal control is u* = -K_aug @ xi and the optimal
        value V*(xi) = -xi' P_aug xi, where xi = [x(t), x(t-dt), ..., x(t-K dt)] is the
        discretised history window (newest first) and (K_aug, P_aug) come from the
        augmented discrete Riccati (src.solvers.delayed_lqr). The earlier code used the
        non-delayed continuous ARE (ignoring A1), which is NOT optimal on a delayed
        plant — kept only as the fallback for non-delayed envs.
        """
        self._delayed_oracle = False
        if not (self.algorithm.actor_oracle or self.algorithm.critic_oracle):
            return
        is_linear_delayed = (type(self.env).__name__ == "JAXDDEEnv"
                             and float(self.env.max_delay) > 0)
        # Nonlinear delayed plants (Mackey-Glass, ...) supply a Jacobian linearisation about their
        # equilibrium via linearised_delayed_matrices; the delayed-LQR built on it is a LINEAR
        # delayed reference (exact for the linearised plant, near-optimal near the equilibrium) and
        # replaces the meaningless non-delayed CARE that ignores the delay entirely.
        has_linearisation = (hasattr(self.env, "linearised_delayed_matrices")
                             and float(self.env.max_delay) > 0)
        if is_linear_delayed or has_linearisation:
            from src.solvers.oracle_agent import delayed_lqr_for_env
            _lin = " (linearised)" if has_linearisation and not is_linear_delayed else ""
            # Same-objective oracle: match the agent's continuous-time discount rate gamma via the
            # per-step factor beta = exp(-gamma dt). Undiscounted (beta=1) when the agent is
            # undiscounted, so the oracle is optimal for the SAME cost the agent minimises.
            beta = float(np.exp(-float(self.env.step_size) * self.discount.gamma)) if self.discount.discounted else 1.0
            self.lqr = delayed_lqr_for_env(self.env, discount_beta=beta)
            if self.discount.discounted:
                print(f"[oracle] discounted delayed-LQR{_lin}: gamma={self.discount.gamma}, beta={beta:.4f}")
            self.K_aug = jnp.array(self.lqr.gain)
            self.P_aug = jnp.array(self.lqr.P)
            self.k_taps = int(self.lqr.k_taps)
            self._delayed_oracle = True
            print(f"[oracle] delayed-LQR{_lin}: {self.k_taps} taps, "
                  f"closed-loop spectral radius {self.lqr.closed_loop_spectral_radius():.4f}")
        else:
            # Same-objective (discounted) non-delayed oracle: the discounted LQR is the ordinary LQR
            # of the shifted system A - (gamma/2) I, i.e. P_gamma = CARE(A - gamma/2 I,
            # B, Q, R), matching the agent's discounted cost. gamma=0 (undiscounted) leaves A unchanged.
            A_np = np.array(self.env.A); B_np = np.array(self.env.B)
            Q_np = np.array(self.env.Q); R_np = np.array(self.env.R)
            gamma_half = (0.5 * self.discount.gamma) if self.discount.discounted else 0.0
            A_shift = A_np - gamma_half * np.eye(A_np.shape[0])
            P = scipy.linalg.solve_continuous_are(A_shift, B_np, Q_np, R_np)
            self.P = jnp.array(P)
            self.optimal_K = jnp.array(np.linalg.inv(R_np) @ B_np.T @ P)
            print(f"[oracle] {'discounted' if self.discount.discounted else 'undiscounted'} non-delayed ARE"
                  + (f' (A - gamma/2 I, gamma/2={gamma_half:.4f})' if self.discount.discounted else '') + '.')

    def _delayed_oracle_window(self) -> jnp.ndarray:
        """Unscaled newest-first history window xi = [x(t), x(t-dt), ..., x(t-K dt)]
        flattened, reconstructed from the env buffer (subsampled to control cadence).
        Matches the (K+1)-tap layout the delayed-LQR gain/value expect."""
        buf = self.wrapper.state.buffer  # type: ignore
        data = np.asarray(buf.data)
        ptr = int(buf.ptr)
        ordered = np.roll(data, -ptr, axis=0)            # oldest .. newest
        newest_first = ordered[::-1]                      # newest .. oldest
        taps = newest_first[::self.env.resolution][:self.k_taps + 1]  # control cadence
        return jnp.asarray(taps).reshape(-1)
    
    def _init_episode_state(
        self, x0: jnp.ndarray | None, eval_callback: Callable[..., Any] | None = None
    ) -> None:
        self.step_counter = 0
        self.current_noise: jax.Array | None = None
        self.episode_noise_trajectory: jax.Array | None = None
        self.x0 = x0
        self.eval_callback = eval_callback
    
    def _init_checkpoints(self) -> None:
        self._best_eval_reward = -float('inf')
        self._best_actor_params = None
        self._best_critic_params = None
        self._best_episode = 0
        self._patience_counter = 0
        self._nan_detected = False
        self.discretization_state = self.training.discretization_state
        self.state_counter = StateCounter(resolution=self.training.discretization_state)

    def _init_signatures(self) -> None:
        max_delay = float(jnp.max(self.env.delay)) if self.env.delay is not None else 0
        window_size_from_delay = int(np.ceil(max_delay / self.env.step_size)) + 1 if max_delay > 0 else 10
        if not self.signature_conf.force_signature_window:
            self.signature_conf.window_size = window_size_from_delay
        else:
            print(f"force_signature_window is True: setting window_size to {self.signature_conf.window_size} to cover max delay of {max_delay}")
        # Representation backbone: signature / raw_history / markovian, selected by
        # ``signature_conf.kind`` and held behind a window buffer exposing the
        # SlidingSignatureJAX surface, so the rest of this class is unchanged. This
        # mirrors ContinuousValueGradient._init_networks: both agents consume the
        # SAME feature maps through the SAME factory, so an actor-critic-versus-
        # value-gradient comparison at fixed representation is not confounded by two
        # different implementations of the representation.
        window_length = self.signature_conf.window_size + 1
        representation = make_representation(
            self.signature_conf.kind,
            window_length=window_length,
            n_state=self.env.N,
            depth=self.signature_conf.depth,
            degree=self.signature_conf.degree,
            time_augmentation=self.signature_conf.time_augmentation,
            origin_augmentation=self.signature_conf.origin_augmentation,
            bias=self.signature_conf.bias,
        )
        # Actor feature map (agent.signature.actor_feature_map). 'raw' (default) strips the
        # polynomial LIFT from the actor's input, leaving the underlying representation the actor
        # network maps directly: the raw state (markovian) or the raw path (raw_history). That lift
        # is a value-function device for the linear-in-features critic; the linear control on a
        # linear plant is a linear functional of the path. The SIGNATURE is itself the representation
        # (linear functionals of it are universal), NOT a polynomial lift -- so the signature actor
        # takes the SIGNATURE (actor_representation left None => reuse the critic's signature map).
        # 'same': actor reuses the critic's feature map for every kind.
        actor_representation = None
        if str(getattr(self.signature_conf, "actor_feature_map", "raw")).lower() == "raw":
            if self.signature_conf.kind == "markovian":
                actor_representation = make_representation(
                    "markovian", window_length=window_length, n_state=self.env.N, degree=1)
            elif self.signature_conf.kind == "raw_history":
                actor_representation = make_representation(
                    "raw_history", window_length=window_length, n_state=self.env.N, degree=1)
            # kind == "signature": leave actor_representation = None -> the actor takes the signature.
        self.representation_buffer = RepresentationBuffer(
            representation, window_length=window_length, n_state=self.env.N,
            actor_representation=actor_representation,
        )
        self._sigma_effective: float | jax.Array = self.noise.sigma

    def _init_networks(self) -> None:
        # `critic` is widened to include CriticFlaxQuadratic so the state-based
        # subclass (CTACJAX in base_jax.py) can override it without a type error.
        self.actor: ActorFlax | ActorFlaxLayerNorm = self._build_actor()
        self.critic: CriticFlax | CriticFlaxLayerNorm | CriticFlaxQuadratic = self._build_critic()
        key_a, key_c = jax.random.split(self.key)
        if self.signature_conf.state_augmentation:
            self.actor_params = self.actor.init(key_a, jnp.zeros(self.representation_buffer.actor_feature_dim + self.env.N))
            self.critic_params = self.critic.init(key_c, jnp.zeros(self.representation_buffer.signature_size + self.env.N))
        else:
            self.actor_params = self.actor.init(key_a, jnp.zeros(self.representation_buffer.actor_feature_dim))
            self.critic_params = self.critic.init(key_c, jnp.zeros(self.representation_buffer.signature_size))

        #optimizers — absorb dt into learning rate for correct continuous-time scaling.
        # Gradient clipping is opt-in via training.clip_gradient (global-norm, off by default).
        # Robbins-Monro 1/(1+k)^p schedule on the ACTOR optimiser only. For the averaged actor
        # (algorithm.actor_averaged) and the policy gradient the actor steps once per episode, so
        # k is the episode index and this is a per-episode RM schedule; with the online actor it is
        # per-step. lr_decay_power = 0 (default) is a constant rate. The critic is left constant
        # (per-step decay would be far more aggressive; see robbins_monro_schedule).
        shared_opt = getattr(self.training, "optimizer", "adam")
        actor_opt = getattr(self.training, "actor_optimizer", None) or shared_opt
        critic_opt = getattr(self.training, "critic_optimizer", None) or shared_opt
        self.actor_optimizer = build_optimizer(
            actor_opt,
            self.training.actor_lr * self.env.step_size,
            clip_gradient=self.training.clip_gradient, b1=0.1,
            decay_power=float(getattr(self.training, "lr_decay_power", 0.0)))
        # Two-timescale (actor-critic): the critic also carries an RM schedule, on its per-step
        # count. lr_decay_power (actor) > critic_lr_decay_power (critic) plus the actor's
        # once-per-episode cadence puts the actor on the slower timescale.
        self.critic_optimizer = build_optimizer(
            critic_opt,
            self.training.critic_lr * self.env.step_size,
            clip_gradient=self.training.clip_gradient,
            decay_power=float(getattr(self.training, "critic_lr_decay_power", 0.0)))
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

    # =========================================================================
    # Network Construction
    # =========================================================================
    
    def _build_actor(self) -> ActorFlax | ActorFlaxLayerNorm:
        """Build the actor network."""
        output_dim = self.env.B.shape[1]
        actor: ActorFlax | ActorFlaxLayerNorm
        if self.network.normalize_sigs:
            actor = ActorFlaxLayerNorm(output_dim=output_dim, stddev=self.network.std_init/10)
        else:
            actor = ActorFlax(output_dim=output_dim, stddev=self.network.std_init/10)
        return actor

    def _build_critic(self) -> CriticFlax | CriticFlaxLayerNorm:
        """Build the critic network."""
        critic: CriticFlax | CriticFlaxLayerNorm
        if self.network.normalize_sigs:
            critic = CriticFlaxLayerNorm(stddev=self.network.std_init)
        else:
            critic = CriticFlax(stddev=self.network.std_init)
        return critic

    # =========================================================================
    # Episode Initialization
    # =========================================================================
    
    def _sample_initial_state(self) -> jnp.ndarray:
        """Sample initial state for an episode."""
        self.key, subkey = jax.random.split(self.key) 
        return jax.random.normal(subkey, shape=(self.env.N,))
    
    def _fill_buffer_initial(self):
        """Load the initial path phi of the delay differential equation into the window.

        The initial condition of a delay differential equation is a PATH
        phi in C([-tau, 0], R^N), not a point; the environment holds its discretisation
        (``wrapper.initial_conditions``), and it is loaded here directly, subsampled to the
        control cadence.

        The window is filled to FULL capacity (window_size + 1). A partial fill would leave
        the reset's zero prefill in the oldest slots, so the window would open on a spurious
        jump from the origin to phi — a path the plant never followed, and one whose signature
        is dominated by that jump. When the subsampled history is shorter than the capacity it
        is FRONT-padded with its earliest element, that is, by the constant extension of phi to
        the left of its own support: the only extension derivable from the environment, and the
        one ContinuousValueGradient._fill_buffer_initial applies, so both agents open every
        episode on the same padding SEMANTICS — their windows agreeing up to the float32 residual
        quantified below — and a comparison between them at fixed representation is not confounded
        by the initial condition. The constant extension IS phi wherever phi is itself constant,
        which is measured to hold on all five campaign cells; on an environment supplying a
        non-constant history_function it would fabricate an input, and must be revisited before
        any such cell enters a study.

        Elements are appended through ``RepresentationBuffer.append``, which casts to the
        buffer's float64 dtype. ContinuousValueGradient instead casts to float32 while writing
        past ``append`` into the raw deque; that cast is a known residual of code-review finding
        F-E1 (commits 07b8d4a / 7b1153e moved the buffers to float64 and missed this path) and
        is NOT mirrored here: the rounding is irreversible, ``DequeBuffer.to_array`` merely
        upcasting the already-rounded values back to float64. The cast is live in production,
        where main_unified.py:63 enables jax_enable_x64 and the env history is therefore float64;
        it is inert only when x64 is off, the history then already being float32. The residual is
        therefore the binary32 representation error of the initial state: it is 0 exactly when
        every entry of x0 is representable in binary32, and the rounding error otherwise. Measured
        max|actor-critic - value-gradient| over the loaded window, x64 enabled, on all five
        campaign cells: 0 on markovian (x0 = [1.0, 0.5]) and platoon (entries 0.0 and 0.5), and
        1.192093e-08 on linear_dde, mg_limit_cycle and mg_chaotic (x0 = [0.8], not representable).
        Rounding the actor-critic window to float32 drives the residual to 0 on all five, which
        identifies this cast as its sole cause. Removing it from value_gradient_jax.py is the
        follow-up that makes the two agents load phi bit-identically; it is deferred because the
        existing five-seed value-gradient results must stay bit-reproducible.
        """
        self.representation_buffer.reset()
        if self.wrapper.state is None:
            return
        _, history_data = self.wrapper.initial_conditions
        subsampled = history_data[::self.env.resolution]
        target_len = self.representation_buffer.window_size + 1
        if len(subsampled) >= target_len:
            initial_path = list(subsampled[-target_len:])
        else:
            earliest = subsampled[0]
            initial_path = [earliest] * (target_len - len(subsampled)) + list(subsampled)
        for state_on_initial_path in initial_path:
            self.representation_buffer.append(state_on_initial_path / self.training.scale)
        # Assigned eagerly rather than left to the lazy dirty-flag path so that the agent's
        # internal state immediately after the fill — window AND cached feature — is identical
        # to ContinuousValueGradient's at the same point.
        self.representation_buffer.current_signature = self.representation_buffer.compute_signature()

    def _zero_control_burn_in_steps(self) -> int:
        """Number of zero-control steps taken after the reset, before the first controlled step."""
        steps = int(self.algorithm.burning_steps)
        if self.algorithm.preheat:
            steps += int(self.representation_buffer.window_size)
        return steps

    def _announce_zero_control_burn_in(self) -> None:
        """Announce, once per agent, that a zero-control burn-in is active.

        The burn-in is retained for the studies that deliberately define their control problem
        to start from the uncontrolled continuation of phi rather than from phi itself: the
        Mackey-Glass configurations drive the plant onto its attractor with burning_steps=100
        before control begins (conf/agent/CTAC_sig_MG_1D.yaml:73,
        conf/agent/CTAC_jax_mackey_glass.yaml:82), and conf/agent/CTAC_sig_chemical.yaml:72 uses
        5. It is NOT the initial condition of the delay
        differential equation: _fill_buffer_initial already loads phi, and the burn-in overwrites
        it with an uncontrolled-evolution path. It also shortens the episode, because the clock
        is not re-based across the burn-in while _is_episode_done terminates on
        wrapper.state.t >= training.max_time.

        Both consequences are invisible in the metrics, so the path announces itself rather than
        altering the object under study in silence.
        """
        if getattr(self, "_zero_control_burn_in_announced", False):
            return
        self._zero_control_burn_in_announced = True
        steps = self._zero_control_burn_in_steps()
        if steps == 0:
            return
        warnings.warn(
            f"Zero-control burn-in active (preheat={self.algorithm.preheat}, "
            f"burning_steps={self.algorithm.burning_steps}, window_size="
            f"{self.representation_buffer.window_size}): {steps} zero-control steps run after each "
            f"reset. The initial path phi loaded by _fill_buffer_initial is overwritten by the "
            f"uncontrolled evolution, so the episode does not start from the delay differential "
            f"equation's initial condition; and the first controlled step occurs at "
            f"t={steps * self.env.step_size:.4g} of training.max_time={self.training.max_time:.4g}, "
            f"the burn-in being charged to the episode horizon. Set preheat=false and "
            f"burning_steps=0 to control from phi at t=0.",
            RuntimeWarning,
            stacklevel=2,
        )

    # =========================================================================
    # Action Selection
    # =========================================================================

    def _make_select_action_fn(self):
        """Create JIT-compiled action selection function."""
        actor = self.actor
        clip_action = self.training.clip_action
        # Action clipping is opt-in, mirroring ContinuousValueGradient._make_action_jit_fn:
        # clip_action None/<=0 means NO clipping. The default is null so that this agent and
        # the value gradient apply the SAME rule to the control; the previous default (10.0)
        # was applied here and nowhere in the value gradient, and it bound on 152/2973 =
        # 5.113% of steps on the raw_history arm of the markovian double-integrator control cell
        # (config since removed; pre-clip |u| = 2014.44) whilst binding on 0/3020 of its
        # {markovian,signature} arms. It therefore acted on
        # ONE ARM of the H1 contrast (raw_history against markovian) on the falsification
        # control cell, where the raw-history window of a Markovian plant is collinear, its
        # feature Gram near-singular and its greedy control consequently unbounded -- the
        # regime under study. A clamp that fires there masks that regime.
        do_clip = clip_action is not None and clip_action > 0
        # Both OU and GP pre-sample the whole episode's noise, so both feed explicit_noise_val.
        smooth_noise = self.noise.smooth or getattr(self.noise, "ou", False)  # Capture static config
        @jax.jit
        def select_action_fn(actor_params, sig, key, sigma, noise_state, dt, explicit_noise_val):
            """
            Unified signature for action selection.
            noise_state: Previous noise value (unused if explicit_noise_val provided)
            dt: Time step (float)
            explicit_noise_val: Pre-computed noise value for GP mode (or None/zeros if unused)
            """
            mu = actor.apply(actor_params, sig)
            key, subkey = jax.random.split(key)
            
            if smooth_noise:
                 noise = explicit_noise_val
            else:
                noise = sigma * jax.random.normal(subkey, shape=mu.shape) # type: ignore

            action = mu + noise
            if do_clip:
                action = jnp.clip(action, -clip_action, clip_action)

            return action, mu, noise, key, noise
        
        return select_action_fn
    
    def _make_compute_values_fn(self):
        """Create JIT-compiled value computation function."""
        critic = self.critic
        
        @jax.jit
        def compute_values_fn(critic_params, features_t, features_next):
            V_t = critic.apply(critic_params, features_t).squeeze() # type: ignore
            V_next = critic.apply(critic_params, features_next).squeeze() # type: ignore
            return V_t, V_next
        
        return compute_values_fn
    
    def _compute_value_function(self, sig):
        return self.critic.apply(self.critic_params, sig).squeeze() # type: ignore

    def _select_action(self, state: jnp.ndarray, dt: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Select action with exploration noise."""
        
        # --- Compute effective noise level based on schedule ---
        if self.noise.schedule == 'adaptive':
            V_t = self._compute_value_function(self.representation_buffer.current_signature)
            V_TARGET = self.discount.V_target
            V_BAD = self.discount.V_bad
            noise_scale = jnp.clip((V_TARGET - V_t) / (V_TARGET - V_BAD + 1e-6), 0.1, 1.0)
            self._sigma_effective = self.noise.sigma * noise_scale
        elif self.noise.schedule == 'linear_decay':
            progress = min(self.episode / max(self.training.n_episodes, 1), 1.0)
            self._sigma_effective = self.noise.sigma * max(0.05, 1.0 - 0.9 * progress)
        else:
            # 'constant'
            self._sigma_effective = self.noise.sigma
        if self.algorithm.actor_oracle:
            if getattr(self, "_delayed_oracle", False):
                mu = -self.K_aug @ self._oracle_xi_t   # delayed-LQR control on the window
            else:
                mu = - self.optimal_K @ jnp.array(self.wrapper.state.x) #type: ignore
            self.key, subkey = jax.random.split(self.key)
            noise = jnp.zeros_like(mu)
            action = mu + noise
            self.current_noise = jnp.zeros_like(noise)
        else:
            if self.current_noise is None:
                 self.current_noise = jnp.zeros(self.env.B.shape[1])

            # Prepare explicit noise value if GP mode is active
            assert self.current_noise is not None  # guaranteed by the block above
            explicit_noise_val = jnp.zeros_like(self.current_noise)
            if (self.noise.smooth or getattr(self.noise, "ou", False)) and self.episode_noise_trajectory is not None:
                # Find index corresponding to current time
                t_idx = int(round(self.wrapper.state.t / self.env.step_size)) #type: ignore
                # Clamp to avoid overflow
                t_idx = min(t_idx, len(self.episode_noise_trajectory) - 1)
                explicit_noise_val = self.episode_noise_trajectory[t_idx]

            action, mu, noise, self.key, self.current_noise = self._jit_select_action(
                self.actor_params, 
                state,
                self.key,
                self._sigma_effective,
                self.current_noise,
                dt,
                explicit_noise_val
            )
        return action, mu, noise
 

    # =========================================================================
    # Episode Termination
    # =========================================================================
    
    def _is_episode_done(self, x: jnp.ndarray, time_only: bool = False) -> bool:
        """Check if episode should terminate.

        Uses jax.device_get once to fetch both t and the norm check in a
        single GPU→CPU synchronisation instead of multiple float() calls.
        """
        if self.wrapper.state is None:
            return False
        if time_only:
            return bool(self.wrapper.state.t >= self.training.max_time)
        # Horizon test, finiteness and norm are fetched in a SINGLE device_get, preserving the
        # one-synchronisation property of the previous jnp.logical_or form.
        horizon_reached, all_finite, state_norm = jax.device_get(
            (self.wrapper.state.t >= self.training.max_time,
             jnp.all(jnp.isfinite(x)),
             jnp.linalg.norm(x))
        )
        # NON-FINITE STATE. A FAILURE GUARD, not a trim: a trajectory containing a NaN or an
        # infinity is already mathematically over. Always fires, never opt-out, announces itself.
        #
        # This agent previously had NO finiteness guard at all, whilst ContinuousValueGradient did
        # (value_gradient_jax.py). The omission was not benign: `nan > threshold` evaluates to
        # False, so a NaN state never tripped the divergence bound either and the episode simply
        # ran on. That is the measured mechanism by which this agent's evaluation on
        # MG_1D_chaotic/raw_history reached ||x|| = 3.37e28 and returned nan, whilst the value
        # gradient on the identical cell, representation and seed stayed finite.
        if not bool(all_finite):
            n_nan = int(jnp.sum(jnp.isnan(x)))
            n_inf = int(jnp.sum(jnp.isinf(x)))
            report_clamp_activation(
                "nonfinite_state_termination/actor_critic",
                code_location="src/agents/actor_critic_jax.py:_is_episode_done",
                bound_description="the state must be finite (no NaN, no inf component)",
                most_extreme_raw_value=float("nan") if n_nan else float("inf"),
                number_of_affected_elements=n_nan + n_inf,
                additional_context=(
                    f"at t={float(self.wrapper.state.t):.4f}: {n_nan} NaN and {n_inf} infinite "
                    f"component(s). The plant has diverged; this is the failure, not a "
                    f"truncation. Every number this episode reports downstream is meaningless."
                ),
                alters_the_value=False,  # a detector, not an intervention
            )
            return True
        # DIVERGENCE BOUND -- OPT-IN, off by default. See the note in configs.DiscountConfig's
        # sibling TrainingConfig.divergence_threshold: a cut episode's cost is not a completed
        # episode's cost, so the bound edits the objective it is meant to measure.
        bound = self.training.divergence_threshold
        if bound is not None and bound > 0 and float(state_norm) > bound:
            report_clamp_activation(
                "divergence_threshold/actor_critic",
                code_location="src/agents/actor_critic_jax.py:_is_episode_done",
                bound_description=f"||x|| <= {bound}",
                most_extreme_raw_value=float(state_norm),
                additional_context=(
                    f"at t={float(self.wrapper.state.t):.4f}; the episode is CUT here, so its "
                    f"accumulated cost covers less than the full horizon "
                    f"max_time={self.training.max_time} and is not comparable with a completed "
                    f"episode's."
                ),
            )
            return True
        return bool(horizon_reached)

    # =========================================================================
    # Single Training Step (orchestration)
    # =========================================================================
    def _train_step(self, x_t: jnp.ndarray, ) -> tuple[jnp.ndarray, StepMetrics, StepContextSignature]:
        """Execute a single training step.
        
        This method orchestrates the training step by calling
        the overridable methods in sequence.
        
        Args:
            x_t: Current state
            
        Returns:
            (x_next, metrics, context): Next state, step metrics, and full context
        """
        x_scaled = x_t / self.training.scale
        # Delayed oracle: capture the history window at x(t) BEFORE the env step
        # (buffer newest = x(t)); reused by the oracle-actor and the oracle-critic V_t.
        if getattr(self, "_delayed_oracle", False):
            self._oracle_xi_t = self._delayed_oracle_window()
        if self.signature_conf.state_augmentation:
            state = jnp.concatenate([self.representation_buffer.current_actor_features, x_scaled])  # type: ignore
        else:
            state = self.representation_buffer.current_actor_features
        # Capture the actor's features at t (before the env-step append) for the once-per-episode
        # actor recording and the online-TD update, mirroring self._oracle_xi_t.
        self._actor_features_t = state
        # Action selection
        dt = self.env.step_size
        action, mu, noise = self._select_action(state, dt)
        
        # Environment step (using wrapper with current state)
        t, x_next, reward = self.wrapper.step(self.wrapper.state, action) # type: ignore
        self.state_counter.add(x_next)  # Count state visit
        if x_next.ndim == 0:
            x_next = jnp.array([x_next])
        else:
            x_next = jnp.array(x_next)
        x_next_scaled = x_next / self.training.scale
        dt = self.env.step_size
        if dt <= 1e-9:
            dt = 1e-4  # Avoid division by zero
        features_t = self.representation_buffer.current_signature
        self.representation_buffer.append(x_next_scaled)
        features_next = self.representation_buffer.current_signature
        if self.signature_conf.state_augmentation:
            features_t = jnp.concatenate([features_t, x_scaled])  # type: ignore
            features_next = jnp.concatenate([features_next, x_next_scaled])  # type: ignore
        # Value function evaluations (JIT-compiled, no float() sync)
        if self.algorithm.critic_oracle:
            # Value = expected discounted reward (reward = -(x'Qx + u'Ru)), so the LQR
            # oracle value is the *negative* quadratic form. For the delayed-LQR it is a
            # functional of the history window, V*(xi) = -xi' P_aug xi (xi_t pre-step,
            # xi_next post-step); for the non-delayed fallback, -x' P x.
            if getattr(self, "_delayed_oracle", False):
                xi_next = self._delayed_oracle_window()  # buffer now at x_next
                V_t = -self._oracle_xi_t @ self.P_aug @ self._oracle_xi_t
                V_next = -xi_next @ self.P_aug @ xi_next
            else:
                V_t = -x_t.T @ self.P @ x_t
                V_next = -x_next.T @ self.P @ x_next
        else:
            V_t, V_next = self._jit_compute_values(self.critic_params, features_t, features_next)
        
        # Build context
        ctx = StepContextSignature(
            x_t=x_t,
            x_scaled=x_scaled,
            x_next=x_next,
            x_next_scaled=x_next_scaled,
            mu=mu,
            noise=noise,
            action=action,
            dt=dt,
            time_series=t, # type: ignore
            V_t=V_t,# type: ignore
            V_next=V_next,# type: ignore
            features_t=features_t,
            features_next=features_next
        )
        
        # Compute reward
        ctx.reward = reward #type: ignore
        
        # Compute V dot and TD error (keep as JAX arrays)
        ctx.V_dot = (ctx.V_next - ctx.V_t) / ctx.dt
        ctx.td_error = ctx.reward + ctx.V_dot - (ctx.V_t * self.discount.gamma if self.discount.discounted else 0.0)
        
        # update networks (returns JAX arrays, no float() sync)
        loss_critic, actor_grad_norm, critic_grad_norm = self._update_networks(ctx)

        # Keep as JAX arrays - avoid float() sync on hot path
        metrics = StepMetrics(
            loss=loss_critic * dt, #type: ignore
            reward=ctx.reward * dt,
            actor_gradient=actor_grad_norm, # type: ignore
            critic_gradient=critic_grad_norm, # type: ignore
        )
        
        return x_next, metrics, ctx
    
    # =========================================================================
    # Network updates (JAX autodiff - modulaire)
    # =========================================================================

    def _make_critic_update_fn(self):
        """Create a JIT-compiled critic update function."""
        critic = self.critic
        optimizer = self.critic_optimizer
        discounted = self.discount.discounted
        gamma = self.discount.gamma

        def critic_loss(critic_params, features_t, features_next, reward, dt):
            V_t = critic.apply(critic_params, features_t).squeeze() # type: ignore
            V_next = critic.apply(jax.lax.stop_gradient(critic_params), features_next).squeeze() # type: ignore
            td_error = reward + (V_next - V_t) / dt
            if discounted:
                td_error = td_error - V_t * gamma
            return 0.5 * td_error ** 2 * dt, td_error
        
        @jax.jit
        def update_fn(critic_params, opt_state, features_t, features_next, reward, dt):
            (loss, td_error), grads = jax.value_and_grad(critic_loss, has_aux=True)(
                critic_params, features_t, features_next, reward, dt
            )
            # Gradient clipping (if enabled) is handled by the optimizer (global-norm).
            updates, new_opt_state = optimizer.update(grads, opt_state, critic_params)
            
            new_params = optax.apply_updates(critic_params, updates)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            return new_params, new_opt_state, loss, td_error, grad_norm
        
        return update_fn
    
    def _make_actor_update_fn(self):
        """Create a JIT-compiled actor update function."""
        actor = self.actor
        optimizer = self.actor_optimizer
        
        def actor_loss(actor_params, features_t, noise, td_error, sigma):
            mu = actor.apply(actor_params, features_t)
            log_prob_grad = noise / (sigma ** 2 + 1e-8)
            return -td_error * jnp.dot(log_prob_grad, mu) # type: ignore
        
        @jax.jit
        def update_fn(actor_params, opt_state, features_t, noise, td_error, sigma, dt):
            grads = jax.grad(actor_loss)(actor_params, features_t, noise, td_error, sigma)
            # Gradient clipping (if enabled) is handled by the optimizer (global-norm).
            updates, new_opt_state = optimizer.update(grads, opt_state, actor_params)

            new_params = optax.apply_updates(actor_params, updates)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            return new_params, new_opt_state, grad_norm

        return update_fn

    # =========================================================================
    # Monte-Carlo policy gradient (REINFORCE, no baseline: A_t = R_t)
    # =========================================================================

    @property
    def _actor_target_is_monte_carlo(self) -> bool:
        """Whether the actor regresses against the realised return rather than the TD error."""
        target = str(getattr(self.algorithm, "actor_target", "td")).lower()
        if target not in ("td", "monte_carlo"):
            raise ValueError(
                f"algorithm.actor_target must be 'td' or 'monte_carlo', got {target!r}. "
                f"'td' is Doya (2000) Equation 20; 'monte_carlo' is REINFORCE with a value "
                f"baseline."
            )
        return target == "monte_carlo"

    @property
    def _actor_updates_once_per_episode(self) -> bool:
        """True when the actor takes a single averaged step per episode rather than online steps.

        Both the Monte-Carlo policy gradient (inherently) and the averaged TD actor-critic
        (algorithm.actor_averaged) update once per episode; they differ only in the signal recorded
        per step -- the realised reward (whence the return R_t) versus the TD error delta_t.
        """
        return self._actor_target_is_monte_carlo or bool(getattr(self.algorithm, "actor_averaged", False))

    def _make_monte_carlo_actor_update_fn(self):
        """Create a JIT-compiled REINFORCE actor update over one whole episode.

        This mirrors ``_make_actor_update_fn`` term for term. The ONLY differences are the signal
        the score function is regressed against -- the realised return A_t = R_t rather than
        the instantaneous temporal-difference error delta_t -- and the fact that a single
        optimiser step is taken per EPISODE over the mean of the per-step losses, which is the
        standard REINFORCE estimator. Keeping the loss algebraically identical is deliberate: the
        two estimators then differ only in their target, so a difference in behaviour is
        attributable to the bootstrap and not to an incidental discrepancy between two
        hand-written losses.

        The policy is Gaussian with mean mu(s; w) and standard deviation sigma, so the score with
        respect to the mean is grad_mu log N(u; mu, sigma^2) = (u - mu)/sigma^2 = noise/sigma^2,
        which is the ``log_prob_grad`` term shared with the temporal-difference form.
        """
        actor = self.actor
        optimizer = self.actor_optimizer
        # DIAGNOSTIC: PG_BASELINE=mean subtracts the batch-mean return as a CONSTANT baseline
        # (variance reduction, still REINFORCE) to test whether PG's ceiling is the no-baseline
        # variance floor. Default 'none' = the shipped no-baseline REINFORCE (A_t = R_t).
        import os as _os
        _use_mean_baseline = _os.environ.get("PG_BASELINE", "none").lower() == "mean"

        def actor_loss(actor_params, features_batch, noise_batch, advantage_batch, sigma):
            # vmap rather than relying on the network broadcasting over a leading batch axis, so
            # the estimator is correct for any actor architecture.
            adv = advantage_batch - jnp.mean(advantage_batch) if _use_mean_baseline else advantage_batch
            mu = jax.vmap(lambda s: actor.apply(actor_params, s))(features_batch)       # (T, m)
            log_prob_grad = noise_batch / (sigma ** 2 + 1e-8)                      # (T, m)
            per_step = -adv * jnp.sum(log_prob_grad * mu, axis=-1)                 # (T,)
            return jnp.mean(per_step)

        @jax.jit
        def update_fn(actor_params, opt_state, features_batch, noise_batch, advantage_batch, sigma):
            grads = jax.grad(actor_loss)(
                actor_params, features_batch, noise_batch, advantage_batch, sigma)
            updates, new_opt_state = optimizer.update(grads, opt_state, actor_params)
            new_params = optax.apply_updates(actor_params, updates)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            return new_params, new_opt_state, grad_norm

        return update_fn

    def _make_actor_snr_fn(self):
        """JIT-compiled actor-gradient signal-to-noise ratio over one episode's recorded steps.

        The once-per-episode update takes jax.grad of the MEAN loss, giving only E[g]. The SNR
        needs the PER-STEP gradients g_t = A_t (n_t/sigma^2) dA_t(w), computed here by vmapping
        jax.grad of the single-step loss, then

            SNR = ||E[g]||^2 / Var[g],   Var[g] = mean_t ||g_t - E[g]||^2  (per-sample noise power).

        A signal-to-noise POWER ratio: SNR > 1 means the averaged gradient's direction carries more
        power than the per-sample fluctuation, i.e. the actor is learning rather than wandering. The
        same estimate for the online TD actor is not produced (it does not accumulate an episode).
        """
        actor = self.actor

        def per_step_loss(actor_params, s, n, a, sigma):
            mu = actor.apply(actor_params, s)
            return -a * jnp.sum((n / (sigma ** 2 + 1e-8)) * mu)

        @jax.jit
        def snr_fn(actor_params, features_batch, noise_batch, advantage_batch, sigma):
            per_step_grad = jax.vmap(
                lambda s, n, a: jax.grad(per_step_loss)(actor_params, s, n, a, sigma)
            )(features_batch, noise_batch, advantage_batch)
            flat = jax.vmap(lambda g: jnp.concatenate(
                [jnp.ravel(x) for x in jax.tree_util.tree_leaves(g)]))(per_step_grad)   # (T, P)
            mean = jnp.mean(flat, axis=0)
            signal = jnp.sum(mean ** 2)
            noise = jnp.mean(jnp.sum((flat - mean) ** 2, axis=1))
            return signal / (noise + 1e-12)

        return snr_fn

    def _make_actor_diag_fn(self):
        """Comprehensive per-episode actor-signal diagnostics for the once-per-episode actors.

        Returns a JIT function of one episode's recorded (sig, noise, advantage) that yields the
        seven scalars needed to diagnose a weak actor signal -- the object of the markovian-cell
        focus. Beyond the per-step SNR (which the SNR function already reports), the decisive one is
        the AVERAGED-UPDATE SNR: the once-per-episode update applies E[g] = mean_t g_t, whose
        precision is not the per-step SNR but

            SNR_update = N_eff * SNR_per-step,

        with N_eff the effective number of INDEPENDENT steps -- reduced below the step count T by the
        temporal correlation the OU exploration deliberately injects. N_eff is estimated from the
        lag-1 autocorrelation rho of the gradient projected on its own mean direction, via the AR(1)
        relation N_eff = T (1 - rho)/(1 + rho). The advantage-noise coupling corr(A_t, n_t) is the
        SOURCE of the signal (E[g] is non-zero only through it); a near-zero coupling means the
        exploration is not producing a measurable advantage response, which is an exploration
        problem, not an averaging one. adv_mean / adv_std separate a biased advantage (PG's
        uncentred return) from a high-variance one; grad_signal is ||E[g]||.
        """
        actor = self.actor

        def per_step_loss(actor_params, s, n, a, sigma):
            mu = actor.apply(actor_params, s)
            return -a * jnp.sum((n / (sigma ** 2 + 1e-8)) * mu)

        @jax.jit
        def diag_fn(actor_params, features_batch, noise_batch, advantage_batch, sigma):
            per_step_grad = jax.vmap(
                lambda s, n, a: jax.grad(per_step_loss)(actor_params, s, n, a, sigma)
            )(features_batch, noise_batch, advantage_batch)
            flat = jax.vmap(lambda g: jnp.concatenate(
                [jnp.ravel(x) for x in jax.tree_util.tree_leaves(g)]))(per_step_grad)   # (T, P)
            T = flat.shape[0]
            mean = jnp.mean(flat, axis=0)
            signal = jnp.sum(mean ** 2)
            noise = jnp.mean(jnp.sum((flat - mean) ** 2, axis=1))
            perstep_snr = signal / (noise + 1e-12)
            # N_eff from the AR(1) lag-1 autocorrelation of the gradient projected on its mean dir.
            ghat = mean / (jnp.sqrt(signal) + 1e-12)
            proj = flat @ ghat
            pc = proj - jnp.mean(proj)
            denom = jnp.sum(pc * pc) + 1e-12
            rho = jnp.clip(jnp.sum(pc[1:] * pc[:-1]) / denom, 0.0, 0.999)
            n_eff = T * (1.0 - rho) / (1.0 + rho)
            update_snr = n_eff * perstep_snr
            # advantage-noise coupling corr(A_t, n_t) on the first action component.
            a_ = advantage_batch.reshape(-1)
            n0 = noise_batch.reshape(noise_batch.shape[0], -1)[:, 0]
            ac_, nc_ = a_ - jnp.mean(a_), n0 - jnp.mean(n0)
            coupling = jnp.sum(ac_ * nc_) / (jnp.sqrt(jnp.sum(ac_ ** 2) * jnp.sum(nc_ ** 2)) + 1e-12)
            return (update_snr, perstep_snr, n_eff, coupling,
                    jnp.mean(a_), jnp.std(a_), jnp.sqrt(signal))

        return diag_fn

    def _record_actor_diag(self, features_batch, noise_batch, advantage) -> None:
        """Run the diagnostic function on one episode's records and cache the scalars.

        Sets ``_last_actor_snr`` (per-step, for backward compatibility with the SNR monitor) and
        fills ``_last_actor_diag`` with the seven diagnostics; a no-op when monitoring is off."""
        if self._jit_actor_diag is None:
            return
        vals = self._jit_actor_diag(
            self.actor_params, features_batch, noise_batch, advantage, self._sigma_effective)
        update_snr, perstep_snr, n_eff, coupling, adv_mean, adv_std, grad_signal = (float(v) for v in vals)
        self._last_actor_snr = perstep_snr
        self._last_actor_diag = {
            "actor_update_snr": update_snr,
            "actor_perstep_snr": perstep_snr,
            "actor_n_eff": n_eff,
            "actor_coupling": coupling,
            "actor_adv_mean": adv_mean,
            "actor_adv_std": adv_std,
            "actor_grad_signal": grad_signal,
        }

    def monte_carlo_returns(self, reward_rates: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Return-to-go R_t of an episode, from the per-step reward RATES.

        The environment's reward is a RATE, r(t) = -(x'Qx + u'Ru); the objective is its integral
        over the episode, which the rest of the code forms as ``reward * step_size``. The
        return-to-go is therefore the right-hand Riemann sum

            R_t = int_t^T r(s) ds  ~=  sum_{k >= t} r_k dt,

        and, when ``discount.discounted`` is set, its exponentially weighted analogue
        R_t = int_t^T exp(-gamma(s-t)) r(s) ds, satisfying the backward recursion
        R_t = r_t dt + exp(-gamma dt) R_{t+1}.

        Exposed publicly (rather than as a private helper) because it is the part of the
        estimator that is checkable against a closed form, and the tests do check it.
        """
        rates = jnp.asarray(reward_rates)
        if getattr(self.discount, "discounted", False):
            decay = float(np.exp(-dt * self.discount.gamma))

            def backward_step(carry, r_k):
                carry = r_k * dt + decay * carry
                return carry, carry

            _, returns = jax.lax.scan(backward_step, 0.0, rates, reverse=True)
            return returns
        # Undiscounted: R_t = sum_{k >= t} r_k dt, by a reversed cumulative sum.
        return jnp.flip(jnp.cumsum(jnp.flip(rates))) * dt

    def _reset_monte_carlo_episode_buffer(self) -> None:
        """Discard the previous episode's recorded steps. Called at every episode start."""
        # Per-step records for the once-per-episode actor update. reward_rate feeds the Monte-Carlo
        # return R_t; td_error feeds the averaged TD actor-critic. Only the one in use is filled.
        self._mc_episode_features: list = []
        self._mc_episode_noise: list = []
        self._mc_episode_reward_rate: list = []
        self._mc_episode_td: list = []

    def _in_actor_warmup(self, episode: int) -> bool:
        """True while the actor is frozen for the critic warm-up (first ``actor_warmup_episodes``)."""
        return episode < int(getattr(self.training, "actor_warmup_episodes", 0))

    def _batched_actor_step(self, features_batch, noise_batch, advantage) -> jnp.ndarray:
        """Accumulate this episode's records; take ONE averaged actor step per K episodes.

        K = training.rollouts_per_update. The K episodes' per-step records are concatenated and the
        averaged gradient E[g] = mean over all K*T steps is applied in a single step, so K independent
        rollouts raise the averaged-update SNR ~K without an OU-correlation penalty (distinct episodes
        draw independent noise), at the same total episode budget. K=1 recovers the per-episode update.
        Returns the gradient norm on a step episode, 0 while still accumulating."""
        K = max(1, int(getattr(self.training, "rollouts_per_update", 1)))
        self._batch_sig.append(features_batch)
        self._batch_noise.append(noise_batch)
        self._batch_adv.append(advantage)
        if len(self._batch_sig) < K:
            return jnp.array(0.0)   # still filling the batch; no optimiser step this episode
        features_all = jnp.concatenate(self._batch_sig)
        noise_all = jnp.concatenate(self._batch_noise)
        adv_all = jnp.concatenate(self._batch_adv)
        self._batch_sig, self._batch_noise, self._batch_adv = [], [], []
        self.actor_params, self.actor_opt_state, grad_norm = self._jit_monte_carlo_actor_update(
            self.actor_params, self.actor_opt_state,
            features_all, noise_all, adv_all, self._sigma_effective,
        )
        return grad_norm

    def _monte_carlo_actor_update(self, episode: int = 0) -> jnp.ndarray:
        """Perform the episode's single averaged actor update. Returns the gradient norm.

        Handles BOTH once-per-episode actors: the Monte-Carlo policy gradient (advantage = the
        realised return-to-go R_t) and the averaged TD actor-critic (advantage = the recorded TD
        errors delta_t). The update itself -- accumulate delta * (n/sigma^2) dA over the episode
        and take one averaged step -- is identical; only the per-step signal differs. A no-op
        (zero gradient norm) for the online TD actor, the oracle actor, or an empty episode.

        During the critic warm-up (``episode < actor_warmup_episodes``) the diagnostics are still
        recorded -- so the beginning of training is visible -- but the optimiser step is skipped:
        the actor is frozen while the critic (updated per step, elsewhere) warms.
        """
        if not self._actor_updates_once_per_episode or self.algorithm.actor_oracle:
            return jnp.array(0.0)
        if not getattr(self, "_mc_episode_features", None):
            return jnp.array(0.0)

        features_batch = jnp.stack([jnp.asarray(s) for s in self._mc_episode_features])
        noise_batch = jnp.stack([jnp.atleast_1d(jnp.asarray(n)) for n in self._mc_episode_noise])

        if not self._actor_target_is_monte_carlo:
            # Averaged TD actor-critic: the per-step signal is the TD error delta_t (Doya Eq 20),
            # accumulated and applied as one averaged step. No return-to-go, no baseline; the
            # critic still learns per step and enters only through delta_t, exactly as in the
            # online form -- only the actor's update cadence changes.
            advantage = jnp.stack([jnp.asarray(d).reshape(()) for d in self._mc_episode_td])
            self._record_actor_diag(features_batch, noise_batch, advantage)
            if self._in_actor_warmup(episode):
                return jnp.array(0.0)   # critic warm-up: actor frozen, critic still learned per step
            return self._batched_actor_step(features_batch, noise_batch, advantage)

        reward_rates = jnp.stack(
            [jnp.asarray(r).reshape(()) for r in self._mc_episode_reward_rate])

        # NO BASELINE. The signal is the realised return itself: A_t = R_t.
        #
        # By the project owner's decision -- "the baseline idea is ok but i dont want to implement
        # it right now and i dont want the algorithm to become actor critic, let keep things
        # simple". The consequence is the point of this learner: its actor never reads the critic,
        # so an H1 verdict measured under it is a statement about the REPRESENTATION rather than
        # about the value machinery. A baseline V(s_t) would put the critic back into the actor's
        # signal and make this a third actor-critic.
        #
        # This also removes a defect rather than merely simplifying. With discount.discounted =
        # false the critic's temporal-difference error is exactly invariant to V -> V + c, so the
        # critic loss supplies no gradient pinning V's absolute level; the level is unidentified
        # and drifts (the signature's constant channel absorbs it). A baseline read off that
        # critic therefore carries an arbitrary offset into the advantage. R_t has no such offset.
        #
        # The cost, stated plainly: no baseline means higher variance. Var[R_t] is not reduced by
        # an action-independent constant, and REINFORCE without one is the high-variance form. If
        # that dominates, the cheapest unbiased remedy is a baseline depending on the state ONLY
        # -- not the episode's own mean return, which depends on the actions taken and would bias
        # the gradient.
        advantage = self.monte_carlo_returns(reward_rates, float(self.env.step_size))

        self._record_actor_diag(features_batch, noise_batch, advantage)
        if self._in_actor_warmup(episode):
            return jnp.array(0.0)   # (warm-up is an AC fix; PG's actor never reads the critic)
        return self._batched_actor_step(features_batch, noise_batch, advantage)

    def _update_networks(self, ctx: StepContextSignature) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update actor and critic networks using JIT-compiled JAX autodiff."""
        step = self.step_counter
        self.step_counter += 1
        features_t = ctx.features_t
        features_next = ctx.features_next
        noise = ctx.noise
        reward = ctx.reward
        dt = ctx.dt
        sigma = self._sigma_effective
        actor_grad_norm = jnp.array(0.)
        critic_grad_norm = jnp.array(0.)
        td_error = ctx.td_error
        c_loss = jnp.array(0.)
        # Critic update (JIT-compiled)
        if not self.algorithm.critic_oracle:
            self.critic_params, self.critic_opt_state, c_loss, td_error, critic_grad_norm = \
                self._jit_critic_update(
                    self.critic_params, self.critic_opt_state, 
                    features_t, features_next, reward, dt
                )
        
        # Actor update (JIT-compiled)
        if not self.algorithm.actor_oracle:
            if self._actor_updates_once_per_episode:
                # Once-per-episode actor (Monte-Carlo return OR averaged TD): the per-step signal is
                # RECORDED here and the single averaged update is performed by
                # _monte_carlo_actor_update from train(). Monte-Carlo records the reward (its return
                # needs the whole episode); averaged TD records the TD error delta_t, which is
                # already known this step. Every step is recorded; actor_update_frequency is
                # ignored in this mode.
                self._mc_episode_features.append(self._actor_features_t)
                self._mc_episode_noise.append(noise)
                if self._actor_target_is_monte_carlo:
                    self._mc_episode_reward_rate.append(reward)
                else:
                    self._mc_episode_td.append(td_error)
            elif step % self.algorithm.actor_update_frequency == 0:
                # Online TD (Doya Eq 20): single-sample update every actor_update_frequency steps.
                self.actor_params, self.actor_opt_state, actor_grad_norm = \
                    self._jit_actor_update(
                        self.actor_params, self.actor_opt_state,
                        self._actor_features_t, noise, td_error, sigma, dt
                    )
        
        # Return JAX arrays - defer float() sync to episode end
        return c_loss, actor_grad_norm, critic_grad_norm

    # =========================================================================
    # Training Loop
    # =========================================================================
    
    def _on_episode_start(self, episode: int, x_init: np.ndarray) -> None:
        """Hook called at the start of each episode. Override for custom logic."""
        # REINFORCE records one entry per step and consumes them at episode end; the buffer must
        # be emptied here so a truncated episode (divergence cut) cannot leak its steps into the
        # next episode's return, which would attribute one episode's rewards to another's actions.
        self._reset_monte_carlo_episode_buffer()
        if getattr(self.noise, "ou", False):
            # Ornstein-Uhlenbeck exploration (Doya 2000), dt-independent correlation. See
            # src/utils/exploration_noise.py and the identical block in ContinuousValueGradient.
            n_points = len(np.arange(0, self.training.max_time + 5 * self.env.step_size,
                                     self.env.step_size))
            self.key, subkey = jax.random.split(self.key)
            self.episode_noise_trajectory = sample_ou_trajectory(
                n_points, self.env.B.shape[1], float(self.env.step_size),
                float(self.noise.tau_n), float(self._sigma_effective), subkey)
        elif self.noise.smooth:
            # Generate pre-sampled Gaussian Process noise
            # Kernel: Squared Exponential (RBF): k(t, t') = sigma^2 * exp(-|t-t'|^2 / (2 * l^2))

            # 1. Define time points
            # Add a small buffer to max_time to avoid index out of bounds at the very last step
            ts = np.arange(0, self.training.max_time + 5 * self.env.step_size, self.env.step_size)
            n_points = len(ts)
            action_dim = self.env.B.shape[1]
            
            # 2. Compute Covariance Matrix (Vectorized RBF)
            # dist_sq[i, j] = (ti - tj)^2
            ts_col = ts[:, np.newaxis]
            dist_sq = (ts_col - ts_col.T)**2
            # Use unit variance for stability, scale later
            K = 1.0 * np.exp(-dist_sq / (2 * self.noise.length_scale**2))
            
            # 3. Add small epsilon for Cholesky stability
            K += 1e-6 * np.eye(n_points)
            
            # 4. Cholesky Decomposition
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback if numerical issues (should be rare with epsilon)
                print("Warning: Cholesky failed, using SVD for GP generation")
                u, s, vh = np.linalg.svd(K)
                L = u @ np.diag(np.sqrt(s))
            
            # 5. Sample White Noise and Transform (using jax.random for reproducibility)
            # We need independent noise for each action dimension
            # Shape: (n_points, action_dim)
            self.key, subkey = jax.random.split(self.key)
            white_noise = jax.random.normal(subkey, shape=(n_points, action_dim))
            # Apply sigma scaling at the very end
            gp_sample = (jnp.array(L) @ white_noise) * self._sigma_effective
            
            self.episode_noise_trajectory = gp_sample
            
    
    def _on_episode_end(self, episode: int, episode_metrics: dict) -> None:
        """Hook called at the end of each episode. Override for custom logic."""
        pass
    
    def _format_progress(self, episode: int, episode_metrics: dict) -> str:
        """Format progress bar description."""
        flags = []
        if self.discount.discounted: 
            flags.append("disc")
        if self.algorithm.semi_gradient: 
            flags.append("semi")
        if self.algorithm.integral_td: 
            flags.append("int")
        flag_str = f"[{','.join(flags)}] " if flags else ""
        
        # Get critic weights from Flax params
        critic_w = np.array(self.critic_params['params']['Dense_0']['kernel']).flatten()
        # Get actor weights from Flax params
        actor_w = np.array(self.actor_params['params']['Dense_0']['kernel']).flatten() #type: ignore
        
        return (
            f"{flag_str}Ep {episode+1} | "
            f"R: {episode_metrics['cost']:.2f}, "
            f"L: {episode_metrics['loss']:.4f}, "
            f"C: {[f'{w:.2f}' for w in critic_w[:6]]}, "
            f"A: {[f'{w:.2f}' for w in actor_w[:6]]}"  # Show first 6 weights
        )

    def update_buffer(self, x: np.ndarray) -> None:
        self.representation_buffer.append(x / self.training.scale)
        if hasattr(self, '_path_data_dirty'):
            setattr(self, '_path_data_dirty', True)
            
    def get_eval_action(self, x_scaled: jnp.ndarray) -> jnp.ndarray:
        action: Any
        if getattr(self.algorithm, 'actor_oracle', False):
            assert self.wrapper.state is not None
            if getattr(self, "_delayed_oracle", False):
                action = -self.K_aug @ self._delayed_oracle_window()
            else:
                action = -self.optimal_K @ jnp.array(self.wrapper.state.x)
        else:
            sig = self.representation_buffer.current_actor_features
            assert self.actor_params is not None
            action = self.actor.apply(self.actor_params, sig)

        action = jnp.asarray(action)
        # Opt-in, as in _make_select_action_fn: null/<=0 means no clipping.
        if self.training.clip_action is not None and self.training.clip_action > 0:
            action = jnp.clip(action, -self.training.clip_action, self.training.clip_action)
        return jnp.array(action)

    def get_value(self) -> float:
        """Compute value function for a given state."""
        if getattr(self.algorithm, 'critic_oracle', False):
            assert self.wrapper.state is not None
            if getattr(self, "_delayed_oracle", False):
                xi = self._delayed_oracle_window()
                return float(-xi @ self.P_aug @ xi)
            x_val = jnp.array(self.wrapper.state.x)
            return float(-x_val.T @ self.P @ x_val)
            
        sig = self.representation_buffer.current_signature
        
        assert self.critic_params is not None
        V_raw = self.critic.apply(self.critic_params, sig)
        V = V_raw[0] if isinstance(V_raw, tuple) else V_raw
        return float(jnp.asarray(V).squeeze())

    def train(self) -> dict:
        """Main training loop.
        
        Returns:
            Dictionary of training metrics
        """
        # Metric storage
        metrics_history: dict[str, Any] = {
            'loss_episodic': [],
            'cost_episodic': [],
            'gradient_actor': [],
            'gradient_critic': [],
            'init_conditions': [],
            'actor_weights': [],
            'critic_weights': [],
            'signature_weights': [],
            'noise': [],
        }
        
        log_interval = self.training.log_interval
        init_log_interval = self.training.init_log_interval
        memory_clear_interval = self.training.memory_clear_interval
        
        iterator = tqdm.trange(self.training.n_episodes, desc="Training", leave=True)
        
        for episode in iterator:
            # Episode initialization
            self.episode = episode
            if not self.algorithm.fix_initial_state:
                x_init = self._sample_initial_state()
            else:
                x_init = self.x0 if self.x0 is not None else jnp.zeros(self.env.N)
            
            if episode % init_log_interval == 0:
                metrics_history['init_conditions'].append(np.array(x_init))
            
            # Reset wrapper with new initial state
            self.key, subkey = jax.random.split(self.key)
            x_t = self.wrapper.reset(subkey, x0=np.array(x_init), t0=0.0)
            self._fill_buffer_initial()  # Refill signature buffer after reset
            self.current_noise = jnp.zeros(self.env.B.shape[1]) # Initialize noise state
            self._on_episode_start(episode, np.array(x_t, dtype=np.float64))
            self._announce_zero_control_burn_in()
            for _ in range(self.algorithm.burning_steps):
                action = jnp.zeros(self.env.B.shape[1]) #burning with zero action
                t, x_t, _ = self.wrapper.step(self.wrapper.state, action) #type: ignore
                self.representation_buffer.append(x_t / self.training.scale)
            if self.algorithm.preheat:
                for _ in range(self.representation_buffer.window_size):
                    if self.signature_conf.state_augmentation:
                        _ = jnp.concatenate([self.representation_buffer.current_signature, x_t / self.training.scale])  # type: ignore
                    else:
                        _ = self.representation_buffer.current_signature
                    # action, _, _ = self._select_action(state, self.env.step_size)
                    action = jnp.zeros(self.env.B.shape[1]) #preheat with zero action
                    t, x_t, _ = self.wrapper.step(self.wrapper.state, action) #type: ignore
                    self.representation_buffer.append(x_t / self.training.scale)
            # Episode accumulators (as JAX arrays to avoid sync)
            episode_loss = jnp.array(0.0)
            episode_cost = jnp.array(0.0)
            actor_grad_sum = jnp.array(0.0)
            critic_grad_sum = jnp.array(0.0)
            n_steps = 0
            
            # Episode loop
            while not self._is_episode_done(x_t):
                x_t, step_metrics, ctx = self._train_step(x_t)
                
                # Accumulate as JAX arrays (no sync)
                episode_loss = episode_loss + step_metrics.loss
                episode_cost = episode_cost + step_metrics.reward
                actor_grad_sum = actor_grad_sum + step_metrics.actor_gradient
                critic_grad_sum = critic_grad_sum + step_metrics.critic_gradient
                n_steps += 1
                
                # Memory management
                if n_steps % memory_clear_interval == 0:
                    self.wrapper._data.clear()
                    self.wrapper._time.clear()
                # Only log signatures occasionally to avoid slowdown
                if episode % 200 == 0 and n_steps % 10 == 0:
                    metrics_history['signature_weights'].append(
                        np.asarray(ctx.features_t).flatten()
                    )
                    metrics_history['noise'].append(
                        np.asarray(ctx.noise).flatten()
                    )
            # Once-per-episode actor (Monte-Carlo return OR averaged TD): the episode is over, so
            # the single averaged update is performed here. A no-op for the online TD actor.
            monte_carlo_grad_norm = self._monte_carlo_actor_update(episode)
            if self._actor_updates_once_per_episode:
                # One update per episode, so report its gradient norm directly rather than the
                # per-step mean (which would be this value divided by n_steps and misleading).
                actor_grad_sum = monte_carlo_grad_norm * max(n_steps, 1)

            # Episode metrics (convert to float only at episode end)
            episode_metrics = {
                'loss': float(episode_loss),
                'cost': float(episode_cost),
                'actor_grad_mean': float(actor_grad_sum) / max(n_steps, 1),
                'critic_grad_mean': float(critic_grad_sum) / max(n_steps, 1),
                'actor_snr': self._last_actor_snr,   # NaN unless training.monitor_snr
                'n_steps': n_steps,
            }
            metrics_history.setdefault('actor_snr', []).append(self._last_actor_snr)
            # Rich actor-signal diagnostics, logged EVERY episode (incl. the first) so the beginning
            # of training is fully resolved. Empty dict when monitor_snr is off.
            for _k, _v in self._last_actor_diag.items():
                episode_metrics[_k] = _v
                metrics_history.setdefault(_k, []).append(_v)
            
            # Check for NaNs
            if (np.isnan(episode_metrics['loss']) or 
                np.isnan(episode_metrics['cost']) or
                np.isnan(episode_metrics['actor_grad_mean']) or
                np.isnan(episode_metrics['critic_grad_mean'])):
                print(f"\n[NaN detected] Episode {episode}: loss={episode_metrics['loss']:.4f}, "
                      f"cost={episode_metrics['cost']:.4f}")
                self._nan_detected = True
                break
            
            self._on_episode_end(episode, episode_metrics)
            
            # Periodic trajectory evaluation (wandb slider)
            if hasattr(self, 'eval_callback') and self.eval_callback is not None:
                self.eval_callback(self, episode)
            
            # Update progress bar
            iterator.set_description(self._format_progress(episode, episode_metrics))
            
            # Log metrics
            if episode % log_interval == 0:
                metrics_history['loss_episodic'].append(float(episode_loss))
                metrics_history['cost_episodic'].append(float(episode_cost))
                metrics_history['gradient_actor'].append(episode_metrics['actor_grad_mean'])
                metrics_history['gradient_critic'].append(episode_metrics['critic_grad_mean'])
                # Extract weights from Flax params
                actor_w = np.array(self.actor_params['params']['Dense_0']['kernel']).copy() #type: ignore
                critic_w = np.array(self.critic_params['params']['Dense_0']['kernel']).copy()
                metrics_history['actor_weights'].append(actor_w)
                if episode % 50 == 0:
                    metrics_history['critic_weights'].append(critic_w)
            
            # --- Periodic noiseless evaluation & best checkpoint ---
            if episode >= self.training.eval_start_episode and episode % self.training.eval_interval == 0:
                eval_reward = self._evaluate_noiseless()
                metrics_history.setdefault('eval_reward', []).append(eval_reward)
                metrics_history.setdefault('eval_episodes', []).append(episode)
                
                if eval_reward > self._best_eval_reward:
                    self._best_eval_reward = eval_reward
                    self._best_actor_params = jax.tree_util.tree_map(
                        lambda x: x.copy(), self.actor_params)
                    self._best_critic_params = jax.tree_util.tree_map(
                        lambda x: x.copy(), self.critic_params)
                    self._best_episode = episode
                    self._patience_counter = 0
                    print(f"  [New best] ep {episode}, eval_reward={eval_reward:.4f}")
                else:
                    self._patience_counter += 1
                
                if self.training.patience > 0 and self._patience_counter >= self.training.patience:
                    print(f"\n[Early stop] No improvement for {self.training.patience} evals. "
                          f"Best at ep {self._best_episode} (reward={self._best_eval_reward:.4f})")
                    break
            
            # Clear wrapper history at end of each episode to prevent memory growth
            self.wrapper._data.clear()
            self.wrapper._time.clear()
            self.wrapper._u_history.clear()
        
        # Restore best checkpoint
        if self._best_critic_params is not None:
            final_reward = float(metrics_history['cost_episodic'][-1]) if metrics_history['cost_episodic'] else -float('inf')
            reason = "NaN detected" if self._nan_detected else "Best checkpoint"
            print(f"\n[{reason}] Restoring params from episode {self._best_episode} "
                  f"(eval_reward={self._best_eval_reward:.4f}, "
                  f"final_reward={final_reward:.4f})")
            self.actor_params = self._best_actor_params
            self.critic_params = self._best_critic_params
        elif self._nan_detected:
            print("\n[Warning] NaN detected but no best checkpoint available!")
        metrics_history['state_counts'] = self.state_counter #type: ignore
        return metrics_history

    def _evaluate_noiseless(self) -> float:
        """Run a noiseless rollout and return the total reward (higher = better).
        
        Uses the actor (mu only, no exploration noise) from the fixed initial condition.
        """
        x_init = self.x0 if self.x0 is not None else jnp.zeros(self.env.N)
        if self.x0 is None:
            import warnings
            warnings.warn(
                "_evaluate_noiseless: x0 is None, evaluating from zeros. "
                "Set agent.x0 or cfg.eval.x0_test for meaningful evaluation.",
                stacklevel=2,
            )
        
        # Save state
        saved_state = self.wrapper.state
        buf = self.representation_buffer.buffer
        saved_buf: Any  # tuple (JAXCircularBuffer state) or deque (DequeBuffer), per branch below
        if hasattr(buf, '_data'):
            saved_buf = (buf._data.copy(), buf._count, buf._head)  # type: ignore[union-attr]
        else:
            from collections import deque
            assert isinstance(buf, DequeBuffer)  # the non-_data branch is the deque buffer
            saved_buf = deque(buf.buffer, maxlen=buf.size)
        saved_sig = self.representation_buffer.current_signature
        
        self.key, subkey = jax.random.split(self.key)
        x_t = self.wrapper.reset(subkey, x0=np.array(x_init), t0=0.0)
        self._fill_buffer_initial()
        self._announce_zero_control_burn_in()

        for _ in range(self.algorithm.burning_steps):
            action = jnp.zeros(self.env.B.shape[1])
            _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
            self.representation_buffer.append(x_t / self.training.scale)
        if self.algorithm.preheat:
            for _ in range(self.representation_buffer.window_size):
                action = jnp.zeros(self.env.B.shape[1])  # zero action, consistent with training
                _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
                self.representation_buffer.append(x_t / self.training.scale)
        
        total_reward = 0.0
        
        while not self._is_episode_done(x_t, time_only=True):
            if self.signature_conf.state_augmentation:
                features_input = jnp.concatenate([self.representation_buffer.current_actor_features, x_t / self.training.scale])
            else:
                features_input = self.representation_buffer.current_actor_features
            
            if self.algorithm.actor_oracle:
                if getattr(self, "_delayed_oracle", False):
                    mu = -self.K_aug @ self._delayed_oracle_window()
                else:
                    mu = -self.optimal_K @ jnp.array(self.wrapper.state.x)  # type: ignore
            else:
                mu = self.actor.apply(self.actor_params, features_input) #type: ignore
            # Opt-in, as in _make_select_action_fn: null/<=0 means no clipping.
            if self.training.clip_action is not None and self.training.clip_action > 0:
                mu = jnp.clip(mu, -self.training.clip_action, self.training.clip_action)
            
            _, x_next, reward = self.wrapper.step(self.wrapper.state, mu)  # type: ignore
            if x_next.ndim == 0:
                x_next = jnp.array([x_next])
            self.representation_buffer.append(x_next / self.training.scale)
            total_reward += float(reward) * self.env.step_size
            x_t = x_next
        
        # Restore state
        self.wrapper.state = saved_state
        if hasattr(buf, '_data'):
            buf._data, buf._count, buf._head = saved_buf  # type: ignore[union-attr]
        else:
            assert isinstance(buf, DequeBuffer)  # the non-_data branch is the deque buffer
            buf.buffer = saved_buf
        self.representation_buffer.current_signature = saved_sig

        return total_reward

    def save(self, filename: str) -> None:
        """Save agent parameters to a file."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'actor_params': self.actor_params,
            'critic_params': self.critic_params,
            'env_params': self.env_params,
            # Config objects (new-style)
            'training': self.training,
            'discount': self.discount,
            'noise': self.noise,
            'signature': self.signature_conf,
            'network': self.network,
            'algorithm': self.algorithm,
            # Flat copies for backward-compat checkpoint loading
            'training_params': configs_to_flat_dict(
                self.training, self.discount, self.noise,
                self.signature_conf, self.network, self.algorithm,
            ),
            'discounted': self.discount.discounted,
            'semi_gradient': self.algorithm.semi_gradient,
            'integral_td': self.algorithm.integral_td,
            'fix_initial_state': self.algorithm.fix_initial_state,
            'decay_noise': self.noise.decay,
            'time_augmentation': self.signature_conf.time_augmentation,
            'depth': self.signature_conf.depth,
        }
        # Save delayed_state flag if it exists (for CTACJAX)
        if hasattr(self, 'delayed_state'):
            save_dict['delayed_state'] = self.algorithm.delayed_state # type: ignore
        
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Agent saved to {filename}")

    def load(self, filename: str) -> None:
        """Load agent parameters from a file.
        
        Args:
            filename: Path to the checkpoint file
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Load parameters
        self.actor_params = data['actor_params']
        self.critic_params = data['critic_params']
        
        print(f"Agent loaded from {filename}")
