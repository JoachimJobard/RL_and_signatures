"""
Continuous-Time Actor-Critic (CTAC) - Modular Implementation

This module builds from the CTAC implementation in src.agents.base
but implements signature methods to control PO/delayed systems.
The states becomes the signature from the history of the system
rather than the current state only.
Configuration via flags rather than subclassing for most variants.

Flags (passed to __init__):
    - discounted: Use discounted TD error (δ = r + V̇ - V/τ)
    - semi_gradient: Use semi-gradient (only features(x), not features(x'))
    - integral_td: Use integral form (δ = r·dt + ΔV)
    - actor_oracle: Fix actor to optimal LQR policy
    - critic_oracle: Fix critic to optimal value function

Overridable methods (for custom behavior):
    - _sample_initial_state(): How to initialize episodes
    - _select_action(x): Action selection with exploration noise
    - _compute_reward(x, u): Reward/cost function
    - _compute_V_dot(ctx): Value derivative estimation
    - _compute_td_error(ctx): TD error formulation
    - _compute_critic_gradient(ctx): Critic gradient
    - _update_critic(td_error, grad, dt): Critic weight update
    - _compute_actor_gradient(ctx): Actor gradient
    - _update_actor(td_error, grad, dt): Actor weight update
    - _is_episode_done(x, t): Episode termination condition

Usage:
    # Oracle critic + semi-gradient + discounted
    agent = CTAC(env, ..., critic_oracle=True, semi_gradient=True, discounted=True)
"""

import jax
import numpy as np
import tqdm
import pickle
from src.utils.dynamic_signature import SlidingSignatureJAX
from src.networks.LQR_actor_critics import ActorFlax, ActorFlaxLayerNorm, CriticFlax, CriticFlaxLayerNorm
from src.utils.step_metrics import StepMetrics
from src.utils.step_context import StepContextSignature
from src.utils.state_counter import StateCounter
from src.configs import (
    TrainingConfig, DiscountConfig, NoiseConfig,
    SignatureConfig, NetworkConfig, AlgorithmConfig,
    from_legacy_params, configs_to_flat_dict,
)
import jax.numpy as jnp
import optax
import scipy


from src.env_rk_jax import JAXDDEEnv, JAXEnvWrapper


class CTACSignatureJAX:
    """Continuous-Time Actor-Critic - Modular Implementation.
    
    All variants controlled via configuration flags:
    - discounted: δ = r + V̇ - V/τ
    - semi_gradient: ∇W uses only features(x)
    - integral_td: δ = r·dt + ΔV
    - actor_oracle: Fix actor to K*
    - critic_oracle: Fix critic to V*
    """
    
    def __init__(
        self,
        env: JAXDDEEnv,
        # --- Legacy positional args (kept for CTACJAX backward compat) ---
        training_params: dict | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        depth: int = 2,
        rng_key: int = 42,
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
        x0: jnp.ndarray | None = None,
        burning_steps: int = 0,
        eval_callback=None,
        # --- New-style config objects (take precedence if provided) ---
        training: TrainingConfig | None = None,
        discount: DiscountConfig | None = None,
        noise: NoiseConfig | None = None,
        signature: SignatureConfig | None = None,
        network: NetworkConfig | None = None,
        algorithm: AlgorithmConfig | None = None,
    ):
        # =================================================================
        # Config resolution: new-style dataclasses or legacy dict+kwargs
        # =================================================================
        if training is not None:
            # New-style path — config objects provided directly
            self.training = training
            self.discount = discount or DiscountConfig()
            self.noise = noise or NoiseConfig()
            self.signature = signature or SignatureConfig()
            self.network = network or NetworkConfig()
            self.algorithm = algorithm or AlgorithmConfig()
        elif training_params is not None:
            # Legacy path — convert dict + kwargs to dataclasses
            (self.training, self.discount, self.noise,
             self.signature, self.network, self.algorithm) = from_legacy_params(
                training_params,
                depth=depth,
                discounted=discounted,
                semi_gradient=semi_gradient,
                integral_td=integral_td,
                fix_initial_state=fix_initial_state,
                decay_noise=decay_noise,
                time_augmentation=time_augmentation,
                state_augmentation=state_augmentation,
                origin_augmentation=origin_augmentation,
                time_origin=time_origin,
                bias=bias,
                actor_oracle=actor_oracle,
                critic_oracle=critic_oracle,
                preheat=preheat,
                actor_update_frequency=actor_update_frequency,
                window_size=window_size,
                smooth_noise=smooth_noise,
                noise_length_scale=noise_length_scale,
                burning_steps=burning_steps,
            )
        else:
            raise ValueError(
                "Either provide config objects (training=..., noise=..., etc.) "
                "or a legacy training_params dict."
            )

        # =================================================================
        # Flat aliases — keeps the rest of the class unchanged
        # =================================================================
        # Training
        self.n_episodes = self.training.n_episodes
        self.max_time = self.training.max_time
        self.actor_lr = self.training.actor_lr
        self.critic_lr = self.training.critic_lr
        self.scale = self.training.scale
        self.clip_gradient = self.training.clip_gradient
        self.clip_action = self.training.clip_action
        self.divergence_threshold = self.training.divergence_threshold
        self.eval_interval = self.training.eval_interval
        self.eval_start_episode = self.training.eval_start_episode
        self.patience = self.training.patience
        self.discretization_state = self.training.discretization_state
        # Discount
        self.discounted = self.discount.discounted
        self.tau = self.discount.tau
        self.V_target = self.discount.V_target
        self.V_bad = self.discount.V_bad
        # Noise
        self.sigma = self.noise.sigma
        self.noise_schedule = self.noise.schedule
        self.decay_noise = self.noise.decay
        self.smooth_noise = self.noise.smooth
        self.noise_length_scale = self.noise.length_scale
        # Signature
        self.depth = self.signature.depth
        self.time_augmentation = self.signature.time_augmentation
        self.time_origin = self.signature.time_origin
        self.origin_augmentation = self.signature.origin_augmentation
        self.state_augmentation = self.signature.state_augmentation
        self.normalize_sigs = self.network.normalize_sigs
        self.force_signature_window = self.signature.force_signature_window
        # Network
        self.std_network = self.network.std_init
        self.normalize_entries = self.network.normalize_entries
        # Algorithm
        self.semi_gradient = self.algorithm.semi_gradient
        self.integral_td = self.algorithm.integral_td
        self.actor_oracle = self.algorithm.actor_oracle
        self.critic_oracle = self.algorithm.critic_oracle
        self.actor_update_frequency = self.algorithm.actor_update_frequency
        self.preheat = self.algorithm.preheat
        self.burning_steps = self.algorithm.burning_steps
        self.fix_initial_state = self.algorithm.fix_initial_state

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
        self.env = env
        self.wrapper = JAXEnvWrapper(env, rng_key=jax.random.PRNGKey(rng_key))
        self.episode = 0

        # Oracle setup
        self.window_size = self.signature.window_size
        if self.actor_oracle:
            P = scipy.linalg.solve_continuous_are(
                self.env.A, self.env.B, self.env.Q, self.env.R
            )
            print("P matrix for optimal LQR policy:\n", P)
            self.P = jnp.array(P)
            self.optimal_K = jnp.array(np.linalg.inv(self.env.R) @ self.env.B.T @ P)

        if self.critic_oracle:
            P = scipy.linalg.solve_continuous_are(
                self.env.A, self.env.B, self.env.Q, self.env.R
            )
            self.P = jnp.array(P)
            print("P matrix for optimal value function:\n", P)

        # Runtime state
        self.step_counter = 0
        self.current_noise = None
        self.episode_noise_trajectory = None
        self.x0 = x0
        self.eval_callback = eval_callback

        # Best checkpoint & early stopping
        self._best_eval_reward = -float('inf')
        self._best_actor_params = None
        self._best_critic_params = None
        self._best_episode = 0
        self._patience_counter = 0
        self._nan_detected = False
        self.state_counter = StateCounter(resolution=self.discretization_state)

        if self.normalize_entries:
            self.scale = self.divergence_threshold
            print("normalize_entries is True: scaling states by ", self.scale)
            print("normalize_sigs is ", self.normalize_sigs)

        # Signature setup
        max_delay = float(jnp.max(self.env.delay)) if self.env.delay is not None else 0
        window_size_from_delay = int(np.ceil(max_delay / self.env.step_size)) + 1 if max_delay > 0 else 10
        if not self.force_signature_window:
            self.window_size = window_size_from_delay
        else:
            print(f"force_signature_window is True: setting window_size to {self.window_size} to cover max delay of {max_delay}")
        self.sliding_signature = SlidingSignatureJAX(
            depth=self.depth, window_size=self.window_size, d=env.N,
            time_augmentation=self.time_augmentation,
            origin_augmentation=self.origin_augmentation,
            time_origin=self.time_origin, bias=self.signature.bias,
        )
        self._sigma_effective = self.sigma

        # Cost matrices
        self.Q = Q
        self.R = R

        # Random generator
        self.key = jax.random.PRNGKey(rng_key)
                
        # Build networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        key_a, key_c = jax.random.split(self.key)
        if self.state_augmentation:
            self.actor_params = self.actor.init(key_a, jnp.zeros(self.sliding_signature.signature_size + self.env.N))
            self.critic_params = self.critic.init(key_c, jnp.zeros(self.sliding_signature.signature_size + self.env.N))
        else:
            self.actor_params = self.actor.init(key_a, jnp.zeros(self.sliding_signature.signature_size))
            self.critic_params = self.critic.init(key_c, jnp.zeros(self.sliding_signature.signature_size))

        #optimizers — absorb dt into learning rate for correct continuous-time scaling
        self.actor_optimizer = optax.adam(self.actor_lr * self.env.step_size, b1=0.1)
        self.critic_optimizer = optax.adam(self.critic_lr * self.env.step_size)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)
        
        # JIT-compiled update functions (created once)
        self._jit_critic_update = self._make_critic_update_fn()
        self._jit_actor_update = self._make_actor_update_fn()
        self._jit_select_action = self._make_select_action_fn()
        self._jit_compute_values = self._make_compute_values_fn()


    # =========================================================================
    # Network Construction
    # =========================================================================
    
    def _build_actor(self) -> ActorFlax | ActorFlaxLayerNorm:
        """Build the actor network."""
        output_dim = self.env.B.shape[1]
        if self.normalize_sigs:
            actor = ActorFlaxLayerNorm(output_dim=output_dim, stddev=self.std_network/10)
        else:
            actor = ActorFlax(output_dim=output_dim, stddev=self.std_network/10)
        return actor

    def _build_critic(self) -> CriticFlax | CriticFlaxLayerNorm:
        """Build the critic network."""       
        if self.normalize_sigs:
            critic = CriticFlaxLayerNorm(stddev=self.std_network)
        else:
            critic = CriticFlax(stddev=self.std_network)
        return critic

    # =========================================================================
    # Episode Initialization
    # =========================================================================
    
    def _sample_initial_state(self) -> jnp.ndarray:
        """Sample initial state for an episode."""
        self.key, subkey = jax.random.split(self.key) 
        return jax.random.normal(subkey, shape=(self.env.N,))
    
    def _fill_buffer_initial(self):
        """Fill the signature buffer with initial states from wrapper."""
        self.sliding_signature.reset()
        if self.wrapper.state is not None:
            _, history_data = self.wrapper.initial_conditions
            # Subsample by resolution to match window size
            # Apply scaling to be consistent with training
            for x in history_data[::self.env.resolution]:
                self.sliding_signature.append(x / self.scale)
            self.sliding_signature._signature_dirty = True

    # =========================================================================
    # Action Selection
    # =========================================================================
    
    def _make_select_action_fn(self):
        """Create JIT-compiled action selection function."""
        actor = self.actor
        clip_action = self.clip_action
        smooth_noise = self.smooth_noise  # Capture static config
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
                 # Use the pre-computed GP noise passed from Python
                 # We ignore 'sigma' here because scaling is already in the GP trajectory
                 noise = explicit_noise_val
            else:
                noise = sigma * jax.random.normal(subkey, shape=mu.shape) # type: ignore

            action = mu + noise
            action = jnp.clip(action, -clip_action, clip_action)
            
            return action, mu, noise, key, noise
        
        return select_action_fn
    
    def _make_compute_values_fn(self):
        """Create JIT-compiled value computation function."""
        critic = self.critic
        
        @jax.jit
        def compute_values_fn(critic_params, sig_t, sig_next):
            V_t = critic.apply(critic_params, sig_t).squeeze() # type: ignore
            V_next = critic.apply(critic_params, sig_next).squeeze() # type: ignore
            return V_t, V_next
        
        return compute_values_fn
    
    def _compute_value_function(self, sig):
        return self.critic.apply(self.critic_params, sig).squeeze() # type: ignore

    def _select_action(self, state, dt: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Select action with exploration noise."""
        
        # --- Compute effective noise level based on schedule ---
        if self.noise_schedule == 'adaptive':
            V_t = self._compute_value_function(self.sliding_signature.current_signature)
            V_TARGET = self.V_target
            V_BAD = self.V_bad
            noise_scale = jnp.clip((V_TARGET - V_t) / (V_TARGET - V_BAD + 1e-6), 0.1, 1.0)
            self._sigma_effective = self.sigma * noise_scale
        elif self.noise_schedule == 'linear_decay':
            progress = min(self.episode / max(self.n_episodes, 1), 1.0)
            self._sigma_effective = self.sigma * max(0.05, 1.0 - 0.9 * progress)
        else:
            # 'constant'
            self._sigma_effective = self.sigma
        if self.actor_oracle:
            mu = - self.optimal_K @ jnp.array(self.wrapper.state.x) #type: ignore
            self.key, subkey = jax.random.split(self.key)
            # noise = self._sigma_effective * jax.random.normal(subkey, shape=mu.shape) # type: ignore
            noise = jnp.zeros_like(mu)
            action = mu + noise
            # action = jnp.clip(action, -self.clip_action, self.clip_action)
            # Update current_noise to zeros to keep consistency if we switch modes
            self.current_noise = jnp.zeros_like(noise)
        else:
            if self.current_noise is None:
                 self.current_noise = jnp.zeros(self.env.B.shape[1])
            
            # Prepare explicit noise value if GP mode is active
            explicit_noise_val = jnp.zeros_like(self.current_noise)
            if self.smooth_noise and self.episode_noise_trajectory is not None:
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
    # Value Derivative Estimation
    # =========================================================================
    
    def _compute_V_dot(self, ctx: StepContextSignature) -> float:
        """Compute time derivative of value function (Euler forward difference)."""
        return (ctx.V_next - ctx.V_t) / ctx.dt

    # =========================================================================
    # TD Error Computation (flag-based)
    # =========================================================================
    
    def _compute_td_error(self, ctx: StepContextSignature) -> float:
        """Compute temporal difference error based on flags."""
        if self.integral_td:
            # Integral form: δ = r·dt + ΔV (- V·dt/τ if discounted)
            delta_V = ctx.V_next - ctx.V_t
            td = ctx.reward * ctx.dt + delta_V
            if self.discounted:
                td -= ctx.V_t * ctx.dt / self.tau
            return td
        else:
            # Differential form: δ = r + V̇ (- V/τ if discounted)
            td = ctx.reward + ctx.V_dot
            if self.discounted:
                td -= ctx.V_t / self.tau
            return td

    # =========================================================================
    # Critic Gradient (flag-based)
    # =========================================================================
    
    
    def critic_loss_fn(self, sig_t, sig_next,x, x_next, reward, dt): 
        if self.state_augmentation:
            sig_t = jnp.concatenate([sig_t, jnp.array(x)])  # type: ignore
            sig_next = jnp.concatenate([sig_next, jnp.array(x_next)])  # type: ignore
        V_t_raw = self.critic.apply(self.critic_params, sig_t)
        V_next_raw = jax.lax.stop_gradient(self.critic.apply(self.critic_params, sig_next))
        V_t = jnp.asarray(V_t_raw[0] if isinstance(V_t_raw, tuple) else V_t_raw).squeeze()
        V_next = jnp.asarray(V_next_raw[0] if isinstance(V_next_raw, tuple) else V_next_raw).squeeze()
        target = (reward + (V_next - V_t) / dt) # type: ignore
        td_error = target
        if self.discounted:
            td_error -= V_t / self.tau
        loss = 0.5 * td_error ** 2
        return loss, td_error


    # =========================================================================
    # Actor Gradient
    # =========================================================================
    
    def actor_loss_fn(self, sig_t, td_error, noise):
        if self.state_augmentation:
            sig_t = jnp.concatenate([sig_t, jnp.array(self.wrapper.state.x)])  # type: ignore
        mu = self.actor.apply(self.actor_params, sig_t) #type: ignore
        loss = - jax.lax.stop_gradient(td_error) * jax.lax.stop_gradient(noise) * mu
        return loss.sum()


    # =========================================================================
    # Episode Termination
    # =========================================================================
    
    def _is_episode_done(self, x: jnp.ndarray, time_only=False) -> bool:
        """Check if episode should terminate.

        Uses jax.device_get once to fetch both t and the norm check in a
        single GPU→CPU synchronisation instead of multiple float() calls.
        """
        if self.wrapper.state is None:
            return False
        if time_only:
            done = self.wrapper.state.t >= self.max_time
        else:
            done = jnp.logical_or(
                self.wrapper.state.t >= self.max_time,
                jnp.linalg.norm(x) > self.divergence_threshold,
            )
        return bool(done)

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
        x_scaled = x_t / self.scale
        if self.state_augmentation:
            state = jnp.concatenate([self.sliding_signature.current_signature, x_scaled])  # type: ignore
        else:
            state = self.sliding_signature.current_signature
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
        x_next_scaled = x_next / self.scale
        dt = self.env.step_size
        if dt <= 1e-9:
            dt = 1e-4  # Avoid division by zero
        sig_t = self.sliding_signature.current_signature
        self.sliding_signature.append(x_next_scaled)
        sig_next = self.sliding_signature.current_signature
        if self.state_augmentation:
            sig_t = jnp.concatenate([sig_t, x_scaled])  # type: ignore
            sig_next = jnp.concatenate([sig_next, x_next_scaled])  # type: ignore
        # Value function evaluations (JIT-compiled, no float() sync)
        if self.critic_oracle:
            V_t = x_t.T @ self.P @ x_t
            V_next = x_next.T @ self.P @ x_next
        else:
            V_t, V_next = self._jit_compute_values(self.critic_params, sig_t, sig_next)
        
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
            sig_t=sig_t,
            sig_next=sig_next
        )
        
        # Compute reward
        ctx.reward = reward #type: ignore
        
        # Compute V dot and TD error (keep as JAX arrays)
        ctx.V_dot = (ctx.V_next - ctx.V_t) / ctx.dt
        ctx.td_error = ctx.reward + ctx.V_dot - (ctx.V_t / self.tau if self.discounted else 0.0)
        
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
        discounted = self.discounted
        tau = self.tau
        
        def critic_loss(critic_params, sig_t, sig_next, reward, dt):
            V_t = critic.apply(critic_params, sig_t).squeeze() # type: ignore
            V_next = critic.apply(jax.lax.stop_gradient(critic_params), sig_next).squeeze() # type: ignore
            td_error = reward + (V_next - V_t) / dt
            if discounted:
                td_error = td_error - V_t / tau
            return 0.5 * td_error ** 2 * dt, td_error
        
        @jax.jit
        def update_fn(critic_params, opt_state, sig_t, sig_next, reward, dt):
            (loss, td_error), grads = jax.value_and_grad(critic_loss, has_aux=True)(
                critic_params, sig_t, sig_next, reward, dt
            )
            # Clip gradients
            grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -10.0, 10.0), grads)
            
            updates, new_opt_state = optimizer.update(grads, opt_state, critic_params)
            
            new_params = optax.apply_updates(critic_params, updates)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            return new_params, new_opt_state, loss, td_error, grad_norm
        
        return update_fn
    
    def _make_actor_update_fn(self):
        """Create a JIT-compiled actor update function."""
        actor = self.actor
        optimizer = self.actor_optimizer
        
        def actor_loss(actor_params, sig_t, noise, td_error, sigma):
            mu = actor.apply(actor_params, sig_t)
            log_prob_grad = noise / (sigma ** 2 + 1e-8)
            return -td_error * jnp.dot(log_prob_grad, mu) # type: ignore
        
        @jax.jit  
        def update_fn(actor_params, opt_state, sig_t, noise, td_error, sigma, dt):
            grads = jax.grad(actor_loss)(actor_params, sig_t, noise, td_error, sigma)
            
            # Clip gradients
            grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -10.0, 10.0), grads)
            
            updates, new_opt_state = optimizer.update(grads, opt_state, actor_params)
            
            new_params = optax.apply_updates(actor_params, updates)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            return new_params, new_opt_state, grad_norm
        
        return update_fn
    
    def _update_networks(self, ctx: StepContextSignature) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update actor and critic networks using JIT-compiled JAX autodiff."""
        step = self.step_counter
        self.step_counter += 1
        sig_t = ctx.sig_t
        sig_next = ctx.sig_next
        noise = ctx.noise
        reward = ctx.reward
        dt = ctx.dt
        sigma = self._sigma_effective
        actor_grad_norm = jnp.array(0.)
        critic_grad_norm = jnp.array(0.)
        td_error = ctx.td_error
        c_loss = jnp.array(0.)
        # Critic update (JIT-compiled)
        if not self.critic_oracle:
            self.critic_params, self.critic_opt_state, c_loss, td_error, critic_grad_norm = \
                self._jit_critic_update(
                    self.critic_params, self.critic_opt_state, 
                    sig_t, sig_next, reward, dt
                )
        
        # Actor update (JIT-compiled)
        if not self.actor_oracle and (step % self.actor_update_frequency == 0):
            self.actor_params, self.actor_opt_state, actor_grad_norm = \
                self._jit_actor_update(
                    self.actor_params, self.actor_opt_state,
                    sig_t, noise, td_error, sigma, dt
                )
        
        # Return JAX arrays - defer float() sync to episode end
        return c_loss, actor_grad_norm, critic_grad_norm

    # =========================================================================
    # Training Loop
    # =========================================================================
    
    def _on_episode_start(self, episode: int, x_init: np.ndarray) -> None:
        """Hook called at the start of each episode. Override for custom logic."""
        if self.smooth_noise:
            # Generate pre-sampled Gaussian Process noise
            # Kernel: Squared Exponential (RBF): k(t, t') = sigma^2 * exp(-|t-t'|^2 / (2 * l^2))
            
            # 1. Define time points
            # Add a small buffer to max_time to avoid index out of bounds at the very last step
            ts = np.arange(0, self.max_time + 5 * self.env.step_size, self.env.step_size)
            n_points = len(ts)
            action_dim = self.env.B.shape[1]
            
            # 2. Compute Covariance Matrix (Vectorized RBF)
            # dist_sq[i, j] = (ti - tj)^2
            ts_col = ts[:, np.newaxis]
            dist_sq = (ts_col - ts_col.T)**2
            # Use unit variance for stability, scale later
            K = 1.0 * np.exp(-dist_sq / (2 * self.noise_length_scale**2))
            
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
        if self.discounted: 
            flags.append("disc")
        if self.semi_gradient: 
            flags.append("semi")
        if self.integral_td: 
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

    def train(self) -> dict:
        """Main training loop.
        
        Returns:
            Dictionary of training metrics
        """
        # Metric storage
        metrics_history = {
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
        
        iterator = tqdm.trange(self.n_episodes, desc="Training", leave=True)
        
        for episode in iterator:
            # Episode initialization
            self.episode = episode
            if not self.fix_initial_state:
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
            for _ in range(self.burning_steps):
                action = jnp.zeros(self.env.B.shape[1]) #burning with zero action
                t, x_t, _ = self.wrapper.step(self.wrapper.state, action) #type: ignore
                self.sliding_signature.append(x_t / self.scale)
            if self.preheat:
                for _ in range(self.sliding_signature.window_size):
                    if self.state_augmentation:
                        _ = jnp.concatenate([self.sliding_signature.current_signature, x_t / self.scale])  # type: ignore
                    else:
                        _ = self.sliding_signature.current_signature
                    # action, _, _ = self._select_action(state, self.env.step_size)
                    action = jnp.zeros(self.env.B.shape[1]) #preheat with zero action
                    t, x_t, _ = self.wrapper.step(self.wrapper.state, action) #type: ignore
                    self.sliding_signature.append(x_t / self.scale)
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
                        np.asarray(ctx.sig_t).flatten()
                    )
                    metrics_history['noise'].append(
                        np.asarray(ctx.noise).flatten()
                    )
            # Episode metrics (convert to float only at episode end)
            episode_metrics = {
                'loss': float(episode_loss),
                'cost': float(episode_cost),
                'actor_grad_mean': float(actor_grad_sum) / max(n_steps, 1),
                'critic_grad_mean': float(critic_grad_sum) / max(n_steps, 1),
                'n_steps': n_steps,
            }
            
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
            if episode >= self.eval_start_episode and episode % self.eval_interval == 0:
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
                
                if self.patience > 0 and self._patience_counter >= self.patience:
                    print(f"\n[Early stop] No improvement for {self.patience} evals. "
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
        buf = self.sliding_signature.buffer
        if hasattr(buf, '_data'):
            saved_buf = (buf._data.copy(), buf._count, buf._head)  # type: ignore[union-attr]
        else:
            from collections import deque
            saved_buf = deque(buf.buffer, maxlen=buf.size)  # type: ignore[union-attr]
        saved_sig = self.sliding_signature.current_signature
        
        self.key, subkey = jax.random.split(self.key)
        x_t = self.wrapper.reset(subkey, x0=np.array(x_init), t0=0.0)
        self._fill_buffer_initial()
        
        for _ in range(self.burning_steps):
            action = jnp.zeros(self.env.B.shape[1])
            _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
            self.sliding_signature.append(x_t / self.scale)
        if self.preheat:
            for _ in range(self.sliding_signature.window_size):
                action = jnp.zeros(self.env.B.shape[1])  # zero action, consistent with training
                _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
                self.sliding_signature.append(x_t / self.scale)
        
        total_reward = 0.0
        
        while not self._is_episode_done(x_t, time_only=True):
            if self.state_augmentation:
                sig_input = jnp.concatenate([self.sliding_signature.current_signature, x_t / self.scale])
            else:
                sig_input = self.sliding_signature.current_signature
            
            if self.actor_oracle:
                mu = -self.optimal_K @ jnp.array(self.wrapper.state.x)  # type: ignore
            else:
                mu = self.actor.apply(self.actor_params, sig_input) #type: ignore
            mu = jnp.clip(mu, -self.clip_action, self.clip_action)
            
            _, x_next, reward = self.wrapper.step(self.wrapper.state, mu)  # type: ignore
            if x_next.ndim == 0:
                x_next = jnp.array([x_next])
            self.sliding_signature.append(x_next / self.scale)
            total_reward += float(reward) * self.env.step_size
            x_t = x_next
        
        # Restore state
        self.wrapper.state = saved_state
        if hasattr(buf, '_data'):
            buf._data, buf._count, buf._head = saved_buf  # type: ignore[union-attr]
        else:
            buf.buffer = saved_buf  # type: ignore[union-attr]
        self.sliding_signature.current_signature = saved_sig
        
        return total_reward

    def save(self, filename: str):
        """Save agent parameters to a file."""
        save_dict = {
            'actor_params': self.actor_params,
            'critic_params': self.critic_params,
            'env_params': self.env_params,
            # Config objects (new-style)
            'training': self.training,
            'discount': self.discount,
            'noise': self.noise,
            'signature': self.signature,
            'network': self.network,
            'algorithm': self.algorithm,
            # Flat copies for backward-compat checkpoint loading
            'training_params': configs_to_flat_dict(
                self.training, self.discount, self.noise,
                self.signature, self.network, self.algorithm,
            ),
            'discounted': self.discounted,
            'semi_gradient': self.semi_gradient,
            'integral_td': self.integral_td,
            'fix_initial_state': self.fix_initial_state,
            'decay_noise': self.decay_noise,
            'time_augmentation': self.time_augmentation,
            'depth': self.depth,
        }
        # Save delayed_state flag if it exists (for CTACJAX)
        if hasattr(self, 'delayed_state'):
            save_dict['delayed_state'] = self.delayed_state # type: ignore
        
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Agent saved to {filename}")

    def load(self, filename: str):
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


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    
    # Simple 2D stable system (oscillator with damping)
    # dx/dt = A*x + B*u, naturellement stable sans contrôle
    A = np.array([[-0.1]])
    B = np.array([[1]])
    A1 = np.zeros_like(A)
    delay = np.array([0])
    x0 = np.array([1.0])
    Q = np.eye(1)
    R = np.eye(1)
    depth = 3
    
    env = JAXDDEEnv(A=A, B=B, Q=Q, R=R, A1=A1, delay=delay, step_size=0.05, resolution=5)
    
    agent = CTACSignatureJAX(
        env=env,
        training_params={
            'n_episodes': 1000,
            'max_time': 1.0,
            'sigma': 0.1,
            'actor_lr': 1e-1,
            'critic_lr': 1e-1,
            'scale': 1.0,
            'clip_gradient': 5.0,
            'clip_action': 5.0,
            'divergence_threshold': 50.0,
            'log_interval': 10,
            'init_log_interval': 50,
            'memory_clear_interval': 20,
        },
        Q=Q,
        R=R,
        depth=depth,
        discounted=False,
        semi_gradient=True,
        integral_td=False,
        fix_initial_state=True,
        decay_noise=True,
        time_augmentation=False,
        bias=False,
        actor_oracle=True,
        critic_oracle=False, 
        preheat=True, 
        state_augmentation=True,
        smooth_noise=True,
    )
    metrics = agent.train()

    import matplotlib.pyplot as plt
    plt.plot(metrics['cost_episodic'])
    plt.xlabel('Episode (x10)')
    plt.ylabel('Episodic Cost')
    plt.title('CTAC Signature Episodic Cost over Training')
    plt.show()

    plt.plot(metrics['loss_episodic'])
    plt.xlabel('Episode (x10)')
    plt.ylabel('Episodic Loss')
    plt.title('CTAC Signature Episodic Loss over Training')
    plt.show()

    plt.figure(figsize=(10, 6))
    critic_weights = np.array(metrics['critic_weights'])
    for i in range(critic_weights.shape[1]):  # Plot all weights
        plt.plot(critic_weights[:, i, 0], label=f'Weight {i}')
    print(critic_weights.shape)
    plt.xlabel('Episode (x10)')
    plt.ylabel('Critic Weight Value')
    plt.title('CTAC Signature Critic Weights over Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics['gradient_critic'])
    plt.xlabel('Episode (x10)')
    plt.ylabel('Critic Gradient Magnitude')
    plt.title('CTAC Signature Critic Gradient Magnitude over Training')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    signature_weights = np.array(metrics['signature_weights'])
    for i in range(min(6, signature_weights.shape[1])):  # Plot first 6 signature features
        plt.plot(signature_weights[:, i], label=f'Signature {i}', alpha=0.7)
    plt.xlabel('Step (x100)')
    plt.ylabel('Signature Feature Value')
    plt.title('CTAC Signature Features over Training')
    plt.legend()
    plt.grid(True)
    plt.show()