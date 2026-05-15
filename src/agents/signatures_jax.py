"""
Continuous-Time Actor-Critic (CTAC) - Modular Implementation

"""

import jax
import numpy as np
import tqdm
import pickle
from pathlib import Path
from src.utils.dynamic_signature import SlidingSignatureJAX
from src.networks.LQR_actor_critics import ActorFlax, ActorFlaxLayerNorm, CriticFlax, CriticFlaxLayerNorm
from src.utils.step_metrics import StepMetrics
from src.utils.step_context import StepContextSignature
from src.utils.state_counter import StateCounter
from src.configs import (
    TrainingConfig, DiscountConfig, NoiseConfig,
    SignatureConfig, NetworkConfig, AlgorithmConfig,
    configs_to_flat_dict,
)
import jax.numpy as jnp
import optax
import scipy


from src.envs.env_rk_jax import JAXDDEEnv, JAXEnvWrapper


class CTACSignatureJAX:
  
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
        eval_callback=None,
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
        self._init_oracle()
        self._init_env(env, rng_key)
        self._init_episode_state(x0, eval_callback=eval_callback)
        self._init_checkpoints()
        self._init_signatures()
        self._init_networks()
        
        # JIT-compiled update functions (created once)
        self._jit_critic_update = self._make_critic_update_fn()
        self._jit_actor_update = self._make_actor_update_fn()
        self._jit_select_action = self._make_select_action_fn()
        self._jit_compute_values = self._make_compute_values_fn()

    def _init_env(self, env: JAXDDEEnv, rng_key: int) -> None:
        self.env = env
        self.wrapper = JAXEnvWrapper(env, rng_key=jax.random.PRNGKey(rng_key))
        self.episode = 0
        if self.network.normalize_entries:
            self.training.scale = self.training.divergence_threshold
            print("normalize_entries is True: scaling states by ", self.training.scale)
            print("normalize_sigs is ", self.network.normalize_sigs)

    def _init_oracle(self)-> None:
        # Oracle setup
        if self.algorithm.actor_oracle:
            P = scipy.linalg.solve_continuous_are(
                self.env.A, self.env.B, self.env.Q, self.env.R
            )
            print("P matrix for optimal LQR policy:\n", P)
            self.P = jnp.array(P)
            self.optimal_K = jnp.array(np.linalg.inv(self.env.R) @ self.env.B.T @ P)

        if self.algorithm.critic_oracle:
            P = scipy.linalg.solve_continuous_are(
                self.env.A, self.env.B, self.env.Q, self.env.R
            )
            self.P = jnp.array(P)
            print("P matrix for optimal value function:\n", P)
    
    def _init_episode_state(self, x0: jnp.ndarray | None, eval_callback=None) -> None:
        self.step_counter = 0
        self.current_noise = None
        self.episode_noise_trajectory = None
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
        self.sliding_signature = SlidingSignatureJAX(
            depth=self.signature_conf.depth, window_size=self.signature_conf.window_size, d=self.env.N,
            time_augmentation=self.signature_conf.time_augmentation,
            origin_augmentation=self.signature_conf.origin_augmentation,
            time_origin=self.signature_conf.time_origin, bias=self.signature_conf.bias,
        )
        self._sigma_effective = self.noise.sigma

    def _init_networks(self) -> None:
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        key_a, key_c = jax.random.split(self.key)
        if self.signature_conf.state_augmentation:
            self.actor_params = self.actor.init(key_a, jnp.zeros(self.sliding_signature.signature_size + self.env.N))
            self.critic_params = self.critic.init(key_c, jnp.zeros(self.sliding_signature.signature_size + self.env.N))
        else:
            self.actor_params = self.actor.init(key_a, jnp.zeros(self.sliding_signature.signature_size))
            self.critic_params = self.critic.init(key_c, jnp.zeros(self.sliding_signature.signature_size))

        #optimizers — absorb dt into learning rate for correct continuous-time scaling
        self.actor_optimizer = optax.adam(self.training.actor_lr * self.env.step_size, b1=0.1)
        self.critic_optimizer = optax.adam(self.training.critic_lr * self.env.step_size)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

    # =========================================================================
    # Network Construction
    # =========================================================================
    
    def _build_actor(self) -> ActorFlax | ActorFlaxLayerNorm:
        """Build the actor network."""
        output_dim = self.env.B.shape[1]
        if self.network.normalize_sigs:
            actor = ActorFlaxLayerNorm(output_dim=output_dim, stddev=self.network.std_init/10)
        else:
            actor = ActorFlax(output_dim=output_dim, stddev=self.network.std_init/10)
        return actor

    def _build_critic(self) -> CriticFlax | CriticFlaxLayerNorm:
        """Build the critic network."""       
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
        """Fill the signature buffer with initial states from wrapper."""
        self.sliding_signature.reset()
        if self.wrapper.state is not None:
            _, history_data = self.wrapper.initial_conditions
            # Subsample by resolution to match window size
            # Apply scaling to be consistent with training
            for x in history_data[::self.env.resolution]:
                self.sliding_signature.append(x / self.training.scale)
            self.sliding_signature._signature_dirty = True

    # =========================================================================
    # Action Selection
    # =========================================================================
    
    def _make_select_action_fn(self):
        """Create JIT-compiled action selection function."""
        actor = self.actor
        clip_action = self.training.clip_action
        smooth_noise = self.noise.smooth  # Capture static config
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
        if self.noise.schedule == 'adaptive':
            V_t = self._compute_value_function(self.sliding_signature.current_signature)
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
            mu = - self.optimal_K @ jnp.array(self.wrapper.state.x) #type: ignore
            self.key, subkey = jax.random.split(self.key)
            noise = jnp.zeros_like(mu)
            action = mu + noise
            self.current_noise = jnp.zeros_like(noise)
        else:
            if self.current_noise is None:
                 self.current_noise = jnp.zeros(self.env.B.shape[1])
            
            # Prepare explicit noise value if GP mode is active
            explicit_noise_val = jnp.zeros_like(self.current_noise)
            if self.noise.smooth and self.episode_noise_trajectory is not None:
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
        if self.algorithm.integral_td:
            # Integral form: δ = r·dt + ΔV (- V·dt/τ if discounted)
            delta_V = ctx.V_next - ctx.V_t
            td = ctx.reward * ctx.dt + delta_V
            if self.discount.discounted:
                td -= ctx.V_t * ctx.dt / self.discount.tau
            return td
        else:
            # Differential form: δ = r + V̇ (- V/τ if discounted)
            td = ctx.reward + ctx.V_dot
            if self.discount.discounted:
                td -= ctx.V_t / self.discount.tau
            return td

    # =========================================================================
    # Critic Gradient (flag-based)
    # =========================================================================
    
    
    def critic_loss_fn(self, sig_t, sig_next,x, x_next, reward, dt): 
        if self.signature_conf.state_augmentation:
            sig_t = jnp.concatenate([sig_t, jnp.array(x)])  # type: ignore
            sig_next = jnp.concatenate([sig_next, jnp.array(x_next)])  # type: ignore
        V_t_raw = self.critic.apply(self.critic_params, sig_t)
        V_next_raw = jax.lax.stop_gradient(self.critic.apply(self.critic_params, sig_next))
        V_t = jnp.asarray(V_t_raw[0] if isinstance(V_t_raw, tuple) else V_t_raw).squeeze()
        V_next = jnp.asarray(V_next_raw[0] if isinstance(V_next_raw, tuple) else V_next_raw).squeeze()
        target = (reward + (V_next - V_t) / dt) # type: ignore
        td_error = target
        if self.discount.discounted:
            td_error -= V_t / self.discount.tau
        loss = 0.5 * td_error ** 2
        return loss, td_error


    # =========================================================================
    # Actor Gradient
    # =========================================================================
    
    def actor_loss_fn(self, sig_t, td_error, noise):
        if self.signature_conf.state_augmentation:
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
            done = self.wrapper.state.t >= self.training.max_time
        else:
            done = jnp.logical_or(
                self.wrapper.state.t >= self.training.max_time,
                jnp.linalg.norm(x) > self.training.divergence_threshold,
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
        x_scaled = x_t / self.training.scale
        if self.signature_conf.state_augmentation:
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
        x_next_scaled = x_next / self.training.scale
        dt = self.env.step_size
        if dt <= 1e-9:
            dt = 1e-4  # Avoid division by zero
        sig_t = self.sliding_signature.current_signature
        self.sliding_signature.append(x_next_scaled)
        sig_next = self.sliding_signature.current_signature
        if self.signature_conf.state_augmentation:
            sig_t = jnp.concatenate([sig_t, x_scaled])  # type: ignore
            sig_next = jnp.concatenate([sig_next, x_next_scaled])  # type: ignore
        # Value function evaluations (JIT-compiled, no float() sync)
        if self.algorithm.critic_oracle:
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
        ctx.td_error = ctx.reward + ctx.V_dot - (ctx.V_t / self.discount.tau if self.discount.discounted else 0.0)
        
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
        tau = self.discount.tau
        
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
        if not self.algorithm.critic_oracle:
            self.critic_params, self.critic_opt_state, c_loss, td_error, critic_grad_norm = \
                self._jit_critic_update(
                    self.critic_params, self.critic_opt_state, 
                    sig_t, sig_next, reward, dt
                )
        
        # Actor update (JIT-compiled)
        if not self.algorithm.actor_oracle and (step % self.algorithm.actor_update_frequency == 0):
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
        if self.noise.smooth:
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
        self.sliding_signature.append(x / self.training.scale)
        if hasattr(self, '_path_data_dirty'):
            setattr(self, '_path_data_dirty', True)
            
    def get_eval_action(self, x_scaled: jnp.ndarray) -> jnp.ndarray:
        if getattr(self.algorithm, 'actor_oracle', False):
            assert self.wrapper.state is not None
            action = -self.optimal_K @ jnp.array(self.wrapper.state.x)
        else:
            sig = self.sliding_signature.current_signature
            assert self.actor_params is not None
            action = self.actor.apply(self.actor_params, sig)
            
        action = jnp.asarray(action)
        action = jnp.clip(action, -self.training.clip_action, self.training.clip_action)
        return jnp.array(action)

    def get_value(self) -> float:
        """Compute value function for a given state."""
        if getattr(self.algorithm, 'critic_oracle', False):
            assert self.wrapper.state is not None
            x_val = jnp.array(self.wrapper.state.x)
            return float(-x_val.T @ self.P @ x_val)
            
        sig = self.sliding_signature.current_signature
        
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
            for _ in range(self.algorithm.burning_steps):
                action = jnp.zeros(self.env.B.shape[1]) #burning with zero action
                t, x_t, _ = self.wrapper.step(self.wrapper.state, action) #type: ignore
                self.sliding_signature.append(x_t / self.training.scale)
            if self.algorithm.preheat:
                for _ in range(self.sliding_signature.window_size):
                    if self.signature_conf.state_augmentation:
                        _ = jnp.concatenate([self.sliding_signature.current_signature, x_t / self.training.scale])  # type: ignore
                    else:
                        _ = self.sliding_signature.current_signature
                    # action, _, _ = self._select_action(state, self.env.step_size)
                    action = jnp.zeros(self.env.B.shape[1]) #preheat with zero action
                    t, x_t, _ = self.wrapper.step(self.wrapper.state, action) #type: ignore
                    self.sliding_signature.append(x_t / self.training.scale)
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
        
        for _ in range(self.algorithm.burning_steps):
            action = jnp.zeros(self.env.B.shape[1])
            _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
            self.sliding_signature.append(x_t / self.training.scale)
        if self.algorithm.preheat:
            for _ in range(self.sliding_signature.window_size):
                action = jnp.zeros(self.env.B.shape[1])  # zero action, consistent with training
                _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
                self.sliding_signature.append(x_t / self.training.scale)
        
        total_reward = 0.0
        
        while not self._is_episode_done(x_t, time_only=True):
            if self.signature_conf.state_augmentation:
                sig_input = jnp.concatenate([self.sliding_signature.current_signature, x_t / self.training.scale])
            else:
                sig_input = self.sliding_signature.current_signature
            
            if self.algorithm.actor_oracle:
                mu = -self.optimal_K @ jnp.array(self.wrapper.state.x)  # type: ignore
            else:
                mu = self.actor.apply(self.actor_params, sig_input) #type: ignore
            mu = jnp.clip(mu, -self.training.clip_action, self.training.clip_action)
            
            _, x_next, reward = self.wrapper.step(self.wrapper.state, mu)  # type: ignore
            if x_next.ndim == 0:
                x_next = jnp.array([x_next])
            self.sliding_signature.append(x_next / self.training.scale)
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
