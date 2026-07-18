from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import pickle
from pathlib import Path

from src.envs.env_rk_jax import JAXEnvWrapper, JAXDDEEnv
from src.networks.LQR_actor_critics import CriticFlax, CriticFlaxLayerNorm
from src.utils.dynamic_signature import DequeBuffer
from src.representations.factory import make_representation, RepresentationBuffer
from src.utils.optim import build_adam
from src.utils.step_context import StepContextSignature
from src.utils.step_metrics import StepMetrics
from src.utils.state_counter import StateCounter
from src.utils.clamp_reporting import report_clamp_activation
from src.utils.exploration_noise import sample_ou_trajectory
from src.configs import (
    TrainingConfig, DiscountConfig, NoiseConfig,
    SignatureConfig, NetworkConfig, AlgorithmConfig,)

class ContinuousValueGradient:
    def __init__(self,
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
        self.eval_callback = eval_callback

        self._init_env(env, rng_key, x0)
        self._init_episode_state()
        self._init_checkpoint_state()
        self._init_networks()

    def _init_env(self, env: JAXDDEEnv, rng_key: int, x0: jnp.ndarray | None) -> None:
        self.env = env
        self.wrapper = JAXEnvWrapper(env, rng_key=jax.random.PRNGKey(rng_key))
        self.Q = env.Q
        self.R = env.R
        self.key = jax.random.PRNGKey(rng_key)
        self.dt = env.step_size
        self.x0 = jnp.array(x0) if x0 is not None else None
        if not self.signature_conf.force_signature_window:
            self.signature_conf.window_size = self._compute_window_size()

    def _compute_window_size(self) -> int:
        max_delay = self.env.delay.max() if self.env.delay is not None else 0.0
        return int(np.ceil(max_delay / self.env.step_size)) + 3 if max_delay > 0 else 10

    def _init_episode_state(self) -> None:
        self.episode = 0
        self.step_counter = 0
        self._sigma_effective: float | jax.Array = self.noise.sigma
        self.episode_noise_trajectory: jax.Array | None = None
        self.current_noise = jnp.zeros(self.env.B.shape[1])
        self._cached_path_data: jnp.ndarray | None = None
        self._path_data_dirty = True

    def _init_checkpoint_state(self) -> None:
        self._best_eval_cost = -float('inf')
        self._best_critic_params = None
        self._best_target_params = None
        self._best_episode = 0
        self._patience_counter = 0
        self.discretization_state = self.training.discretization_state
        self.state_counter = StateCounter(resolution=self.training.discretization_state)

    def _init_networks(self) -> None:
        # Representation backbone: signature / raw_history / markovian, behind a
        # window buffer with a SlidingSignatureJAX-compatible surface so the rest of
        # this agent is unchanged. The feature map is the linear-readout input and
        # the function the value-gradient control law differentiates (dV/dx(t)).
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
        self.sliding_signature = RepresentationBuffer(
            representation, window_length=window_length, n_state=self.env.N,
        )
        self.critic = self._build_network()
        self.target = self._build_network()
        # Gradient clipping is opt-in via training.clip_gradient (global-norm, off by default).
        # Robbins-Monro schedule on the critic: for the value gradient the control IS the critic
        # gradient, so the critic is the policy and this is the canonical value-function RM
        # (single-timescale). Per-step k. critic_lr_decay_power = 0 (default) is a constant rate.
        self.optimizer = build_adam(
            self.training.critic_lr * self.env.step_size,
            clip_gradient=self.training.clip_gradient,
            decay_power=float(getattr(self.training, "critic_lr_decay_power", 0.0)))

        key_critic, key_target, self.key = jax.random.split(self.key, 3)
        self.critic_params = self.critic.init(key_critic, jnp.zeros((self.sliding_signature.signature_size,)))
        self.target_params = self.target.init(key_target, jnp.zeros((self.sliding_signature.signature_size,)))
        self.critic_opt_state = self.optimizer.init(self.critic_params)

        self._gradient_value_fn = self.get_gradient_value_fn()
        self._select_action_jit = self._make_action_jit_fn()
        self._jit_critic_update = self._make_critic_update_fn()

        # LSTD critic solver (opt-in). Accumulators for the continuous-time TD fixed
        # point E[phi*delta]=0; solved directly each episode end (see _on_episode_end).
        self._lstd = bool(getattr(self.algorithm, "lstd", False))
        if self._lstd:
            d = int(self.sliding_signature.signature_size)
            self._lstd_M = np.zeros((d, d))
            self._lstd_b = np.zeros(d)
            self._lstd_mu = np.zeros(d)        # running feature mean (centring vector)
            self._lstd_phisum = np.zeros(d)    # accumulates this episode's feature sum
            self._lstd_C = np.zeros((d, d))    # centred feature 2nd moment (for truncation PCA)
            self._lstd_n = 0.0
            self._lstd_reg = float(getattr(self.algorithm, "lstd_reg", 1e-3))
            self._lstd_forget = float(getattr(self.algorithm, "lstd_forget", 0.7))
            self._lstd_rank = int(getattr(self.algorithm, "lstd_rank", 0))

    def _build_network(self):
        if self.network.normalize_layers:
            return CriticFlaxLayerNorm(hidden_dims=self.network.hidden_dims, stddev=self.network.std_init)
        else:
            return CriticFlax(hidden_dims=self.network.hidden_dims, stddev=self.network.std_init)
    
    def _sample_initial_state(self) -> jnp.ndarray:
        """Sample initial state for an episode."""
        self.key, subkey = jax.random.split(self.key) 
        return jax.random.normal(subkey, shape=(self.env.N,))
    
    def _compute_value_function(self, sig: jnp.ndarray) -> jnp.ndarray:
        """Compute value function from signature."""
        return self.critic.apply(self.critic_params, sig) # type: ignore

    def _fill_buffer_initial(self):
        """Fill the signature buffer to FULL capacity with initial states.
        
        IMPORTANT: This ensures the buffer is always at window_size+1 to avoid
        JAX recompilation due to shape changes.
        """
        self.sliding_signature.reset()
        if self.wrapper.state is not None:
            _, history_data = self.wrapper.initial_conditions
            # Subsample by resolution
            subsampled = history_data[::self.env.resolution]
            
            # Ensure we fill the buffer completely (window_size + 1 elements)
            target_len = self.sliding_signature.window_size + 1
            # float64, not float32 (finding F-E1, commits 07b8d4a/7b1153e moved the buffers to
            # float64 and missed this path). The pipeline runs under jax_enable_x64
            # (main_unified.py:63), and the buffer's own dtype is float64, so the previous float32
            # cast narrowed ONLY the initial history and then upcast the already-rounded values
            # back to float64. Keeping it made the value gradient and the actor-critic start from
            # initial paths differing by ~1.2e-8 (a confound in the learner comparison), and
            # rounded the very quantity whose conditioning this study measures (raw_history's Gram
            # on platoon is ~3320-dimensional and near-singular; float32 rounding in the gradient
            # extraction can change the answer). Removed. The existing main_unified runs predate
            # this and are reproducible from their own commit, not from HEAD.
            if len(subsampled) >= target_len:
                # Use the last target_len elements
                for x in subsampled[-target_len:]:
                    self.sliding_signature.buffer.append(np.asarray(x/self.training.scale, dtype=np.float64)) #type: ignore
            else:
                # Pad with first element to reach target_len
                first_val = np.asarray(subsampled[0]/self.training.scale, dtype=np.float64)
                padding_needed = target_len - len(subsampled)
                for _ in range(padding_needed):
                    self.sliding_signature.buffer.append(first_val) #type: ignore
                for x in subsampled:
                    self.sliding_signature.buffer.append(np.asarray(x/self.training.scale, dtype=np.float64)) #type: ignore
            
            self.sliding_signature.current_signature = self.sliding_signature.compute_signature()
        self._path_data_dirty = True  # Invalidate cache
    
    
    def get_gradient_value_fn(self):
        critic = self.critic
        sig_fn = self.sliding_signature._jit_compute_sig
        @jax.jit
        def grad_V_fn_wrt_path(critic_params, path_data):
            def value_from_path(p):
                sig = sig_fn(p)
                return critic.apply(critic_params, sig)
            grad_path = jax.grad(value_from_path)(path_data)
            return grad_path
        return grad_V_fn_wrt_path
    
    def _make_action_jit_fn(self):
        grad_V_fn = self._gradient_value_fn
        get_B_fn = self.env.get_B
        clip_action = self.training.clip_action
        # Action clipping is opt-in: clip_action None/<=0 means NO clipping (default
        # preference). For a well-initialised critic the value-gradient control law
        # u = 1/2 R^-1 B^T dV/dx is naturally bounded (dV/dx ~ 0 at init), so no clip
        # is needed; the flag is only a safeguard for diverging plants.
        do_clip = clip_action is not None and clip_action > 0
        @jax.jit
        def select_action(critic_params, path_data, R, x_current):
            B = get_B_fn(x_current)
            grad_path = grad_V_fn(critic_params, path_data)
            end_gradient = grad_path[-1]
            u = 1/2*jnp.linalg.inv(R) @ B.T @ end_gradient #Doya LQR development
            if do_clip:
                u = jnp.clip(u, -clip_action, clip_action)
            return u, end_gradient
        return select_action
    
    def _get_path_data(self) -> jnp.ndarray:
        """Get path data as JAX array, using cached version if available."""
        if self._cached_path_data is None or self._path_data_dirty:
            self._cached_path_data = jnp.array(self.sliding_signature.buffer.to_array())
            self._path_data_dirty = False
        return self._cached_path_data
    
    def _select_action(self):
        # --- Compute effective noise level based on schedule ---
        if self.noise.schedule == 'adaptive':
            # Adaptive: scale sigma based on V(current) relative to running stats
            V_t = self._compute_value_function(self.sliding_signature.current_signature)
            V_TARGET = self.discount.V_target
            V_BAD = self.discount.V_bad
            noise_scale = jnp.clip((V_TARGET - V_t) / (V_TARGET - V_BAD + 1e-6), 0.1, 1.0)
            self._sigma_effective = self.noise.sigma * noise_scale
        elif self.noise.schedule == 'linear_decay':
            # Linear decay from sigma to sigma_min over training
            progress = min(self.episode / max(self.training.n_episodes, 1), 1.0)
            self._sigma_effective = self.noise.sigma * max(0.2, 1.0 - 0.9 * progress)
        else:
            # 'constant' - fixed sigma throughout
            self._sigma_effective = self.noise.sigma
        
        pre_sampled = getattr(self.noise, "ou", False) or self.noise.smooth
        explicit_noise_val = jnp.zeros(self.env.B.shape[1])
        if pre_sampled and self.episode_noise_trajectory is not None:
            t_idx = int(round(self.wrapper.state.t / self.env.step_size)) #type: ignore
            # Clamp to avoid overflow
            t_idx = min(t_idx, len(self.episode_noise_trajectory) - 1)
            explicit_noise_val = self.episode_noise_trajectory[t_idx]
        path_data = self._get_path_data()
        mu, end_gradient = self._select_action_jit(
                self.critic_params, path_data, self.R, self.wrapper.state.x) #type: ignore
        if pre_sampled:
            noise = explicit_noise_val
        else:
            self.key, subkey = jax.random.split(self.key)
            noise = jax.random.normal(subkey, shape=mu.shape) * self._sigma_effective
        action = mu + noise
        return action, mu, noise, end_gradient

    
    def _make_critic_update_fn(self):
        """Create a JIT-compiled critic update function."""
        critic = self.critic
        optimizer = self.optimizer
        discounted = self.discount.discounted
        tau = self.discount.tau
        tau_polyak = self.training.tau_polyak
        
        def critic_loss(critic_params, target_params, sig_t, sig_next, reward, dt):
            V_t = critic.apply(critic_params, sig_t).squeeze() # type: ignore
            V_next = critic.apply(jax.lax.stop_gradient(target_params), sig_next).squeeze() # type: ignore
            td_error = reward + (V_next - V_t) / dt
            if discounted:
                td_error = td_error - V_t / tau
            return 0.5 * td_error ** 2 * dt, td_error
        @jax.jit
        def update_fn(critic_params, target_params, opt_state, sig_t, sig_next, reward, dt):
            (loss, td_error), grads = jax.value_and_grad(critic_loss, has_aux=True)(
                critic_params, target_params, sig_t, sig_next, reward, dt
            )
            # Gradient clipping (if enabled) is handled by the optimizer (global-norm).
            updates, new_opt_state = optimizer.update(grads, opt_state, critic_params)
            
            new_params_critic = optax.apply_updates(critic_params, updates)
            new_params_target = optax.incremental_update(
                new_params_critic, target_params, step_size=tau_polyak)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            return new_params_critic, new_params_target, new_opt_state, loss, td_error, grad_norm
        return update_fn
    

    def _update_networks(self, ctx: StepContextSignature) -> tuple:
        """Update actor and critic networks using JIT-compiled JAX autodiff."""
        self.step_counter += 1
        # Signatures are already JAX arrays from sliding_signature
        sig_t = ctx.sig_t
        sig_next = ctx.sig_next
        reward = ctx.reward
        dt = ctx.dt
        if getattr(self, "_lstd", False):
            # Accumulate the LSTD system; the critic is solved at episode end. The
            # continuous-time TD residual is delta = r + theta^T[(phi_next-phi_t)/dt
            # - phi_t/tau], so the fixed point E[phi*delta]=0 gives M theta = -b with
            # M = sum phi_t[(phi_next-phi_t)/dt - phi_t/tau]^T, b = sum phi_t r.
            #
            # CENTRED features phi_c = phi - mu (running mean, frozen per episode). The
            # signature's time-augmentation channel contributes a large DETERMINISTIC
            # constant block (the clock advances identically every window), which is the
            # additive-constant gauge of the value function. In a bias-free critic it
            # dominates the feature second moment (effective rank -> 1, cond ~1e15) and is
            # an undamped TD direction. Centring removes it (gauge fix) WITHOUT changing the
            # affine span, so the Arribas density property is preserved. The value-gradient
            # control uses d/dx (theta^T phi_c) = theta^T dphi/dx (mu is constant), so the
            # kernel solved here drives the control unchanged.
            phi_t = np.asarray(sig_t, dtype=np.float64).reshape(-1)
            phi_n = np.asarray(sig_next, dtype=np.float64).reshape(-1)
            phi_c = phi_t - self._lstd_mu                          # centred test feature
            diff = (phi_n - phi_t) / float(dt) - phi_c / float(self.discount.tau)
            self._lstd_M += np.outer(phi_c, diff)
            self._lstd_b += phi_c * float(reward)
            self._lstd_C += np.outer(phi_c, phi_c)                 # centred 2nd moment (PCA)
            self._lstd_phisum += phi_t                             # for next episode's mean
            self._lstd_n += 1.0
            zero = jnp.array(0.0)
            return zero, zero
        # Critic update (JIT-compiled)
        self.critic_params, self.target_params, self.critic_opt_state, c_loss, td_error, critic_grad_norm = \
            self._jit_critic_update(
                self.critic_params, self.target_params, self.critic_opt_state,
                sig_t, sig_next, reward, dt)
        # Return JAX arrays - defer float() to episode end
        return c_loss, critic_grad_norm
    
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
        # Action selection
        dt = self.env.step_size
        action, mu, noise, end_gradient = self._select_action() 
        
        # Environment step (using wrapper with current state)
        t, x_next, reward = self.wrapper.step(self.wrapper.state, action) # type: ignore
        # x_next is already a JAX array from wrapper.step
        if x_next.ndim == 0:
            x_next = x_next.reshape(1)
        x_next_scaled = x_next / self.training.scale
        dt = self.env.step_size
        if dt <= 1e-9:
            dt = 1e-4  # Avoid division by zero
        sig_t = self.sliding_signature.current_signature
        self.sliding_signature.append(x_next_scaled)
        self._path_data_dirty = True  # Invalidate cache after append
        sig_next = self.sliding_signature.current_signature
        # Note: V_t and V_next computation removed - not needed for training
        # They are computed inside the JIT-compiled critic update
        
        # Build context (lightweight - just references, no computation)
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
            V_t=0.0,  # Not needed for training
            V_next=0.0,  # Not needed for training
            sig_t=sig_t,
            sig_next=sig_next
        )
        
        # Compute reward
        ctx.reward = reward #type: ignore
        
        # Compute V dot and TD error - deferred, not needed for core training
        ctx.V_dot = 0.0
        ctx.td_error = 0.0
        
        # update networks
        loss_critic, critic_grad_norm = self._update_networks(ctx)

        # Keep as JAX arrays to avoid CPU sync - only convert at episode end
        actor_grad_norm = jnp.linalg.norm(end_gradient)
        metrics = StepMetrics(
            loss=loss_critic * dt,
            reward=ctx.reward * dt,
            actor_gradient=actor_grad_norm,
            critic_gradient=critic_grad_norm,
        )
        
        return x_next, metrics, ctx
    


    def _on_episode_start(self, episode: int, x_init: np.ndarray) -> None:
        """Hook called at the start of each episode. Override for custom logic."""
        if getattr(self, "_lstd", False):
            # Refresh the centring mean from the previous episode's features (frozen
            # during the upcoming episode so the LSTD system is consistent), then apply
            # exponential forgetting so it tracks the (improving) policy.
            if self._lstd_n > 0:
                self._lstd_mu = self._lstd_phisum / self._lstd_n
            self._lstd_phisum = np.zeros_like(self._lstd_phisum)
            self._lstd_n = 0.0
            self._lstd_M *= self._lstd_forget
            self._lstd_b *= self._lstd_forget
            self._lstd_C *= self._lstd_forget
        if getattr(self.noise, "ou", False):
            # Ornstein-Uhlenbeck exploration (Doya 2000): correlation exp(-dt/tau_n) over a step, a
            # function of physical time and independent of dt. Pre-sampled for the whole episode so
            # the per-step select_action can index into it (same contract as the GP path below).
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
    
    def update_buffer(self, x: np.ndarray) -> None:
        self.sliding_signature.append(x / self.training.scale)
        if hasattr(self, '_path_data_dirty'):
            setattr(self, '_path_data_dirty', True)
            
    def get_eval_action(self, x_scaled: jnp.ndarray) -> jnp.ndarray:
        data_path = jnp.array(self.sliding_signature.buffer.to_array())
        assert self.wrapper.state is not None
        assert self.critic_params is not None
        action, _ = self._select_action_jit(self.critic_params, data_path, self.env.R, self.wrapper.state.x)
        return jnp.array(action)

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
            'all_grads_sig': [],
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
            # The DDE initial condition is the initial PATH phi on [-tau, 0], already
            # loaded into the signature window by _fill_buffer_initial() from the env
            # history buffer (set at reset via history_function, or constant x_init).
            # Control begins at t=0 from phi: NO zero-control burn-in, which would
            # overwrite phi with an uncontrolled-evolution path (the system would drift
            # into its attractor before control) and advance the episode clock.
            self._t_episode_start = float(self.wrapper.state.t)  # = t0 = 0
            # Episode accumulators (as JAX arrays to avoid sync)
            episode_loss = jnp.array(0.0)
            episode_cost = jnp.array(0.0)
            actor_grad_sum = jnp.array(0.0)
            critic_grad_sum = jnp.array(0.0)
            all_sigs_grads = []
            n_steps = 0
            
            # Episode loop
            while not self._is_episode_done(x_t):
                x_t, step_metrics, ctx = self._train_step(x_t)
                
                # Accumulate as JAX arrays (no sync)
                episode_loss = episode_loss + step_metrics.loss
                episode_cost = episode_cost + step_metrics.reward
                critic_grad_sum = critic_grad_sum + step_metrics.critic_gradient
                actor_grad_sum = actor_grad_sum + step_metrics.actor_gradient
                all_sigs_grads.append(step_metrics.actor_gradient)
                n_steps += 1
                
                # Memory management
                if n_steps % memory_clear_interval == 0:
                    self.wrapper._data.clear()
                    self.wrapper._time.clear()
                # Log detailed metrics less frequently to avoid overhead
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
            
            self._on_episode_end(episode, episode_metrics)
            
            # --- NaN detection: stop training and restore best checkpoint ---
            if (np.isnan(episode_metrics['loss']) or np.isnan(episode_metrics['cost'])
                    or any(jnp.any(jnp.isnan(p)) for p in jax.tree_util.tree_leaves(self.critic_params))):
                print(f"\n[NaN detected] at episode {episode}. Stopping training.")
                break
            
            # Periodic trajectory evaluation (wandb slider)
            if self.eval_callback is not None:
                self.eval_callback(self, episode)
            
            # Update progress bar
            iterator.set_description(self._format_progress(episode, episode_metrics))
            
            # Log metrics
            if episode % log_interval == 0:
                metrics_history['loss_episodic'].append(episode_loss)
                metrics_history['cost_episodic'].append(episode_cost)
                metrics_history['gradient_actor'].append(episode_metrics['actor_grad_mean'])
                metrics_history['gradient_critic'].append(episode_metrics['critic_grad_mean'])
                # Extract weights from Flax params
                critic_w = np.array(self.critic_params['params']['Dense_0']['kernel']).copy()
                if episode % 50 == 0:
                    metrics_history['critic_weights'].append(critic_w)
                metrics_history['all_grads_sig'].append(np.array(all_sigs_grads))  # No actor in value gradient
            
            # --- Periodic noiseless evaluation & best checkpoint ---
            if episode % self.training.eval_interval == 0:
                eval_cost = self._evaluate_noiseless()
                metrics_history.setdefault('eval_cost', []).append(eval_cost)
                metrics_history.setdefault('eval_episodes', []).append(episode)
                
                if eval_cost > self._best_eval_cost: #CAREFUL, cost is in fact reward here, so higher is better
                    self._best_eval_cost = eval_cost
                    self._best_critic_params = jax.tree_util.tree_map(
                        lambda x: x.copy(), self.critic_params)
                    self._best_target_params = jax.tree_util.tree_map(
                        lambda x: x.copy(), self.target_params)
                    self._best_episode = episode
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1
                
                if self.training.patience > 0 and self._patience_counter >= self.training.patience:
                    print(f"\n[Early stop] No improvement for {self.training.patience} evals. "
                          f"Best at ep {self._best_episode} (cost={self._best_eval_cost:.4f})")
                    break
        
        # Restore best checkpoint
        if self._best_critic_params is not None:
            final_cost = float(metrics_history['cost_episodic'][-1]) if metrics_history['cost_episodic'] else float('inf')
            improvement = ((final_cost - self._best_eval_cost) 
                          / (abs(self._best_eval_cost) + 1e-8) * 100)
            print(f"\n[Best checkpoint] Restoring params from episode {self._best_episode} "
                  f"(eval_cost={self._best_eval_cost:.4f}, "
                  f"final_cost={final_cost:.4f}, "
                  f"delta={-improvement:+.1f}%)")
            self.critic_params = self._best_critic_params
            self.target_params = self._best_target_params
        
        return metrics_history


    def _is_episode_done(self, x: jnp.ndarray, time_only: bool = False) -> bool:
        """Check if episode should terminate."""
        t = float(self.wrapper.state.t) if self.wrapper.state is not None else 0.0
        # max_time measures CONTROLLED time: the horizon clock (_t_episode_start) is
        # set at the start of each rollout's control loop. Control begins at t=0 from
        # the initial path (no burn-in), so _t_episode_start = 0; the elapsed-time form
        # is kept so the horizon is robust to any pre-control stepping a caller adds.
        if t - getattr(self, "_t_episode_start", 0.0) >= self.training.max_time:
            return True
        if not time_only:
            # NON-FINITE STATE. This is a FAILURE GUARD, not a trim: a trajectory containing a NaN
            # or an infinity is already mathematically over -- there is nothing left to integrate,
            # every subsequent state is non-finite, and every downstream number is meaningless. It
            # therefore always fires, is never opt-out, and announces itself loudly, so that a
            # diverging run is DIAGNOSED rather than silently producing nan.
            if not bool(jnp.all(jnp.isfinite(x))):
                n_nan = int(jnp.sum(jnp.isnan(x)))
                n_inf = int(jnp.sum(jnp.isinf(x)))
                report_clamp_activation(
                    "nonfinite_state_termination/value_gradient",
                    code_location="src/agents/value_gradient_jax.py:_is_episode_done",
                    bound_description="the state must be finite (no NaN, no inf component)",
                    most_extreme_raw_value=float("nan") if n_nan else float("inf"),
                    number_of_affected_elements=n_nan + n_inf,
                    additional_context=(
                        f"at t={t:.4f}: {n_nan} NaN and {n_inf} infinite component(s). The plant "
                        f"has diverged; this is the failure, not a truncation. Every number this "
                        f"episode reports downstream is meaningless."
                    ),
                    alters_the_value=False,  # a detector, not an intervention
                )
                return True
            # DIVERGENCE BOUND -- OPT-IN, and off by default. None/<=0 means the episode runs to
            # its horizon whatever ||x|| does. A cut episode's accumulated cost is not the cost of
            # a completed one, so this bound edits the objective it is meant to measure; and a run
            # rescued from divergence cannot be diagnosed. Kept only as a compute safeguard for a
            # caller who deliberately opts in, and it announces itself when it binds.
            bound = self.training.divergence_threshold
            if bound is not None and bound > 0:
                state_norm = float(jnp.linalg.norm(x))
                if state_norm > bound:
                    report_clamp_activation(
                        "divergence_threshold/value_gradient",
                        code_location="src/agents/value_gradient_jax.py:_is_episode_done",
                        bound_description=f"||x|| <= {bound}",
                        most_extreme_raw_value=state_norm,
                        additional_context=(
                            f"at t={t:.4f}; the episode is CUT here, so its accumulated cost "
                            f"covers less than the full horizon max_time={self.training.max_time} "
                            f"and is not comparable with a completed episode's."
                        ),
                    )
                    return True
        return False

    def _format_progress(self, episode: int, episode_metrics: dict) -> str:
        """Format progress bar description."""
        flags = []
        if self.discount.discounted: 
            flags.append("disc")
        flag_str = f"[{','.join(flags)}] " if flags else ""
        
        # Get critic weights from Flax params
        critic_w = np.array(self.critic_params['params']['Dense_0']['kernel']).flatten()
        
        return (
            f"{flag_str}Ep {episode+1} | "
            f"R: {episode_metrics['cost']:.2f}, "
            f"L: {episode_metrics['loss']:.4f}, "
            f"C: {[f'{w:.2f}' for w in critic_w[:6]]}, " 
        )

    def _evaluate_noiseless(self) -> float:
        """Run a short noiseless rollout and return the total cost.
        
        Uses the current critic to select actions (mu only, no noise)
        from the fixed initial condition.
        """
        x_init = self.x0 if self.x0 is not None else jnp.zeros(self.env.N)
        if self.x0 is None:
            import warnings
            warnings.warn(
                "_evaluate_noiseless: x0 is None, evaluating from zeros. "
                "Set agent.x0 or cfg.eval.x0_test for meaningful evaluation.",
                stacklevel=2,
            )
        
        # Save state. _fill_buffer_initial below resets the window to a NEW buffer,
        # so save the buffer object and reassign it back afterwards to restore the
        # training-time window exactly (the previous code restored a detached copy
        # and left the training buffer in the post-eval state — a latent bug).
        saved_state = self.wrapper.state
        saved_buffer = self.sliding_signature.buffer
        saved_sig = self.sliding_signature.current_signature
        saved_dirty = self._path_data_dirty
        saved_cached = self._cached_path_data
        
        self.key, subkey = jax.random.split(self.key)
        x_t = self.wrapper.reset(subkey, x0=np.array(x_init), t0=0.0)
        # Initial condition = the initial path phi, loaded into the window by
        # _fill_buffer_initial (no zero-control burn-in; mirrors train()).
        self._fill_buffer_initial()

        # Controlled horizon starts at t=0 (initial path already loaded; no burn-in).
        self._t_episode_start = float(self.wrapper.state.t)  # = 0
        total_cost = 0.0

        while not self._is_episode_done(x_t, time_only=True):
            path_data = self._get_path_data()
            mu, _ = self._select_action_jit(
                self.critic_params, path_data, self.R, self.wrapper.state.x) #type: ignore
            _, x_next, reward = self.wrapper.step(self.wrapper.state, mu)
            if x_next.ndim == 0:
                x_next = x_next.reshape(1)
            self.sliding_signature.append(x_next / self.training.scale)
            self._path_data_dirty = True
            total_cost += float(reward) * self.env.step_size
            x_t = x_next
        
        # Restore state (reassign the saved training-time buffer object).
        self.wrapper.state = saved_state
        self.sliding_signature.buffer = saved_buffer
        self.sliding_signature.current_signature = saved_sig
        self._path_data_dirty = saved_dirty
        self._cached_path_data = saved_cached
        
        return total_cost

    def _on_episode_end(self, episode: int, episode_metrics: dict) -> None:
        """Hook called at the end of each episode. Override for custom logic."""
        if getattr(self, "_lstd", False):
            # Solve the regularised LSTD system theta = -(M + reg*I)^-1 b and write it
            # into the (bias-free, linear) critic. No target network / optimiser state
            # is used in LSTD; critic and target share the solved parameters.
            d = self._lstd_b.shape[0]
            rank = int(getattr(self, "_lstd_rank", 0))
            if rank > 0 and self._lstd_n > 0:
                # Truncated-SVD LSTD: project onto the top-k principal components of the
                # CENTRED feature covariance (the directions that actually carry data
                # variance), solve the LSTD there, and map back. This discards the
                # rank-deficient null/noise subspace that wrecks the full-D solve, rather
                # than amplifying it (as per-feature whitening did). With U_k the top-k
                # eigenvectors: M_k = U_k^T M U_k, b_k = U_k^T b, theta = U_k theta_k.
                C = self._lstd_C / max(self._lstd_n, 1.0)
                evals, evecs = np.linalg.eigh((C + C.T) / 2.0)
                Uk = evecs[:, -rank:]                       # top-k eigenvectors
                Mk = Uk.T @ self._lstd_M @ Uk
                bk = Uk.T @ self._lstd_b
                scale = np.trace(Mk) / rank if rank > 0 else 1.0
                try:
                    theta_k = -np.linalg.solve(Mk + self._lstd_reg * abs(scale) * np.eye(rank), bk)
                except np.linalg.LinAlgError:
                    theta_k = -np.linalg.lstsq(Mk + self._lstd_reg * abs(scale) * np.eye(rank),
                                               bk, rcond=None)[0]
                theta = Uk @ theta_k
            else:
                try:
                    theta = -np.linalg.solve(self._lstd_M + self._lstd_reg * np.eye(d), self._lstd_b)
                except np.linalg.LinAlgError:
                    theta = -np.linalg.lstsq(self._lstd_M + self._lstd_reg * np.eye(d),
                                             self._lstd_b, rcond=None)[0]
            if np.all(np.isfinite(theta)):
                kernel = self.critic_params["params"]["Dense_0"]["kernel"]
                new_kernel = jnp.asarray(theta.reshape(kernel.shape), dtype=kernel.dtype)
                self.critic_params = {"params": {"Dense_0": {"kernel": new_kernel}}}
                self.target_params = self.critic_params

    def save(self, filename: str) -> None:
        """Save critic/target parameters and configs to a file."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'critic_params': self.critic_params,
            'target_params': self.target_params,
            'env_params': {
                'A': np.array(self.env.A),
                'B': np.array(self.env.B),
                'Q': np.array(self.env.Q),
                'R': np.array(self.env.R),
                'A1': np.array(self.env.A1),
                'delay': np.array(self.env.delay) if self.env.delay is not None else None,
                'step_size': self.env.step_size,
                'resolution': self.env.resolution,
            },
            'training': self.training,
            'discount': self.discount,
            'noise': self.noise,
            'signature': self.signature_conf,
            'network': self.network,
            'algorithm': self.algorithm,
        }

        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Agent saved to {filename}")

    def load(self, filename: str) -> None:
        """Load critic/target parameters from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.critic_params = data['critic_params']
        self.target_params = data['target_params']
        print(f"Agent loaded from {filename}")

# =====================================================
# ================== Only in Inference=================
# =====================================================

    def get_value(self) -> float:
        """Compute value function for a current state."""
        sig = self.sliding_signature.current_signature
        return float(jnp.asarray(self.critic.apply(self.critic_params, sig)).squeeze()) # type: ignore