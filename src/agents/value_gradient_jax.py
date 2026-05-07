import jax 
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from src.envs.env_rk_jax import JAXEnvWrapper
from src.networks.value_gradient_nets import CriticFlax, CriticFlaxLayerNorm
from src.utils.dynamic_signature import SlidingSignatureJAX
from src.utils.step_context import StepContextSignature
from src.utils.step_metrics import StepMetrics
from src.configs import (
    TrainingConfig, DiscountConfig, NoiseConfig,
    SignatureConfig, NetworkConfig, AlgorithmConfig,)

class ContinuousValueGradient:
    def __init__(self,
        env,
        Q: jnp.ndarray,
        R: jnp.ndarray,
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
        self.eval_callback = eval_callback

        self._init_env(env, Q, R, rng_key, x0)
        self._init_episode_state()
        self._init_checkpoint_state()
        self._init_networks()

    def _init_env(self, env, Q, R, rng_key: int, x0) -> None:
        self.env = env
        self.wrapper = JAXEnvWrapper(env, rng_key=jax.random.PRNGKey(rng_key))
        self.Q = Q if Q is not None else env.Q
        self.R = R if R is not None else env.R
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
        self._sigma_effective = self.noise.sigma
        self.episode_noise_trajectory = None
        self.current_noise = jnp.zeros(self.env.B.shape[1])
        self._cached_path_data: jnp.ndarray | None = None
        self._path_data_dirty = True

    def _init_checkpoint_state(self) -> None:
        self._best_eval_cost = -float('inf')
        self._best_critic_params = None
        self._best_target_params = None
        self._best_episode = 0
        self._patience_counter = 0

    def _init_networks(self) -> None:
        self.sliding_signature = SlidingSignatureJAX(
            depth=self.signature_conf.depth, window_size=self.signature_conf.window_size, d=self.env.N,
            time_augmentation=self.signature_conf.time_augmentation,
            origin_augmentation=self.signature_conf.origin_augmentation, bias=self.signature_conf.bias,
        )
        self.critic = self._build_network()
        self.target = self._build_network()
        self.optimizer = optax.adam(learning_rate=self.training.critic_lr * self.env.step_size)

        key_critic, key_target, self.key = jax.random.split(self.key, 3)
        self.critic_params = self.critic.init(key_critic, jnp.zeros((self.sliding_signature.signature_size,)))
        self.target_params = self.target.init(key_target, jnp.zeros((self.sliding_signature.signature_size,)))
        self.critic_opt_state = self.optimizer.init(self.critic_params)

        self._gradient_value_fn = self.get_gradient_value_fn()
        self._select_action_jit = self._make_action_jit_fn()
        self._jit_critic_update = self._make_critic_update_fn()

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
            if len(subsampled) >= target_len:
                # Use the last target_len elements
                for x in subsampled[-target_len:]:
                    self.sliding_signature.buffer.append(np.asarray(x/self.training.scale, dtype=np.float32)) #type: ignore
            else:
                # Pad with first element to reach target_len
                first_val = np.asarray(subsampled[0]/self.training.scale, dtype=np.float32)
                padding_needed = target_len - len(subsampled)
                for _ in range(padding_needed):
                    self.sliding_signature.buffer.append(first_val) #type: ignore
                for x in subsampled:
                    self.sliding_signature.buffer.append(np.asarray(x/self.training.scale, dtype=np.float32)) #type: ignore
            
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
        @jax.jit
        def select_action(critic_params, path_data, R, x_current):
            B = get_B_fn(x_current)
            grad_path = grad_V_fn(critic_params, path_data)
            end_gradient = grad_path[-1]
            u = 1/2*jnp.linalg.inv(R) @ B.T @ end_gradient #Doya LQR development
            u = jnp.clip(u, -clip_action, clip_action) #arbitrary but should be enough
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
        
        explicit_noise_val = jnp.zeros(self.env.B.shape[1])
        if self.noise.smooth and self.episode_noise_trajectory is not None:
            t_idx = int(round(self.wrapper.state.t / self.env.step_size)) #type: ignore
            # Clamp to avoid overflow
            t_idx = min(t_idx, len(self.episode_noise_trajectory) - 1)
            explicit_noise_val = self.episode_noise_trajectory[t_idx]
        path_data = self._get_path_data()
        mu, end_gradient = self._select_action_jit(
                self.critic_params, path_data, self.R, self.wrapper.state.x) #type: ignore
        if self.noise.smooth:
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
            # Clip gradients
            grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -10.0, 10.0), grads)
            
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
            for _ in range(self.algorithm.burning_steps):
                action = jnp.zeros(self.env.B.shape[1]) #burning with zero action
                t, x_t, _ = self.wrapper.step(self.wrapper.state, action) #type: ignore
                self.sliding_signature.append(x_t / self.training.scale)
                self._path_data_dirty = True
            if self.algorithm.preheat:
                for _ in range(self.sliding_signature.window_size):
                    # action, _, _ = self._select_action(state, self.env.step_size)
                    action = jnp.zeros(self.env.B.shape[1]) #preheat with zero action
                    # path_data = self._get_path_data()
                    # action, _ = self._select_action_jit(self.critic_params, path_data, self.R, self.wrapper.state.x) #type: ignore
                    t, x_t, _ = self.wrapper.step(self.wrapper.state, action) #type: ignore
                    self.sliding_signature.append(x_t / self.training.scale)
                    self._path_data_dirty = True
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
        if t >= self.training.max_time:
            return True
        if not time_only:
            if jnp.any(jnp.isnan(x)):
                return True
            if jnp.linalg.norm(x) > self.training.divergence_threshold:
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
        
        # Save state
        saved_state = self.wrapper.state
        buf = self.sliding_signature.buffer
        if hasattr(buf, '_data'):
            # JAXCircularBuffer
            saved_buf = (buf._data.copy(), buf._count, buf._head)  # type: ignore[union-attr]
        else:
            # DequeBuffer
            from collections import deque
            saved_buf = deque(buf.buffer, maxlen=buf.size)  # type: ignore[union-attr]
        saved_sig = self.sliding_signature.current_signature
        saved_dirty = self._path_data_dirty
        saved_cached = self._cached_path_data
        
        self.key, subkey = jax.random.split(self.key)
        x_t = self.wrapper.reset(subkey, x0=np.array(x_init), t0=0.0)
        self._fill_buffer_initial()
        
        for _ in range(self.algorithm.burning_steps):
            action = jnp.zeros(self.env.B.shape[1])
            _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
            self.sliding_signature.append(x_t / self.training.scale)
            self._path_data_dirty = True
        if self.algorithm.preheat:
            for _ in range(self.sliding_signature.window_size):
                action = jnp.zeros(self.env.B.shape[1])  # zero action, consistent with training
                _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
                self.sliding_signature.append(x_t / self.training.scale)
                self._path_data_dirty = True
        
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
        
        # Restore state
        self.wrapper.state = saved_state
        if hasattr(buf, '_data'):
            buf._data, buf._count, buf._head = saved_buf  # type: ignore[union-attr]
        else:
            buf.buffer = saved_buf  # type: ignore[union-attr]
        self.sliding_signature.current_signature = saved_sig
        self._path_data_dirty = saved_dirty
        self._cached_path_data = saved_cached
        
        return total_cost

    def _on_episode_end(self, episode: int, episode_metrics: dict) -> None:
        """Hook called at the end of each episode. Override for custom logic."""
        pass

