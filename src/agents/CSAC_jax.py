"""Implementation of CSAC from Huimin Han (Continuous Soft Actor-Critic) using JAX.
"""


import distrax
import numpy as np
import tqdm
from envs.env_rk_jax import JAXDDEEnv, JAXEnvWrapper
from src.utils.dynamic_signature import SlidingSignatureJAX
import jax
import jax.numpy as jnp
import optax
import scipy
from src.networks.LQR_actor_critics import ActorFlaxLayerNormSTD, ActorFlaxSTD, AlphaModel, CriticFlax, CriticFlaxLayerNorm
from src.utils.experience_replay_buffer import ExperienceReplayBuffer, Transition
from flax.training.train_state import TrainState
from configs import (
    TrainingConfig, DiscountConfig, NoiseConfig,
    SignatureConfig, NetworkConfig, AlgorithmConfig,
    ReplayBufferConfig, from_legacy_params, configs_to_flat_dict,
)


class CSAC:
    def __init__(self,
        env: JAXDDEEnv,
        # --- Legacy positional args (kept for backward compat) ---
        training_params: dict | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        # signature params
        depth: int = 2,
        rng_key: int = 42,
        # Variant flags
        discounted: bool = False,
        semi_gradient: bool = False,
        integral_td: bool = False,
        fix_initial_state: bool = False,
        decay_noise: bool = True,
        time_augmentation: bool = True,
        state_augmentation: bool = False,
        origin_augmentation: bool = True,
        bias: bool = True,
        actor_oracle: bool = False,
        critic_oracle: bool = False,
        preheat: bool = True, 
        actor_update_frequency: int = 1,
        window_size: int = 10,
        smooth_noise: bool = False,
        noise_length_scale: float = 0.2,
        lbda: float = 0.1,
        eval_callback=None,
        # --- New-style config objects (take precedence if provided) ---
        training: TrainingConfig | None = None,
        discount: DiscountConfig | None = None,
        noise: NoiseConfig | None = None,
        signature: SignatureConfig | None = None,
        network: NetworkConfig | None = None,
        algorithm: AlgorithmConfig | None = None,
        replay_buffer: ReplayBufferConfig | None = None,
    ):
        # =================================================================
        # Config resolution: new-style dataclasses or legacy dict+kwargs
        # =================================================================
        if training is not None:
            self.training = training
            self.discount = discount or DiscountConfig()
            self.noise = noise or NoiseConfig()
            self.signature = signature or SignatureConfig()
            self.network = network or NetworkConfig()
            self.algorithm = algorithm or AlgorithmConfig()
            self.replay_buffer_cfg = replay_buffer or ReplayBufferConfig()
        elif training_params is not None:
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
                bias=bias,
                actor_oracle=actor_oracle,
                critic_oracle=critic_oracle,
                preheat=preheat,
                actor_update_frequency=actor_update_frequency,
                window_size=window_size,
                smooth_noise=smooth_noise,
                noise_length_scale=noise_length_scale,
            )
            self.replay_buffer_cfg = ReplayBufferConfig(
                capacity=training_params.get('replay_buffer_size', 100_000),
                batch_size=training_params.get('batch_size', 256),
                n_updates_per_epoch=training_params.get('n_updates_per_epochs', 128),
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
        self.tau_polyak = self.training.tau_polyak
        # Discount
        self.discounted = self.discount.discounted
        self.discount_factor_tau = self.discount.tau
        # Noise
        self.sigma = self.noise.sigma
        self.decay_noise = self.noise.decay
        self.smooth_noise = self.noise.smooth
        self.noise_length_scale = self.noise.length_scale
        self._sigma_effective = self.sigma
        # Signature
        self.depth = self.signature.depth
        self.time_augmentation = self.signature.time_augmentation
        self.origin_augmentation = self.signature.origin_augmentation
        self.state_augmentation = self.signature.state_augmentation
        # Network
        self.hidden_dims = self.network.hidden_dims
        self.normalize_entries = self.network.normalize_entries
        self.normalize_sigs = self.network.normalize_sigs
        # Algorithm
        self.semi_gradient = self.algorithm.semi_gradient
        self.integral_td = self.algorithm.integral_td
        self.fix_initial_state = self.algorithm.fix_initial_state
        self.actor_oracle = self.algorithm.actor_oracle
        self.critic_oracle = self.algorithm.critic_oracle
        self.preheat = self.algorithm.preheat
        self.actor_update_frequency = self.algorithm.actor_update_frequency
        # Replay buffer
        self.replay_buffer_size = self.replay_buffer_cfg.capacity
        self.batch_size = self.replay_buffer_cfg.batch_size
        self.n_updates_per_epochs = self.replay_buffer_cfg.n_updates_per_epoch
        # CSAC-specific
        self.lbda = lbda

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
        self.dt = self.env.step_size
        self.step_counter = 0
        self.current_noise = None
        self.episode_noise_trajectory = None
        self.mesh_grids = int(self.max_time / self.dt) * 2

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

        if self.normalize_entries:
            self.scale = self.divergence_threshold
            print("normalize_entries is True: scaling states by ", self.scale)
            print("normalize_sigs is ", self.normalize_sigs)

        # Signature
        max_delay = float(jnp.max(self.env.delay)) if self.env.delay is not None else 0
        window_size_delay = (int(np.ceil(max_delay / self.env.step_size)) + 1) * 4 if max_delay > 0 else 50
        print("Using window_size =", window_size_delay, "for signature computation.")
        self.sliding_signature = SlidingSignatureJAX(
            depth=self.depth, window_size=window_size_delay, d=env.N,
            time_augmentation=self.time_augmentation,
            origin_augmentation=self.origin_augmentation,
            bias=self.signature.bias,
        )
        
        # Cost matrices
        self.Q = Q if Q is not None else env.Q
        self.R = R if R is not None else env.R
        
        # Random generator
        self.key = jax.random.PRNGKey(rng_key)
                
        # Build networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.critic_target = self._build_critic()
        key_a, key_c, key_tc = jax.random.split(self.key, 3)
        if self.state_augmentation:
            self.actor_params = self.actor.init(key_a, jnp.zeros(self.sliding_signature.signature_size + self.env.N))
            self.critic_params = self.critic.init(key_c, jnp.zeros(self.sliding_signature.signature_size + self.env.N))
            self.target_critic_params = self.critic_target.init(key_tc, jnp.zeros(self.sliding_signature.signature_size + self.env.N))
        else:
            self.actor_params = self.actor.init(key_a, jnp.zeros(self.sliding_signature.signature_size))
            self.critic_params = self.critic.init(key_c, jnp.zeros(self.sliding_signature.signature_size))
            self.target_critic_params = self.critic_target.init(key_tc, jnp.zeros(self.sliding_signature.signature_size))
        # Entropy
        self.target_entropy = -float(self.env.B.shape[1]) 
        self.alpha_model = AlphaModel()
        self.alpha_state = TrainState.create(
            apply_fn=self.alpha_model.apply,
            params=self.alpha_model.init(self.key),
            tx=optax.adam(self.actor_lr * self.dt)
        )

        # Optimizers
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_gradient),
            optax.adam(self.actor_lr * self.dt)
        )
        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_gradient),
            optax.adam(self.critic_lr * self.dt)
        )

        # JIT functions
        self._jit_select_action = self._make_action_jit_fn()
        self._jit_compute_value = self._make_compute_value()
        self._jit_compute_value_target = self._make_compute_value_target()
        # States
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor_params,
            tx=self.actor_optimizer
        )
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic_params,
            tx=self.critic_optimizer
        )
        # Backward pass
        self._jit_update_step = self._make_update_step()

        # Replay buffer
        dummy_transition = Transition(
                s=jnp.zeros((self.sliding_signature.signature_size + (self.env.N if self.state_augmentation else 0),)),
                a=jnp.zeros((self.env.B.shape[1],)),
                mu=jnp.zeros((self.env.B.shape[1],)),
                log_pi=jnp.zeros(()),
                r=0.0,
                s_next=jnp.zeros((self.sliding_signature.signature_size + (self.env.N if self.state_augmentation else 0),)),
                done=False,
                x_t=jnp.zeros((self.env.N,)),
                x_next=jnp.zeros((self.env.N,)),
            )
        self.replay_buffer = ExperienceReplayBuffer(capacity=self.replay_buffer_size, 
                                                    dummy_transition=dummy_transition)
    
    def _build_actor(self) -> ActorFlaxSTD | ActorFlaxLayerNormSTD:
        """Build the actor network."""
        output_dim = self.env.B.shape[1]
        if self.normalize_sigs:
            actor = ActorFlaxLayerNormSTD(output_dim=output_dim)
        else:
            actor = ActorFlaxSTD(output_dim=output_dim)
        return actor

    def _build_critic(self) -> CriticFlax | CriticFlaxLayerNorm:
        """Build the critic network."""       
        if self.normalize_sigs:
            critic = CriticFlaxLayerNorm()
        else:
            critic = CriticFlax()
        return critic
    
    
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
            for x in history_data[::self.env.resolution]:
                self.sliding_signature.buffer.append(np.asarray(x, dtype=np.float32))
            self.sliding_signature.current_signature = self.sliding_signature.compute_signature()

    def _make_action_jit_fn(self):
        actor = self.actor
        clip_action = self.clip_action
        @jax.jit
        def select_action_fn(actor_params, state, key):
            mu, sigma = actor.apply(actor_params, state)
            dist = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma) # type: ignore
            # dist = distrax.Transformed(distribution=dist, bijector=distrax.Block(distrax.Tanh(), ndims=mu.shape[0]))  # type: ignore
            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)
            log_pi = dist.log_prob(action)
            action = action * clip_action
            return action, key, mu, log_pi
        return select_action_fn
    
    def _make_compute_value(self):
        critic = self.critic
        @jax.jit
        def compute_value_fn(critic_params, state_t, state_next):
            V_t:jnp.ndarray = critic.apply(critic_params, state_t) #type: ignore
            V_next:jnp.ndarray = critic.apply(critic_params, state_next) #type: ignore
            return V_t, V_next
        return compute_value_fn
    
    def _make_compute_value_target(self):
        critic_target = self.critic_target
        @jax.jit
        def compute_value_target_fn(critic_target_params, state_next):
            V_next_target:jnp.ndarray = critic_target.apply(critic_target_params, state_next) #type: ignore
            return V_next_target
        return compute_value_target_fn
    
    def _select_action(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        action, self.key, mu, log_pi = self._jit_select_action(self.actor_params, state, self.key)
        return action, mu, log_pi

    def _make_update_step(self):
        dt = self.dt
        tau_polyak = self.tau_polyak
        tau_discount_factor = self.discount_factor_tau
        target_entropy = self.target_entropy
        actor_apply = self.actor.apply
        critic_apply = self.critic.apply
        critic_target_apply = self.critic_target.apply
        @jax.jit
        def update_step(actor_state, critic_state, target_critic_params, alpha_state, batch, key):
            key, subkey = jax.random.split(key)
            def loss_fn(a_params, c_params, alpha_params):
                state, action, reward, next_state, done = batch.s, batch.a, batch.r, batch.s_next, batch.done
                
                log_alpha = self.alpha_model.apply(alpha_params)
                lbda = jnp.exp(log_alpha) #type: ignore
                
                V_t = critic_apply(c_params, state)
                V_next_target = critic_target_apply(target_critic_params, next_state)
                
                mu, sigma = actor_apply(a_params, state)
                # sigma = jnp.clip(sigma, 1e-3, 10)  # Prevent numerical issues
                dist = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma) #type: ignore
                # Apply Tanh transformation to match action selection (critical for correct log_prob)
                # dist = distrax.Transformed(distribution=dist, bijector=distrax.Block(distrax.Tanh(), ndims=mu.shape[-1])) #type: ignore
                # Actions are scaled by clip_action, so unscale before computing log_prob
                log_pi_buffer = dist.log_prob(action)
                integral_flow = reward * dt - jax.lax.stop_gradient(lbda) * jax.lax.stop_gradient(log_pi_buffer) * dt \
                    - 1/tau_discount_factor * V_t * dt
                delta_M = V_next_target * (1 - done) - V_t + integral_flow
                # jax.debug.print("LogPi Stats -> Mean: {m}, Min: {mi}, Max: {ma}", 
                #                 m=jnp.mean(log_pi_buffer), 
                #                 mi=jnp.min(log_pi_buffer), 
                #                 ma=jnp.max(log_pi_buffer))
                # jax.debug.print("action Stats -> Mean: {m}, Min: {mi}, Max: {ma}", 
                #                 m=jnp.mean(action), 
                #                 mi=jnp.min(action), 
                #                 ma=jnp.max(action))
                # jax.debug.print("V_t Stats -> Mean: {m}, Min: {mi}, Max: {ma}", 
                #                 m=jnp.mean(V_t), 
                #                 mi=jnp.min(V_t), 
                #                 ma=jnp.max(V_t))
                # jax.debug.print("V_next_target Stats -> Mean: {m}, Min: {mi}, Max: {ma}", 
                #                 m=jnp.mean(V_next_target), 
                #                 mi=jnp.min(V_next_target), 
                #                 ma=jnp.max(V_next_target))
                # jax.debug.print("delta_M Stats -> Mean: {m}, Min: {mi}, Max: {ma}", 
                #                 m=jnp.mean(delta_M), 
                #                 mi=jnp.min(delta_M), 
                #                 ma=jnp.max(delta_M))
                # jax.debug.print("sigma Stats -> Mean: {m}, Min: {mi}, Max: {ma}", 
                #                 m=jnp.mean(sigma), 
                #                 mi=jnp.min(sigma), 
                #                 ma=jnp.max(sigma))  

                key, subkey_sample = jax.random.split(subkey)
                new_action_raw = dist.sample(seed=subkey_sample)
                # new_action_tanh = jnp.tanh(new_action_raw)
                # # new_action = new_action_raw * clip_action
                new_action_raw = jax.lax.stop_gradient(new_action_raw)
                log_prob_gaussian = dist.log_prob(new_action_raw)
                # correction = 2 * (jnp.log(2) - new_action_raw - jax.nn.softplus(-2 * new_action_raw))
                log_pi_new = log_prob_gaussian #-jnp.sum(correction, axis=-1)
                
                critic_loss = jnp.mean(0.5 * (delta_M ** 2)/dt/(1+1/tau_discount_factor*dt))
                actor_loss = -jnp.mean(log_pi_new * ((jax.lax.stop_gradient(delta_M)/dt - jax.lax.stop_gradient(lbda))))
                alpha_loss = -jnp.mean(log_alpha * (jax.lax.stop_gradient(log_pi_new) + target_entropy)) #type: ignore
                total_loss = actor_loss + critic_loss + alpha_loss
                
                return total_loss, (actor_loss, critic_loss, alpha_loss, lbda, jnp.mean(log_pi_new), jnp.mean(delta_M))
            
            # Use value_and_grad to get both losses and gradients
            (total_loss, aux), grads = jax.value_and_grad(loss_fn, argnums=(0,1,2), has_aux=True)(
                actor_state.params, critic_state.params, alpha_state.params
            )
            actor_grads, critic_grads, alpha_grads = grads
            # Gradient clipping is now handled by the optimizer chain in apply_gradients
            actor_loss, critic_loss, alpha_loss, alpha, mean_log_pi, mean_delta_M = aux
            
            new_actor_state = actor_state.apply_gradients(grads=actor_grads)
            new_critic_state = critic_state.apply_gradients(grads=critic_grads)
            new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
            new_target_critic_params = optax.incremental_update(new_critic_state.params, target_critic_params, tau_polyak)
            
            # Return metrics as a dict
            metrics = {
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'alpha_loss': alpha_loss,
                'alpha': alpha,
                'entropy': -mean_log_pi,
                'delta_M': mean_delta_M,
                'grad_norm_actor': optax.global_norm(actor_grads),
                'grad_norm_critic': optax.global_norm(critic_grads),
                'grad_norm_alpha': optax.global_norm(alpha_grads),
                'weights_actor_mu': jnp.mean(jnp.abs(new_actor_state.params['params']['mu head']['kernel'])),
                'weights_actor_std': jnp.mean(jnp.abs(new_actor_state.params['params']['log_std head']['kernel'])),
                'weights_critic': jnp.mean(jnp.abs(new_critic_state.params['params']['Dense_0']['kernel'])),
                'actor_grads_mu': jnp.mean(jnp.abs(actor_grads['params']['mu head']['kernel'])),
                'actor_grads_std': jnp.mean(jnp.abs(actor_grads['params']['log_std head']['kernel'])),
            }
            return new_actor_state, new_critic_state, new_alpha_state, new_target_critic_params, metrics
        return update_step
            
            # def _compute_delta_martingale( 
            #                             state_t: jnp.ndarray, 
            #                             state_next: jnp.ndarray, 
            #                             log_pi:jnp.ndarray, 
            #                             reward: float, 
            #                             done: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
            #     V_t = self._jit_compute_value(self.critic_params, state_t)
            #     V_next_target = self._jit_compute_value(self.target_critic_params, state_next)
            #     delta_M = V_next_target * (1 - done) - V_t + \
            #         reward * self.dt - self.lbda * log_pi * self.dt - \
            #             self.discount_factor_tau * V_t * self.dt
            #     return delta_M
    
    def _update_networks(self):
        """Update actor and critic networks using a batch from the replay buffer."""
        # Sample batch
        if self.replay_buffer.size < self.batch_size:
            return {}
        self.key, subkey, subkey_sample = jax.random.split(self.key, 3)
        batch = self.replay_buffer.sample(self.batch_size, subkey_sample)
        (self.actor_state, 
         self.critic_state,
         self.alpha_state,
         self.target_critic_params,
         metrics) = self._jit_update_step(
             self.actor_state, self.critic_state, self.target_critic_params, 
             self.alpha_state, batch, subkey
         )
        
        return metrics
  
    def _fill_replay_buffer(self):
        x_t = self.wrapper.state.x # type: ignore
        env_state = self.wrapper.state # type: ignore
        x_scaled = x_t / self.scale if self.normalize_entries else x_t
        if self.state_augmentation:
            sig_state = jnp.concatenate([self.sliding_signature.current_signature, x_scaled])
        else:
            sig_state = self.sliding_signature.current_signature
        action, mu, log_pi = self._select_action(sig_state)
        t_next, x_next, reward = self.wrapper.step(env_state, action) #type: ignore
        x_next_scaled = x_next / self.scale if self.normalize_entries else x_next
        done = self._is_done(t_next, x_next)
        self.sliding_signature.append(x_next_scaled)
        if self.state_augmentation:
            sig_state_next = jnp.concatenate([self.sliding_signature.current_signature, x_next_scaled])
        else:
            sig_state_next = self.sliding_signature.current_signature
        # Store transition in replay buffer
        transition = Transition(
            s=sig_state,
            a=action,
            mu=mu,
            log_pi=log_pi,
            r=reward,
            s_next=sig_state_next,
            done=done,
            x_t=x_t,
            x_next=x_next,
        )
        self.replay_buffer.add(transition)
        # Update signature buffer
        return x_next, reward, done

    def update_buffer(self, x: np.ndarray) -> None:
        self.sliding_signature.append(x / self.training.scale)
        if hasattr(self, '_path_data_dirty'):
            setattr(self, '_path_data_dirty', True)

    def get_eval_action(self, x_scaled: jnp.ndarray) -> jnp.ndarray:
        if getattr(self, 'actor_oracle', False) or getattr(self.algorithm, 'actor_oracle', False):
            return jnp.array(-self.optimal_K @ jnp.array(self.wrapper.state.x))
            
        sig = self.sliding_signature.current_signature
        if getattr(self.signature_conf, 'state_augmentation', False):
            sig_input = jnp.concatenate([sig, x_scaled])
        else:
            sig_input = sig
        mu, _ = self.actor.apply(self.actor_params, sig_input)
        return jnp.array(mu * getattr(self.training, 'clip_action', 1.0))

    def train(self) -> dict:
        """Train the agent and return metrics dict."""
        ep_iter = tqdm.trange(self.n_episodes, desc="Training Episodes")
        
        # Metrics storage
        all_metrics: dict[str, list] = {
            'rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'alphas': [],
            'entropies': [],
            'delta_M': [],
            'grad_norm_actor': [],
            'grad_norm_critic': [],
            'grad_norm_actor_mu': [],
            'grad_norm_actor_std': [],
            'weights_actor_mu': [],
            'weights_actor_std': [],
            'weights_critic': [],
            'signature_coefficients': [],
        }
        
        for epoch in ep_iter:
            done = True
            reward_per_episode = 0.0
            list_rewards = []
            for k in range(self.mesh_grids):
                if done:
                    list_rewards.append(reward_per_episode)
                    reward_per_episode = 0.0
                    self._on_episode_start()
                x_next, reward, done = self._fill_replay_buffer()
                signature_coeffs = self.sliding_signature.current_signature
                all_metrics['signature_coefficients'].append(signature_coeffs)
                reward_per_episode += reward * self.dt
            
            # Network updates with metrics tracking
            metrics_epoch: dict[str, list] = {}
            for _ in range(self.n_updates_per_epochs):
                metrics = self._update_networks()
                if metrics:
                    for key, val in metrics.items():
                        metrics_epoch.setdefault(key, []).append(float(val))
            
            # Average metrics over epoch
            avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items()}
            
            # Store metrics
            list_rewards.pop(0)  # Remove first incomplete episode reward
            all_metrics['rewards'] += list_rewards
            all_metrics['actor_losses'].append(avg_metrics.get('actor_loss', 0.0))
            all_metrics['critic_losses'].append(avg_metrics.get('critic_loss', 0.0))
            all_metrics['alpha_losses'].append(avg_metrics.get('alpha_loss', 0.0))
            all_metrics['alphas'].append(avg_metrics.get('alpha', 0.0))
            all_metrics['entropies'].append(avg_metrics.get('entropy', 0.0))
            all_metrics['delta_M'].append(avg_metrics.get('delta_M', 0.0))
            all_metrics['grad_norm_actor'].append(avg_metrics.get('grad_norm_actor', 0.0))
            all_metrics['grad_norm_critic'].append(avg_metrics.get('grad_norm_critic', 0.0))
            
            # Map detailed metrics
            all_metrics['grad_norm_actor_mu'].append(avg_metrics.get('actor_grads_mu', 0.0))
            all_metrics['grad_norm_actor_std'].append(avg_metrics.get('actor_grads_std', 0.0))
            all_metrics['weights_actor_mu'].append(avg_metrics.get('weights_actor_mu', 0.0))
            all_metrics['weights_actor_std'].append(avg_metrics.get('weights_actor_std', 0.0))
            all_metrics['weights_critic'].append(avg_metrics.get('weights_critic', 0.0))
            
            # Update progress bar
            
            # Periodic trajectory evaluation (wandb slider)
            if hasattr(self, 'eval_callback') and self.eval_callback is not None:
                self.eval_callback(self, epoch)
            
            ep_iter.set_postfix({
                "reward": f"{reward_per_episode:.2f}",
                "α": f"{avg_metrics.get('alpha', 0):.3f}",
                "c_loss": f"{avg_metrics.get('critic_loss', 0):.3f}",
            })
        
        return all_metrics
              
    
    def _is_done(self, t: float, state: jnp.ndarray) -> bool:
        """Check if episode is done."""
        if jnp.linalg.norm(state) > self.divergence_threshold:
            return True
        if t >= self.max_time:
            return True
        return False

    def _on_episode_start(self):
        if self.fix_initial_state:
            state = jnp.array(self.env_params['A'].shape[0] * [1.0])
        else:
            state = self._sample_initial_state()
        self.key, subkey = jax.random.split(self.key)
        self.wrapper.reset(subkey, x0=state)
        self._fill_buffer_initial()
        self.step_counter = 0
        if self.preheat:
            for _ in range(self.sliding_signature.window_size):
                if self.state_augmentation:
                    sig_state = jnp.concatenate([self.sliding_signature.current_signature, self.wrapper.state.x / self.scale if self.normalize_entries else self.wrapper.state.x]) # type: ignore
                else:
                    sig_state = self.sliding_signature.current_signature
                action, mu, log_pi = self._select_action(sig_state)
                _, next_state, _ = self.wrapper.step(self.wrapper.state, action)
                self.sliding_signature.append(next_state / self.scale if self.normalize_entries else next_state)
    


