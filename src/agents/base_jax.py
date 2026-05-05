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
from jax.numpy import ndarray
import numpy as np
from src.agents.signatures_jax import CTACSignatureJAX
from src.utils.dynamic_signature import SlidingSignatureJAX
from src.networks.LQR_actor_critics import CriticFlaxQuadratic
from src.utils.step_metrics import StepMetrics
from src.utils.step_context import StepContextSignature, StepContextDelayed
from src.configs import (
    TrainingConfig, DiscountConfig, NoiseConfig,
    SignatureConfig, NetworkConfig, AlgorithmConfig,
)
import jax.numpy as jnp
import optax
import scipy

from src.env_rk_jax import JAXDDEEnv


class CTACJAX(CTACSignatureJAX):
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
        env,
        # --- Legacy positional args (kept for backward compat) ---
        training_params: dict | None = None,
        Q=None,
        R=None,
        rng_key=42,
        discounted=False,
        semi_gradient=False,
        integral_td=False,
        actor_oracle=False,
        critic_oracle=False,
        fix_initial_state=False,
        decay_noise=False,
        preheat=True,
        time_augmentation=False,
        bias=True,
        window_size=20,
        delayed_state=False,
        whole_state_delay=False,
        burning_steps=0,
        actor_update_frequency=1,
        # --- New-style config objects (take precedence if provided) ---
        training: TrainingConfig | None = None,
        discount: DiscountConfig | None = None,
        noise: NoiseConfig | None = None,
        signature: SignatureConfig | None = None,
        network: NetworkConfig | None = None,
        algorithm: AlgorithmConfig | None = None,
    ):
        # Resolve actor/critic oracle early (needed before super().__init__)
        if training is not None:
            _algo = algorithm or AlgorithmConfig()
            self.actor_oracle = _algo.actor_oracle
            self.critic_oracle = _algo.critic_oracle
        else:
            self.actor_oracle = actor_oracle
            self.critic_oracle = critic_oracle

        # Force depth=2 for non-signature version
        if signature is not None:
            signature.depth = 2

        super().__init__(
            env,
            training_params=training_params,
            Q=Q,
            R=R,
            depth=2,  # depth=2 for non-signature version
            rng_key=rng_key,
            discounted=discounted,
            semi_gradient=semi_gradient,
            integral_td=integral_td,
            fix_initial_state=fix_initial_state,
            decay_noise=decay_noise,
            time_augmentation=time_augmentation,
            preheat=preheat,
            bias=bias,
            window_size=window_size,
            burning_steps=burning_steps,
            actor_update_frequency=actor_update_frequency,
            # Forward config objects to parent
            training=training,
            discount=discount,
            noise=noise,
            signature=signature,
            network=network,
            algorithm=algorithm,
        )
        # Read delayed-state flags from config (new) or kwargs (legacy)
        self.delayed_state = self.algorithm.delayed_state if training is not None else delayed_state
        self.whole_state_delay = self.algorithm.whole_state_delay if training is not None else whole_state_delay
        self.key, key_a, key_c = jax.random.split(self.key, 3)
        # Read window_size override for non-signature CTAC (parent may have set it from delay)
        max_delay = self.env.delay.max() if self.env.delay is not None else 0.0
        window_size = int(np.ceil(max_delay / self.env.step_size)) + 3 if max_delay > 0 else 10 #for evaluation purposes
        if not self.force_signature_window:
            self.window_size = window_size
        else:
            print(f"force_signature_window is True: setting window_size to {self.window_size} to cover max delay of {max_delay}")
        # Recreate sliding_signature with correct window_size (parent created it before we could override)
        self.sliding_signature = SlidingSignatureJAX(
            depth=self.depth, window_size=self.window_size, d=self.env.N,
            time_augmentation=self.time_augmentation,
            origin_augmentation=self.origin_augmentation,
            time_origin=self.time_origin, bias=self.signature.bias,
        )
        # State dimension: N if no delay, 2*N if delayed (current + delayed state)
        state_dim = self.env.N * 2 if delayed_state else self.env.N
        if self.whole_state_delay:
            # Use sliding_signature buffer size (agent step resolution), not env.buffer_size (solver resolution)
            state_dim = self.sliding_signature.buffer.size * self.env.N
        dummy_state = jnp.zeros((state_dim,))
        # Override critic with quadratic features (like base.py)
        self.critic = CriticFlaxQuadratic()
        
        self.actor_params = self.actor.init(key_a, dummy_state)
        self.critic_params = self.critic.init(key_c, dummy_state)
        # Re-initialize optimizers for the new parameters
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

        if self.actor_oracle or self.critic_oracle:
            self.P = jnp.array(scipy.linalg.solve_continuous_are(self.env.A, self.env.B, self.Q, self.R))
        
        # Override JIT functions for state-based (not signature-based) version
        self._jit_select_action = self._make_select_action_fn()
        self._jit_compute_values = self._make_compute_values_fn()
        if not self.critic_oracle:
            self._jit_critic_update = self._make_critic_update_fn()
        if not self.actor_oracle:
            self._jit_actor_update = self._make_actor_update_fn()
    
    def _make_select_action_fn(self):
        """Create JIT-compiled action selection for state-based input."""
        clip_action = self.clip_action
        
        if self.actor_oracle:
            K_opt = -jnp.linalg.solve(jnp.array(self.R), jnp.array(self.env.B).T @ self.P)
            
            @jax.jit
            def select_action_fn(actor_params, x, key, sigma):
                mu = K_opt @ x
                key, subkey = jax.random.split(key)
                noise = sigma * jax.random.normal(subkey, shape=mu.shape)
                action = mu + noise
                action = jnp.clip(action, -clip_action, clip_action)
                return action, mu, noise, key
            return select_action_fn
        else:
            actor = self.actor
            
            @jax.jit
            def select_action_fn(actor_params, x, key, sigma):
                mu = actor.apply(actor_params, x)
                key, subkey = jax.random.split(key)
                noise = sigma * jax.random.normal(subkey, shape=mu.shape) # type: ignore
                action = mu + noise
                action = jnp.clip(action, -clip_action, clip_action)
                return action, mu, noise, key
            return select_action_fn
    
    def _make_compute_values_fn(self):
        """Create JIT-compiled value computation for state-based input."""
        if self.critic_oracle:
            P = self.P
            
            @jax.jit
            def compute_values_fn(critic_params, x_t, x_next):
                V_t = -x_t.T @ P @ x_t
                V_next = -x_next.T @ P @ x_next
                return V_t, V_next
            return compute_values_fn
        else:
            critic = self.critic
            
            @jax.jit
            def compute_values_fn(critic_params, x_t, x_next):
                V_t = critic.apply(critic_params, x_t).squeeze() # type: ignore
                V_next = critic.apply(critic_params, x_next).squeeze() # type: ignore
                return V_t, V_next
            return compute_values_fn
    
    def _make_critic_update_fn(self):
        """Create JIT-compiled critic update for state-based input."""
        critic = self.critic
        critic_optimizer = self.critic_optimizer
        tau = self.tau
        discounted = self.discounted
        semi_gradient = self.semi_gradient
        clip_gradient = self.clip_gradient
        
        @jax.jit
        def critic_update_fn(critic_params, critic_opt_state, x_t, x_next, reward, dt):
            def loss_fn(params):
                V_t = critic.apply(params, x_t).squeeze() # type: ignore
                if semi_gradient:
                    V_next = critic.apply(jax.lax.stop_gradient(params), x_next).squeeze() # type: ignore
                else:
                    V_next = critic.apply(params, x_next).squeeze() # type: ignore
                V_dot = (V_next - V_t) / dt
                discount_term = V_t / tau if discounted else 0.0
                td_error = reward + V_dot - discount_term
                loss = 0.5 * td_error ** 2 * dt
                return loss, td_error
            
            (loss, td_error), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_params)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            grads = jax.tree_util.tree_map(
                lambda g: jnp.clip(g, -clip_gradient, clip_gradient), grads
            )
            updates, new_opt_state = critic_optimizer.update(grads, critic_opt_state, critic_params)
            new_params = optax.apply_updates(critic_params, updates)
            return new_params, new_opt_state, loss, td_error, grad_norm
        
        return critic_update_fn
    
    def _make_actor_update_fn(self):
        """Create JIT-compiled actor update for state-based input."""
        actor = self.actor
        actor_optimizer = self.actor_optimizer
        clip_gradient = self.clip_gradient
        
        @jax.jit
        def actor_update_fn(actor_params, actor_opt_state, x, noise, td_error, sigma, dt):
            def loss_fn(params):
                mu = actor.apply(params, x)
                grad_log_policy = noise / (sigma ** 2)
                loss = -td_error * jnp.sum(grad_log_policy * mu)
                return loss
            
            loss, grads = jax.value_and_grad(loss_fn)(actor_params)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            grads = jax.tree_util.tree_map(
                lambda g: jnp.clip(g, -clip_gradient, clip_gradient), grads
            )
            updates, new_opt_state = actor_optimizer.update(grads, actor_opt_state, actor_params)
            new_params = optax.apply_updates(actor_params, updates)
            return new_params, new_opt_state, grad_norm
        
        return actor_update_fn
    
    def _select_action(self, x: jnp.ndarray, dt: float = 0.0) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: # type: ignore
        """Select action based on current state x."""
        if self.noise_schedule == 'linear_decay':
            progress = min(self.episode / max(self.n_episodes, 1), 1.0)
            self._sigma_effective = self.sigma * max(0.05, 1.0 - 0.9 * progress)
        else:
            # 'constant' (adaptive not applicable for state-based agent)
            self._sigma_effective = self.sigma
        action, mu, noise, self.key = self._jit_select_action(
            self.actor_params, 
            x,
            self.key,
            self._sigma_effective
        )
        return action, mu, noise

    def _train_step(self, x_t: ndarray) -> tuple[jnp.ndarray, StepMetrics, StepContextSignature|StepContextDelayed]:
        """Execute a single training step (state-based, no signature)."""
        x_scaled = x_t / self.scale
        if self.whole_state_delay:
            # Use the full buffer path (already scaled) as feature vector
            buf = self.sliding_signature.buffer
            x_scaled = jnp.array(buf.to_array()).flatten()
            x_t = x_scaled * self.scale  # unscaled version for logging
        elif self.delayed_state:
            if x_t.shape[0] == self.env.N: #state not augmented!
                x_t = jnp.concatenate([x_t, self.wrapper.current_delayed_state], axis=0)
                x_scaled = x_t / self.scale
        # Action selection (use scaled state)
        action, mu, noise = self._select_action(x_scaled)
        # Environment step (using wrapper with current state)
        t, x_next, reward = self.wrapper.step(self.wrapper.state, action) # type: ignore
        self.state_counter.add(x_next)
        if x_next.ndim == 0 and not self.delayed_state:
            x_next = jnp.array([x_next])
        else:
            x_next = jnp.array(x_next)
        # Update buffer with new state (before building x_next_scaled)
        if self.whole_state_delay:
            self.sliding_signature.append(x_next / self.scale)
            buf = self.sliding_signature.buffer
            x_next_scaled = jnp.array(buf.to_array()).flatten()
        else:
            x_next_scaled = x_next / self.scale
        if self.delayed_state:
            x_next_delayed = self.wrapper.current_delayed_state
            x_next = jnp.concatenate([x_next, x_next_delayed], axis=0)
            x_next_delayed_scaled = x_next_delayed / self.scale
            x_next_scaled = jnp.concatenate([x_next_scaled, x_next_delayed_scaled], axis=0)
        dt = self.env.step_size
        if dt <= 1e-9:
            dt = 1e-4  # Avoid division by zero
        
        V_t, V_next = self._jit_compute_values(self.critic_params, x_scaled, x_next_scaled)
        
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
            V_t=V_t,
            V_next=V_next,
            sig_t = jnp.zeros(1),  # unused
            sig_next = jnp.zeros(1),  # unused
        )
        
        # Compute reward
        ctx.reward = reward #type: ignore
        
        # Compute V dot and TD error (keep as JAX arrays)
        ctx.V_dot = (ctx.V_next - ctx.V_t) / ctx.dt
        ctx.td_error = ctx.reward + ctx.V_dot - (ctx.V_t / self.tau if self.discounted else 0.0)
        
        # update networks
        loss_critic, actor_grad_norm, critic_grad_norm = self._update_networks(ctx)

        # Keep as JAX arrays — defer float() to episode end
        metrics = StepMetrics(
            loss=loss_critic * dt,
            reward=ctx.reward * dt,
            actor_gradient=actor_grad_norm, # type: ignore
            critic_gradient=critic_grad_norm, # type: ignore
        )
        
        return x_next, metrics, ctx
    
    def _update_networks(self, ctx: StepContextSignature) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update actor and critic networks using state-based JIT functions.

        Returns JAX arrays to avoid GPU→CPU sync every step.
        Conversion to float is deferred to episode-end metrics.
        """
        x_t_scaled = ctx.x_scaled
        x_next_scaled = ctx.x_next_scaled
        noise = ctx.noise
        reward = ctx.reward
        dt = ctx.dt
        sigma = self._sigma_effective
        step = self.step_counter
        self.step_counter += 1
        
        # Critic update
        if not self.critic_oracle:
            self.critic_params, self.critic_opt_state, c_loss, td_error, critic_grad_norm = \
            self._jit_critic_update(
                self.critic_params,
                self.critic_opt_state,
                x_t_scaled, x_next_scaled, reward, dt)
        else:
            td_error = ctx.td_error
            c_loss = jnp.array(0.0)
            critic_grad_norm = jnp.array(0.0)
        
        # Actor update
        if not self.actor_oracle and step%self.actor_update_frequency==0:
            self.actor_params, self.actor_opt_state, actor_grad_norm = \
            self._jit_actor_update(
                self.actor_params,
                self.actor_opt_state,
                x_t_scaled, noise, td_error, sigma, dt
            )
        else:
            actor_grad_norm = jnp.array(0.0)
        
        return c_loss, actor_grad_norm, critic_grad_norm

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
                action = jnp.zeros(self.env.B.shape[1])
                _, x_t, _ = self.wrapper.step(self.wrapper.state, action)  # type: ignore
                self.sliding_signature.append(x_t / self.scale)
        
        total_reward = 0.0
        
        while not self._is_episode_done(x_t, time_only=True):
            if self.actor_oracle:
                mu = -self.optimal_K @ jnp.array(self.wrapper.state.x)  # type: ignore
            else:
                if hasattr(self, 'delayed_state') and self.delayed_state: # type: ignore
                    # Delayed state agents
                    x_delayed = self.wrapper.current_delayed_state
                    x_augmented = jnp.concatenate([self.wrapper.state.x / self.scale, x_delayed / self.scale], axis=0)
                    mu = self.actor.apply(self.actor_params, x_augmented) # type: ignore
                elif hasattr(self, 'whole_state_delay') and self.whole_state_delay: # type: ignore
                    # Whole state delay agents — buffer already contains scaled states
                    buf = self.sliding_signature.buffer
                    x_augmented = jnp.array(buf.to_array()).flatten()
                    mu = self.actor.apply(self.actor_params, x_augmented) # type: ignore
                else:
                    mu = self.actor.apply(self.actor_params, self.wrapper.state.x / self.scale) # type: ignore
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
        
        return total_reward

# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    
    # Simple 2D stable system (oscillator with damping)
    # dx/dt = A*x + B*u, naturellement stable sans contrôle
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    A1 = np.zeros_like(A)
    delay = np.array([0.0])
    x0 = np.array([1.0])
    Q=np.eye(1)
    R=np.eye(1)
    depth = 2
    
    env = JAXDDEEnv(A=A, B=B, Q=Q, R=R, A1=A1, delay=delay, step_size=0.1, resolution=10)
    
    agent = CTACJAX(
        env=env,
        training_params={
            'n_episodes': 500,
            'max_time': 5.0,
            'sigma': 0.2,
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'scale': 1.0,
            'clip_gradient': 5.0,
            'clip_action': 5.0,
            'divergence_threshold': 50.0,
            'log_interval': 20,
            'init_log_interval': 50,
            'memory_clear_interval': 20,
        },
        Q=Q,
        R=R,
        discounted=False,
        semi_gradient=True,
        integral_td=False,
        fix_initial_state=True,
        decay_noise=True,
        time_augmentation=False,
        bias=False,
        actor_oracle=False,  # Use learned actor
        critic_oracle=False,  # Use learned critic
    )
    metrics = agent.train()

    import matplotlib.pyplot as plt
    plt.plot(metrics['cost_episodic'])
    plt.xlabel('Episode (x50)')
    plt.ylabel('Episodic Cost')
    plt.title('CTAC Signature Episodic Cost over Training')
    plt.show()

    plt.plot(metrics['loss_episodic'])
    plt.xlabel('Episode (x50)')
    plt.ylabel('Episodic Loss')
    plt.title('CTAC Signature Episodic Loss over Training')
    plt.show()