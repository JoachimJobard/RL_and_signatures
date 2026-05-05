"""
Continuous-Time Actor-Critic (CTAC) - Modular Implementation

This module provides a modular baseline for CT actor-critic algorithms.
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

from typing import Optional
import numpy as np
import scipy
import tqdm
from src.networks.LQR_actor_critics import ActorNetwork, CriticNetwork, CriticOracle
from src.utils.step_metrics import StepMetrics
from src.utils.step_context import StepContext

from src.env_rk import Environment  


class CTAC:
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
        env: Environment,
        actor_params: dict,
        critic_params: dict,
        training_params: dict,
        Q: np.ndarray,
        R: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        # Variant flags
        discounted: bool = False,
        semi_gradient: bool = False,
        integral_td: bool = False,
        actor_oracle: bool = False,
        critic_oracle: bool = False,
        fix_initial_state: bool = False,
        decay_noise: bool = True,
    ):
        # Store environment parameters for potential recreation
        # self.env_params = {
        #     'A': env.A.copy(),
        #     'B': env.B.copy(),
        #     'A1': env.A1.copy(),
        #     'delay': env.delay.copy(),
        #     'x0': env.x0.copy(),
        #     'step_size': env.step_size,
        #     'resolution': env.resolution,
        # }
        self.env = env
        self.episode = 0
        
        # Variant flags
        self.discounted = discounted
        self.semi_gradient = semi_gradient
        self.integral_td = integral_td
        self.actor_oracle = actor_oracle
        self.critic_oracle = critic_oracle
        self.fix_initial_state = fix_initial_state
        
        # Training hyperparameters
        self.training_params = training_params
        self.n_episodes = training_params.get('n_episodes', 1000)
        self.max_time = training_params.get('max_time', 20.0)
        self.sigma = training_params.get('sigma', 0.1)
        self.tau = training_params.get('tau', 1.0)
        self.actor_lr = training_params.get('actor_lr', 1e-3)
        self.critic_lr = training_params.get('critic_lr', 1e-3)
        self.scale = training_params.get('scale', 1.0)
        self.clip_gradient = training_params.get('clip_gradient', 10.0)
        self.clip_action = training_params.get('clip_action', 10.0)
        self.divergence_threshold = training_params.get('divergence_threshold', 50.0)
        self.decay_noise = decay_noise  # Use constructor argument, not training_params
        
        # Effective sigma (updated during training if decay_noise=True)
        self._sigma_effective = self.sigma
        
        # Cost matrices
        self.Q = Q
        self.R = R
        
        # Random generator
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Compute optimal solution if needed for oracles
        self._P_opt = None
        self._K_opt = None
        if self.actor_oracle or self.critic_oracle:
            self._compute_optimal_solution()
        
        # Build networks
        self.actor = self._build_actor(actor_params)
        self.critic = self._build_critic(critic_params)

    def _compute_optimal_solution(self):
        """Compute optimal P and K for oracles."""
        self._P_opt = scipy.linalg.solve_continuous_are(
            self.env.A, self.env.B, self.Q, self.R
        )
        self._K_opt = np.linalg.inv(self.R) @ self.env.B.T @ self._P_opt

    # =========================================================================
    # Network Construction
    # =========================================================================
    
    def _build_actor(self, params: dict) -> ActorNetwork:
        """Build the actor network."""
        input_dim = params.get('input_dim', self.env.N)
        output_dim = params.get('output_dim', self.env.B.shape[1])
        actor = ActorNetwork(input_dim, output_dim, rng=self.rng)
        
        if self.actor_oracle:
            actor.W = -self._K_opt.T # type: ignore
        
        return actor

    def _build_critic(self, params: dict) -> CriticNetwork:
        """Build the critic network."""
        input_dim = params.get('input_dim', self.env.N)
        
        if self.critic_oracle:
            return CriticOracle(input_dim, P=self._P_opt, rng=self.rng) # type: ignore
        
        return CriticNetwork(input_dim, rng=self.rng)

    # =========================================================================
    # Episode Initialization
    # =========================================================================
    
    def _sample_initial_state(self) -> np.ndarray:
        """Sample initial state for an episode."""
        if self.fix_initial_state:
            return np.ones(self.env.N)*3
        return self.rng.standard_normal(self.env.N)

    # =========================================================================
    # Action Selection
    # =========================================================================
    
    def _select_action(self, x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select action with exploration noise."""
        mu = self.actor(x_scaled)
        if self.decay_noise:
            self._sigma_effective = self.sigma * max(0.0001, 1.0 - self.episode / self.n_episodes)
        else:
            self._sigma_effective = self.sigma
        noise = self._sigma_effective * self.rng.standard_normal(mu.shape)
        action = mu + noise
        action = np.clip(action, -self.clip_action, self.clip_action)
        return action, mu, noise

    # =========================================================================
    # Reward Computation
    # =========================================================================
    
    def _compute_reward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: #no more used, in the environment
        """Compute instantaneous reward (negative cost for LQR)."""
        return -1*(x.T @ self.Q @ x + u.T @ self.R @ u)

    # =========================================================================
    # Value Derivative Estimation
    # =========================================================================
    
    def _compute_V_dot(self, ctx: StepContext) -> float:
        """Compute time derivative of value function (Euler forward difference)."""
        return (ctx.V_next - ctx.V_t) / ctx.dt

    # =========================================================================
    # TD Error Computation (flag-based)
    # =========================================================================
    
    def _compute_td_error(self, ctx: StepContext) -> float:
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
    
    def _compute_critic_gradient(self, ctx: StepContext) -> np.ndarray:
        """Compute gradient for critic update based on flags."""
        features_current = self.critic._compute_features(ctx.x_scaled)
        
        if self.semi_gradient:
            return features_current
        
        # Full gradient
        features_next = self.critic._compute_features(ctx.x_next_scaled)
        grad = features_current - features_next
        
        if self.discounted:
            grad += (ctx.dt / self.tau) * features_current
        
        return grad

    # =========================================================================
    # Critic Update (flag-based)
    # =========================================================================
    
    def _update_critic(self, td_error: float, gradient: np.ndarray, dt: float) -> np.ndarray:
        """Update critic weights."""
        if self.critic_oracle:
            return np.zeros_like(gradient)
        update = gradient * td_error
        update = np.clip(update, -self.clip_gradient, self.clip_gradient)
        self.critic.W += self.critic_lr * update * dt
        return update

    # =========================================================================
    # Actor Gradient
    # =========================================================================
    
    def _compute_actor_gradient(self, ctx: StepContext) -> np.ndarray:
        """Compute gradient for actor update (Gullapalli/REINFORCE-style).
        
        Uses the effective sigma (which may decay) for proper gradient scaling.
        """
        return np.outer(ctx.x_scaled, ctx.noise) / (self._sigma_effective ** 2)

    # =========================================================================
    # Actor Update (flag-based)
    # =========================================================================
    
    def _update_actor(self, td_error: float, gradient: np.ndarray, dt: float) -> np.ndarray:
        """Update actor weights."""
        if self.actor_oracle:
            return np.zeros_like(gradient)
        
        update = gradient * td_error
        update = np.clip(update, -self.clip_gradient, self.clip_gradient)
        self.actor.W += self.actor_lr * update * dt
        return update

    # =========================================================================
    # Episode Termination
    # =========================================================================
    
    def _is_episode_done(self, x: np.ndarray, t: float) -> bool:
        """Check if episode should terminate."""
        if t >= self.max_time:
            return True
        if np.linalg.norm(x) > self.divergence_threshold:
            return True
        return False

    # =========================================================================
    # Single Training Step (orchestration)
    # =========================================================================
    
    def _train_step(self, x_t: np.ndarray) -> tuple[np.ndarray, StepMetrics, StepContext]:
        """Execute a single training step.
        
        This method orchestrates the training step by calling
        the overridable methods in sequence.
        
        Args:
            x_t: Current state
            
        Returns:
            (x_next, metrics, context): Next state, step metrics, and full context
        """
        x_scaled = x_t / self.scale
        
        # Action selection
        action, mu, noise = self._select_action(x_scaled)
        
        # Environment step
        time_series, data_series, reward = self.env.step(action)
        x_next = data_series
        if x_next.ndim == 0:
            x_next = np.array([x_next])
        x_next_scaled = x_next / self.scale
        dt = self.env.step_size
        if dt <= 1e-9:
            dt = 1e-4  # Avoid division by zero
        
        # Value function evaluations
        V_t = self.critic(np.array(x_scaled))
        V_next = self.critic(x_next_scaled)
        
        # Build context
        ctx = StepContext(
            x_t=x_t,
            x_scaled=x_scaled,
            x_next=x_next,
            x_next_scaled=x_next_scaled,
            mu=mu,
            noise=noise,
            action=action,
            dt=dt,
            time_series=time_series, # type: ignore
            data_series=data_series, # type: ignore
            V_t=V_t,
            V_next=V_next,
        )
        
        # Compute reward
        ctx.reward = reward #type: ignore
        
        # Compute V dot and TD error
        ctx.V_dot = self._compute_V_dot(ctx)
        ctx.td_error = self._compute_td_error(ctx)
        
        # Critic update
        critic_grad = self._compute_critic_gradient(ctx)
        critic_update = self._update_critic(ctx.td_error, critic_grad, dt)
        
        # Actor update
        actor_grad = self._compute_actor_gradient(ctx)
        actor_update = self._update_actor(ctx.td_error, actor_grad, dt)
        
        # Compute metrics
        loss = 0.5 * ctx.td_error ** 2
        metrics = StepMetrics(
            loss=loss * dt,
            reward=ctx.reward * dt,
            actor_gradient=actor_update,
            critic_gradient=critic_update,
        )
        
        return x_next, metrics, ctx

    # =========================================================================
    # Training Loop
    # =========================================================================
    
    def _on_episode_start(self, episode: int, x_init: np.ndarray) -> None:
        """Hook called at the start of each episode. Override for custom logic."""
        pass
    
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
        if self.actor_oracle: 
            flags.append("A*")
        if self.critic_oracle: 
            flags.append("C*")
        flag_str = f"[{','.join(flags)}] " if flags else ""
        
        return (
            f"{flag_str}Ep {episode+1} | "
            f"R: {episode_metrics['cost'][0]:.2f}, "
            f"L: {episode_metrics['loss'][0]:.4f}, "
            f"C: {[f'{w:.2f}' for w in self.critic.W]}"
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
        }
        
        log_interval = self.training_params.get('log_interval', 50)
        init_log_interval = self.training_params.get('init_log_interval', 5000)
        memory_clear_interval = self.training_params.get('memory_clear_interval', 20)
        
        iterator = tqdm.trange(self.n_episodes, desc="Training", leave=True)
        
        for episode in iterator:
            # Episode initialization
            self.episode = episode
            x_t = self._sample_initial_state()
            
            if episode % init_log_interval == 0:
                metrics_history['init_conditions'].append(x_t.copy())
            
            self.env.x0 = x_t
            self.env.reset()
            self._on_episode_start(episode, x_t)
            
            # Episode accumulators
            episode_loss = 0.0
            episode_cost = 0.0
            actor_grad_sum = 0.0
            critic_grad_sum = 0.0
            n_steps = 0
            
            # Episode loop
            while not self._is_episode_done(x_t, self.env.t):
                x_t, step_metrics, ctx = self._train_step(x_t)
                
                episode_loss += step_metrics.loss
                episode_cost += step_metrics.reward
                actor_grad_sum += step_metrics.actor_gradient
                critic_grad_sum += step_metrics.critic_gradient
                n_steps += 1
                
                # Memory management
                if n_steps % memory_clear_interval == 0:
                    self.env._data.clear()
                    self.env._time.clear()
            
            # Episode metrics
            episode_metrics = {
                'loss': episode_loss,
                'cost': episode_cost,
                'actor_grad_mean': actor_grad_sum / max(n_steps, 1),
                'critic_grad_mean': critic_grad_sum / max(n_steps, 1),
                'n_steps': n_steps,
            }
            
            self._on_episode_end(episode, episode_metrics)
            
            # Update progress bar
            iterator.set_description(self._format_progress(episode, episode_metrics))
            
            # Log metrics
            if episode % log_interval == 0:
                metrics_history['loss_episodic'].append(episode_loss)
                metrics_history['cost_episodic'].append(episode_cost)
                metrics_history['gradient_actor'].append(episode_metrics['actor_grad_mean'])
                metrics_history['gradient_critic'].append(episode_metrics['critic_grad_mean'])
                metrics_history['actor_weights'].append(self.actor.W.copy())
                metrics_history['critic_weights'].append(self.critic.W.copy())
        
        return metrics_history


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    
    # Setup simple 1D LQR
    A = np.array([[-0.1]])
    B = np.array([[1.0]])
    A1 = np.zeros_like(A)
    delay = np.array([0.0])
    x0 = np.array([1.0])
    Q=np.eye(1),
    R=np.eye(1)
    
    env = Environment(A=A, B=B, A1=A1, delay=delay, x0=x0, step_size=0.01, resolution=10, Q=Q, R=R)
    
    training_params = {
        'n_episodes': 200,
        'max_time': 5.0,
        'sigma': 0.5,
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'log_interval': 50,
        'tau': 1.0,
    }
    
    print("=" * 60)
    print("Testing CTAC (undiscounted, full gradient)")
    print("=" * 60)
    agent = CTAC(
        env=env,
        actor_params={},
        critic_params={},
        training_params=training_params,
        Q=np.eye(1),
        R=np.eye(1),
    )
    metrics = agent.train()
    print(f"Final critic: {agent.critic.W}, actor: {agent.actor.W.flatten()}\n")
    
    print("=" * 60)
    print("Testing CTAC with discounted=True")
    print("=" * 60)
    env.reset(hard_reset=True)
    agent = CTAC(
        env=env,
        actor_params={},
        critic_params={},
        training_params=training_params,
        Q=np.eye(1),
        R=np.eye(1),
        discounted=True,
        semi_gradient=True,
        fix_initial_state=True,
    )
    metrics = agent.train()
    print(f"Final critic: {agent.critic.W}, actor: {agent.actor.W.flatten()}\n")
    
    print("=" * 60)
    print("Testing CTAC with semi_gradient=True")
    print("=" * 60)
    env.reset(hard_reset=True)
    agent = CTAC(
        env=env,
        actor_params={},
        critic_params={},
        training_params=training_params,
        Q=np.eye(1),
        R=np.eye(1),
        semi_gradient=True,
        fix_initial_state=True,
    )
    metrics = agent.train()
    print(f"Final critic: {agent.critic.W}, actor: {agent.actor.W.flatten()}\n")
    
    print("=" * 60)
    print("Testing CTAC with actor_oracle=True + semi_gradient=True")
    print("=" * 60)
    env.reset(hard_reset=True)
    agent = CTAC(
        env=env,
        actor_params={},
        critic_params={},
        training_params=training_params,
        Q=np.eye(1),
        R=np.eye(1),
        actor_oracle=True,
        semi_gradient=True,
    )
    metrics = agent.train()
    print(f"Final critic: {agent.critic.W}, actor: {agent.actor.W.flatten()}\n")
    
    print("=" * 60)
    print("Testing CTAC with critic_oracle=True + discounted=True")
    print("=" * 60)
    env.reset(hard_reset=True)
    agent = CTAC(
        env=env,
        actor_params={},
        critic_params={},
        training_params=training_params,
        Q=np.eye(1),
        R=np.eye(1),
        critic_oracle=True,
        discounted=True,
    )
    metrics = agent.train()
    print(f"Final critic: {agent.critic.W}, actor: {agent.actor.W.flatten()}\n")