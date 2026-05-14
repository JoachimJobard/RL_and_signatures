"""
Unified evaluation module for all CTAC agent variants.

Provides common evaluation functions that work with any agent implementing
the TrainableAgent protocol.

This replaces: evaluate_signatures.py, evaluate_base_jax.py, evaluate_CSAC.py
"""

from collections import defaultdict
import numpy as np
import jax
import jax.numpy as jnp
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib

from src.utils.dynamic_signature import SlidingSignature
matplotlib.use('Agg')  # Non-interactive backend for wandb logging
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Any, Protocol, runtime_checkable, Callable
import pickle
from pathlib import Path


# =============================================================================
# Agent Protocol for Evaluation
# =============================================================================

@runtime_checkable
class EvaluableAgent(Protocol):
    """Protocol for agents that can be evaluated."""
    env: Any
    wrapper: Any
    training: Any
    algorithm: Any  
    sliding_signature: SlidingSignature
    signature_conf: Any
    
    def _fill_buffer_initial(self) -> None: ...
    def update_buffer(self, x: np.ndarray) -> None: ...
    def get_eval_action(self, x_scaled: np.ndarray) -> np.ndarray: ...
    def get_value(self) -> float: ...

def _resolve_burning_steps(agent: EvaluableAgent, burning_steps: int | None) -> int:
    """Resolve evaluation burn-in steps from override or agent defaults."""
    if burning_steps is not None:
        return max(0, int(burning_steps))

    if hasattr(agent, 'algorithm') and hasattr(agent.algorithm, 'burning_steps'):
        try:
            return max(0, int(agent.algorithm.burning_steps))  
        except (TypeError, ValueError):
            pass

    return 0


# =============================================================================
# Training Metrics Logging
# =============================================================================

def log_training_metrics(metrics: dict, log_interval: int = 1) -> None:
    """Log training metrics to wandb (subsampled for efficiency)."""
    cost = metrics.get('cost_episodic', [])
    loss = metrics.get('loss_episodic', [])
    grad_critic = metrics.get('gradient_critic', [])
    grad_actor = metrics.get('gradient_actor', [])
    
    for i in range(0, len(cost), log_interval):
        log_dict = {"train/cost": cost[i] if i < len(cost) else 0}
        if i < len(loss):
            log_dict["train/loss"] = loss[i]
        if i < len(grad_critic):
            log_dict["train/critic_grad"] = grad_critic[i]
        if i < len(grad_actor):
            log_dict["train/actor_grad"] = grad_actor[i]
        wandb.log(log_dict, step=i)


def plot_training_metrics(metrics: dict) -> go.Figure:
    """Create comprehensive dashboard of training metrics."""
    # Determine available metrics
    has_actor = 'actor_weights' in metrics and len(metrics.get('actor_weights', [])) > 0
    has_signature = 'signature_weights' in metrics and len(metrics.get('signature_weights', [])) > 0
    
    n_cols = 3
    n_rows = 2 if (has_actor or has_signature) else 1
    
    subplot_titles = ['Episodic Cost', 'Episodic Loss', 'Gradient Magnitude']
    if n_rows == 2:
        subplot_titles.extend(['Critic Weights', 'Actor Weights' if has_actor else 'N/A', 
                               'Signature Features' if has_signature else 'N/A'])
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12, horizontal_spacing=0.08
    )
    
    # Row 1: Scalars
    if 'cost_episodic' in metrics:
        fig.add_trace(go.Scatter(y=metrics['cost_episodic'], mode='lines', name='Cost'), row=1, col=1)
    if 'loss_episodic' in metrics:
        fig.add_trace(go.Scatter(y=metrics['loss_episodic'], mode='lines', name='Loss'), row=1, col=2)
    if 'gradient_critic' in metrics:
        fig.add_trace(go.Scatter(y=metrics['gradient_critic'], mode='lines', name='Critic Grad'), row=1, col=3)
    if 'gradient_actor' in metrics:
        fig.add_trace(go.Scatter(y=metrics['gradient_actor'], mode='lines', name='Actor Grad', 
                                  line=dict(dash='dash')), row=1, col=3)
    
    # Row 2: Weights
    if n_rows == 2:
        if 'critic_weights' in metrics and len(metrics['critic_weights']) > 0:
            critic_weights = np.array(metrics['critic_weights'])
            for i in range(min(5, critic_weights.shape[1])):
                fig.add_trace(go.Scatter(y=critic_weights[:, i, 0], mode='lines', 
                                        name=f'C{i}', opacity=0.7), row=2, col=1)
        
        if has_actor:
            actor_weights = np.array(metrics['actor_weights'])
            for i in range(min(6, actor_weights.shape[1])):
                fig.add_trace(go.Scatter(y=actor_weights[:, i, 0], mode='lines', 
                                        name=f'A{i}', opacity=0.7), row=2, col=2)
        
        if has_signature:
            sig_weights = np.array(metrics['signature_weights'])
            max_points = min(1000, sig_weights.shape[0])
            for i in range(min(6, sig_weights.shape[1])):
                fig.add_trace(go.Scatter(y=sig_weights[:max_points, i], mode='lines', 
                                        name=f'Sig{i}', opacity=0.7), row=2, col=3)
    
    fig.update_layout(height=350 * n_rows, width=1200, title_text='Training Metrics', showlegend=False)
    return fig


def get_training_summary(metrics: dict) -> dict:
    """Compute summary statistics from training metrics."""
    summary = {}
    
    if 'cost_episodic' in metrics and len(metrics['cost_episodic']) > 0:
        cost = np.array(metrics['cost_episodic'])
        summary["train/final_cost_mean"] = float(np.mean(cost[-10:]))
        summary["train/final_cost_std"] = float(np.std(cost[-10:]))
        summary["train/min_cost"] = float(np.min(cost))
        summary["train/total_episodes"] = len(cost)
    
    if 'loss_episodic' in metrics and len(metrics['loss_episodic']) > 0:
        loss = np.array(metrics['loss_episodic'])
        summary["train/final_loss_mean"] = float(np.mean(loss[-10:]))
    
    return summary


# =============================================================================
# Trajectory Simulation
# =============================================================================

def simulate_trajectory(
    agent: EvaluableAgent,
    x0: np.ndarray,
    T_sim: float,
    seed: int = 123,
    burning_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a trajectory using the trained agent.
    
    Works with any agent that has:
    - env, wrapper, scale attributes
    - _fill_buffer_initial() method
    - update_buffer() and get_eval_action() handling the internal state
    
    Parameters
    ----------
    agent : EvaluableAgent
        Trained agent
    x0 : np.ndarray
        Initial state
    T_sim : float
        Simulation time
    seed : int
        Random seed
    burning_steps : int | None
        Number of zero-control burn-in steps before logging trajectories.
        If None, uses agent default (if available).
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (states, actions, times)
    """
    env = agent.env
    n_steps = int(T_sim / env.step_size)
    
    # Reset agent state
    if hasattr(agent, 'sliding_signature'):
        agent.sliding_signature.reset() 
    
    key = jax.random.PRNGKey(seed)
    x_t = agent.wrapper.reset(key, x0=x0, t0=0.0)
    agent._fill_buffer_initial()
    
    burn_steps = _resolve_burning_steps(agent, burning_steps)

    # Burn-in: advance system with zero action before trajectory logging.
    for _ in range(burn_steps):
        action_burn = jnp.zeros(env.B.shape[1])
        _, x_t, _ = agent.wrapper.step(agent.wrapper.state, action_burn)
        agent.update_buffer(np.array(x_t))
    
    # Initialize states and times AFTER preheat
    states = [np.array(x_t).flatten()]
    actions = []
    times = [float(agent.wrapper.state.t)]
    
    for step in range(n_steps):
        x_scaled = x_t / agent.training.scale
        
        # Delegate action generation to the agent itself
        action = agent.get_eval_action(x_scaled)
        action = np.clip(np.array(action).flatten(), -10, 10)
        actions.append(action)
        
        t, x_next, _ = agent.wrapper.step(agent.wrapper.state, action)
        
        # Delegate buffer updates to the agent
        agent.update_buffer(np.array(x_next))
        
        x_t = x_next
        states.append(np.array(x_next).flatten())
        times.append(float(t))
        
    return np.array(states), np.array(actions), np.array(times)


def simulate_uncontrolled_trajectory(
    agent: EvaluableAgent,
    x0: np.ndarray,
    T_sim: float,
    seed: int = 123,
    burning_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an uncontrolled trajectory (u=0)."""
    env = agent.env
    n_steps = int(T_sim / env.step_size)
    action_dim = env.B.shape[1]
    
    # Reset agent state
    if hasattr(agent, 'sliding_signature'):
        agent.sliding_signature.reset() 
    
    key = jax.random.PRNGKey(seed)
    x_t = agent.wrapper.reset(key, x0=x0, t0=0.0)
    agent._fill_buffer_initial()
    
    burn_steps = _resolve_burning_steps(agent, burning_steps)

    # Burn-in: advance uncontrolled system before logging trajectories.
    for _ in range(burn_steps):
        action_burn = jnp.zeros(action_dim)
        _, x_t, _ = agent.wrapper.step(agent.wrapper.state, action_burn)
        agent.update_buffer(np.array(x_t))
    
    # Initialize states and times AFTER preheat
    states = [np.array(x_t).flatten()]
    times = [float(agent.wrapper.state.t)]
    
    for step in range(n_steps):
        action_zero = jnp.zeros(action_dim)
        t, x_next, _ = agent.wrapper.step(agent.wrapper.state, action_zero)
        
        # Update buffer so it stays consistent (same as controlled trajectory)
        agent.update_buffer(np.array(x_next))
        
        x_t = x_next
        states.append(np.array(x_next).flatten())
        times.append(float(t))
    
    return np.array(states), np.array(times)


# =============================================================================
# Cost Computation
# =============================================================================

def compute_trajectory_cost(
    states: np.ndarray, 
    actions: np.ndarray, 
    Q: np.ndarray, 
    R: np.ndarray, 
    step_size: float,
    x_target: np.ndarray | None = None
) -> float:
    """Compute total cost for a trajectory.
    
    Parameters
    ----------
    x_target : np.ndarray, optional
        Target state. If None, assumes target is origin (0).
    """
    if x_target is None:
        x_target = np.zeros(states.shape[1])
    x_target = np.array(x_target).flatten()
    
    total = 0.0
    for i in range(len(actions)):
        error = states[i] - x_target
        ut = actions[i].reshape(-1)
        total += (error.T @ Q @ error + ut.T @ R @ ut) * step_size
    return float(total)


# =============================================================================
# Comparison Figures
# =============================================================================

def compare_with_no_control(
    agent: EvaluableAgent,
    x0: np.ndarray,
    T_sim: float,
    seed: int = 456,
    burning_steps: int | None = None,
) -> tuple[go.Figure, dict]:
    """Compare agent vs uncontrolled system with figure and metrics."""
    env = agent.env
    Q, R = np.array(env.Q), np.array(env.R)
    step_size = env.step_size
    
    # Get x_target if environment has one (e.g., Mackey-Glass), otherwise None
    x_target = np.array(env.x_target).flatten() if hasattr(env, 'x_target') else None
    has_target = x_target is not None and np.any(x_target != 0)
    
    # For cost computation, use x_target if available, else origin
    x_ref = x_target if x_target is not None else np.zeros(x0.shape[0])
    
    states_agent, actions_agent, times = simulate_trajectory(
        agent,
        x0,
        T_sim,
        seed,
        burning_steps=burning_steps,
    )
    states_no_ctrl, _ = simulate_uncontrolled_trajectory(
        agent,
        x0,
        T_sim,
        seed,
        burning_steps=burning_steps,
    )
    
    # Compute costs (relative to x_ref)
    n_actions = len(actions_agent)
    cost_agent = [
        (states_agent[i] - x_ref).T @ Q @ (states_agent[i] - x_ref) + 
        actions_agent[i].reshape(-1).T @ R @ actions_agent[i].reshape(-1) 
        for i in range(n_actions)
    ]
    cost_no_ctrl = [
        (states_no_ctrl[i] - x_ref).T @ Q @ (states_no_ctrl[i] - x_ref) 
        for i in range(min(n_actions, len(states_no_ctrl)-1))
    ]
    cum_cost_agent = np.cumsum(cost_agent) * step_size
    cum_cost_no_ctrl = np.cumsum(cost_no_ctrl) * step_size
    
    # Figure - adapt subplot title based on whether we have a target
    error_title = 'Error from Target' if has_target else 'State Norm'
    fig = make_subplots(rows=2, cols=3, subplot_titles=(
        'State Evolution', 'Control Actions', 'Cumulative Cost',
        'Phase Portrait', error_title, 'Instantaneous Cost'
    ), vertical_spacing=0.12, horizontal_spacing=0.08)
    
    # State evolution (with target line only if has_target)
    for i in range(states_agent.shape[1]):
        fig.add_trace(go.Scatter(x=times, y=states_agent[:, i], mode='lines', 
                                name=f'Agent x{i}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=states_no_ctrl[:, i], mode='lines', 
                                name=f'NoCtrl x{i}', line=dict(dash='dash')), row=1, col=1)
        # Add target line only if target exists and is non-zero
        if has_target and i < len(x_ref):
            fig.add_trace(go.Scatter(x=[times[0], times[-1]], y=[x_ref[i], x_ref[i]], 
                                    mode='lines', name=f'Target x{i}', 
                                    line=dict(dash='dot', color='red')), row=1, col=1)
    
    # Control actions
    for i in range(actions_agent.shape[1]):
        fig.add_trace(go.Scatter(x=times[:-1], y=actions_agent[:, i], mode='lines', 
                                name=f'Agent u{i}'), row=1, col=2)
    # Cumulative cost
    fig.add_trace(go.Scatter(x=times[:len(cum_cost_agent)], y=cum_cost_agent, 
                            mode='lines', name='Agent'), row=1, col=3)
    fig.add_trace(go.Scatter(x=times[:len(cum_cost_no_ctrl)], y=cum_cost_no_ctrl, 
                            mode='lines', name='NoCtrl', line=dict(dash='dash')), row=1, col=3)
    
    # Phase portrait (if 2D or more) - else show state over time
    if states_agent.shape[1] >= 2:
        fig.add_trace(go.Scatter(x=states_agent[:, 0], y=states_agent[:, 1], 
                                mode='lines', name='Agent'), row=2, col=1)
        fig.add_trace(go.Scatter(x=states_no_ctrl[:, 0], y=states_no_ctrl[:, 1], 
                                mode='lines', name='NoCtrl', line=dict(dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=[x0[0]], y=[x0[1]], mode='markers', 
                                name='x0', marker=dict(size=10)), row=2, col=1)
        # Add target point only if has_target and 2D
        if has_target and len(x_ref) >= 2:
            fig.add_trace(go.Scatter(x=[x_ref[0]], y=[x_ref[1]], mode='markers', 
                                    name='Target', marker=dict(size=12, symbol='star', color='red')), row=2, col=1)
    else:
        # 1D case: show state over time
        fig.add_trace(go.Scatter(x=times, y=states_agent[:, 0], mode='lines', 
                                name='Agent'), row=2, col=1)
        fig.add_trace(go.Scatter(x=times, y=states_no_ctrl[:, 0], mode='lines', 
                                name='NoCtrl', line=dict(dash='dash')), row=2, col=1)
    
    # Error/norm from reference (target if exists, else origin)
    error_agent = np.linalg.norm(states_agent - x_ref, axis=1)
    error_no_ctrl = np.linalg.norm(states_no_ctrl - x_ref, axis=1)
    fig.add_trace(go.Scatter(x=times, y=error_agent, mode='lines', name='Agent'), row=2, col=2)
    fig.add_trace(go.Scatter(x=times, y=error_no_ctrl, mode='lines', 
                            name='NoCtrl', line=dict(dash='dash')), row=2, col=2)
    
    # Instantaneous cost
    fig.add_trace(go.Scatter(x=times[:len(cost_agent)], y=cost_agent, mode='lines', 
                            name='Agent', opacity=0.7), row=2, col=3)
    fig.add_trace(go.Scatter(x=times[:len(cost_no_ctrl)], y=cost_no_ctrl, mode='lines', 
                            name='NoCtrl', line=dict(dash='dash'), opacity=0.7), row=2, col=3)
    
    # Title with target info only if has_target
    target_str = f", target={x_ref}" if has_target else ""
    fig.update_layout(height=700, width=1200, title_text=f'Agent vs No Control (x0={x0}{target_str})', showlegend=True)
    
    # Metrics
    cost_reduction_pct = 100 * (cum_cost_no_ctrl[-1] - cum_cost_agent[-1]) / cum_cost_no_ctrl[-1] \
        if cum_cost_no_ctrl[-1] > 0 else 0
    
    metrics: dict[str, float | list[float]] = {
        "eval/total_cost_agent": float(cum_cost_agent[-1]),
        "eval/total_cost_no_control": float(cum_cost_no_ctrl[-1]),
        "eval/cost_reduction_pct": float(cost_reduction_pct),
        "eval/final_error_agent": float(error_agent[-1]),
        "eval/final_error_no_control": float(error_no_ctrl[-1]),
    }
    # Only add x_target to metrics if it exists
    if has_target:
        metrics["eval/x_target"] = np.asarray(x_ref, dtype=float).tolist()
    
    return fig, metrics


def evaluate_multiple_trajectories(
    agent: EvaluableAgent,
    x0_list: list[np.ndarray],
    T_sim: float,
    base_seed: int = 100,
    burning_steps: int | None = None,
) -> tuple[go.Figure, dict]:
    """Evaluate on multiple initial conditions with figure."""
    env = agent.env
    Q, R = np.array(env.Q), np.array(env.R)
    step_size = env.step_size
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('State Norms', 'Cumulative Costs', 'Final Costs'))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    costs, final_norms = [], []
    for idx, x0 in enumerate(x0_list):
        states, actions, times = simulate_trajectory(
            agent,
            x0,
            T_sim,
            base_seed + idx,
            burning_steps=burning_steps,
        )
        cost = compute_trajectory_cost(states, actions, Q, R, step_size)
        costs.append(cost)
        final_norms.append(np.linalg.norm(states[-1]))
        
        norm = np.linalg.norm(states, axis=1)
        cum_cost = np.cumsum([
            states[i].T @ Q @ states[i] + actions[i].reshape(-1).T @ R @ actions[i].reshape(-1) 
            for i in range(len(actions))
        ]) * step_size
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(x=times, y=norm, mode='lines', 
                                name=f'x0={idx}', line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=times[:-1], y=cum_cost, mode='lines', 
                                line=dict(color=color), showlegend=False), row=1, col=2)
    
    fig.add_trace(go.Bar(x=[f'x0_{i}' for i in range(len(costs))], y=costs, 
                        marker_color=colors[:len(costs)]), row=1, col=3)
    fig.update_layout(height=400, width=1200, title_text='Multiple Initial Conditions')
    
    metrics = {
        "eval/multi_cost_mean": float(np.mean(costs)),
        "eval/multi_cost_std": float(np.std(costs)),
        "eval/multi_final_norm_mean": float(np.mean(final_norms)),
        "eval/n_trajectories": len(x0_list),
    }
    return fig, metrics


# =============================================================================
# Periodic Trajectory Snapshots (wandb slider)
# =============================================================================

def create_trajectory_snapshot(
    agent: EvaluableAgent,
    x0: np.ndarray,
    T_sim: float,
    episode: int,
    seed: int = 789,
    burning_steps: int | None = None,
) -> matplotlib.figure.Figure:
    """
    Create a matplotlib figure showing state trajectories and control actions.
    
    Used for periodic evaluation during training: logged to wandb with
    episode as the step, which creates an interactive slider in the wandb UI.
    
    Parameters
    ----------
    agent : EvaluableAgent
        The agent (at current training state)
    x0 : np.ndarray
        Initial state for evaluation
    T_sim : float
        Simulation horizon
    episode : int
        Current training episode (used in the title)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    env = agent.env
    Q, R_mat = np.array(env.Q), np.array(env.R)
    
    # Get target if the environment has one
    x_target = np.array(env.x_target).flatten() if hasattr(env, 'x_target') else None
    has_target = x_target is not None and np.any(x_target != 0)
    x_ref = x_target if x_target is not None else np.zeros(x0.shape[0])
    
    # --- Simulate controlled + uncontrolled ---
    states_agent, actions_agent, times = simulate_trajectory(
        agent,
        x0,
        T_sim,
        seed,
        burning_steps=burning_steps,
    )
    states_no_ctrl, times_nc = simulate_uncontrolled_trajectory(
        agent,
        x0,
        T_sim,
        seed,
        burning_steps=burning_steps,
    )
    
    state_dim = states_agent.shape[1]
    action_dim = actions_agent.shape[1]
    
    # --- Cumulative cost ---
    n_actions = len(actions_agent)
    cost_agent = np.array([
        (states_agent[i] - x_ref) @ Q @ (states_agent[i] - x_ref)
        + actions_agent[i].reshape(-1) @ R_mat @ actions_agent[i].reshape(-1)
        for i in range(n_actions)
    ])
    cost_no_ctrl = np.array([
        (states_no_ctrl[i] - x_ref) @ Q @ (states_no_ctrl[i] - x_ref)
        for i in range(min(n_actions, len(states_no_ctrl) - 1))
    ])
    cum_cost_agent = np.cumsum(cost_agent) * env.step_size
    cum_cost_no_ctrl = np.cumsum(cost_no_ctrl) * env.step_size
    
    # --- Create figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Episode {episode} — x0={np.round(x0, 2)}', fontsize=13)
    
    # (0,0) State evolution
    ax = axes[0, 0]
    for i in range(state_dim):
        ax.plot(times, states_agent[:, i], label=f'Agent $x_{i}$')
        ax.plot(times, states_no_ctrl[:, i], '--', alpha=0.5, label=f'No ctrl $x_{i}$')
        if has_target and i < len(x_ref):
            ax.axhline(x_ref[i], color='red', ls=':', alpha=0.4)
    ax.set_title('State evolution')
    ax.set_xlabel('t')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # (0,1) Control actions
    ax = axes[0, 1]
    t_actions = times[:len(actions_agent)]
    for i in range(action_dim):
        ax.plot(t_actions, actions_agent[:, i], label=f'$u_{i}$')
    ax.set_title('Control actions')
    ax.set_xlabel('t')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # (1,0) Cumulative cost
    ax = axes[1, 0]
    ax.plot(times[:len(cum_cost_agent)], cum_cost_agent, label='Agent')
    ax.plot(times[:len(cum_cost_no_ctrl)], cum_cost_no_ctrl, '--', label='No control')
    ax.set_title('Cumulative cost')
    ax.set_xlabel('t')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # (1,1) Error norm from target/origin
    ax = axes[1, 1]
    error_agent = np.linalg.norm(states_agent - x_ref, axis=1)
    error_no_ctrl = np.linalg.norm(states_no_ctrl - x_ref, axis=1)
    ax.plot(times, error_agent, label='Agent')
    ax.plot(times_nc, error_no_ctrl, '--', label='No control')
    ax.set_title('Error from target' if has_target else 'State norm')
    ax.set_xlabel('t')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def log_trajectory_snapshot(
    agent: EvaluableAgent,
    x0: np.ndarray,
    T_sim: float,
    episode: int,
    seed: int = 789,
    wandb_key: str = "eval/trajectory_evolution",
    burning_steps: int | None = None,
) -> None:
    """
    Evaluate the agent, create a snapshot figure, and log it to wandb.
    
    When called repeatedly with increasing `episode`, wandb creates
    an interactive slider to visualize how the control evolves over training.
    
    Parameters
    ----------
    agent : EvaluableAgent
        Agent at current training state
    x0 : np.ndarray
        Initial condition for evaluation
    T_sim : float
        Simulation horizon
    episode : int
        Current episode (used as wandb step for the slider)
    seed : int
        Random seed
    wandb_key : str
        wandb logging key
    """
    fig = create_trajectory_snapshot(
        agent,
        x0,
        T_sim,
        episode,
        seed,
        burning_steps=burning_steps,
    )
    # Use a custom step key so snapshots don't conflict with the global wandb step
    step_key = wandb_key + "_step"
    wandb.log({wandb_key: wandb.Image(fig), step_key: episode})
    plt.close(fig)


def make_eval_callback(
    x0: np.ndarray,
    T_sim: float,
    eval_interval: int = 100,
    seed: int = 789,
    wandb_key: str = "eval/trajectory_evolution",
    burning_steps: int | None = None,
) -> Callable:
    """
    Factory that returns a callback suitable for setting on an agent.
    
    The returned callable has signature ``callback(agent, episode)``
    and logs a trajectory snapshot to wandb every ``eval_interval`` episodes.
    
    Usage
    -----
    >>> agent.eval_callback = make_eval_callback(x0, T_sim, eval_interval=100)
    >>> agent.train()  # callback is invoked automatically inside the loop
    
    Parameters
    ----------
    x0 : np.ndarray
        Evaluation initial condition
    T_sim : float
        Simulation time
    eval_interval : int
        Evaluate every N episodes
    seed : int
        Random seed for trajectory simulation
    wandb_key : str
        wandb log key (creates the slider panel)
        
    Returns
    -------
    Callable[[EvaluableAgent, int], None]
    """
    # Define a custom x-axis for the snapshot images so they don't
    # collide with the global wandb step used by log_training_metrics
    step_key = wandb_key + "_step"
    wandb.define_metric(step_key, hidden=True)
    wandb.define_metric(wandb_key, step_metric=step_key)

    def _callback(agent: EvaluableAgent, episode: int) -> None:
        if episode % eval_interval != 0:
            return
        log_trajectory_snapshot(
            agent,
            x0,
            T_sim,
            episode,
            seed,
            wandb_key,
            burning_steps=burning_steps,
        )
    
    return _callback


# =============================================================================
# Data Export
# =============================================================================

def collect_evaluation_data(
    agent: EvaluableAgent,
    x0: np.ndarray,
    T_sim: float,
    seed: int = 456,
    burning_steps: int | None = None,
) -> dict:
    """Collect all evaluation data for external plotting."""
    env = agent.env
    Q, R = np.array(env.Q), np.array(env.R)
    step_size = env.step_size
    
    states_agent, actions_agent, times = simulate_trajectory(
        agent,
        x0,
        T_sim,
        seed,
        burning_steps=burning_steps,
    )
    states_no_ctrl, _ = simulate_uncontrolled_trajectory(
        agent,
        x0,
        T_sim,
        seed,
        burning_steps=burning_steps,
    )
    
    n_actions = len(actions_agent)
    cost_agent = [
        states_agent[i].T @ Q @ states_agent[i] + 
        actions_agent[i].reshape(-1).T @ R @ actions_agent[i].reshape(-1) 
        for i in range(n_actions)
    ]
    cost_no_ctrl = [
        states_no_ctrl[i].T @ Q @ states_no_ctrl[i] 
        for i in range(min(n_actions, len(states_no_ctrl)-1))
    ]
    
    return {
        'times': times,
        'states_agent': states_agent,
        'states_no_ctrl': states_no_ctrl,
        'actions': actions_agent,
        'cost_agent': np.array(cost_agent),
        'cost_no_ctrl': np.array(cost_no_ctrl),
        'x0': x0,
        'T_sim': T_sim,
        'Q': Q,
        'R': R,
        'step_size': step_size,
        'cum_cost_agent': np.cumsum(cost_agent) * step_size,
        'cum_cost_no_ctrl': np.cumsum(cost_no_ctrl) * step_size,
    }


def collect_multiple_trajectories_data(
    agent: EvaluableAgent,
    x0_list: list[np.ndarray],
    T_sim: float,
    base_seed: int = 100,
    burning_steps: int | None = None,
) -> dict:
    """Collect data for multiple trajectories."""
    all_data = []
    for idx, x0 in enumerate(x0_list):
        data = collect_evaluation_data(
            agent,
            x0,
            T_sim,
            base_seed + idx,
            burning_steps=burning_steps,
        )
        all_data.append(data)
    return {'trajectories': all_data, 'n_trajectories': len(x0_list)}


def save_evaluation_data(data: dict, filepath: str | Path) -> None:
    """Save evaluation data to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Evaluation data saved to {filepath}")


def load_evaluation_data(filepath: str | Path) -> dict:
    """Load evaluation data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_training_metrics(metrics: dict, filepath: str | Path) -> None:
    """Save training metrics history to a separate pickle file.
    
    Converts all list values to numpy arrays for efficient storage.
    Stored keys typically include:
        - cost_episodic, loss_episodic
        - gradient_critic, gradient_actor
        - critic_weights, actor_weights, signature_weights
    
    Parameters
    ----------
    metrics : dict
        Raw metrics dictionary returned by agent.train()
    filepath : str | Path
        Output path (e.g. "run_training_metrics.pkl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {}
    for key, value in metrics.items():
        if isinstance(value, list) and len(value) > 0:
            try:
                data[key] = np.array(value)
            except (ValueError, TypeError):
                data[key] = value
        else:
            data[key] = value
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Training metrics saved to {filepath}")


def load_training_metrics(filepath: str | Path) -> dict:
    """Load training metrics from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_statistics_visited_states(metrics, discretization_state) -> go.Figure:
    state_counts = metrics['state_counts'].counter
    tuple_size = len(next(iter(state_counts)))

    fig = make_subplots(
        rows=1,
        cols=tuple_size,
        subplot_titles=[f"State Dimension {i}" for i in range(tuple_size)]
    )

    for dim in range(tuple_size):
        agg = defaultdict(int)
        for state, count in state_counts.items():
            agg[state[dim]] += count

        x = list(agg.keys())
        x = [elem*discretization_state for elem in x]
        y = list(agg.values())

        fig.add_trace(
            go.Bar(x=x, y=y, name=f"Dim {dim}"),
            row=1,
            col=dim + 1
        )
    
    fig.update_layout(height=400, width=1200, title_text='Visited States Distribution')
    return fig