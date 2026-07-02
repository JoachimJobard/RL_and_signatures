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
import matplotlib

from src.utils.dynamic_signature import SlidingSignature
from src.utils.plot_style import (
    STROKE_TRAINED, STROKE_REFERENCE, STROKE_AUXILIARY,
    sequential_colors, prepare_figure,
)
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


def plot_training_metrics(metrics: dict) -> matplotlib.figure.Figure:
    """Create a comprehensive dashboard of training metrics (Matplotlib).

    All plotted curves are measured training quantities and therefore solid
    (``STROKE_TRAINED``); the actor-gradient overlay is the sole exception, drawn
    dashed only to distinguish it from the critic gradient in a shared panel.
    """
    has_actor = 'actor_weights' in metrics and len(metrics.get('actor_weights', [])) > 0
    has_signature = 'signature_weights' in metrics and len(metrics.get('signature_weights', [])) > 0

    n_cols = 3
    n_rows = 2 if (has_actor or has_signature) else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows + 1.2),
                             squeeze=False)
    fig.suptitle("Training metrics", fontsize=13)

    # Row 1: scalar training signals.
    ax = axes[0, 0]
    if 'cost_episodic' in metrics:
        ax.plot(metrics['cost_episodic'], STROKE_TRAINED, label='Cost')
    ax.set_title('Episodic cost')
    ax.set_ylabel(r"$\sum_t (x^\top Q x + u^\top R u)$")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if 'loss_episodic' in metrics:
        ax.plot(metrics['loss_episodic'], STROKE_TRAINED, label='Loss')
    ax.set_title('Episodic loss')
    ax.set_ylabel(r"$\frac{1}{2}\,\delta^2$")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if 'gradient_critic' in metrics:
        ax.plot(metrics['gradient_critic'], STROKE_TRAINED, label='Critic grad')
    if 'gradient_actor' in metrics:
        ax.plot(metrics['gradient_actor'], STROKE_REFERENCE, label='Actor grad')
    ax.set_title('Gradient magnitude')
    ax.set_ylabel(r"$\|\nabla\|_2$")
    ax.grid(True, alpha=0.3)

    # Row 2: weight / feature trajectories (sequential colour over the index).
    if n_rows == 2:
        ax = axes[1, 0]
        if 'critic_weights' in metrics and len(metrics['critic_weights']) > 0:
            cw = np.array(metrics['critic_weights'])
            cols = sequential_colors(min(5, cw.shape[1]))
            for i in range(min(5, cw.shape[1])):
                ax.plot(cw[:, i, 0], STROKE_TRAINED, color=cols[i], alpha=0.8,
                        label=f'C{i}')
        ax.set_title('Critic weights')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if has_actor:
            aw = np.array(metrics['actor_weights'])
            cols = sequential_colors(min(6, aw.shape[1]))
            for i in range(min(6, aw.shape[1])):
                ax.plot(aw[:, i, 0], STROKE_TRAINED, color=cols[i], alpha=0.8,
                        label=f'A{i}')
        ax.set_title('Actor weights' if has_actor else 'N/A')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        if has_signature:
            sw = np.array(metrics['signature_weights'])
            max_points = min(1000, sw.shape[0])
            cols = sequential_colors(min(6, sw.shape[1]))
            for i in range(min(6, sw.shape[1])):
                ax.plot(sw[:max_points, i], STROKE_TRAINED, color=cols[i], alpha=0.8,
                        label=f'Sig{i}')
        ax.set_title('Signature features' if has_signature else 'N/A')
        ax.grid(True, alpha=0.3)

    for col in range(n_cols):
        axes[n_rows - 1, col].set_xlabel("episode")

    # External legend (collected across panels) + formula box, with overlap check.
    prepare_figure(
        fig, fname="figure_training_metrics", axes=list(axes.ravel()),
        reserve_bottom=0.22, legend_fontsize=7,
        formula=(r"$\delta = r + \dot V - V/\tau$ (continuous-time TD error); "
                 r"running cost $x^\top Q x + u^\top R u$. All curves are measured "
                 r"training quantities (solid stroke)."),
    )
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
    t0 = float(agent.wrapper.state.t)
    states = [np.array(x_t).flatten()]
    actions = []
    times = [0.0]
    
    for step in range(n_steps):
        x_scaled = x_t / agent.training.scale
        
        # Delegate action generation to the agent itself
        action = agent.get_eval_action(x_scaled)
        action = np.array(action).flatten()
        actions.append(action)
        
        t, x_next, _ = agent.wrapper.step(agent.wrapper.state, action)
        
        # Delegate buffer updates to the agent
        agent.update_buffer(np.array(x_next))
        
        x_t = x_next
        states.append(np.array(x_next).flatten())
        times.append(float(t) - t0)
        
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
    t0 = float(agent.wrapper.state.t)
    states = [np.array(x_t).flatten()]
    times = [0.0]
    
    for step in range(n_steps):
        action_zero = jnp.zeros(action_dim)
        t, x_next, _ = agent.wrapper.step(agent.wrapper.state, action_zero)
        
        # Update buffer so it stays consistent (same as controlled trajectory)
        agent.update_buffer(np.array(x_next))
        
        x_t = x_next
        states.append(np.array(x_next).flatten())
        times.append(float(t) - t0)
    
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
        error_next = states[i+1] - x_target
        ut = actions[i].reshape(-1)
        total += (error_next.T @ Q @ error_next + ut.T @ R @ ut) * step_size
    return float(total)


# =============================================================================
# Comparison Figures
# =============================================================================

def plot_agent_vs_no_control_from_data(data: dict, *, title: str | None = None) -> matplotlib.figure.Figure:
    """Build the agent-vs-no-control figure from collected data alone (no agent),
    so the figure can be rebuilt and restyled a posteriori from the saved data
    (see collect_evaluation_data). ``title`` overrides the default figure title.

    Stroke convention: controlled (agent) trajectory solid (``STROKE_TRAINED``),
    uncontrolled baseline dashed (``STROKE_REFERENCE``), target / set-point dotted
    (``STROKE_AUXILIARY``).
    """
    times = np.asarray(data['times'])
    states_agent = np.asarray(data['states_agent'])
    states_no_ctrl = np.asarray(data['states_no_ctrl'])
    actions_agent = np.asarray(data['actions'])
    cost_agent = np.asarray(data['cost_agent'])
    cost_no_ctrl = np.asarray(data['cost_no_ctrl'])
    cum_cost_agent = np.asarray(data['cum_cost_agent'])
    cum_cost_no_ctrl = np.asarray(data['cum_cost_no_ctrl'])
    x0 = np.asarray(data['x0'])
    x_ref = np.asarray(data['x_ref'])
    has_target = bool(data['has_target'])
    n_dim = states_agent.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # (0,0) State evolution — agent solid, uncontrolled dashed, target dotted.
    ax = axes[0, 0]
    for i in range(n_dim):
        ax.plot(times, states_agent[:, i], STROKE_TRAINED, label=f'Agent $x_{i}$')
        ax.plot(times, states_no_ctrl[:, i], STROKE_REFERENCE, alpha=0.7,
                label=f'No ctrl $x_{i}$')
        if has_target and i < len(x_ref):
            ax.axhline(x_ref[i], color='red', ls=STROKE_AUXILIARY, alpha=0.6,
                       label=f'Target $x_{i}$')
    ax.set_title('State evolution'); ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x(t)$")
    ax.grid(True, alpha=0.3)

    # (0,1) Control actions (trained, solid).
    ax = axes[0, 1]
    for i in range(actions_agent.shape[1]):
        ax.plot(times[:len(actions_agent)], actions_agent[:, i], STROKE_TRAINED,
                label=f'$u_{i}$')
    ax.set_title('Control actions'); ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$u(t)$")
    ax.grid(True, alpha=0.3)

    # (0,2) Cumulative cost.
    ax = axes[0, 2]
    ax.plot(times[:len(cum_cost_agent)], cum_cost_agent, STROKE_TRAINED, label='_nolegend_')
    ax.plot(times[:len(cum_cost_no_ctrl)], cum_cost_no_ctrl, STROKE_REFERENCE,
            alpha=0.7, label='_nolegend_')
    ax.set_title('Cumulative cost'); ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\int_0^t c\,ds$")
    ax.grid(True, alpha=0.3)

    # (1,0) Phase portrait (>=2D) or state-over-time (1D).
    ax = axes[1, 0]
    if n_dim >= 2:
        ax.plot(states_agent[:, 0], states_agent[:, 1], STROKE_TRAINED, label='_nolegend_')
        ax.plot(states_no_ctrl[:, 0], states_no_ctrl[:, 1], STROKE_REFERENCE,
                alpha=0.7, label='_nolegend_')
        ax.plot([x0[0]], [x0[1]], marker='o', ms=8, ls='none', color='k', label='$x_0$')
        if has_target and len(x_ref) >= 2:
            ax.plot([x_ref[0]], [x_ref[1]], marker='*', ms=12, ls='none',
                    color='red', label='Target')
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    else:
        ax.plot(times, states_agent[:, 0], STROKE_TRAINED, label='_nolegend_')
        ax.plot(times, states_no_ctrl[:, 0], STROKE_REFERENCE, alpha=0.7, label='_nolegend_')
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x(t)$")
    ax.set_title('Phase portrait' if n_dim >= 2 else 'State')
    ax.grid(True, alpha=0.3)

    # (1,1) Error / norm from reference (target if present, else origin).
    ax = axes[1, 1]
    error_agent = np.linalg.norm(states_agent - x_ref, axis=1)
    error_no_ctrl = np.linalg.norm(states_no_ctrl - x_ref, axis=1)
    ax.plot(times, error_agent, STROKE_TRAINED, label='_nolegend_')
    ax.plot(times, error_no_ctrl, STROKE_REFERENCE, alpha=0.7, label='_nolegend_')
    ax.set_title('Error from target' if has_target else 'State norm')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|x-x_{\mathrm{ref}}\|_2$" if has_target else r"$\|x\|_2$")
    ax.grid(True, alpha=0.3)

    # (1,2) Instantaneous cost.
    ax = axes[1, 2]
    ax.plot(times[:len(cost_agent)], cost_agent, STROKE_TRAINED, alpha=0.8, label='_nolegend_')
    ax.plot(times[:len(cost_no_ctrl)], cost_no_ctrl, STROKE_REFERENCE, alpha=0.7,
            label='_nolegend_')
    ax.set_title('Instantaneous cost'); ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$c(t)$")
    ax.grid(True, alpha=0.3)

    target_str = f", target={x_ref}" if has_target else ""
    default_title = f"Agent vs no control ($x_0={x0}{target_str}$)"
    fig.suptitle(title if title is not None else default_title, fontsize=12)

    prepare_figure(
        fig, fname="figure_agent_vs_no_control", axes=list(axes.ravel()),
        reserve_bottom=0.22, legend_fontsize=7,
        formula=(r"Controlled (agent) trajectory: solid; uncontrolled baseline: "
                 r"dashed; target / set-point: dotted. Instantaneous cost "
                 r"$c=(x-x_{\mathrm{ref}})^\top Q (x-x_{\mathrm{ref}}) + u^\top R u$; "
                 r"cumulative cost $\int_0^t c\,ds$."),
    )
    return fig


def compare_with_no_control(
    agent: EvaluableAgent,
    x0: np.ndarray,
    T_sim: float,
    seed: int = 456,
    burning_steps: int | None = None,
) -> tuple[matplotlib.figure.Figure, dict]:
    """Compare agent vs uncontrolled system: collect the trajectory data, then build
    the figure from it. Returns (figure, metrics); the same data drives the replot."""
    data = collect_evaluation_data(agent, x0, T_sim, seed, burning_steps=burning_steps)
    return plot_agent_vs_no_control_from_data(data), data['eval_metrics']


def plot_multiple_trajectories_from_data(multi_data: dict, *, title: str | None = None) -> matplotlib.figure.Figure:
    """Build the multiple-initial-conditions figure from collected data alone (no
    agent), so it can be rebuilt and restyled a posteriori. ``title`` overrides the
    default. Colour encodes the initial-condition index (sequential viridis)."""
    trajectories = multi_data['trajectories']
    n = len(trajectories)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = sequential_colors(n)

    totals = []
    for idx, d in enumerate(trajectories):
        times = np.asarray(d['times'])
        states = np.asarray(d['states_agent'])
        cum_cost = np.asarray(d['cum_cost_agent'])
        norm = np.linalg.norm(states, axis=1)
        totals.append(float(cum_cost[-1]) if len(cum_cost) else 0.0)
        color = colors[idx]
        # Trained trajectories are solid; the sweep over initial conditions is in colour.
        axes[0].plot(times, norm, STROKE_TRAINED, color=color, label=f'$x_0$={idx}')
        axes[1].plot(times[:len(cum_cost)], cum_cost, STROKE_TRAINED, color=color)

    axes[2].bar([f'$x_0$={i}' for i in range(n)], totals, color=colors[:n])

    axes[0].set_title('State norms'); axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$\|x(t)\|_2$"); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Cumulative costs'); axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$\int_0^t (x^\top Q x + u^\top R u)\,ds$")
    axes[1].grid(True, alpha=0.3)
    axes[2].set_title('Final costs'); axes[2].set_xlabel("initial condition")
    axes[2].set_ylabel("total cost"); axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', labelrotation=45)

    fig.suptitle(title if title is not None else 'Multiple initial conditions', fontsize=12)
    prepare_figure(
        fig, fname="figure_multiple_trajectories", axes=list(axes.ravel()),
        reserve_bottom=0.26, legend_fontsize=7,
        formula=(r"One controlled trajectory per initial condition (solid; colour = "
                 r"initial-condition index, viridis). Total cost "
                 r"$\int_0^T (x^\top Q x + u^\top R u)\,ds$."),
    )
    return fig


def _qualitative_colors(n: int) -> list[str]:
    """Return ``n`` distinct qualitative colours for a *categorical* axis.

    Used to distinguish representations (markovian / raw-history / signature),
    which form a categorical set, not a scalar sweep — so a qualitative palette
    (``tab10``) is correct here, whereas a sweep over a hyperparameter would use
    :func:`sequential_colors`. Falls back to cycling ``tab20`` past 10 entries.
    """
    from matplotlib.colors import to_hex
    cmap = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    m = 10 if n <= 10 else 20
    return [to_hex(cmap(i % m)) for i in range(n)]


def plot_representation_comparison_from_data(
    variants: list[tuple[str, dict]],
    *,
    title: str | None = None,
    state_index: int = 0,
) -> matplotlib.figure.Figure:
    """Overlay the closed-loop trajectories of several representations on shared axes.

    Each entry of ``variants`` is ``(label, eval_data)`` where ``eval_data`` is the
    dict produced by :func:`collect_evaluation_data` (saved as ``eval.pkl``). This
    is the cross-representation comparison figure — distinct from the per-variant
    :func:`plot_agent_vs_no_control_from_data` — answering "how does the controlled
    trajectory differ across representations on the same plant".

    Panels (2x2): controlled state component ``state_index`` (or its norm for a
    multi-dimensional state), control ``u_0(t)``, error from target
    ``||x - x_ref||_2``, and cumulative cost ``int_0^t c\\,ds``.

    Stroke convention (repo-wide): the controlled trajectories are *trained* outputs
    and therefore solid (``STROKE_TRAINED``), each representation in a distinct
    qualitative colour; the shared uncontrolled baseline is a reference and is dashed
    (``STROKE_REFERENCE``); the target / set-point is auxiliary and is dotted
    (``STROKE_AUXILIARY``). The representation axis is categorical, so colour is
    qualitative (``tab10``), not a sequential sweep.
    """
    if not variants:
        raise ValueError("plot_representation_comparison_from_data: no variants given")

    labels = [lab for lab, _ in variants]
    colors = _qualitative_colors(len(variants))

    # Env-level fields are shared across variants (same plant); read from the first.
    ref0 = variants[0][1]
    has_target = bool(ref0["has_target"])
    x_ref = np.asarray(ref0["x_ref"])
    n_dim = np.asarray(ref0["states_agent"]).shape[1]
    multi_dim = n_dim > 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) Controlled state: component `state_index` for 1D, else the L2 norm.
    ax = axes[0, 0]
    for (lab, d), c in zip(variants, colors):
        t = np.asarray(d["times"])
        s = np.asarray(d["states_agent"])
        y = np.linalg.norm(s, axis=1) if multi_dim else s[:, state_index]
        ax.plot(t, y, STROKE_TRAINED, color=c, label=lab)
    # Shared uncontrolled baseline (reference, dashed) drawn once.
    s_nc = np.asarray(ref0["states_no_ctrl"])
    t_nc = np.asarray(ref0["times"])
    y_nc = np.linalg.norm(s_nc, axis=1) if multi_dim else s_nc[:, state_index]
    ax.plot(t_nc, y_nc, STROKE_REFERENCE, color="0.4", alpha=0.8, label="no control")
    if has_target and not multi_dim and state_index < len(x_ref):
        ax.axhline(float(x_ref[state_index]), color="red", ls=STROKE_AUXILIARY,
                   alpha=0.7, label="target")
    ax.set_title(r"Controlled state $\|x(t)\|_2$" if multi_dim
                 else rf"Controlled state $x_{state_index}(t)$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|x(t)\|_2$" if multi_dim else rf"$x_{state_index}(t)$")
    ax.grid(True, alpha=0.3)

    # (0,1) Control signal u_0(t).
    ax = axes[0, 1]
    for (lab, d), c in zip(variants, colors):
        a = np.asarray(d["actions"])
        t = np.asarray(d["times"])[:len(a)]
        ax.plot(t, a[:, 0], STROKE_TRAINED, color=c, label=lab)
    ax.set_title(r"Control action $u_0(t)$")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$u_0(t)$")
    ax.grid(True, alpha=0.3)

    # (1,0) Error from target (or state norm if no target).
    ax = axes[1, 0]
    for (lab, d), c in zip(variants, colors):
        ax.plot(np.asarray(d["times"]), np.asarray(d["error_agent"]),
                STROKE_TRAINED, color=c, label=lab)
    ax.plot(t_nc, np.asarray(ref0["error_no_ctrl"]), STROKE_REFERENCE,
            color="0.4", alpha=0.8, label="no control")
    ax.set_title("Error from target" if has_target else "State norm")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|x-x_{\mathrm{ref}}\|_2$" if has_target else r"$\|x\|_2$")
    ax.grid(True, alpha=0.3)

    # (1,1) Cumulative cost (log scale: spans orders of magnitude across reps).
    ax = axes[1, 1]
    for (lab, d), c in zip(variants, colors):
        cc = np.asarray(d["cum_cost_agent"])
        t = np.asarray(d["times"])[:len(cc)]
        ax.plot(t, cc, STROKE_TRAINED, color=c, label=lab)
    cc_nc = np.asarray(ref0["cum_cost_no_ctrl"])
    ax.plot(t_nc[:len(cc_nc)], cc_nc, STROKE_REFERENCE, color="0.4", alpha=0.8,
            label="no control")
    ax.set_title("Cumulative cost")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\int_0^t c\,ds$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(title if title is not None else
                 "Controlled trajectories by representation", fontsize=13)
    prepare_figure(
        fig, fname="figure_representation_comparison", axes=list(axes.ravel()),
        reserve_bottom=0.20, legend_fontsize=8,
        formula=(r"Controlled trajectory per representation: solid (colour = "
                 r"representation); shared uncontrolled baseline: dashed; target: "
                 r"dotted. Cost $c=(x-x_{\mathrm{ref}})^\top Q (x-x_{\mathrm{ref}}) "
                 r"+ u^\top R u$."),
    )
    return fig


def evaluate_multiple_trajectories(
    agent: EvaluableAgent,
    x0_list: list[np.ndarray],
    T_sim: float,
    base_seed: int = 100,
    burning_steps: int | None = None,
) -> tuple[matplotlib.figure.Figure, dict]:
    """Evaluate on multiple initial conditions: collect the data, then build the
    figure from it. Returns (figure, metrics); the same data drives the replot."""
    data = collect_multiple_trajectories_data(agent, x0_list, T_sim, base_seed,
                                              burning_steps=burning_steps)
    return plot_multiple_trajectories_from_data(data), data['metrics']


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
        (states_agent[i+1] - x_ref) @ Q @ (states_agent[i+1] - x_ref)
        + actions_agent[i].reshape(-1) @ R_mat @ actions_agent[i].reshape(-1)
        for i in range(n_actions)
    ])
    cost_no_ctrl = np.array([
        (states_no_ctrl[i+1] - x_ref) @ Q @ (states_no_ctrl[i+1] - x_ref)
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


def conform_initial_state(x0: Any, env_dim: int | None) -> np.ndarray:
    """Resize an initial-condition vector to the environment's state dimension.

    The configured ``eval.x0_test`` may not match the dimension of the selected
    environment (e.g. the 2D default ``[1, 1]`` with the 1D Mackey-Glass env).
    Conforming it here keeps the in-training snapshot evaluation robust to that
    mismatch instead of raising a shape error inside the env step.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    if env_dim is not None and x0.shape[0] != env_dim:
        x0 = np.resize(x0, env_dim)
    return x0


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
    x0 = conform_initial_state(x0, getattr(agent.env, "N", None))
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


def make_training_snapshot_callback(
    x0: np.ndarray,
    T_sim: float,
    run_dir: str | Path,
    interval: int = 100,
    seed: int = 789,
    burning_steps: int | None = None,
) -> Callable:
    """Lightweight, wandb-free on-disk training snapshot.

    Returns a ``callback(agent, episode)`` that, every ``interval`` episodes, rolls
    out one short trajectory and OVERWRITES a compact 3-panel PNG in ``run_dir``:
      (1) controlled state x(t), (2) control signal u(t), (3) eval-cost vs episode.
    It also writes ``training_snapshot.npz`` with the snapshot arrays so the panels
    can be replotted/restyled later. Deliberately lightweight: a single overwritten
    PNG at low dpi, the figure is closed each time, and only a coarse cost history
    is kept — open ``run_dir/training_snapshot.png`` to watch progress live.
    """
    run_dir = Path(run_dir)
    x0 = np.asarray(x0, dtype=float)
    episodes: list[int] = []
    eval_costs: list[float] = []

    def _callback(agent: EvaluableAgent, episode: int) -> None:
        if interval <= 0 or episode % interval != 0:
            return
        env = agent.env
        Q, R = np.array(env.Q), np.array(env.R)
        step_size = env.step_size
        x_target = np.array(env.x_target).flatten() if hasattr(env, 'x_target') else None
        x_ref = x_target if x_target is not None else np.zeros(env.N)

        x0c = conform_initial_state(x0, getattr(env, "N", None))
        states, actions, times = simulate_trajectory(
            agent, x0c, T_sim, seed, burning_steps=burning_steps,
        )
        cost = np.array([
            (states[i + 1] - x_ref).T @ Q @ (states[i + 1] - x_ref)
            + actions[i].reshape(-1).T @ R @ actions[i].reshape(-1)
            for i in range(len(actions))
        ])
        episodes.append(int(episode))
        eval_costs.append(float(np.sum(cost) * step_size))

        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        for d in range(states.shape[1]):
            axes[0].plot(times, states[:, d], "-", lw=1.0)
        if x_target is not None and np.any(x_ref != 0):
            for d in range(len(x_ref)):
                axes[0].axhline(float(x_ref[d]), ls=":", color="r", lw=0.8)
        axes[0].set_title(f"controlled state (ep {episode})")
        axes[0].set_xlabel("t"); axes[0].set_ylabel("x(t)")
        for d in range(actions.shape[1]):
            axes[1].plot(times[:len(actions)], actions[:, d], "-", lw=1.0)
        axes[1].set_title("control signal u(t)")
        axes[1].set_xlabel("t"); axes[1].set_ylabel("u(t)")
        axes[2].plot(episodes, eval_costs, "-o", lw=1.0, ms=3)
        axes[2].set_title("eval cost vs episode")
        axes[2].set_xlabel("episode"); axes[2].set_ylabel(r"$\int_0^T c\,dt$")
        fig.tight_layout()
        fig.savefig(run_dir / "training_snapshot.png", dpi=80)
        plt.close(fig)

        np.savez(
            run_dir / "training_snapshot.npz",
            episode=np.array(episodes), eval_cost=np.array(eval_costs),
            times=times, states=states, actions=actions, x_ref=np.asarray(x_ref),
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
    """Collect every array and metric needed to (re)build the agent-vs-no-control
    figure, so the figure can be rebuilt and restyled a posteriori from the saved
    data alone (never recomputed). This is the single source of the trajectory
    arrays; the figure builder ``plot_agent_vs_no_control_from_data`` reads from
    this dict and the run path saves it (see main_unified)."""
    env = agent.env
    Q, R = np.array(env.Q), np.array(env.R)
    step_size = env.step_size

    x_target = np.array(env.x_target).flatten() if hasattr(env, 'x_target') else None
    has_target = bool(x_target is not None and np.any(x_target != 0))
    x_ref = x_target if x_target is not None else np.zeros(x0.shape[0])

    states_agent, actions_agent, times = simulate_trajectory(
        agent, x0, T_sim, seed, burning_steps=burning_steps,
    )
    states_no_ctrl, _ = simulate_uncontrolled_trajectory(
        agent, x0, T_sim, seed, burning_steps=burning_steps,
    )

    n_actions = len(actions_agent)
    cost_agent = np.array([
        (states_agent[i + 1] - x_ref).T @ Q @ (states_agent[i + 1] - x_ref)
        + actions_agent[i].reshape(-1).T @ R @ actions_agent[i].reshape(-1)
        for i in range(n_actions)
    ])
    cost_no_ctrl = np.array([
        (states_no_ctrl[i + 1] - x_ref).T @ Q @ (states_no_ctrl[i + 1] - x_ref)
        for i in range(min(n_actions, len(states_no_ctrl) - 1))
    ])
    cum_cost_agent = np.cumsum(cost_agent) * step_size
    cum_cost_no_ctrl = np.cumsum(cost_no_ctrl) * step_size
    error_agent = np.linalg.norm(states_agent - x_ref, axis=1)
    error_no_ctrl = np.linalg.norm(states_no_ctrl - x_ref, axis=1)

    reduction = (100.0 * (cum_cost_no_ctrl[-1] - cum_cost_agent[-1]) / cum_cost_no_ctrl[-1]
                 if len(cum_cost_no_ctrl) and cum_cost_no_ctrl[-1] > 0 else 0.0)
    eval_metrics: dict[str, Any] = {
        "eval/total_cost_agent": float(cum_cost_agent[-1]) if len(cum_cost_agent) else 0.0,
        "eval/total_cost_no_control": float(cum_cost_no_ctrl[-1]) if len(cum_cost_no_ctrl) else 0.0,
        "eval/cost_reduction_pct": float(reduction),
        "eval/final_error_agent": float(error_agent[-1]),
        "eval/final_error_no_control": float(error_no_ctrl[-1]),
    }
    if has_target:
        eval_metrics["eval/x_target"] = np.asarray(x_ref, dtype=float).tolist()

    return {
        'times': times,
        'states_agent': states_agent,
        'states_no_ctrl': states_no_ctrl,
        'actions': actions_agent,
        'cost_agent': cost_agent,
        'cost_no_ctrl': cost_no_ctrl,
        'cum_cost_agent': cum_cost_agent,
        'cum_cost_no_ctrl': cum_cost_no_ctrl,
        'error_agent': error_agent,
        'error_no_ctrl': error_no_ctrl,
        'x0': np.asarray(x0),
        'x_target': x_target,
        'x_ref': x_ref,
        'has_target': has_target,
        'T_sim': T_sim,
        'Q': Q,
        'R': R,
        'step_size': step_size,
        'eval_metrics': eval_metrics,
    }


def collect_multiple_trajectories_data(
    agent: EvaluableAgent,
    x0_list: list[np.ndarray],
    T_sim: float,
    base_seed: int = 100,
    burning_steps: int | None = None,
) -> dict:
    """Collect all data + metrics for multiple trajectories, so the figure is
    rebuildable a posteriori from the saved data alone."""
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
    totals = [float(d['cum_cost_agent'][-1]) if len(d['cum_cost_agent']) else 0.0
              for d in all_data]
    final_norms = [float(np.linalg.norm(d['states_agent'][-1])) for d in all_data]
    metrics = {
        "eval/multi_cost_mean": float(np.mean(totals)) if totals else 0.0,
        "eval/multi_cost_std": float(np.std(totals)) if totals else 0.0,
        "eval/multi_final_norm_mean": float(np.mean(final_norms)) if final_norms else 0.0,
        "eval/n_trajectories": len(x0_list),
    }
    return {'trajectories': all_data, 'n_trajectories': len(x0_list), 'metrics': metrics}


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
    
    data: dict[Any, Any] = {}
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


def get_statistics_visited_states(
    metrics: dict, discretization_state: float
) -> matplotlib.figure.Figure:
    state_counts = metrics['state_counts'].counter
    if not state_counts:
        # No states were recorded (e.g. a very short run): return an annotated
        # empty figure rather than crashing on next(iter(...)).
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title('Visited states distribution (no data)')
        ax.set_axis_off()
        prepare_figure(
            fig, fname="figure_visited_states", reserve_bottom=0.18,
            formula=r"No visited-state counts were recorded for this run.")
        return fig
    tuple_size = len(next(iter(state_counts)))

    fig, axes = plt.subplots(1, tuple_size, figsize=(4 * tuple_size, 4.2),
                             squeeze=False)
    axes = axes[0]
    bar_colors = sequential_colors(tuple_size)
    for dim in range(tuple_size):
        agg: defaultdict[Any, int] = defaultdict(int)
        for state, count in state_counts.items():
            agg[state[dim]] += count
        x = [elem * discretization_state for elem in agg.keys()]
        y = list(agg.values())
        ax = axes[dim]
        ax.bar(x, y, width=discretization_state, color=bar_colors[dim],
               align='center')
        ax.set_title(f"State dimension {dim}")
        ax.set_xlabel(rf"$x_{dim}$")
        ax.set_ylabel("visit count")
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Visited states distribution', fontsize=12)
    prepare_figure(
        fig, fname="figure_visited_states", axes=list(axes), reserve_bottom=0.20,
        formula=(r"Empirical visitation histogram per state dimension over training "
                 r"(states binned at resolution $\Delta x$). Diagnostic / auxiliary."),
    )
    return fig