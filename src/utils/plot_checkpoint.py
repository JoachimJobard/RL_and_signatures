"""
Script pour visualiser les métriques et les trajectoires d'un agent RL sauvegardé.
Usage :
    python plot_checkpoint.py --agent_path outputs/2026-01-21/12-00-00/checkpoint_agent.pkl --metrics_path outputs/2026-01-21/12-00-00/metrics.pkl
    
Pour utiliser avec un agent CTACJAX qui a delayed_state=True, l'environnement doit être reconstruit.
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from src.env_rk_jax import JAXDDEEnv
from src.agents.signatures_jax import CTACSignatureJAX
from src.agents.base_jax import CTACJAX


def plot_training_metrics(metrics: dict):
    """Plot training metrics: cost, loss, gradients."""
    cost = np.array(metrics['cost_episodic'])
    loss = np.array(metrics['loss_episodic'])
    grad_critic = np.array(metrics.get('gradient_critic', []))
    grad_actor = np.array(metrics.get('gradient_actor', []))
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    axs[0, 0].plot(cost)
    axs[0, 0].set_title('Episodic Cost')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Cost')
    axs[0, 0].grid(True, alpha=0.3)
    
    axs[0, 1].plot(loss)
    axs[0, 1].set_title('Episodic Loss')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True, alpha=0.3)
    
    if grad_critic.size > 0:
        axs[1, 0].plot(grad_critic)
        axs[1, 0].set_title('Critic Gradient')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Grad Critic')
        axs[1, 0].grid(True, alpha=0.3)
    
    if grad_actor.size > 0:
        axs[1, 1].plot(grad_actor)
        axs[1, 1].set_title('Actor Gradient')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Grad Actor')
        axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150)
    plt.show()


def plot_weights(metrics: dict):
    """Plot evolution of network weights."""
    if 'critic_weights' in metrics and len(metrics['critic_weights']) > 0:
        critic_weights = np.array(metrics['critic_weights'])
        plt.figure(figsize=(10, 5))
        for i in range(min(critic_weights.shape[1], 6)):
            plt.plot(critic_weights[:, i, 0], label=f'Critic W{i}', alpha=0.7)
        plt.title('Critic Weights Evolution')
        plt.xlabel('Episode')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('critic_weights.png', dpi=150)
        plt.show()
    
    if 'actor_weights' in metrics and len(metrics['actor_weights']) > 0:
        actor_weights = np.array(metrics['actor_weights'])
        plt.figure(figsize=(10, 5))
        for i in range(min(actor_weights.shape[1], 6)):
            plt.plot(actor_weights[:, i, 0], label=f'Actor W{i}', alpha=0.7)
        plt.title('Actor Weights Evolution')
        plt.xlabel('Episode')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('actor_weights.png', dpi=150)
        plt.show()
    
    if 'signature_weights' in metrics and len(metrics['signature_weights']) > 0:
        sig_weights = np.array(metrics['signature_weights'])
        plt.figure(figsize=(10, 5))
        n_points = min(1000, sig_weights.shape[0])
        for i in range(min(sig_weights.shape[1], 6)):
            plt.plot(sig_weights[:n_points, i], label=f'Sig {i}', alpha=0.7)
        plt.title('Signature Features Evolution')
        plt.xlabel('Step')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('signature_features.png', dpi=150)
        plt.show()


def simulate_and_plot_trajectory(agent, x0: np.ndarray, T_sim: float, seed: int = 123):
    """Simulate trajectory and plot states, actions, and delayed states if applicable."""
    from evaluate_base_jax import simulate_trajectory, simulate_uncontrolled_trajectory
    
    env = agent.env
    has_delayed_state = getattr(agent, 'delayed_state', False)
    
    states, actions, times, delayed_states = simulate_trajectory(agent, x0, T_sim, seed)
    states_no_ctrl, times_no_ctrl, _ = simulate_uncontrolled_trajectory(agent, x0, T_sim, seed)
    
    Q, R = np.array(env.Q), np.array(env.R)
    step_size = env.step_size
    
    # Figure 1: States comparison
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    for i in range(states.shape[1]):
        axs[0, 0].plot(times, states[:, i], label=f'Agent x{i}')
        axs[0, 0].plot(times_no_ctrl, states_no_ctrl[:, i], '--', label=f'NoCtrl x{i}', alpha=0.7)
    axs[0, 0].set_title('State Evolution')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('State')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Actions
    for i in range(actions.shape[1]):
        axs[0, 1].plot(times[:-1], actions[:, i], label=f'u{i}')
    axs[0, 1].set_title('Control Actions')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Action')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Cumulative cost
    cost_agent = [states[i].T @ Q @ states[i] + actions[i].T @ R @ actions[i] for i in range(len(actions))]
    cost_no_ctrl = [states_no_ctrl[i].T @ Q @ states_no_ctrl[i] for i in range(len(states_no_ctrl) - 1)]
    cum_cost_agent = np.cumsum(cost_agent) * step_size
    cum_cost_no_ctrl = np.cumsum(cost_no_ctrl) * step_size
    
    axs[1, 0].plot(times[:-1], cum_cost_agent, label='Agent')
    axs[1, 0].plot(times_no_ctrl[:-1], cum_cost_no_ctrl, '--', label='No Control', alpha=0.7)
    axs[1, 0].set_title('Cumulative Cost')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Cost')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # State norm
    norm_agent = np.linalg.norm(states, axis=1)
    norm_no_ctrl = np.linalg.norm(states_no_ctrl, axis=1)
    axs[1, 1].plot(times, norm_agent, label='Agent')
    axs[1, 1].plot(times_no_ctrl, norm_no_ctrl, '--', label='No Control', alpha=0.7)
    axs[1, 1].set_title('State Norm')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('||x||')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150)
    plt.show()
    
    # Figure 2: Delayed state verification (if applicable)
    if has_delayed_state and delayed_states is not None:
        delay_val = float(np.max(np.array(env.delay)))
        delay_steps = int(delay_val / step_size)
        
        fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot delayed states
        for i in range(delayed_states.shape[1]):
            axs2[0].plot(times, delayed_states[:, i], label=f'x{i}(t-τ)')
        axs2[0].set_title(f'Delayed State x(t-τ), τ={delay_val:.2f}')
        axs2[0].set_xlabel('Time')
        axs2[0].set_ylabel('x(t-τ)')
        axs2[0].legend()
        axs2[0].grid(True, alpha=0.3)
        
        # Verification: x_delayed(t) should equal x(t - tau)
        if delay_steps > 0 and delay_steps < len(times):
            x_shifted = states[:-delay_steps, :]
            x_delayed_check = delayed_states[delay_steps:, :]
            times_check = times[delay_steps:]
            min_len = min(len(x_shifted), len(x_delayed_check))
            
            axs2[1].plot(times_check[:min_len], x_shifted[:min_len, 0], label='x(t-τ) expected')
            axs2[1].plot(times_check[:min_len], x_delayed_check[:min_len, 0], '--', label='x_delayed actual')
            axs2[1].set_title('Delay Verification')
            axs2[1].set_xlabel('Time')
            axs2[1].legend()
            axs2[1].grid(True, alpha=0.3)
            
            # Error
            error = np.abs(x_shifted[:min_len] - x_delayed_check[:min_len])
            axs2[2].plot(times_check[:min_len], error[:, 0], 'r-', label='|error|')
            axs2[2].set_title(f'Delay Error (max={np.max(error):.2e})')
            axs2[2].set_xlabel('Time')
            axs2[2].set_ylabel('|x_expected - x_actual|')
            axs2[2].legend()
            axs2[2].grid(True, alpha=0.3)
            
            print(f"\n=== Delay Verification ===")
            print(f"Delay τ = {delay_val:.3f}")
            print(f"Max error: {np.max(error):.2e}")
            print(f"Mean error: {np.mean(error):.2e}")
            print(f"Delay correctly implemented: {np.max(error) < 1e-5}")
        
        plt.tight_layout()
        plt.savefig('delayed_state_verification.png', dpi=150)
        plt.show()
    
    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Final cost (agent): {cum_cost_agent[-1]:.4f}")
    print(f"Final cost (no ctrl): {cum_cost_no_ctrl[-1]:.4f}")
    cost_reduction = 100 * (cum_cost_no_ctrl[-1] - cum_cost_agent[-1]) / cum_cost_no_ctrl[-1] if cum_cost_no_ctrl[-1] > 0 else 0
    print(f"Cost reduction: {cost_reduction:.1f}%")
    print(f"Final norm (agent): {norm_agent[-1]:.4f}")


def rebuild_agent_from_checkpoint(agent_path: str):
    """Rebuild agent from checkpoint with environment."""
    with open(agent_path, 'rb') as f:
        data = pickle.load(f)
    
    env_params = data.get('env_params', {})
    training_params = data.get('training_params', {})
    
    # Rebuild environment
    env = JAXDDEEnv(
        A=env_params['A'],
        B=env_params['B'],
        A1=env_params.get('A1', np.zeros_like(env_params['A'])),
        delay=env_params.get('delay', np.array([0.0])),
        Q=env_params['Q'],
        R=env_params['R'],
        step_size=env_params.get('step_size', 0.05),
        resolution=env_params.get('resolution', 5),
    )
    
    # Check if this is a signature-based or base agent
    # Base agent has depth=2, signature agents have depth > 2
    depth = training_params.get('depth', 2)
    
    # Try to determine agent type from saved params shape
    actor_params = data['actor_params']
    
    # For CTACJAX (base), use simpler constructor
    # Check if delayed_state flag was saved
    delayed_state = data.get('delayed_state', False)
    
    agent = CTACJAX(
        env=env,
        training_params=training_params,
        Q=env_params['Q'],
        R=env_params['R'],
        rng_key=42,
        discounted=training_params.get('discounted', False),
        semi_gradient=training_params.get('semi_gradient', True),
        integral_td=training_params.get('integral_td', False),
        fix_initial_state=training_params.get('fix_initial_state', True),
        decay_noise=training_params.get('decay_noise', False),
        time_augmentation=training_params.get('time_augmentation', False),
        bias=training_params.get('bias', True),
        delayed_state=delayed_state,
    )
    
    # Load weights
    agent.actor_params = data['actor_params']
    agent.critic_params = data['critic_params']
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='Visualize RL agent checkpoints')
    parser.add_argument('--agent_path', type=str, required=True, help='Path to agent checkpoint (.pkl)')
    parser.add_argument('--metrics_path', type=str, required=True, help='Path to metrics file (.pkl)')
    parser.add_argument('--x0', type=float, nargs='+', default=None, help='Initial condition for trajectory')
    parser.add_argument('--T_sim', type=float, default=10.0, help='Simulation duration')
    parser.add_argument('--no_trajectory', action='store_true', help='Skip trajectory simulation')
    args = parser.parse_args()

    # Load metrics
    print(f"Loading metrics from {args.metrics_path}")
    with open(args.metrics_path, 'rb') as f:
        metrics = pickle.load(f)

    # Plot training metrics
    print("Plotting training metrics...")
    plot_training_metrics(metrics)
    plot_weights(metrics)

    # Trajectory simulation
    if not args.no_trajectory:
        print(f"Loading agent from {args.agent_path}")
        agent = rebuild_agent_from_checkpoint(args.agent_path)
        
        # Determine x0
        if args.x0 is not None:
            x0 = np.array(args.x0)
        elif 'init_conditions' in metrics and len(metrics['init_conditions']) > 0:
            x0 = np.array(metrics['init_conditions'][0])
        else:
            x0 = np.ones(agent.env.N)
        
        print(f"Simulating trajectory with x0={x0}, T_sim={args.T_sim}")
        simulate_and_plot_trajectory(agent, x0, args.T_sim)


if __name__ == '__main__':
    main()
