from typing import Optional
from legacy_code.env_rk import Environment # type: ignore
import numpy as np
import tqdm



class Critic1D:
    """1D critic for sanity checks: V(x) = α * x²
    """
    def __init__(self, alpha_init: float = 0.0, rng:Optional[np.random.Generator]=None):
        self.alpha = alpha_init  # Scalar parameter
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def __call__(self, x): 
        # V(x) = α * x²
        return self.alpha * x**2
    
    def gradient(self, x):
        # ∂V/∂α = x²
        return x**2


class CriticSanityCheck1D:
    """
    Sanity Check pour le Critic seul sur un système 1D.
    
    Système: ẋ = -x (stable, pas de contrôle)
    Reward: r(x) = -x²
    Politique: u = 0 (pas d'action)
    
    Value Function analytique avec discount τ:
        V(x) = -τ/(2τ+1) * x²
    
    Le critic doit apprendre α* = -τ/(2τ+1)
    
    TD Error: δ(t) = r(t) - V(t)/τ + V̇(t)
    Au point fixe: δ = 0 partout
    """
    
    def __init__(self, 
                 env: Environment,
                 tau: float = 1.0,
                 critic_lr: float = 1e-3,
                 n_episodes: int = 1000,
                 max_time: float = 5.0,
                 rng: Optional[np.random.Generator] = None):
        
        self.env = env
        self.tau = tau
        self.critic_lr = critic_lr
        self.n_episodes = n_episodes
        self.max_time = max_time
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Critic: V(x) = α * x²
        self.critic = Critic1D(alpha_init=0.0, rng=self.rng)
        
        # Analytical solution: α* = -τ/(2τ+1)
        self.alpha_star = -self.tau / (2 * self.tau + 1)
        
    def _compute_reward(self, x):
        """r(x) = -x²"""
        return -x**2
    
    def train(self):
        """
        Entraîne le critic pour apprendre V(x) = α* x²
        
        TD Error: δ = r - V/τ + V̇
        
        Semi-gradient TD: on minimise δ² en ne dérivant que V (pas V̇)
        ∂(½δ²)/∂α ≈ δ · ∂δ/∂α = δ · ∂(-V/τ)/∂α = δ · (-x²/τ)
        
        Update: α ← α - η · δ · (-x²/τ) = α + η · δ · x²/τ
        
        On peut simplifier en utilisant juste x² comme feature:
        α ← α + η · δ · x²
        """
        
        alpha_history = []
        td_error_history = []
        
        iterator = tqdm.trange(self.n_episodes, desc="Critic Training")
        
        for episode in iterator:
            # Random initial state
            x_t = self.rng.standard_normal() * 2.0
            self.env.x0 = np.array([x_t])
            self.env.reset(hard_reset=False)
            
            done = False
            episode_td_errors = []
            
            while not done:
                # Pas d'action (u = 0)
                action = np.array([0.0])
                
                # Environment step
                time_series, data = self.env.step(action)
                x_next = data[-1][0]  # Scalar
                dt = time_series[-1] - time_series[0]
                
                if dt < 1e-9:
                    dt = 1e-4
                
                # Reward
                reward = self._compute_reward(x_t)
                
                # Value function
                V_t = self.critic(x_t)
                V_next = self.critic(x_next)
                V_dot = (V_next - V_t) / dt
                
                # TD Error: δ = r - V/τ + V̇
                td_error = reward - V_t / self.tau + V_dot
                
                # Semi-gradient update: α ← α + η · δ · x² · dt
                # Le + car ∂δ/∂α = -x²/τ donc -∂δ/∂α = +x²/τ ∝ +x²
                gradient = self.critic.gradient(x_t)  # = x²
                self.critic.alpha += self.critic_lr * td_error * gradient * dt
                
                episode_td_errors.append(td_error**2)
                x_t = x_next
                
                # Memory cleanup
                if len(self.env._data) > 20:
                    self.env._data.clear()
                    self.env._time.clear()
                
                if self.env.t >= self.max_time:
                    done = True
                if abs(x_t) < 1e-6:  # Converged to 0
                    done = True
            
            # Logging
            if episode % 10 == 0:
                alpha_history.append(self.critic.alpha)
                td_error_history.append(np.mean(episode_td_errors) if episode_td_errors else 0)
            
            iterator.set_description(
                f"Episode {episode+1} | α={self.critic.alpha:.4f} | "
                f"α*={self.alpha_star:.4f} | TD²={np.mean(episode_td_errors):.2e}"
            )
        
        results = {
            'alpha_history': alpha_history,
            'td_error_history': td_error_history,
            'alpha_learned': self.critic.alpha,
            'alpha_star': self.alpha_star,
            'error': abs(self.critic.alpha - self.alpha_star),
            'tau': self.tau
        }
        
        return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("TEST: Critic Sanity Check (Fixed Policy Evaluation)")
    print("=" * 60)
    print()
    print("Système: ẋ = -x (stable, 1D)")
    print("Reward: r(x) = -x²")
    print("Politique: u = 0 (gelée)")
    print()
    print("Value Function analytique: V(x) = α* x²")
    print("où α* = -τ/(2τ+1)")
    print()
    print("Le Critic doit apprendre α → α*")
    print("=" * 60)
    
    # Setup environment 1D: ẋ = -x
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    A1 = np.zeros_like(A)
    delay = np.array([0.0])
    x0 = np.array([1.0])
    step_size = 0.01
    resolution = 10
    
    env = Environment(
        A=A, B=B, A1=A1, delay=delay, x0=x0,
        step_size=step_size, resolution=resolution
    )
    
    # Test avec différentes valeurs de τ
    tau_values = [0.5, 1.0, 2.0, 5.0]
    results_all = []
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, tau in enumerate(tau_values):
        print(f"\n{'='*40}")
        print(f"Test avec τ = {tau}")
        alpha_star = -tau / (2 * tau + 1)
        print(f"α* = -τ/(2τ+1) = {alpha_star:.6f}")
        print(f"{'='*40}")
        
        # Reset environment
        env.reset(hard_reset=True)
        
        agent = CriticSanityCheck1D(
            env=env,
            tau=tau,
            critic_lr=5e-3,
            n_episodes=500,
            max_time=5.0,
            rng=np.random.default_rng(42)
        )
        
        results = agent.train()
        results_all.append(results)
        
        print(f"\nRÉSULTATS pour τ={tau}:")
        print(f"  α* (analytique) = {results['alpha_star']:.6f}")
        print(f"  α  (appris)     = {results['alpha_learned']:.6f}")
        print(f"  Erreur          = {results['error']:.6f}")
        print(f"  Convergé        = {'OUI ✓' if results['error'] < 0.05 else 'NON ✗'}")
        
        # Plot
        ax = axes[i]
        steps = np.arange(len(results['alpha_history'])) * 10
        ax.plot(steps, results['alpha_history'], 'b-', linewidth=2, label='α appris')
        ax.axhline(y=results['alpha_star'], color='r', linestyle='--', 
                   linewidth=2, label=f"α* = {results['alpha_star']:.4f}")
        ax.set_xlabel('Épisode')
        ax.set_ylabel('α (paramètre du Critic)')
        ax.set_title(f'τ = {tau} | Erreur finale = {results["error"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sanity Check Critic: Apprentissage de V(x) = αx² pour ẋ = -x', fontsize=14)
    plt.tight_layout()
    plt.savefig('critic_sanity_check.png', dpi=150)
    plt.show()
    
    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    all_passed = True
    for r in results_all:
        status = "✓ PASS" if r['error'] < 0.05 else "✗ FAIL"
        all_passed = all_passed and (r['error'] < 0.05)
        print(f"τ = {r['tau']:.1f}: α* = {r['alpha_star']:.4f}, "
              f"α = {r['alpha_learned']:.4f}, erreur = {r['error']:.4f} [{status}]")
    
    print("=" * 60)
    if all_passed:
        print("✓ TOUS LES TESTS PASSÉS!")
        print("  Le TD error continu est correctement implémenté:")
        print("  δ = r - V/τ + V̇")
    else:
        print("✗ CERTAINS TESTS ONT ÉCHOUÉ!")
        print("  Vérifier: 1) signe de V̇, 2) terme de discount V/τ, 3) learning rate")
    print("=" * 60)