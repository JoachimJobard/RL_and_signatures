from typing import Optional
import scipy
from env_rk import Environment # type: ignore
import numpy as np
import tqdm
import gc

class CTActorCriticOracle:
    def __init__(self, 
                 env: Environment, 
                 actor_params: dict, critic_params: dict, training_params: dict, 
                 Q:np.ndarray, R:np.ndarray, 
                 rng:Optional[np.random.Generator]=None,
                 gradient_method: str = 'euler',
                 init_method: str = 'zero',
                 semi_gradient: bool = True):
        
        self.env = env
        self.env_params = {
            'A': env.A.copy(),
            'B': env.B.copy(),
            'A1': env.A1.copy(),
            'delay': env.delay.copy(),
            'x0': env.x0.copy(),
            'step_size': env.step_size,
            'resolution': env.resolution
        }
        self.init_method = init_method
        self.gradient_method = gradient_method
        self.training_params = training_params
        self.delta_index = training_params.get('delta_index', 1)
        self.tau = training_params.get('tau', 1.01)
        self.actor_lr = training_params.get('actor_lr', 1e-3)
        self.critic_lr = training_params.get('critic_lr', 1e-3)
        self.Q = Q
        self.scale = training_params.get('scale', 1.0)
        self.R = R
        self.rng = rng if rng is not None else np.random.default_rng()
        self.ricatti_solution = scipy.linalg.solve_continuous_are(self.env.A, self.env.B, self.Q, self.R)
        self.P = self.ricatti_solution
        self.actor = self._build_actor(actor_params)
        self.critic = self._build_critic(critic_params)
        
        
    def _build_actor(self, params: dict):
        input_dim = params.get('input_dim', self.env.N)
        output_dim = params.get('output_dim', self.env.B.shape[1])
        net = ActorNetwork(input_dim, output_dim, rng=self.rng)
        net.W *= 0.01 
        if self.init_method == 'zero':
            net.W = np.zeros_like(net.W)  # Initialize actor weights to zero
        elif self.init_method == 'close_to_solution':
            net.W = (-np.linalg.inv(self.R) @ self.env.B.T @ self.P).T  + self.rng.standard_normal(net.W.shape)*0.1
        # net.W = net.W.reshape(input_dim, output_dim)
        return net
    
    def _build_critic(self, params: dict):
        input_dim = params.get('input_dim', self.env.N)
        net = CriticOracle(input_dim, self.P, rng=self.rng)
        
        #verify ricatti solution
        assert(self.env.A.T @ self.P + self.P @ self.env.A - self.P @ self.env.B @ np.linalg.inv(self.R) @ self.env.B.T @ self.P + self.Q).max() <1e-5
        net.W = self.P
        net.W = np.zeros_like(net.W)  # Initialize critic weights to zero
        print(f"Critic initialized with Ricatti solution, max P element: {np.abs(net.W).max():.4f}")
        return net
    
    def _recreate_environment(self):
        del self.env
        self.env = Environment(
            **self.env_params
        )
        gc.collect()

    def train(self):
        n_episodes = self.training_params.get('n_episodes', 1000)
        sigma = self.training_params.get('sigma', 0.1)
        iterator_episodes = tqdm.trange(n_episodes, desc="Training Episodes", leave=True)
        loss_episodic_list=[]
        cost_episodic_list = []
        gradient_actor_list = []
        gradient_critic_list = []
        init_conditions_list = []
        weights_actor_list = []
        weights_critic_list = []
        difference_critic_list = []
        real_gradient = False
        differentiation_index = -1
        integral_td = False
        if self.gradient_method == 'real_gradient':
            print("Using real gradient method for actor update.")
            real_gradient = True
        elif self.gradient_method == 'precise_derivative':
            print("Using precise derivative method for actor update.")
            differentiation_index = 1
        elif self.gradient_method == 'integral_td':
            print("Using integral TD method for actor update.")
            integral_td = True
        elif self.gradient_method == 'precise_integral_td':
            print("Using precise integral TD method for actor update.")
            differentiation_index = 1
            integral_td = True
        for episode in iterator_episodes:
            # decay = max(0, 1 - episode / n_episodes /early_stop) #linear noise decay
            x_t = self.rng.standard_normal(self.env.N) #*0.1
            # x_t = np.ones(self.env.N) #fixed initial condition for testing
            if episode %5000==0:
                init_conditions_list.append(x_t)
            # if episode %1000 == 0:
            #     self._recreate_environment()
            self.env.x0 = x_t
            self.env.reset(hard_reset=False)
            done = False
            loss_sum = 0
            episodic_cost = 0
            gradient_actor_episode = 0
            gradient_critic_episode = 0
            iter = 0
            difference_vdot = 0
            while not done:
                #Actor decision and noise
                iter += 1
                x_scaled = x_t / self.scale
                mu = self.actor(x_scaled)  # Actor sees scaled input
                noise = sigma * self.rng.standard_normal(self.env.B.shape[1])
                action_applied = mu + noise
                action_applied = np.clip(action_applied, -10, 10)
                effective_noise = action_applied - mu

                #environment step
                time_serie_step, data_step = self.env.step(action_applied)
                
                x_next = data_step[-1]  # x_next is in ORIGINAL scale
                x_next_computations = data_step[differentiation_index]
                dt = time_serie_step[differentiation_index] - time_serie_step[0]
                dt_rl_algo = time_serie_step[-1] - time_serie_step[0]
                if dt <= 1e-9:
                    print("CAREFUL: dt too small, setting to 1e-4")
                    dt = 1e-4
                
                #reward
                reward = self._compute_reward(x_t, action_applied)
                
                #TD Error
                V_t = self.critic(x_scaled)
                V_next = self.critic(x_next_computations / self.scale)  # Critic sees scaled input
                V_dot = (V_next - V_t)/dt
                V_dot_real = -2 * x_t.T @ self.P @ (self.env.A @ x_t + self.env.B @ action_applied)
                difference_vdot = V_dot - V_dot_real
                V_dot = V_dot_real * real_gradient + (1 - real_gradient) * V_dot #use real gradient or not 

                td_error = reward*dt_rl_algo*(integral_td) + reward*(1-integral_td) + V_dot*dt_rl_algo*(integral_td) + V_dot*(1-integral_td) #- (1/self.tau) * V_t 
                
                #Critic update
                loss_critic = 1/2*td_error**2
                
                #actor update - gradient w.r.t. scaled input
                gradient_actor = np.outer(x_scaled, effective_noise) / (sigma**2)
                update_actor = gradient_actor * td_error
                self.actor.W += self.actor_lr * np.clip(update_actor, -100, 100)

                x_t = x_next
                #metrics
                episodic_cost += dt * reward 
                loss_sum += loss_critic * dt
                gradient_actor_episode += update_actor
                if iter % 20 == 0:
                    self.env._data.clear()
                    self.env._time.clear()
                if self.env.t >= self.training_params.get('max_time', 20):
                    done = True
                if np.linalg.norm(x_t) > 50:
                    done = True
            iterator_episodes.set_description(f"Episode {episode+1}, | Episodic reward: {episodic_cost:.2f}, Episodic Loss: {loss_sum:.2f}, actor weight: {[f'{w:.2f}' for w in self.actor.W.flatten()]}, difference V_dot: {abs(difference_vdot):.6f}")
            if episode % 20 == 0:
                weights_actor_list.append(self.actor.W.copy())
                loss_episodic_list.append(loss_sum)
                cost_episodic_list.append(episodic_cost)
                gradient_actor_list.append(gradient_actor_episode/iter) 
                gradient_critic_list.append(gradient_critic_episode/iter)
                weights_critic_list.append(self.critic.W.copy())


        
        metrics = {
            'loss_episodic': loss_episodic_list,
            'cost_episodic': cost_episodic_list,
            'gradient_actor': gradient_actor_list,
            'gradient_critic': gradient_critic_list,
            'init_conditions': init_conditions_list, 
            "actor_weights": weights_actor_list,
            "critic_weights": weights_critic_list,
            "difference_critic": difference_critic_list
        }
        return metrics


    def _compute_reward(self, position, action):
        # Pour bandit test: reward quadratique -(u - μ*)²
        target = np.ones_like(action)  # μ* = 1
        return -np.sum((action - target)**2)  # Quadratic, not norm!


class BanditTest:
    """
    Static Bandit Test according to Doya:
    - No state (x fixed or ignored)
    - V = 0, V_dot = 0
    - δ(t) = r(t) = -(u - μ*)²
    - Update: w += η * δ * ∂log(π)/∂w
    """
    def __init__(self, 
                 target_mu: float = 1.0,
                 actor_lr: float = 1e-2,
                 sigma: float = 0.5,
                 n_steps: int = 10000,
                 rng: Optional[np.random.Generator] = None,
                 decay: bool = False):
        
        self.target_mu = target_mu
        self.actor_lr = actor_lr
        self.sigma = sigma
        self.n_steps = n_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.decay = decay
        
        # Actor = just a bias (no state)
        self.mu = 0.0  # Parameter to learn
    
    def train(self):
        iterator = tqdm.trange(self.n_steps, desc="Bandit Test")
        mu_history = []
        reward_history = []
        
        for step in iterator:
            # Policy: u = μ + σ*n
            noise = self.rng.standard_normal()
            if self.decay:
                current_sigma = self.sigma * (1 - step / self.n_steps)
            else:
                current_sigma = self.sigma
            u = self.mu + current_sigma * noise
            
            # Quadratic reward: r = -(u - μ*)²
            reward = -((u - self.target_mu) ** 2)
            
            # TD error = reward (since V=0, V_dot=0 for bandit)
            delta = reward
            
            # Gradient of log π(u|μ) = (u - μ)/σ² = noise/σ²
            # ∂log p(u)/∂μ = (u - μ)/σ² = noise/σ²
            grad_log_pi = noise / (self.sigma ** 2)
            
            # Update according to Doya: dμ/dt = η * δ * ∂log(π)/∂μ
            self.mu += self.actor_lr * delta * grad_log_pi
            
            if step % 100 == 0:
                mu_history.append(self.mu)
                reward_history.append(reward)
            
            iterator.set_description(
                f"μ={self.mu:.4f}, target={self.target_mu}, r={reward:.4f}"
            )
        
        return {
            'mu_history': mu_history,
            'reward_history': reward_history,
            'final_mu': self.mu,
            'target': self.target_mu,
            'error': abs(self.mu - self.target_mu)
        }


class ActorNetwork:
    def __init__(self, input_dim, output_dim, rng:Optional[np.random.Generator]=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = self.rng.standard_normal((input_dim, output_dim))*0.01
    def __call__(self, x): 
        return np.dot(x, self.W)

class CriticOracle:
    def __init__(self, input_dim, P, rng:Optional[np.random.Generator]=None):
        self.input_dim = input_dim
        self.feature_dim = input_dim + input_dim * (input_dim - 1) // 2
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = P
    def __call__(self, x): 
        return -x.T @ self.W @ x
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("TEST 1: Static Bandit (actor update verification)")
    print("=" * 60)
    print("Goal: μ must converge to target_mu = 1.0")
    print("According to Doya: δ = r, update = η * δ * ∂log(π)/∂w")
    print("=" * 60)
    
    bandit = BanditTest(
        target_mu=1.0,
        actor_lr=1e-3,
        sigma=0.1,
        n_steps=5000, 
        decay=True
    )
    
    results = bandit.train()
    
    print(f"\n{'='*40}")
    print("RESULTS:")
    print(f"  Target μ* = {results['target']}")
    print(f"  Learned μ = {results['final_mu']:.4f}")
    print(f"  Error     = {results['error']:.4f}")
    print(f"  Converged = {'YES ✓' if results['error'] < 0.1 else 'NO ✗'}")
    print(f"{'='*40}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(results['mu_history'])
    axes[0].axhline(y=results['target'], color='r', linestyle='--', 
                    label=f"Target μ*={results['target']}")
    axes[0].set_xlabel('Step (x100)')
    axes[0].set_ylabel('μ (actor parameter)')
    axes[0].set_title('Actor Parameter Convergence')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(results['reward_history'])
    axes[1].axhline(y=0, color='r', linestyle='--', label='Optimal reward')
    axes[1].set_xlabel('Step (x100)')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Reward Evolution')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('bandit_test_doya.png')
    plt.show()
    
    if results['error'] < 0.1:
        print("\n✓ Bandit test PASSED!")
        print("  The actor update equation works correctly.")
    else:
        print("\n✗ Bandit test FAILED!")
        print("  Check: 1) gradient sign, 2) learning rate, 3) reward formula")