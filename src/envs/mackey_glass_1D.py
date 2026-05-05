import jax.numpy as jnp
import jax


from src.env_rk_jax import EnvState, JAXDDEEnv, JAXEnvWrapper
from src.utils.solver_buffer_jax import buffer_append, get_delayed_interpolated

class MackeyGlass1DEnv(JAXDDEEnv):
    def __init__(
        self,
        delay: float = 17.0,
        step_size: float = 0.01,
        resolution: int = 10,
        Q: jnp.ndarray = jnp.array([[1.0]]),
        R: jnp.ndarray = jnp.array([[1.0]]),
        n: int = 10,
        p: float = 0.2,
        mu: float = 0.1,
        x_target: float = 0.5,
    ):
        """
        Mackey-Glass 1D environment modeled as a delay differential equation (DDE).
        
        Parameters
        ----------
        delay : float
            Time delay in the system.
        step_size : float
            Integration step size.
        resolution : int
            Number of sub-steps for integration within each step.
        """
        super().__init__(
            A=jnp.array([[0.0]]),
            B=jnp.array([[1.0]]),
            A1=jnp.array([[0.0]]),
            delay=jnp.array([delay]),
            Q=Q,
            R=R,
            step_size=step_size,
            resolution=resolution,
        )
        self.n = n
        self.p = p
        self.mu = mu
        self.x_target = jnp.array([x_target])  # Ensure it's an array
        
        # Compute natural equilibrium: x^n = p/mu - 1
        if p > mu:
            self.natural_equilibrium = (p/mu - 1) ** (1/n)
        else:
            self.natural_equilibrium = 0.0



    def get_B(self, x):
        """State-independent input matrix B(x) = [[1.0]] (u enters as +u)."""
        return jnp.array([[1.0]])

    def dynamics(self, x, buffer, u, dt_offset_fraction):
        x_delayed = jnp.zeros(self.N)
        if self.has_delay:
            base_delay_steps = self.delay / self.solver_step_size
            adjusted_delay_steps = base_delay_steps - dt_offset_fraction
            x_delayed = get_delayed_interpolated(buffer, adjusted_delay_steps)
        mg_nonlinear = self.p * (x_delayed) / (1 + (x_delayed) ** self.n) + u
        return -(self.mu) * x + mg_nonlinear
    
    def step(self, state: EnvState, u: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray,]:
        eps=1e-3
        def body_fun(curr_state, _):
            # Utilise self.runge_kutta4 qui appelle self.dynamics
            x_next = self.runge_kutta4(curr_state.x, curr_state.buffer, u)
            next_buffer = buffer_append(curr_state.buffer, x_next)
            
            new_s = curr_state._replace(
                x=x_next, 
                t=curr_state.t + self.solver_step_size, 
                buffer=next_buffer
            )
            return new_s, None

        final_state, _ = jax.lax.scan(body_fun, state, None, length=self.resolution)

        error = final_state.x - self.x_target  # x_target is now an array
        delta_u = u - state.last_u  # type: ignore
        if self.x_target is not None or not jnp.all(self.x_target == 0):
            cost = error.T @ self.Q @ error + delta_u.T @ self.R @ delta_u + eps * (u.T @ self.R @ u)
        else:
            cost = final_state.x.T @ self.Q @ final_state.x + u.T @ self.R @ u
        reward = -cost.squeeze() 

        final_state = final_state._replace(last_u=u)
        
        return final_state, final_state.x, reward
    

class MackeyGlass1DEnvWrapper(JAXEnvWrapper):
    def __init__(
        self,
        delay: float = 17.0,
        step_size: float = 0.01,
        resolution: int = 10,
        Q: jnp.ndarray = jnp.array([[1.0]]),
        R: jnp.ndarray = jnp.array([[1.0]]),
        n: int = 10,
        p: float = 0.2,
        mu: float = 0.1,
        x0 = 0.1
    ):
        env = MackeyGlass1DEnv(
            delay=delay,
            step_size=step_size,
            resolution=resolution,
            Q=Q,
            R=R,
            n=n,
            p=p,
            mu=mu,
        )
        super().__init__(env)
    
# TESTING =========================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 16
})
    # Configuration
    X_TARGET = 0.0  # Target valuet
    X0 = 0.8  # Initial condition near non-trivial equilibrium for oscillations
    delay= 30.0
    step_size = 0.1
    env = MackeyGlass1DEnv(
        delay=delay,
        step_size=step_size,
        resolution=10,
        Q=jnp.array([[1.0]]),
        R=jnp.array([[0.1]]),
        n=10,
        p=0.2,
        mu=0.1,
        x_target=X_TARGET,
    )
    def hist_function(x):
        return jnp.array([X0])  # Non-zero history to seed oscillations
    env.history_function = hist_function
    print(f"=== Mackey-Glass 1D Environment ===")
    print(f"Natural equilibrium: x* = {env.natural_equilibrium:.4f}")
    print(f"Target: x_target = {X_TARGET}")
    print(f"Initial condition: x0 = {X0}")
    
    env_wrapped = JAXEnvWrapper(env, rng_key=42)
    rng_key = jax.random.PRNGKey(0)     
    env_wrapped.reset(rng_key, x0=jnp.array([X0]), t0=0.0, history_function=hist_function)
    
    state_list = []
    time_list = []  
    reward_list = []
    delayed_state = []
    
    for _ in range(10000):
        action = jnp.array([0.0])  # No control
        t, x, reward = env_wrapped.step(env_wrapped.state, action)
        state_list.append(x)
        time_list.append(t)
        reward_list.append(reward)
        base_delay_steps = env.delay / env.solver_step_size
        adjusted_delay_steps = base_delay_steps
        x_delayed = get_delayed_interpolated(env_wrapped.state.buffer, adjusted_delay_steps)  # Store the delayed state for analysis
        delayed_state.append(x_delayed)

    state_array = jnp.stack(state_list)
    reward_array = jnp.array(reward_list)
    time_array = jnp.array(time_list)
    delayed_array = jnp.array(delayed_state)
    
    # Plot 1: State evolution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time_array, state_array, label='State x(t)')
    plt.axhline(y=X_TARGET, color='r', linestyle='--', label=f'Target = {X_TARGET}')
    plt.axhline(y=env.natural_equilibrium, color='g', linestyle=':', label=f'Equilibrium = {env.natural_equilibrium:.2f}')
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Mackey-Glass 1D Dynamics (No Control)")
    plt.legend()
    
    # Plot 2: Reward
    plt.subplot(1, 2, 2)
    plt.plot(time_array, reward_array, label='Reward')
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title("Reward over Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    plt.plot(time_array, state_array, label='State x(t)')
    plt.xlabel("Time")
    plt.ylabel("State")
    # plt.title("Mackey-Glass 1D Dynamics (No Control)")
    plt.legend()
    # plt.savefig("../master_thesis/kth-typst-template/figures/mackey_glass_1D_dynamics_n_10_tau_8.pdf")
    plt.show()

    # attractor analysis
    plt.figure(figsize=(6, 6))
    plt.plot(delayed_array[int(delay/step_size)+1:], state_array[int(delay/step_size)+1:], label='Attractor Trajectory')
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.axis('off')
    # plt.legend()
    plt.tight_layout()
    plt.savefig("../master_thesis/kth-typst-template/figures/mackey_glass_1D_attractor_n_10_tau_30.pdf")
    plt.show()

    
    print(f"\nFinal state: {state_array[-1, 0]:.4f}")
    print(f"Final error from target: {abs(state_array[-1, 0] - X_TARGET):.4f}")
    
        
