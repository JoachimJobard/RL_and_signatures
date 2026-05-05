import jax.numpy as jnp
import jax


from src.env_rk_jax import JAXDDEEnv, JAXEnvWrapper
from src.utils.solver_buffer_jax import get_delayed_interpolated

class ChemicalReactionEnv(JAXDDEEnv):
    def __init__(
        self,
        delay: float = 1,
        step_size: float = 0.01,
        resolution: int = 10,
        Q: jnp.ndarray = jnp.array([[1.0]]),
        R: jnp.ndarray = jnp.array([[1.0]]),
        x_target: float = 0.0,
    ):
        """
        Chemical Reaction environment modeled as a delay differential equation (DDE).
        
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
            A=jnp.zeros((4, 4)),
            B=jnp.zeros((4, 2)),
            A1=jnp.zeros((4, 4)),
            delay=jnp.array([delay]),
            Q=Q,
            R=R,
            step_size=step_size,
            resolution=resolution,
        )

    def dynamics(self, x, buffer, u, dt_offset_fraction):
        R1 = (x[0] + 0.5)*jnp.exp(25*x[1]/(x[1]+2))
        R2 = (x[2] + 0.25)*jnp.exp(25*x[3]/(x[3]+2))
        x_delayed = jnp.zeros(self.N)
        if self.has_delay:
            base_delay_steps = self.delay / self.solver_step_size
            adjusted_delay_steps = base_delay_steps - dt_offset_fraction
            x_delayed = get_delayed_interpolated(buffer, adjusted_delay_steps)
        x_1_dot = 0.5 - x[0] - R1
        x_2_dot = -2*(x[1]+0.25) - u[0]*(x[1]+0.25) + R1
        x_3_dot = x_delayed[0] - x[2] - R2 + 0.25
        x_4_dot = x_delayed[1] - 2*x[3] - u[1]*(x[3]+0.25) + R2 - 0.25
        return jnp.array([x_1_dot, x_2_dot, x_3_dot, x_4_dot])

    def get_B(self, x):
        """State-dependent input matrix B(x)."""
        return jnp.array([[0.0, 0.0], [-(x[1]+0.25), 0.0], [0.0, 0.0], [0.0, -(x[3]+0.25)]])

    
    
    
    # def step(self, state: EnvState, u: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray,]:
    #     def body_fun(curr_state, _):
    #         # Utilise self.runge_kutta4 qui appelle self.dynamics
    #         x_next = self.runge_kutta4(curr_state.x, curr_state.buffer, u)
    #         next_buffer = buffer_append(curr_state.buffer, x_next)
            
    #         new_s = curr_state._replace(
    #             x=x_next, 
    #             t=curr_state.t + self.solver_step_size, 
    #             buffer=next_buffer
    #         )
    #         return new_s, None

    #     final_state, _ = jax.lax.scan(body_fun, state, None, length=self.resolution)
    #     cost = final_state.x.T @ self.Q @ final_state.x + u.T @ self.R @ u
    #     reward = -cost.squeeze() 

    #     final_state = final_state._replace(last_u=u)
        
        # return final_state, final_state.x, reward
    

    
# TESTING =========================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Configuration
    X0 = jnp.array([0.15, -0.03, 0.1, 0.0])  # Initial condition
    
    env = ChemicalReactionEnv(
        delay=0.15,
        step_size=0.01,
        resolution=10,
        Q=jnp.eye(4),
        R=jnp.eye(2),

    )
    
    env_wrapped = JAXEnvWrapper(env, rng_key=42)
    rng_key = jax.random.PRNGKey(0)     
    env_wrapped.reset(rng_key, x0=X0, t0=0.0)
    
    state_list = []
    time_list = []  
    reward_list = []
    
    for _ in range(200):
        action = jnp.zeros((2,))  # No control
        t, x, reward = env_wrapped.step(env_wrapped.state, action)
        state_list.append(x)
        time_list.append(t)
        reward_list.append(reward)
    
    state_array = jnp.stack(state_list)
    reward_array = jnp.array(reward_list)
    time_array = jnp.array(time_list)
    
    # Plot 1: State evolution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for i in range(4):
        plt.plot(time_array, state_array[:, i], label=f'State x{i+1}(t)')
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Chemical Reaction Dynamics (No Control)")
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
    
    print(f"\nFinal state: {state_array[-1, 0]:.4f}")
    print(f"Final error from target: {abs(state_array[-1, 0]):.4f}")
    
        
