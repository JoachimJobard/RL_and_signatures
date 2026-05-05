import jax
import jax.numpy as jnp
from typing import Any, NamedTuple

import numpy as np
from src.utils.solver_buffer_jax import BufferState, buffer_append, get_delayed_interpolated

class EnvState(NamedTuple):
    x: jax.Array
    t: float
    buffer: BufferState
    last_u: jnp.ndarray | None = None

class JAXDDEEnv:
    def __init__(self, A, B, A1, delay, Q, R, step_size, resolution, reward_kernel=None):
        self.A = jnp.array(A)
        self.B = jnp.array(B)
        self.A1 = jnp.array(A1) if A1 is not None else jnp.zeros_like(self.A)
        self.delay = jnp.array(delay) if delay is not None else None
        self.Q = jnp.array(Q)
        self.R = jnp.array(R)
        self.step_size = step_size
        self.resolution = resolution
        self.solver_step_size = step_size / resolution
        self.N = self.A.shape[0]
        self.max_delay = jnp.max(self.delay) if self.delay is not None else 0
        self.has_delay = self.delay is not None and jnp.any(self.delay > 0)
        self.reward_kernel = reward_kernel

        if self.max_delay > 0:
            self.buffer_size = int(jnp.ceil(self.max_delay / self.solver_step_size)) + 1
        else:
            self.buffer_size = 2
    
    def _get_delayed_state(self, buffer: BufferState) -> jnp.ndarray:
        if self.delay is None or jnp.all(self.delay == 0):
            return jnp.zeros(self.N)
        delay_steps_vec = self.delay / self.solver_step_size
        return get_delayed_interpolated(buffer, delay_steps_vec)
    
    def get_B(self, x):
        """Return input matrix B. Override for state-dependent B(x)."""
        return self.B

    def dynamics(self, x, buffer, u, dt_offset_fraction):
        x_delayed = jnp.zeros(self.N)
        if self.has_delay:
            base_delay_steps = self.delay / self.solver_step_size
            adjusted_delay_steps = base_delay_steps - dt_offset_fraction
            x_delayed = get_delayed_interpolated(buffer, adjusted_delay_steps)
        return self.A @ x + self.B @ u + self.A1 @ x_delayed
    
    def runge_kutta4(self, x_t, buffer, u):
        h = self.solver_step_size
        k1 = self.dynamics(x_t, buffer, u, 0.0)
        k2 = self.dynamics(x_t + 0.5 * h * k1, buffer, u, 0.5)
        k3 = self.dynamics(x_t + 0.5 * h * k2, buffer, u, 0.5)
        k4 = self.dynamics(x_t + h * k3, buffer, u, 1.0)
        return x_t + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def reset(self, rng_key, x0=None, t0=0.0, history_function=None):
        self.t = t0
        if x0 is not None:
            self.x0 = jnp.array(x0)
        else:
            self.x0 = jnp.zeros(self.N)        
        self.history_function = history_function
        buffer = BufferState.create(self.buffer_size, self.N)
        indexes = jnp.arange(self.buffer_size)
        times = t0 - (self.buffer_size - 1 - indexes) * self.solver_step_size 
        if history_function is not None:
            data_history = jax.vmap(history_function)(times)
            # Ensure 2D shape (buffer_size, N) even if history_function returns a scalar
            if data_history.ndim == 1:
                data_history = data_history[:, None]
        else:
            data_history = jnp.tile(self.x0, (self.buffer_size, 1))
        buffer = buffer._replace(data=data_history, ptr=0)
        return EnvState(x=self.x0, t=t0, buffer=buffer, last_u=jnp.zeros(self.B.shape[1]))
    
    def step(self, state: EnvState, u: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
        def body_fun(curr_state, _):
            x_next = self.runge_kutta4(curr_state.x, curr_state.buffer, u)
            next_buffer = buffer_append(curr_state.buffer, x_next)
            new_s = curr_state._replace(x=x_next, t=curr_state.t + self.solver_step_size, buffer=next_buffer)
            return new_s, None
        final_state, _ = jax.lax.scan(body_fun, state, None, length=self.resolution)
        reward = -(final_state.x.T @ self.Q @ final_state.x + u.T @ self.R @ u)
        return final_state._replace(last_u=u), final_state.x, reward
    
    def compute_reward(self, state: EnvState, u: jnp.ndarray) -> jnp.ndarray:
        reward_kernel = self.reward_kernel
        def body_fun(curr_state: EnvState, _) -> jnp.ndarray:
            reward = jax.lax.cond(reward_kernel is not None,
                         lambda s: (s.x.T @ self.Q @ s.x + u.T @ self.R @ u),
                         lambda s: -(s.x.T @ self.Q @ s.x + u.T @ self.R @ u),
                         curr_state)
            return reward
        reward = jax.lax.scan(body_fun, state, None, length=self.resolution)[0]
        return reward

class JAXEnvWrapper:
    def __init__(self, env: JAXDDEEnv, rng_key=None):
        self.env = env
        self.key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        self._jit_step = jax.jit(self.env.step)
        self._jit_reset = jax.jit(self.env.reset)

        self._data = []
        self._time = []
        self._u_history = []
        self.state = None

        self.history_function = None
    
    def reset(self, rng_key, x0=None, t0=0.0, history_function=None):
        self._data = []
        self._time = []
        self._u_history = []
        self.history_function = history_function

        self.key, subkey = jax.random.split(rng_key)
        if x0 is not None:
            x0 = jnp.array(x0)
            
        self.state = self.env.reset(subkey, x0=x0, t0=t0, history_function=history_function)
        
        self._data.append(np.array(self.state.x))
        self._time.append(float(self.state.t))
        return self.state.x
    
    def step(self, state: EnvState | Any, u: jnp.ndarray) -> tuple[float, jnp.ndarray, float]:
        if self.state is None:
            raise ValueError("Environment must be reset before stepping.")
        u_jax = jnp.array(u)
        self.state, x_next, reward = self._jit_step(state, u_jax)
        # NOTE: _data/_time/_u_history recording is deferred to avoid
        # GPU→CPU synchronisation on every step.  Call record_step()
        # explicitly when the data is actually needed (e.g. end of episode).
        return self.state.t, x_next, reward

    def record_step(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
        """Manually record a step for post-episode analysis (forces GPU→CPU)."""
        self._data.append(np.array(x))
        self._time.append(float(self.state.t))
        self._u_history.append(np.array(u))
    
    
    @property
    def time(self):
        return np.array(self._time)

    @property
    def time_flat(self):
        return np.array(self._time).flatten()

    @property
    def data(self):
        return np.array(self._data)

    @property
    def data_flat(self):
        return np.array(self._data).flatten()
    
    @property
    def controls(self):
        return np.array(self._u_history)
    
    @property
    def current_delayed_state(self):
        if self.state is None:
             raise ValueError("Environment not initialized")
        return jnp.array(self.env._get_delayed_state(self.state.buffer))

    @property
    def initial_conditions(self):
        """
        Rebuild the full history from the JAX buffer.
        """
        if self.state is None:
             raise ValueError("Environment not initialized")

        buffer = self.state.buffer
        data = np.array(buffer.data)
        ptr = int(buffer.ptr)
        capacity = int(buffer.capacity)
        ordered_data = np.roll(data, -ptr, axis=0)
        t_now = self._time[0]
        dt = self.env.solver_step_size
        indices = np.arange(capacity)
        past_times = t_now - (capacity - 1 - indices) * dt
        
        return past_times, ordered_data   
