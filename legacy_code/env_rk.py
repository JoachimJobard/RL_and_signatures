"""Env to patch jitcdde problems, runge kutta 4"""
import numpy as np
from typing import Callable, Optional
from legacy_code.solver_buffer import NumpyRingBuffer

# from src.history import HistoryUtils

class Environment:
    """
    Creates a simulation environment for differential delayed equations
    ---------
    Parameters:
    A (np.ndarray): the system matrix
    B (np.ndarray): the control matrix
    A1 (Optional[np.ndarray]): the delayed system matrix
    delay (Optional[np.ndarray]): the delay for each dimension
    step_size (float): the integration step size
    x0 (Optional[np.ndarray]): the initial condition
    t (float): the initial time
    resolution (int): the number of solver steps per environment step
    """
    def __init__(self, 
                 A: np.ndarray, 
                 B: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray, 
                 A1: Optional[np.ndarray] = None, 
                 delay: Optional[np.ndarray] = None, 
                 step_size: float = 1., 
                 x0: Optional[np.ndarray] = None,
                 history_function: Optional[Callable] = None, 
                 t: float = 0.,
                 resolution: int = 20) -> None:
        
        print("initializing env rk")
        
        # Dynamics parameters
        self.A = A
        self.B = B
        self.A1 = A1
        self.N = self.A.shape[0]
        self.Q = Q
        self.R = R
        
        # Time-related parameters
        self.t = t
        self.step_size = step_size
        self.resolution = resolution
        self.solver_step_size = step_size / resolution

        # Delay parameters
        self.delay = delay
        
        #buffer
        if self.delay is not None and np.max(self.delay) != 0:
            # +1 to have enough room for interpolation at exact delay
            size_buffer = int(np.ceil(np.max(self.delay) / self.solver_step_size)) + 1
        else:
            size_buffer = 1
        self._buffer = NumpyRingBuffer(capacity=size_buffer, dim=self.N)
  
        # Initial condition
        self.t0 = t
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.N)
        self.history_function = history_function
        self._init_condition()
        
        # Storage for simulation results
        self._data = []
        self._time = []
        self._current_time = []
        
        
        # Validation checks
        self._validate_dimensions()

        #initialize condition
        if self.history_function is not None:
            self._init_condition()
        
    
    def _validate_dimensions(self) -> None:
        """Validate matrix dimensions and parameters."""
        assert self.A.shape[0] == self.B.shape[0], "A should have as many rows as B has rows"
        assert self.A.shape[1] == self.A.shape[0], "A should be square matrix"
        assert self.x0 is None or len(self.x0) == self.N, "x0 dimension should match A and B"
        
        if self.A1 is not None:
            assert self.delay is not None, "If A1 is provided, delay must be provided too"
            assert len(self.delay) == self.N, "delay should have same row number as A and B"
            assert self.A1.shape[0] == self.A.shape[0], "A1 should have same row number as A and B"
            assert self.A1.shape[1] == self.A1.shape[0], "A1 should be square matrix"
    
    def _init_condition(self) -> None:
        """Initialize the buffer with history function if provided."""
        if self.delay is None or max(self.delay) == 0:
            self._buffer.append(self.x0)
            return
        past_times = np.linspace(-max(self.delay) + self.t0, self.t0, num=self._buffer.size)
        if self.history_function is not None:
            for t in past_times:
                x_past = self.history_function(t) # type: ignore
                self._buffer.append(x_past)
        else:
            for _ in past_times:
                self._buffer.append(self.x0)

    def runge_kutta4(self, x_t, x_delayed, u):
        """Performs one step of the 4th order Runge-Kutta method."""
        h = self.solver_step_size
        current_t = self.t
        k1 = self.dynamics(x_t, x_delayed, u)
        x_tau_2 = self._get_delayed_state_at(current_t - np.max(self.delay) + 0.5 * h) #type: ignore
        k2 = self.dynamics(x_t + 0.5 * h * k1, x_tau_2, u)
        k3 = self.dynamics(x_t + 0.5 * h * k2, x_tau_2, u)
        x_tau_4 = self._get_delayed_state_at(current_t - np.max(self.delay) + h) #type: ignore
        k4 = self.dynamics(x_t + h * k3, x_tau_4, u)
        dx = (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next = x_t + dx
        self.t += h
        self._buffer.append(x_next)
        return x_next
        
    def get_B(self, x):
        """Return input matrix B. Override for state-dependent B(x)."""
        return self.B

    def dynamics(self, x, x_delayed, u):
        """Computes the system dynamics."""
        dxdt = self.A @ x + self.A1 @ x_delayed + self.B @ u
        return dxdt

    def _get_delayed_state(self) -> np.ndarray:
        """Get the delayed state with interpolation."""
        if self.delay is None or np.max(self.delay) == 0:
            return np.zeros(self.N)
        
        # Calculate fractional delay steps for interpolation
        delay_steps = np.max(self.delay) / self.solver_step_size
        return self._buffer.get_delayed_interpolated(delay_steps)
    def _get_delayed_state_at(self, query_time: float) -> np.ndarray:
        """Get the delayed state at a specific query time with interpolation."""
        if self.delay is None or np.max(self.delay) == 0:
            return np.zeros(self.N)
        
        # Calculate the time difference
        time_diff = self.t - query_time
        if time_diff < 0:
            raise ValueError("Query time is in the future.")
        
        # Calculate fractional delay steps for interpolation
        delay_steps = time_diff / self.solver_step_size
        return self._buffer.get_delayed_interpolated(delay_steps)
    
    def step(self, u):
        """
        Advances the simulation by one step using the provided control input.
        Args:
            u (np.ndarray): The control input for the current step.
        """
        self._current_time = []
        self.u = u
        reward = 0.0
        x_t = self._buffer.get_current()
        for _ in range(self.resolution):
            x_tau = self._get_delayed_state()
            x_t = self.runge_kutta4(x_t, x_tau, u)
            reward += self.compute_reward(x_t, u)
            self._current_time.append(self.t)
        self._data.append(x_t.copy())
        self._time.append(self.t)
        return self.t, x_t, reward/self.resolution
    
    def reset(self, hard_reset: bool = False) -> None:
        """Resets the simulation environment to the initial state."""
        self.u = [0. for i in range(self.B.shape[1])]
        self._current_time = []
        self.t = self.t0
        self._buffer.reset()
        self._data.clear()
        self._time.clear()
        self._buffer.append(self.x0)
        self._init_condition()
    
    def compute_reward(self, x, u) -> float:
        """Computes the reward based on the current state and control input."""
        return -(x.T @self.Q@ x + u.T@self.R @ u)

    @property
    def time(self):
        return np.array(self._time)
    # for the solution to be stick together, there is some duplicated entries in the time vector, to be checked later
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
    def time_series_of_last_step(self):
        return np.array(self._current_time)
    @property
    def initial_conditions(self):
        if self.delay is None or max(self.delay) == 0:
            raise ValueError("No delay defined in the system.")
        if self.history_function is None:
            raise ValueError("No history function defined in the system.")
        past_times = np.linspace(-max(self.delay) + self.t0, self.t0, num=int(self.resolution/self.step_size))
        past_data = np.array([self.history_function(t) for t in past_times])
        return past_times, past_data
