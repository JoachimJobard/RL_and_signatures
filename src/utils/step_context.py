from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

#============================================================================
#  Standard version of StepContext
#============================================================================
@dataclass
class StepContext:
    """Container for all data from a single training step.
    
    This context is passed to overridable methods to provide 
    access to any needed information without changing method signatures.
    """
    # State information
    x_t: np.ndarray           # Current state (original scale)
    x_scaled: np.ndarray      # Current state (scaled)
    x_next: np.ndarray        # Next state (original scale)
    x_next_scaled: np.ndarray # Next state (scaled)
    
    # Action information
    mu: np.ndarray            # Policy mean (deterministic part)
    noise: np.ndarray         # Exploration noise
    action: np.ndarray        # Applied action (mu + noise, clipped)
    
    # Time information
    dt: float                 # Time step duration
    time_series: np.ndarray   # Full time series from env step
    data_series: list         # Full data series from env step
    
    # Value information
    V_t: float                # Value at current state
    V_next: float             # Value at next state
    
    # Computed quantities (filled during step)
    reward: float = 0.0
    V_dot: float = 0.0
    td_error: float = 0.0

#============================================================================
#  Signature version of StepContext
#============================================================================

@dataclass
class StepContextSignature:
    """Container for all data from a single training step, signature version.
    
    This context is passed to overridable methods to provide 
    access to any needed information without changing method signatures.
    """
    # State information
    x_t: np.ndarray | jnp.ndarray          # Current position (original scale)
    x_scaled: np.ndarray | jnp.ndarray      # Current position (scaled)
    x_next: np.ndarray | jnp.ndarray        # Next position (original scale)
    x_next_scaled: np.ndarray | jnp.ndarray # Next position (scaled)
    sig_t: jnp.ndarray        # Current signature
    sig_next: jnp.ndarray     # Next signature
    
    # Action information
    mu: np.ndarray | jnp.ndarray            # Policy mean (deterministic part)
    noise: np.ndarray | jnp.ndarray         # Exploration noise
    action: np.ndarray | jnp.ndarray        # Applied action (mu + noise, clipped)
    
    # Time information
    dt: float                 # Time step duration
    time_series: np.ndarray   # Full time series from env step
    
    # Value information
    V_t: float                # Value at current state
    V_next: float             # Value at next state
    
    # Computed quantities (filled during step)
    reward: float = 0.0
    V_dot: float = 0.0
    td_error: float = 0.0

@dataclass
class StepContextDelayed(StepContextSignature):
    """Container for all data from a single training step, signature version.
    
    This context is passed to overridable methods to provide 
    access to any needed information without changing method signatures.
    """
    # State information
    delayed_x_t: np.ndarray | jnp.ndarray | None = None        # Current delayed position (original scale)
    delayed_x_next: np.ndarray | jnp.ndarray | None = None       # Next delayed position (original scale)