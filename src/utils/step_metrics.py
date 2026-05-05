from dataclasses import dataclass

import numpy as np


@dataclass 
class StepMetrics:
    """Metrics collected from a single training step.
    
    Fields may be JAX arrays (to avoid GPU→CPU sync) or plain floats.
    """
    loss: float | np.ndarray
    reward: float | np.ndarray
    actor_gradient: np.ndarray
    critic_gradient: np.ndarray
    noise: np.ndarray | None = None