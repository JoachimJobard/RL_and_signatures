import jax
import jax.numpy as jnp
import numpy as np
from flax import struct as flax_struct

@flax_struct.dataclass
class Transition:
    s: jnp.ndarray
    a: jnp.ndarray
    mu: jnp.ndarray
    log_pi: jnp.ndarray
    r: float
    s_next: jnp.ndarray
    done: bool
    x_t: jnp.ndarray
    x_next: jnp.ndarray

class ExperienceReplayBuffer:
    def __init__(self, capacity: int, dummy_transition: Transition):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        def make_storage(x):
            # Handle scalars (float, int, bool) that don't have .shape
            x_arr = np.asarray(x)
            return np.zeros((capacity,) + x_arr.shape, dtype=x_arr.dtype)
        
        # Use numpy arrays for mutable in-place storage
        self.storage = jax.tree_util.tree_map(make_storage, dummy_transition)
    
    def add(self, transition: Transition):
        def set_item(storage_arr, new_data):
            storage_arr[self.ptr] = new_data
        
        jax.tree_util.tree_map(set_item, self.storage, transition)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, key: jax.Array) -> Transition:
        index_jax = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        indices = np.array(index_jax)
        batch = jax.tree_util.tree_map(
            lambda arr: jnp.array(arr[indices]),
            self.storage
        )
        return batch