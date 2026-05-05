from typing import NamedTuple
import jax.numpy as jnp

class BufferState(NamedTuple):
    """immuable state of ring buffer (jax version)
    Atributes:
        data(jnp.array): buffer data
        ptr(int): pointer to next write position
        capacity(int): maximum capacity of the buffer
        dim(int): dimension of each element
    """
    data: jnp.ndarray
    ptr: int
    capacity: int
    dim: int

    @classmethod
    def create(cls, capacity: int, dim: int, initial_value: jnp.ndarray | None = None)->"BufferState":
        "initialise the buffer"
        if initial_value is None:
            initial_value = jnp.zeros((capacity, dim))
        data = jnp.zeros((capacity, dim)) + initial_value
        return cls(data=data, ptr=0, capacity=capacity, dim=dim)
    
def buffer_append(state: BufferState, value: jnp.ndarray) -> "BufferState":
    "Add a value to the buffer, return new state"
    new_data = state.data.at[state.ptr].set(value)
    new_ptr = (state.ptr + 1) % state.capacity
    return state._replace(data=new_data, ptr=new_ptr)

def get_current(state: BufferState) -> jnp.ndarray:
    "Get the most recent value from the buffer"
    index = (state.ptr - 1) % state.capacity
    return state.data[index]

def get_delayed_interpolated(state: BufferState, delay_steps: float) -> jnp.ndarray:
    """Get delayed value with linear interpolation for fractional delay_steps.
    
    Supports both scalar and vector delay_steps (per-dimension delays).
    """
    delay_int = jnp.floor(delay_steps).astype(int)
    delay_frac = delay_steps - delay_int
    index_newer = (state.ptr - 1 - delay_int) % state.capacity
    index_older = (state.ptr - 1 - (delay_int + 1)) % state.capacity
    
    if state.data.ndim == 1:
        # 1D buffer: scalar indexing
        x_newer = state.data[index_newer]
        x_older = state.data[index_older]
    else:
        # 2D buffer: vectorized per-dimension indexing
        dim = state.data.shape[1]
        delay_steps_arr = jnp.broadcast_to(jnp.atleast_1d(delay_steps), (dim,))
        delay_int = jnp.floor(delay_steps_arr).astype(int)
        delay_frac = delay_steps_arr - delay_int
        index_newer = (state.ptr - 1 - delay_int) % state.capacity
        index_older = (state.ptr - 1 - (delay_int + 1)) % state.capacity
        col_index = jnp.arange(dim)
        x_newer = state.data[index_newer, col_index]
        x_older = state.data[index_older, col_index]

    return (1 - delay_frac) * x_newer + delay_frac * x_older
        