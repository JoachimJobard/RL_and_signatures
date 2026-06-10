from collections import defaultdict
import jax
import numpy as np

ArrayLike = np.ndarray | jax.Array


class StateCounter:
    '''
    Class to count the number of time a state has been visited,
    given a resolution. This resolutionis not specifically linked
    to the environment/learning algorithm.
    '''
    def __init__(self, resolution: float) -> None:
        self.resolution = resolution
        self.counter: defaultdict[tuple[int, ...], int] = defaultdict(int)

    def get_key(self, state: ArrayLike) -> tuple[int, ...]:
        discretized = np.floor(state / self.resolution).astype(int)
        return tuple(discretized)

    def add(self, state: ArrayLike) -> None:
        key = self.get_key(state)
        self.counter[key] += 1

    def get_count(self, state: ArrayLike) -> int:
        key = self.get_key(state)
        return self.counter[key]
    