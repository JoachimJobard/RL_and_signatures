from collections import defaultdict
import numpy as np

class StateCounter:
    '''
    Class to count the number of time a state has been visited,
    given a resolution. This resolutionis not specifically linked
    to the environment/learning algorithm.
    '''
    def __init__(self, resolution: float,):
        self.resolution = resolution
        self.counter = defaultdict(int)
    
    def get_key(self, state):
        discretized = np.floor(state / self.resolution).astype(int)
        return tuple(discretized)
    
    def add(self, state):
        key = self.get_key(state)
        self.counter[key] += 1
    
    def get_count(self, state):
        key = self.get_key(state)
        return self.counter[key]
    