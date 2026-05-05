import numpy as np

class NumpyRingBuffer:
    """
    Simple buffer for solver environment 

    Parameters:
    capacity (int): Maximum number of elements in the buffer
    dim (int): Dimension of each element
    """
    def __init__(self, capacity, dim) -> None:
        self.size = capacity
        self.dim = dim
        self.buffer = np.zeros((capacity, dim), dtype=np.float64)
        self.ptr = 0
        self.full = False
        self._len = 0

    def append(self, value: np.ndarray) -> None:
        self.buffer[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.size #Circular buffer
        if self.ptr == 0:
            self.full = True
        self._len = min(self._len + 1, self.size)
    
    def get_delayed(self, delay_steps: int) -> np.ndarray:
        """Get state from delay_steps ago.
        
        delay_steps=0 returns current state (most recent)
        delay_steps=1 returns state 1 step ago
        etc.
        """
        if delay_steps >= self._len:
            raise IndexError(f"Not enough elements in buffer for the requested delay. "
                           f"Requested {delay_steps}, have {self._len}")
        # ptr points to next write position, so ptr-1 is most recent
        index = (self.ptr - 1 - delay_steps) % self.size
        return self.buffer[index]
    
    def get_delayed_interpolated(self, delay_steps: float) -> np.ndarray:
        """Get delayed value with linear interpolation for fractional delay_steps."""
        if delay_steps >= self._len:
            raise IndexError(f"Not enough elements in buffer for the requested delay. "
                           f"Requested {delay_steps}, have {self._len}")
        
        # Integer and fractional parts
        delay_int = int(delay_steps)
        delay_frac = delay_steps - delay_int
        
        if delay_frac < 1e-9:
            return self.get_delayed(delay_int)
        
        # Need delay_int + 1 to be valid
        if delay_int + 1 >= self._len:
            return self.get_delayed(delay_int)
        
        # Interpolate between two adjacent points
        x_newer = self.get_delayed(delay_int)       # Newer point
        x_older = self.get_delayed(delay_int + 1)   # Older point
        
        # Linear interpolation
        return (1 - delay_frac) * x_newer + delay_frac * x_older
    
    def get_current(self) -> np.ndarray:
        if self._len == 0:
            raise IndexError("Buffer is empty.")
        index = (self.ptr - 1) % self.size
        return self.buffer[index]
    
    def reset(self) -> None:
        self.ptr = 0
        self.full = False
        self._len = 0
        self.buffer.fill(0)
    
    def get_complete_buffer(self) -> np.ndarray:
        clone_buffer = np.zeros_like(self.buffer)
        clone_buffer[self._len - self.ptr:] = self.buffer[:self.ptr]
        clone_buffer[:self._len - self.ptr] = self.buffer[self.ptr:]
        return clone_buffer 
    def __len__(self) -> int:
        return self._len
