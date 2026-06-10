import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
from typing import Callable
from numpy.typing import DTypeLike
from signax.module import SignatureTransform


ArrayLike = np.ndarray | jax.Array


class DequeBuffer:
    def __init__(self, size: int, dtype: DTypeLike = np.float64) -> None:
        self.size = size
        self.dtype = dtype
        self.buffer: deque = deque(maxlen=size)
    def append(self, item: ArrayLike) -> None:
        self.buffer.append(item)
    def __len__(self) -> int:
        return len(self.buffer)
    def __getitem__(self, idx: int) -> ArrayLike:
        return self.buffer[idx]
    def __str__(self) -> str:
        return f"DequeBuffer(size={len(self)}/{self.size}, items={list(self.buffer)})"
    def __repr__(self) -> str:
        return self.__str__()
    def to_array(self) -> np.ndarray:
        """Convert buffer to numpy array efficiently."""
        return np.array(self.buffer, dtype=self.dtype)


class SlidingSignature:
    """Original SlidingSignature for non-JAX code."""
    
    def __init__(self, depth: int, window_size: int, d: int, time_augmentation: bool = False, bias: bool = False):
        self.depth = depth
        self.d = d + time_augmentation
        self.window_size = window_size
        self.time_augmentation = time_augmentation
        self.signature_size = (self.d ** (self.depth + 1) - 1) // (self.d - 1) - 1 + bias
        self.current_signature = jnp.zeros(self.signature_size)
        self.buffer = DequeBuffer(size=window_size + 1)
        self.signature_transform = SignatureTransform(depth=depth)
        self.current_signature = jnp.zeros(self.signature_size)
        self.bias = bias
    
    def reset(self):
        """Clear buffer and reset signature to zeros."""
        self.buffer = DequeBuffer(size=self.window_size + 1)
        self.current_signature = jnp.zeros(self.signature_size)
    
    def append(self, item):
        self.buffer.append(item)
        self.current_signature = self.compute_signature()
    
    def compute_signature(self):
        n = len(self.buffer)
        if n < 2:
            return jnp.zeros(self.signature_size) if not self.bias else jnp.concatenate([jnp.array([1.0]), jnp.zeros(self.signature_size - 1)])
        
        data = jnp.array([self.buffer[i] for i in range(n)])
        
        if self.time_augmentation:
            relative_times = jnp.linspace(-1.0, 0.0, n).reshape(-1, 1)  # normalized times
            data = jnp.concatenate([relative_times, data], axis=1)
        
        if self.bias:
            return jnp.concatenate([jnp.array([1.0]), self.signature_transform(data)])
        else:
            return self.signature_transform(data)


class JAXCircularBuffer:
    """JAX-native circular buffer to avoid Python ↔ JAX conversions."""
    
    def __init__(self, size: int, d: int, dtype: DTypeLike = jnp.float64) -> None:
        self.size = size
        self.d = d
        self.dtype = dtype
        # Pre-allocated JAX array
        self._data = jnp.zeros((size, d), dtype=dtype)  # type: ignore[arg-type]
        self._count = 0  # Number of items currently in buffer
        self._head = 0   # Next write position

    def append(self, item: ArrayLike) -> None:
        """Append item to circular buffer."""
        item_arr = jnp.asarray(item, dtype=self.dtype)  # type: ignore[arg-type]
        self._data = self._data.at[self._head].set(item_arr)
        self._head = (self._head + 1) % self.size
        self._count = min(self._count + 1, self.size)

    def to_array(self) -> jnp.ndarray:
        """Get buffer contents in chronological order."""
        if self._count < self.size:
            # Buffer not full yet - return from start
            return self._data[:self._count]
        else:
            # Buffer full - roll to get chronological order
            return jnp.roll(self._data, -self._head, axis=0)

    def __len__(self) -> int:
        return self._count

    def reset(self) -> None:
        """Reset buffer to empty state."""
        self._data = jnp.zeros((self.size, self.d), dtype=self.dtype)  # type: ignore[arg-type]
        self._count = 0
        self._head = 0


class SlidingSignatureJAX:
    """Optimized SlidingSignature with JIT-compiled signature computation."""
    
    def __init__(self, 
                 depth: int, 
                 window_size: int, 
                 d: int, 
                 time_augmentation: bool = False, 
                 bias: bool = False, 
                 origin_augmentation: bool = False,
                 time_origin: float = 1.0,
                 use_jax_buffer: bool = False,  # Default to deque for better perf
                 dtype: DTypeLike = np.float64) -> None:  # float64 by default (JAX x64 enabled); float32 selectable for comparison
        self.depth = depth
        self.dtype = dtype
        self.origin_augmentation = origin_augmentation
        self.d = d + time_augmentation + origin_augmentation * d
        self.d_raw = d  # Store raw dimension for buffer
        self.time_origin = time_origin
        self.window_size = window_size
        self.time_augmentation = time_augmentation
        self.signature_size = sum([self.d**i for i in range(1, self.depth + 1)]) + (1 if bias else 0)
        self.use_jax_buffer = use_jax_buffer
        
        # Use DequeBuffer by default (faster for small windows)
        self.buffer: DequeBuffer | JAXCircularBuffer
        if use_jax_buffer:
            self.buffer = JAXCircularBuffer(size=window_size + 1, d=d, dtype=dtype)
        else:
            self.buffer = DequeBuffer(size=window_size + 1, dtype=dtype)

        self.signature_transform = SignatureTransform(depth=depth)
        self.bias = bias
        
        # Pre-compile signature computation
        self._jit_compute_sig = self._make_compute_signature_fn()
        
        # Cache for empty signatures
        if bias:
            self._empty_sig = jnp.concatenate([jnp.array([1.0]), jnp.zeros(self.signature_size - 1)])
        else:
            self._empty_sig = jnp.zeros(self.signature_size)
        
        self._current_signature = self._empty_sig
        self._signature_dirty = True
    
    def _make_compute_signature_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Create JIT-compiled signature computation with fixed-size input."""
        sig_transform = self.signature_transform
        time_aug = self.time_augmentation
        origin_aug = self.origin_augmentation
        bias = self.bias
        time_origin = self.time_origin
        # Fixed size for JIT - signature computation always uses full window
        fixed_size = self.window_size + 1
        
        @jax.jit
        def compute_sig(data: jnp.ndarray) -> jnp.ndarray:
            """Compute the signature of the augmented windowed path.

            Implements the path augmentation of Perez Arribas (2018), "Derivatives
            pricing using signature payoffs", Definition p.5:

                X_hat_t = ( t , X_t , (X_0 / T) * t ) ,   t in [0, T],

            three channels in this order:
              (1) a strictly monotone time channel t. By Theorem 4.2 (and its proof,
                  p.6) this monotone coordinate makes the signature injective — it
                  determines the augmented path uniquely — which is what licenses the
                  universal approximation of payoffs/value functions by linear
                  functionals of the signature.
              (2) the state path X_t itself.
              (3) a linear "basepoint" ramp (X_0 / T) * t, running 0 at the window
                  start up to X_0 (the earliest, lower-integration-limit state) at the
                  window end; its total increment is exactly X_0, which injects the
                  absolute level so that level-dependent functionals stay linear in the
                  signature (paper p.6 examples).

            `data[0]` is the earliest (window-start) sample, i.e. X_0. T is the
            (normalized) window length `time_origin` (default 1.0). The signature is
            invariant under translating a channel by a constant, so writing the ramp
            as 0 -> X_0 here is equivalent to any constant-shifted variant of the same
            increment, and the time-channel normalization only rescales that channel
            (absorbed by the downstream critic/actor layer). Channels are ordered
            (t, X, ramp) to match the paper.
            """
            n = data.shape[0]
            if origin_aug:
                # Channel 3: (X_0 / T) * t, ramping 0 -> X_0 over the window (Perez Arribas p.5).
                elapsed_fraction = jnp.linspace(0.0, 1.0, n).reshape(-1, 1)
                origin = data[0] * elapsed_fraction
                data = jnp.concatenate([data, origin], axis=1)
            if time_aug:
                # Channel 1: monotone time t in [0, T], prepended (Perez Arribas p.5).
                time_channel = jnp.linspace(0.0, time_origin, n).reshape(-1, 1)
                data = jnp.concatenate([time_channel, data], axis=1)
            sig = sig_transform(data)
            if bias:
                return jnp.concatenate([jnp.array([1.0]), sig])
            return sig
        
        return compute_sig
    
    def reset(self, prefill_zeros: bool = True) -> None:
        """Clear buffer and reset signature to zeros.
        
        Args:
            prefill_zeros: If True, pre-fill buffer with zeros to avoid JIT recompilation
                          due to changing buffer size during warmup.
        """
        if self.use_jax_buffer:
            self.buffer = JAXCircularBuffer(size=self.window_size + 1, d=self.d_raw, dtype=self.dtype)
        else:
            self.buffer = DequeBuffer(size=self.window_size + 1, dtype=self.dtype)

        # Pre-fill buffer to avoid recompilation from size changes
        if prefill_zeros:
            zero_item = np.zeros(self.d_raw, dtype=self.dtype)
            for _ in range(self.window_size + 1):
                self.buffer.append(zero_item)
        
        self._current_signature = self._empty_sig
        self._signature_dirty = True
    
    def append(self, item):
        """Append item and recompute signature."""
        if self.use_jax_buffer:
            self.buffer.append(item)  # JAXCircularBuffer handles conversion
        else:
            self.buffer.append(np.asarray(item, dtype=self.dtype))
        # Lazy computation - signature will be computed on access
        self._signature_dirty = True
    
    @property
    def current_signature(self) -> jnp.ndarray:
        """Get current signature, computing if needed."""
        if self._signature_dirty:
            self._current_signature = self.compute_signature()
            self._signature_dirty = False
        return self._current_signature
    
    @current_signature.setter
    def current_signature(self, value):
        self._current_signature = value
        self._signature_dirty = False
    
    def compute_signature(self) -> jnp.ndarray:
        """Compute signature from buffer."""
        n = len(self.buffer)
        if n < 2:
            return self._empty_sig
        
        # Get buffer contents - already JAX array if using JAXCircularBuffer
        data = self.buffer.to_array()
        if not isinstance(data, jnp.ndarray):
            data = jnp.array(data)
        return self._jit_compute_sig(data)