from typing import Optional
from src.utils.dynamic_signature import SlidingSignature

import numpy as np
import flax.linen as nn
import jax.numpy as jnp


class ActorNetwork:
    def __init__(self, input_dim: int, output_dim: int, rng: Optional[np.random.Generator] = None) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = self.rng.standard_normal((input_dim, output_dim))*0.01
    def __call__(self, x): 
        return self.W.T@x

class CriticNetwork:
    def __init__(self, input_dim: int, rng: Optional[np.random.Generator] = None) -> None:
        self.input_dim = input_dim
        self.feature_dim = input_dim + input_dim * (input_dim - 1) // 2
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = self.rng.standard_normal(self.feature_dim)*0.01
    def _compute_features(self, x):
        features = []
        for i in range(self.input_dim):
            for j in range(i, self.input_dim):
                features.append(x[i] * x[j])
        return np.array(features)
    def __call__(self, x): return np.dot(self.W, self._compute_features(x))

class CriticOracle:
    def __init__(self, input_dim: int, P: np.ndarray, rng: Optional[np.random.Generator] = None) -> None:
        self.input_dim = input_dim
        self.feature_dim = input_dim + input_dim * (input_dim - 1) // 2
        self.rng = rng if rng is not None else np.random.default_rng()
        self.P = P
        # W pour compatibilité avec l'affichage (flatten de P pour 1D)
        self.W = np.array([P[0, 0]]) if input_dim == 1 else P.flatten()
    
    def _compute_features(self, x):
        """Même interface que CriticNetwork pour compatibilité."""
        features = []
        for i in range(self.input_dim):
            for j in range(i, self.input_dim):
                features.append(x[i] * x[j])
        return np.array(features)
    
    def __call__(self, x): 
        return -x.T @ self.P @ x

class CriticSignature:
    def __init__(self, input_dim: int, sliding_signature: SlidingSignature, rng: Optional[np.random.Generator] = None) -> None:
        self.input_dim = input_dim
        self.sliding_signature = sliding_signature
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = self.rng.standard_normal(self.sliding_signature.signature_size)*0.01
    def __call__(self, signature): 
        return np.dot(self.W, signature)
    
class ActorSignature:
    def __init__(self, output_dim: int, sliding_signature: SlidingSignature, rng: Optional[np.random.Generator] = None) -> None:
        self.output_dim = output_dim
        self.sliding_signature = sliding_signature
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = self.rng.standard_normal((self.sliding_signature.signature_size, output_dim))*0.01
    def __call__(self, signature): 
        return self.W.T @ signature
    
# Every parametric layer below carries param_dtype=jnp.float64. Flax defaults param_dtype to
# float32 REGARDLESS of jax_enable_x64, so without this the network weights are float32 even though
# the data pipeline (buffers, env, signatures, gradients) runs in float64 under main_unified.py:63.
# The study measures conditioning of the gradient extraction d_x V, which on the high-dimensional
# platoon cell passes through a near-singular readout where float32 weights can change the answer;
# full float64 is therefore the deliberate, consistent choice. Applied to every campaign module
# (ActorFlax, CriticFlax and their LayerNorm variants). Changes all three learners symmetrically,
# so it adds no confound between them; it does change absolute results against the pre-2026-07-17
# float32-weight runs, which remain reproducible from their own commit.
_PARAM_DTYPE = jnp.float64


class ActorFlax(nn.Module):
    output_dim: int
    stddev: float = 0.01
    @nn.compact
    def __call__(self, input):
        return nn.Dense(features=self.output_dim,
                        use_bias=False,
                        param_dtype=_PARAM_DTYPE,
                        kernel_init=nn.initializers.normal(stddev=self.stddev))(input)

class CriticFlax(nn.Module):
    stddev: float = 0.01
    hidden_dims: tuple = ()  # Default to no hidden layers

    @nn.compact
    def __call__(self, input):
        x = input
        # Create hidden layers if specified
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim, param_dtype=_PARAM_DTYPE,
                         kernel_init=nn.initializers.orthogonal(scale=np.sqrt(2)))(x)
            x = nn.relu(x)  # Common activation for hidden layers

        out = nn.Dense(features=1,
                       use_bias=False,
                       param_dtype=_PARAM_DTYPE,
                       kernel_init=nn.initializers.normal(stddev=self.stddev))(x)
        return out.squeeze()

class CriticFlaxLayerNorm(nn.Module):
    stddev: float = 0.01
    hidden_dims: tuple = ()  # Default to no hidden layers
    @nn.compact
    def __call__(self, input):
        x = input
        x = nn.LayerNorm(param_dtype=_PARAM_DTYPE)(x)
        # Create hidden layers if specified
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim, param_dtype=_PARAM_DTYPE,
                         kernel_init=nn.initializers.orthogonal(scale=np.sqrt(2)))(x)
            x = nn.relu(x)  # Common activation for hidden layers

        out = nn.Dense(features=1,
                       use_bias=False,
                       param_dtype=_PARAM_DTYPE,
                       kernel_init=nn.initializers.normal(stddev=self.stddev))(x)
        return out.squeeze()

class ActorFlaxLayerNorm(nn.Module):
    output_dim: int
    stddev: float = 0.01
    @nn.compact
    def __call__(self, input):
        x = nn.LayerNorm(param_dtype=_PARAM_DTYPE)(input)
        return nn.Dense(features=self.output_dim,
                        use_bias=False,
                        param_dtype=_PARAM_DTYPE,
                        kernel_init=nn.initializers.normal(stddev=self.stddev))(x)

class CriticFlaxQuadratic(nn.Module):
    """Critic with quadratic features for LQR problems.
    
    Computes V(x) = W @ features(x) where features are quadratic:
    [x_0^2, x_0*x_1, ..., x_0*x_n, x_1^2, x_1*x_2, ..., x_n^2]
    """
    stdev: float = 0.01
    @nn.compact
    def __call__(self, x):
        import jax.numpy as jnp
        # Compute quadratic features: x_i * x_j for i <= j
        n = x.shape[-1] if x.ndim > 0 else 1
        features = []
        for i in range(n):
            for j in range(i, n):
                features.append(x[i] * x[j])
        features_array = jnp.stack(features)

        out = nn.Dense(features=1,
                       use_bias=False,
                       kernel_init=nn.initializers.normal(stddev=self.stdev))(features_array)
        return out.squeeze()

class ActorFlaxSTD(nn.Module):
    output_dim: int
    stddev: float = 0.01
    @nn.compact
    def __call__(self, input): 
        mu = nn.Dense(features=self.output_dim,
                        use_bias=False,
                        kernel_init=nn.initializers.normal(stddev=self.stddev),
                        bias_init=nn.initializers.zeros,
                        name='mu head')(input)
        log_std = nn.Dense(features=self.output_dim,
                           kernel_init=nn.initializers.orthogonal(scale=0.01),
                           bias_init=nn.initializers.constant(0.),
                            name='log_std head')(input)
        LOG_STD_MIN = -5
        LOG_STD_MAX = 0
        log_std = jnp.tanh(log_std) 
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = jnp.exp(log_std)
        return mu, std

    
class ActorFlaxLayerNormSTD(nn.Module):
    output_dim: int
    stddev: float = 0.01
    # initial_log_std: float = -0.2  # Default initial log std
    @nn.compact
    def __call__(self, input): 
        x = nn.LayerNorm()(input)
        mu = nn.Dense(features=self.output_dim,
                        kernel_init=nn.initializers.normal(stddev=self.stddev),
                        bias_init=nn.initializers.zeros,
                        name='mu head')(x)
        LOG_STD_MIN = -5
        LOG_STD_MAX = 0
        log_std = nn.Dense(features=self.output_dim,
                           use_bias=False,
                           kernel_init=nn.initializers.orthogonal(scale=0.01),
                           bias_init=nn.initializers.constant(0.),
                            name='log_std head')(x)
        log_std = jnp.tanh(log_std) 
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = jnp.exp(log_std)
        return mu, std

class AlphaModel(nn.Module):
    initial_log_alpha: float = 0.0
    @nn.compact
    def __call__(self):
        log_alpha = self.param('log_alpha', nn.initializers.constant(self.initial_log_alpha), ())
        return jnp.clip(log_alpha, -5, 2.0)