import flax.linen as nn
import numpy as np

class CriticFlax(nn.Module):
    hidden_dims: tuple = ()  # Default to no hidden layers
    stddev: float = 0.01
    @nn.compact
    def __call__(self, input):
        x = input
        # Create hidden layers if specified
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim, kernel_init=nn.initializers.orthogonal(scale=np.sqrt(2)))(x)
            x = nn.relu(x)  # Common activation for hidden layers
            
        out = nn.Dense(features=1,
                       use_bias=False,
                       kernel_init=nn.initializers.normal(stddev=self.stddev))(x)
        return out.squeeze()

class CriticFlaxLayerNorm(nn.Module):
    hidden_dims: tuple = ()  # Default to no hidden layers
    stddev: float = 0.01
    @nn.compact
    def __call__(self, input):
        x = input
        x = nn.LayerNorm()(x)
        # Create hidden layers if specified
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim, kernel_init=nn.initializers.orthogonal(scale=np.sqrt(2)))(x)
            x = nn.relu(x)  # Common activation for hidden layers
            
        out = nn.Dense(features=1,
                       use_bias=False,
                       kernel_init=nn.initializers.normal(stddev=self.stddev))(x)
        return out.squeeze()
    