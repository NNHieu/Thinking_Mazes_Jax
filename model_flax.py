from flax import linen as nn
import jax
import flax
import jax.numpy as jnp

class BasicBlock2D(nn.Module):
    @nn.compact
    def __call__(self,x):
        conv1 = nn.Conv(features=128, kernel_size=(3, 3))
        conv2 = nn.Conv(features=128, kernel_size=(3, 3))

        h = nn.relu(conv1(x))
        h = conv2(h) + x
        h = nn.relu(x)
        
        return h

class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    
    iter_block = BasicBlock2D()
    head = nn.Sequential((
        nn.Conv(features=32, kernel_size=(3, 3)),
        nn.relu,
        nn.Conv(features=8, kernel_size=(3, 3)),
        nn.relu,
        nn.Conv(features=2, kernel_size=(3, 3)),
    ))
    
    x = flax .fori_loop(0, 30, lambda i, h: iter_block(h), x)
    x = head(x)

    return x