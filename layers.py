import numpy as np
import jax.numpy as jnp
import flax.linen as nn


class GLU(nn.Module):
    dim:int
    @nn.compact
    def __call__(self, x):
        out, gate = jnp.split(x,2, axis=self.dim)
        return out * nn.sigmoid(gate)
    
class ConformerConvModule(nn.Module):
  output_dim : int = 256
  kernel_size : int = 31
  @nn.compact
  def __call__(self, x):
    x = nn.LayerNorm()(x)
    x = nn.Conv(self.output_dim * 4, [1])(x)
    x = GLU(dim=2)(x)
    x = nn.Conv(self.output_dim, kernel_size = [self.kernel_size])(x)
    x = nn.swish(x)
    x = nn.Conv(self.output_dim, kernel_size = [self.kernel_size])(x)
    return x

class EncoderBlock(nn.Module):
  emb_dim : int = 256
  n_heads : int = 8
  

  @nn.compact
  def __call__(self, input):
    x = input
    x = nn.LayerNorm()(x)
    x = nn.SelfAttention(num_heads=self.n_heads,out_features=self.emb_dim)(x)
    x = x + input
    
    y = x
    y = ConformerConvModule(self.emb_dim)(y)
    y = y + x

    return y
