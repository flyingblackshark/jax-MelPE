
import flax.linen as nn

from .layers import *

class MelPETransformer(nn.Module):
  embed_dim : int = 256
  output_dim : int = 360
  n_enc_layers : int = 24

  def setup(self):
    self.encoder_layers = [EncoderBlock() for i in range(self.n_enc_layers)]
    self.fc_out = nn.Dense(self.output_dim)
    self.stack = nn.Sequential([
            nn.Conv(self.embed_dim, [3]),
            nn.GroupNorm(num_groups=4),
            nn.leaky_relu,
            nn.Conv(self.embed_dim, [3])])
    self.norm = nn.LayerNorm()
  def __call__(self, src_input):
    src_input = src_input.transpose(0,2,1)
    x = self.stack(src_input)
    for i in range(len(self.encoder_layers)):
      x = self.encoder_layers[i](x)
    x = self.norm(x)
    x = self.fc_out(x)
    x = nn.sigmoid(x)
    return x

  