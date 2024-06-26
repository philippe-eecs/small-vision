import flax.linen as nn
import jax.numpy as jnp
import jax
import math

from typing import Any, Callable, Tuple

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

class TimeEmb(nn.Module):
  hidden_size: int
  dtype: jnp.dtype = jnp.float32
  dropout_prob: float = 0.0
  unconditional_int: int = 0

  @nn.compact
  def __call__(self, time, train=False):
    if train and self.dropout_prob > 0.0:
      drop_ids = jax.random.bernoulli(self.make_rng('time_cfg'), p=self.dropout_prob, shape=time.shape)
      time = jnp.where(drop_ids, self.unconditional_int, time)

    assert len(time.shape) == 2.
    half_dim = self.hidden_size // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=self.dtype) * -emb)
    emb = time.astype(self.dtype) * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return emb

class LabelEmbedder(nn.Module):
  """
  Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
  """
  hidden_size: int = 768
  num_classes: int = 1000
  class_dropout_prob: float = 0.1
  
  @nn.compact
  def __call__(self, labels, train=False):
    if train:
      drop_ids = jax.random.bernoulli(self.make_rng('cfg'), p=self.class_dropout_prob, shape=labels.shape)
      labels = jnp.where(drop_ids, self.num_classes, labels)
    
    embedding = nn.Embed(num_embeddings=self.num_classes + 1, features=self.hidden_size, name='embedding')(labels)
    return embedding

class EmbeddingTrunk(nn.Module):
  width: int
  mlp_factor: int = 2
  
  @nn.compact
  def __call__(self, cond, train=False):
    cond = nn.Dense(features=self.width * self.mlp_factor)(cond)
    cond = nn.silu(cond)
    cond = nn.Dense(features=self.width)(cond)
    return cond