from typing import Optional, Sequence
import jax
import flax.linen as nn
import jax.numpy as jnp

from big_vision.models.vit import Encoder
from big_vision.models.embeddings import TimeEmb, LabelEmbedder, EmbeddingTrunk

def random_masking(x, mask_ratio, rng_key):
  N, L, _ = x.shape  # batch, length, dim
  len_keep = int(L * (1 - mask_ratio))

  # Generate uniform random noise and shuffle ids
  noise = jax.random.uniform(rng_key, (N, L))
  ids_shuffle = jnp.argsort(noise, axis=1)
  ids_restore = jnp.argsort(ids_shuffle, axis=1)

  # Create the mask for the first subset
  ids_keep = ids_shuffle[:, :len_keep]
  # Reshape ids_keep for broadcasting and take along axis
  ids_keep = ids_keep[:, :, None]  # Reshape to (N, idx, 1) for proper broadcasting
  x_masked = jnp.take_along_axis(x, ids_keep, axis=1)

  # Generate the binary mask: 0 is keep, 1 is remove
  mask = jnp.ones((N, L))
  mask = mask.at[:, :len_keep].set(0)
  mask = jax.vmap(lambda m, idx: jnp.take_along_axis(m, idx, axis=0))(mask, ids_restore)
  return x_masked, mask, ids_restore

def sequence_mask_to_image_mask(sequence_mask, patch_size, img_size):
  """Converts a sequence mask to an image mask."""
  num_patches_height, num_patches_width = img_size // patch_size, img_size // patch_size
  reshaped_mask = sequence_mask.reshape((-1, num_patches_height, num_patches_width))
  mask_array = jnp.repeat(jnp.repeat(reshaped_mask, patch_size, axis=1), patch_size, axis=2)
  mask_array = jnp.expand_dims(mask_array, axis=-1)
  return mask_array

class _ViTAE(nn.Module):
  num_classes: int = None #Initialize label embedding for fine-tuning. Pre-training is self-supervised
  channels: int = 3
  img_size: int = 64
  patch_size: Sequence[int] = (4, 4)
  width: int = 768
  depth: int = 12
  dec_depth: int = 4
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = True
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"
  adaln: bool = False
  cfg_dropout_rate: float = 0.1
  num_cls: int = 4
  no_decay_list: Sequence[str] = ('cls', 'image_mask_embedding', 'bias')

  def setup(self):
    self.sinusoidal_pos_emb = TimeEmb(self.width, dtype=self.dtype_mm)
    self.time_trunk = EmbeddingTrunk(self.width, 2)
    if self.num_classes is not None:
      self.label_emb = LabelEmbedder(hidden_size=self.width, class_dropout_prob=self.cfg_dropout_rate, num_classes=self.num_classes) #Initialize for now, but only used during fine-tuning
      self.label_trunk = EmbeddingTrunk(self.width, 2)
    self.cls = self.param("cls", nn.initializers.zeros, (1, self.num_cls, self.width), self.dtype_mm)
    self.image_embed = nn.Conv(
        self.width, self.patch_size, strides=self.patch_size,
        padding="VALID", name="embedding", dtype=self.dtype_mm)
    
    self.h = self.w = self.img_size // self.patch_size[0]
    self.posemb = self.param("pos_embedding", nn.initializers.normal(stddev=1/jnp.sqrt(self.h * self.w)), (1, self.h * self.w, self.width), self.dtype_mm)
    self.dec_posemb = self.param("dec_pos_embedding", nn.initializers.normal(stddev=1/jnp.sqrt(self.h * self.w)), (1, self.h * self.w, self.width), self.dtype_mm)
    self.image_mask_embedding = self.param("image_mask_embedding", nn.initializers.normal(stddev=0.02, dtype=self.dtype_mm), (1, 1, self.width),)

    self.encoder = Encoder(
        depth=self.depth,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        adaln=self.adaln,
        name="Encoder")
    
    self.decoder = Encoder(
        depth=self.dec_depth,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        adaln=self.adaln,
        name="Decoder")
    
    if self.adaln:
      self.final_modulation = nn.Dense(self.width * 2, kernel_init=nn.initializers.zeros, name="final_modulation", dtype=self.dtype_mm)
    self.final_conv = nn.ConvTranspose(self.channels * 2, kernel_size=self.patch_size, strides=self.patch_size,
        padding="VALID", name="final_conv", dtype=self.dtype_mm,
        kernel_init=jax.nn.initializers.normal(0.02))
  
  def embed(self, image, t=None, y=None, train=False):
    image = jnp.asarray(image, self.dtype_mm)
    x = self.image_embed(image)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    if t is None:
      t = jnp.zeros((image.shape[0], 1), dtype=jnp.int32) #0 is for no noise or unconditional time
    if y is None and self.num_classes is not None:
      y = jnp.ones((n, ), dtype=jnp.int32) * self.num_classes
      y_cond = self.label_emb(y, train=train)
      y_cond = self.label_trunk(y_cond, train=train)
    elif y is not None:
      assert self.num_classes is not None, 'num_classes must be provided if y is not None'
      y_cond = self.label_emb(y, train=train)
      y_cond = self.label_trunk(y_cond, train=train)
    else:
      y_cond = jnp.zeros((n, self.width), dtype=self.dtype_mm)
    
    time_cond = self.sinusoidal_pos_emb(t, train=train)  # [B. dim]
    time_cond = self.time_trunk(time_cond, train=train)
    
    if not self.adaln:
      cond = time_cond + y_cond
    else:
      cond = nn.silu(time_cond + y_cond)
    return x, cond
  
  def encode(self, x, cond, *, mask=0.0, train=False):
    out = {}
    n, _, _ = x.shape
    x = x + self.posemb
    
    if mask > 0.0:
      x, image_mask, ids_restore = random_masking(x, mask, self.make_rng("mae_noise"))
      out['mask'] = sequence_mask_to_image_mask(image_mask, self.patch_size[0], self.img_size)
    else:
      ids_restore = None
      out['mask'] = None

    x = jnp.concatenate([jnp.tile(self.cls, [n, 1, 1]), x], axis=1)
    x, _ = self.encoder(
        x, cond=cond, deterministic=not train)
    rep = x[:, :self.num_cls].mean(axis=1) #Average the class tokens
    encoded = x[:, self.num_cls:]
    out["pre_logits"] = rep
    return rep, encoded, ids_restore, out
  
  def decode(self, rep, x, cond, ids_restore=None, mask=0.0, train=False):
    n = x.shape[0]

    if ids_restore is not None:
      masked_image_x = jnp.broadcast_to(
        self.image_mask_embedding,
        (n,
        ids_restore.shape[1] - int(ids_restore.shape[1] * (1.0 - mask)),
        self.width,
        ))

      x = jnp.concatenate([x, masked_image_x], axis=1)
      x = jnp.take_along_axis(x, ids_restore[:, :, None], axis=1)

    x = (x + self.dec_posemb)
    x = jnp.concatenate([rep[:, None, :], x], axis=1)
    x, _ = self.decoder(x, cond=cond, deterministic=not train)
    x = x[:, 1:, :]

    if self.adaln:
      cond = self.final_modulation(cond)
      cond = cond[:, jnp.newaxis, :]
      shift, scale = jnp.split(cond, 2, axis=-1)
      x = x * (1 + scale) + shift

    x = x.reshape((n, self.h, self.w, self.width))
    x = self.final_conv(x)
    return x

  def __call__(self, image, *, t=None, y=None, cfg_scale=None, mask=0.0, train=False):
    if cfg_scale is not None:
      assert y is not None, 'y must be provided if cfg_scale is not None'
      assert self.num_classes is not None, 'num_classes must be provided if cfg_scale is not None'
      assert not train, 'cfg_scale is only used during inference'
      n = image.shape[0]
      image = jnp.concatenate([image, image.copy()], axis=0)
      t = jnp.concatenate([t, t.copy()], axis=0)
      null_y = jnp.ones((n, ), dtype=jnp.int32) * self.num_classes
      y = jnp.concatenate([y, null_y], axis=0)
      n = image.shape[0]

    x, cond = self.embed(image, t=t, y=y, train=train)
    rep, encoded, ids_restore, out = self.encode(x, cond, mask=mask, train=train)
    pred = self.decode(rep, encoded, cond, ids_restore=ids_restore, mask=mask, train=train)

    if cfg_scale is not None:
      unconditional = pred[n//2:]
      conditional = pred[:n//2]
      pred = unconditional + cfg_scale * (conditional - unconditional)

    return pred, out
  

def decode_variant(variant):
  """Converts a string like "B" or "B/32" into a params dict."""
  if variant is None:
    return {}

  v, patch = variant, {}
  if "/" in variant:
    v, patch = variant.split("/")
    patch = {"patch_size": (int(patch), int(patch))}

  return { #Default params for MAE
      # pylint:disable=line-too-long
      "width": {"S": 384, "B": 768, "L": 1024}[v],
      "depth": {"S": 12, "B": 12, "L": 24}[v],
      "dec_depth": {"S": 4, "B": 4, "L": 8}[v],
      "num_heads": {"S": 6, "B": 12, "L": 16}[v],
      # pylint:enable=line-too-long
      **patch
  }

def Model(*, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _ViTAE(**{**decode_variant(variant), **kw})
