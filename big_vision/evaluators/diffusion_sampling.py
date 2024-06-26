# pylint: disable=consider-using-from-import

import functools
import jax
import jax.numpy as jnp
from tqdm import tqdm



# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@functools.cache
def get_eval_fn(predict_fn):
  """Produces eval function, also applies pmap."""
  @jax.jit
  def _sample_fn(train_state, rng):
    dict = predict_fn(train_state, rng)
    return dict
  return _sample_fn


class Evaluator:
  """Sampling evaluator."""

  def __init__(self, predict_fn, batch_size, total_samples=1000, *, devices):
    self.eval_fn = get_eval_fn(predict_fn)
    self.total_samples = total_samples

  def run(self, train_state):
    """Computes all metrics."""
    rng = jax.random.PRNGKey(0) #Fix rng so samples are consistent
    total_images = 0
    fid_samples = None
    ys = None
    while total_images < self.total_samples:
      if jax.process_count() == 1:
        dict = jax.device_get(
            self.eval_fn(train_state, rng))
      else:
        gather = jax.experimental.multihost_utils.process_allgather(
          self.eval_fn(train_state, rng))
        dict = jax.device_get(gather)
      rng, _ = jax.random.split(dict["rng"])
      if fid_samples is None:
        fid_samples = dict["fid_samples"]
        ys = dict["ys"]
      else:
        fid_samples = jnp.concatenate([fid_samples, dict["fid_samples"]], axis=0)
        if ys is not None:
          ys = jnp.concatenate([ys, dict["ys"]], axis=0)
      wandbimage_examples = dict["wandbimage_examples"]
      total_images += dict["fid_samples"].shape[0]

    yield ("fid_samples", {"samples": fid_samples, "ys": ys})
    yield ("wandbimage_examples", wandbimage_examples)
    
    