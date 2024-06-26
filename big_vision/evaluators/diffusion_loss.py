# pylint: disable=consider-using-from-import

import functools
import big_vision.datasets.core as ds_core
import big_vision.input_pipeline as input_pipeline
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import jax
import jax.numpy as jnp

# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@functools.cache
def get_eval_fn(predict_fn):
  """Produces eval function, also applies pmap."""
  @jax.jit
  def _loss_fn(train_state, batch):
    loss, x_t, x_0, x_0_eps = predict_fn(train_state, batch)
    x_t = jnp.clip(x_t, -1, 1)
    x_0 = jnp.clip(x_0, -1, 1)
    x_0_eps = jnp.clip(x_0_eps, -1, 1)
    return loss, x_t, x_0, x_0_eps
  return _loss_fn


class Evaluator:
  """Sampling evaluator."""

  def __init__(self, predict_fn, data, pp_fn, batch_size,
               cache_final=True, cache_raw=False, prefetch=1,
               label_key='labels', *, devices):
    data = ds_core.get(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), pp_fn, batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_global(self.ds, devices, prefetch)
    self.eval_fn = get_eval_fn(predict_fn)
    self.label_key = label_key

  def run(self, train_state):
    """Computes all metrics."""
    total_loss, nseen = 0, 0
    for i, batch in zip(range(self.steps), self.data_iter):
      if jax.process_count() == 1:
        loss, x_t, x_0, x_0_eps = jax.device_get(self.eval_fn(train_state, batch))
      else:
        gather = jax.experimental.multihost_utils.process_allgather(
          self.eval_fn(train_state, batch))
        loss, x_t, x_0, x_0_eps = jax.device_get(gather)
      total_loss += loss
      nseen += 1
      if i == 0:
        first_batch_images = batch['image']
        first_batch_x_t = x_t
        first_batch_pred_x_0 = x_0
        first_batch_pred_x_0_eps = x_0_eps

    yield ('loss', total_loss / nseen)
    yield ('batch', first_batch_images) #Just to see what's in the batch
    yield ('wandbimage_x_t', first_batch_x_t)
    yield ('wandbimage_pred_x', first_batch_pred_x_0)
    yield ('wandbimage_pred_x_eps', first_batch_pred_x_0_eps)