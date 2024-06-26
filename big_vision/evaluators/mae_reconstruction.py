# Copyright 2023 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluator for the classfication task."""
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
  def _patch_predict_fn(train_state, batch):
    image_output, image_mask = predict_fn(train_state, batch)
    true_image = batch['image']
    masked_image = true_image * (1 - image_mask)
    predicted_image = image_output
    predicted_image_combined = image_output * image_mask + true_image * (1 - image_mask)
    loss = jnp.mean((predicted_image * image_mask  - true_image * image_mask) ** 2) / jnp.mean(image_mask)
    predicted_image_combined = jnp.clip(predicted_image_combined, -1, 1)
    return loss, masked_image, predicted_image_combined
  return _patch_predict_fn


class Evaluator:
  """Classification evaluator."""

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
      #If multihost, we need to do allgather
      if jax.process_count() == 1:
        loss, masked_image, predicted_image_combined = jax.device_get(
            self.eval_fn(train_state, batch))
      else:
        gather = jax.experimental.multihost_utils.process_allgather(
          self.eval_fn(train_state, batch))
        loss, masked_image, predicted_image_combined = jax.device_get(gather)
      total_loss += loss
      nseen += 1
      if i == 0:
        first_batch_images = batch['image']
        first_batch_masked_images = masked_image
        first_batch_predicted_images_combined = predicted_image_combined

    yield ('loss', total_loss / nseen)
    yield ('batch', first_batch_images) #Just to see what's in the batch
    yield ('masked_wandbimage', first_batch_masked_images)
    yield ('predicted_wandbimage', first_batch_predicted_images_combined)