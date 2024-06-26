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

"""Most common few-shot eval configuration."""

import ml_collections as mlc


def get_fewshot_lsr(target_resolution=224, resize_resolution=256,
                    runlocal=False, **kw):
  """Returns a standard-ish fewshot eval configuration."""
  kw.setdefault('representation_layer', 'pre_logits')
  kw.setdefault('pred', 'predict')
  kw.setdefault('datasets', {'imagenet': ('imagenet2012', 'train[:100000]', 'validation')})
  kw.setdefault('shots', (100,))
  kw.setdefault('l2_reg', 1024)
  kw.setdefault('num_seeds', 1)
  kw.setdefault('prefix', '')  # No prefix as we already use a/ z/ and zz/

  # Backward-compatible default:
  if not any(f'log_{x}' in kw for x in ['steps', 'percent', 'examples', 'epochs']):  # pylint: disable=line-too-long
    kw['log_steps'] = 25_000

  config = mlc.ConfigDict(kw)
  config.type = 'fewshot_lsr'
  config.datasets = kw['datasets']


  config.pp_train = (f'decode|resize_small({resize_resolution}, antialias=True)|'
                     f'central_crop({target_resolution})|'
                     f'value_range(-1,1)|keep("image", "label")')
  config.pp_eval = (f'decode|resize_small({resize_resolution}, antialias=True)|'
                    f'central_crop({target_resolution})|'
                    f'value_range(-1,1)|keep("image", "label")')
  config.display_first = []
  return config
