import jax
from jax import lax
import scipy
from jax.nn import initializers
import jax.numpy as jnp
from flax.linen.module import merge_param
import flax.linen as nn
from jax.experimental import mesh_utils
import big_vision.sharding as bv_sharding
from typing import Callable, Iterable, Optional, Tuple, Union, Any
import functools
import pickle
from tqdm import tqdm
import requests
import os
import tempfile
import numpy as np
from tensorflow.io import gfile
import pickle 

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any

def compute_statistics(arr, inception_forward, batch_size=1024):
    assert len(arr.shape) == 4, 'Input array must have shape [B, H, W, C]'
    assert arr.dtype == np.uint8, 'Input array must be of type jnp.uint8'
    num_batches = int(arr.shape[0] // batch_size) + int(arr.shape[0] % batch_size != 0)
    acts = []
    softmax_outputs = []
    for i in range(num_batches):
        x = arr[i * batch_size : (i + 1) * batch_size].copy()
        x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear') # Resize to 299x299
        pred = inception_forward(x)
        pred = jax.device_get(pred) # Bring to CPU
        acts.append(pred['acts'].squeeze(axis=1).squeeze(axis=1))
        softmax_outputs.append(jax.nn.softmax(pred['logits'], axis=1))
    acts = np.concatenate(acts, axis=0)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    softmax_outputs = np.concatenate(softmax_outputs, axis=0)
    inception_score = compute_inception_score(softmax_outputs)
    return mu, sigma, inception_score

def compute_inception_score(p_yx, num_splits=10):
    split_scores = []
    n = p_yx.shape[0]
    for i in range(num_splits):
        part = p_yx[i * n // num_splits: (i + 1) * n // num_splits]
        p_y = np.mean(part, axis=0, keepdims=True)
        kl_div = part * (np.log(part) - np.log(p_y))
        kl_div = np.sum(kl_div, axis=1)
        split_scores.append(np.exp(np.mean(kl_div)))
    return np.mean(split_scores)

def compute_frechet_distance(mu1, mu2, sigma1, sigma2):
    # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_1d(sigma1)
    sigma2 = np.atleast_1d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2)).real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def create_fid_score_fn(batch_size, ref_arr_path):
    model = InceptionV3(pretrained=True)
    _, rng_init = jax.random.split(jax.random.PRNGKey(0))
    params_shape = jax.eval_shape(model.init, rng_init, jnp.ones((1, 299, 299, 3)))
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = jax.sharding.Mesh(devices, ("data",))
    params_sharding = bv_sharding.infer_sharding(
        params_shape, mesh, axis_name="data",
        strategy="replicated",
        extra_strategy_args={},
        )
    params = jax.jit(model.init, out_shardings=params_sharding)(rng_init, jnp.ones((1, 299, 299, 3)))
    apply_fn = functools.partial(model.apply, train=False)

    def inception_forward(arr):
        x = arr / 255.0
        x = 2 * x - 1
        pred = apply_fn(params, jax.lax.stop_gradient(x))
        return pred

    inception_forward = jax.jit(inception_forward)
    with gfile.GFile(ref_arr_path, 'rb') as f:
        stats = np.load(f, allow_pickle=True).item()
    mu1, sigma1 = stats['mu'], stats['sigma']
    def compute_scores(arr):
        mu2, sigma2, inception_score = compute_statistics(arr, inception_forward, batch_size)
        fid_score = compute_frechet_distance(mu1, mu2, sigma1, sigma2)
        return fid_score, inception_score
    return compute_scores

def download(url, ckpt_dir=None):
    name = url[url.rfind('/') + 1 : url.rfind('?')]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'jax_fid')
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file

def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]

class InceptionV3(nn.Module):
    """
    InceptionV3 network.
    Reference: https://arxiv.org/abs/1512.00567
    Ported mostly from: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

    Attributes:
        include_head (bool): If True, include classifier head.
        num_classes (int): Number of classes.
        pretrained (bool): If True, use pretrained weights. 
        transform_input (bool): If True, preprocesses the input according to the method with which it
                                was trained on ImageNet.
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
        dtype (str): Data type.
    """
    include_head: bool=False
    num_classes: int=1000
    pretrained: bool=False
    transform_input: bool=False
    aux_logits: bool=False
    ckpt_path: str='https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1'
    dtype: str='float32'

    def setup(self):
        if self.pretrained:
            ckpt_file = download(self.ckpt_path)
            self.params_dict = pickle.load(open(ckpt_file, 'rb'))
            self.num_classes_ = 1000
        else:
            self.params_dict = None
            self.num_classes_ = self.num_classes

    @nn.compact
    def __call__(self, x, train=True, rng=jax.random.PRNGKey(0)):
        """
        Args:
            x (tensor): Input image, shape [B, H, W, C].
            train (bool): If True, training mode.
            rng (jax.random.PRNGKey): Random seed.
        """
        x = self._transform_input(x)
        x = BasicConv2d(out_channels=32,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        params_dict=get(self.params_dict, 'Conv2d_1a_3x3'),
                        dtype=self.dtype)(x, train)
        x = BasicConv2d(out_channels=32,
                        kernel_size=(3, 3),
                        params_dict=get(self.params_dict, 'Conv2d_2a_3x3'),
                        dtype=self.dtype)(x, train)
        x = BasicConv2d(out_channels=64,
                        kernel_size=(3, 3),
                        padding=((1, 1), (1, 1)),
                        params_dict=get(self.params_dict, 'Conv2d_2b_3x3'),
                        dtype=self.dtype)(x, train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = BasicConv2d(out_channels=80,
                        kernel_size=(1, 1),
                        params_dict=get(self.params_dict, 'Conv2d_3b_1x1'),
                        dtype=self.dtype)(x, train)
        x = BasicConv2d(out_channels=192,
                        kernel_size=(3, 3),
                        params_dict=get(self.params_dict, 'Conv2d_4a_3x3'),
                        dtype=self.dtype)(x, train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = InceptionA(pool_features=32,
                       params_dict=get(self.params_dict, 'Mixed_5b'),
                       dtype=self.dtype)(x, train)
        x = InceptionA(pool_features=64,
                       params_dict=get(self.params_dict, 'Mixed_5c'),
                       dtype=self.dtype)(x, train)
        x = InceptionA(pool_features=64,
                       params_dict=get(self.params_dict, 'Mixed_5d'),
                       dtype=self.dtype)(x, train)
        x = InceptionB(params_dict=get(self.params_dict, 'Mixed_6a'),
                       dtype=self.dtype)(x, train)
        x = InceptionC(channels_7x7=128,
                       params_dict=get(self.params_dict, 'Mixed_6b'),
                       dtype=self.dtype)(x, train)
        x = InceptionC(channels_7x7=160,
                       params_dict=get(self.params_dict, 'Mixed_6c'),
                       dtype=self.dtype)(x, train)
        x = InceptionC(channels_7x7=160,
                       params_dict=get(self.params_dict, 'Mixed_6d'),
                       dtype=self.dtype)(x, train)
        x = InceptionC(channels_7x7=192,
                       params_dict=get(self.params_dict, 'Mixed_6e'),
                       dtype=self.dtype)(x, train)
        aux = None
        if self.aux_logits and train:
            aux = InceptionAux(num_classes=self.num_classes_,
                               params_dict=get(self.params_dict, 'AuxLogits'),
                               dtype=self.dtype)(x, train)
        x = InceptionD(params_dict=get(self.params_dict, 'Mixed_7a'),
                       dtype=self.dtype)(x, train)
        x = InceptionE(avg_pool, params_dict=get(self.params_dict, 'Mixed_7b'),
                       dtype=self.dtype)(x, train)
        # Following the implementation by @mseitzer, we use max pooling instead
        # of average pooling here.
        # See: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py#L320
        x = InceptionE(nn.max_pool, params_dict=get(self.params_dict, 'Mixed_7c'),
                       dtype=self.dtype)(x, train)
        x = jnp.mean(x, axis=(1, 2), keepdims=True)
        acts = x.copy()
        x = nn.Dropout(rate=0.5)(x, deterministic=not train, rng=rng)
        x = jnp.reshape(x, newshape=(x.shape[0], -1))
        logits = Dense(features=self.num_classes_,
                  params_dict=get(self.params_dict, 'fc'),
                  dtype=self.dtype)(x)
        outs = {'logits': logits, 'acts': acts}
        return outs

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = jnp.expand_dims(x[..., 0], axis=-1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = jnp.expand_dims(x[..., 1], axis=-1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = jnp.expand_dims(x[..., 2], axis=-1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = jnp.concatenate((x_ch0, x_ch1, x_ch2), axis=-1)
        return x


class Dense(nn.Module):
    features: int
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features,
                     kernel_init=self.kernel_init if self.params_dict is None else lambda *_ : jnp.array(self.params_dict['kernel']),
                     bias_init=self.bias_init if self.params_dict is None else lambda *_ : jnp.array(self.params_dict['bias']))(x)
        return x


class BasicConv2d(nn.Module):
    out_channels: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    strides: Optional[Iterable[int]]=(1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]]='valid'
    use_bias: bool=False
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(features=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    use_bias=self.use_bias,
                    kernel_init=self.kernel_init if self.params_dict is None else lambda *_ : jnp.array(self.params_dict['conv']['kernel']),
                    bias_init=self.bias_init if self.params_dict is None else lambda *_ : jnp.array(self.params_dict['conv']['bias']),
                    dtype=self.dtype)(x)
        if self.params_dict is None:
            x = BatchNorm(epsilon=0.001,
                          momentum=0.1,
                          use_running_average=not train,
                          dtype=self.dtype)(x)
        else:
            x = BatchNorm(epsilon=0.001,
                          momentum=0.1,
                          bias_init=lambda *_ : jnp.array(self.params_dict['bn']['bias']),
                          scale_init=lambda *_ : jnp.array(self.params_dict['bn']['scale']),
                          mean_init=lambda *_ : jnp.array(self.params_dict['bn']['mean']),
                          var_init=lambda *_ : jnp.array(self.params_dict['bn']['var']),
                          use_running_average=not train,
                          dtype=self.dtype)(x)
        x = jax.nn.relu(x)
        return x


class InceptionA(nn.Module):
    pool_features: int
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch1x1 = BasicConv2d(out_channels=64,
                                kernel_size=(1, 1),
                                params_dict=get(self.params_dict, 'branch1x1'),
                                dtype=self.dtype)(x, train)
        branch5x5 = BasicConv2d(out_channels=48,
                                kernel_size=(1, 1),
                                params_dict=get(self.params_dict, 'branch5x5_1'),
                                dtype=self.dtype)(x, train)
        branch5x5 = BasicConv2d(out_channels=64,
                                kernel_size=(5, 5),
                                padding=((2, 2), (2, 2)),
                                params_dict=get(self.params_dict, 'branch5x5_2'),
                                dtype=self.dtype)(branch5x5, train)

        branch3x3dbl = BasicConv2d(out_channels=64,
                                   kernel_size=(1, 1),
                                   params_dict=get(self.params_dict, 'branch3x3dbl_1'),
                                   dtype=self.dtype)(x, train)
        branch3x3dbl = BasicConv2d(out_channels=96,
                                   kernel_size=(3, 3),
                                   padding=((1, 1), (1, 1)),
                                   params_dict=get(self.params_dict, 'branch3x3dbl_2'),
                                   dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl = BasicConv2d(out_channels=96,
                                   kernel_size=(3, 3),
                                   padding=((1, 1), (1, 1)),
                                   params_dict=get(self.params_dict, 'branch3x3dbl_3'),
                                   dtype=self.dtype)(branch3x3dbl, train)

        branch_pool = avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        branch_pool = BasicConv2d(out_channels=self.pool_features,
                                  kernel_size=(1, 1),
                                  params_dict=get(self.params_dict, 'branch_pool'),
                                  dtype=self.dtype)(branch_pool, train)
        
        output = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=-1)
        return output


class InceptionB(nn.Module):
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch3x3 = BasicConv2d(out_channels=384,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                params_dict=get(self.params_dict, 'branch3x3'),
                                dtype=self.dtype)(x, train)

        branch3x3dbl = BasicConv2d(out_channels=64,
                                   kernel_size=(1, 1),
                                   params_dict=get(self.params_dict, 'branch3x3dbl_1'),
                                   dtype=self.dtype)(x, train)
        branch3x3dbl = BasicConv2d(out_channels=96,
                                   kernel_size=(3, 3),
                                   padding=((1, 1), (1, 1)),
                                   params_dict=get(self.params_dict, 'branch3x3dbl_2'),
                                   dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl = BasicConv2d(out_channels=96,
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   params_dict=get(self.params_dict, 'branch3x3dbl_3'),
                                   dtype=self.dtype)(branch3x3dbl, train)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        output = jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=-1)
        return output


class InceptionC(nn.Module):
    channels_7x7: int
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch1x1 = BasicConv2d(out_channels=192,
                                kernel_size=(1, 1),
                                params_dict=get(self.params_dict, 'branch1x1'),
                                dtype=self.dtype)(x, train)
            
        branch7x7 = BasicConv2d(out_channels=self.channels_7x7,
                                kernel_size=(1, 1),
                                params_dict=get(self.params_dict, 'branch7x7_1'),
                                dtype=self.dtype)(x, train)
        branch7x7 = BasicConv2d(out_channels=self.channels_7x7,
                                kernel_size=(1, 7),
                                padding=((0, 0), (3, 3)),
                                params_dict=get(self.params_dict, 'branch7x7_2'),
                                dtype=self.dtype)(branch7x7, train)
        branch7x7 = BasicConv2d(out_channels=192,
                                kernel_size=(7, 1),
                                padding=((3, 3), (0, 0)),
                                params_dict=get(self.params_dict, 'branch7x7_3'),
                                dtype=self.dtype)(branch7x7, train)

        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7,
                                   kernel_size=(1, 1),
                                   params_dict=get(self.params_dict, 'branch7x7dbl_1'),
                                   dtype=self.dtype)(x, train)
        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7,
                                   kernel_size=(7, 1),
                                   padding=((3, 3), (0, 0)),
                                   params_dict=get(self.params_dict, 'branch7x7dbl_2'),
                                   dtype=self.dtype)(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7,
                                   kernel_size=(1, 7),
                                   padding=((0, 0), (3, 3)),
                                   params_dict=get(self.params_dict, 'branch7x7dbl_3'),
                                   dtype=self.dtype)(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7,
                                   kernel_size=(7, 1),
                                   padding=((3, 3), (0, 0)),
                                   params_dict=get(self.params_dict, 'branch7x7dbl_4'),
                                   dtype=self.dtype)(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7,
                                   kernel_size=(1, 7),
                                   padding=((0, 0), (3, 3)),
                                   params_dict=get(self.params_dict, 'branch7x7dbl_5'),
                                   dtype=self.dtype)(branch7x7dbl, train)

        branch_pool = avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        branch_pool = BasicConv2d(out_channels=192,
                                  kernel_size=(1, 1),
                                  params_dict=get(self.params_dict, 'branch_pool'),
                                  dtype=self.dtype)(branch_pool, train)
        
        output = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=-1)
        return output


class InceptionD(nn.Module):
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch3x3 = BasicConv2d(out_channels=192,
                                kernel_size=(1, 1),
                                params_dict=get(self.params_dict, 'branch3x3_1'),
                                dtype=self.dtype)(x, train)
        branch3x3 = BasicConv2d(out_channels=320,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                params_dict=get(self.params_dict, 'branch3x3_2'),
                                dtype=self.dtype)(branch3x3, train)
            
        branch7x7x3 = BasicConv2d(out_channels=192,
                                  kernel_size=(1, 1),
                                  params_dict=get(self.params_dict, 'branch7x7x3_1'),
                                  dtype=self.dtype)(x, train)
        branch7x7x3 = BasicConv2d(out_channels=192,
                                  kernel_size=(1, 7),
                                  padding=((0, 0), (3, 3)),
                                  params_dict=get(self.params_dict, 'branch7x7x3_2'),
                                  dtype=self.dtype)(branch7x7x3, train)
        branch7x7x3 = BasicConv2d(out_channels=192,
                                  kernel_size=(7, 1),
                                  padding=((3, 3), (0, 0)),
                                  params_dict=get(self.params_dict, 'branch7x7x3_3'),
                                  dtype=self.dtype)(branch7x7x3, train)
        branch7x7x3 = BasicConv2d(out_channels=192,
                                  kernel_size=(3, 3),
                                  strides=(2, 2),
                                  params_dict=get(self.params_dict, 'branch7x7x3_4'),
                                  dtype=self.dtype)(branch7x7x3, train)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        
        output = jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=-1)
        return output


class InceptionE(nn.Module):
    pooling: Callable
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch1x1 = BasicConv2d(out_channels=320,
                                kernel_size=(1, 1),
                                params_dict=get(self.params_dict, 'branch1x1'),
                                dtype=self.dtype)(x, train)
          
        branch3x3 = BasicConv2d(out_channels=384,
                                kernel_size=(1, 1),
                                params_dict=get(self.params_dict, 'branch3x3_1'),
                                dtype=self.dtype)(x, train)
        branch3x3_a = BasicConv2d(out_channels=384,
                                  kernel_size=(1, 3),
                                  padding=((0, 0), (1, 1)),
                                  params_dict=get(self.params_dict, 'branch3x3_2a'),
                                  dtype=self.dtype)(branch3x3, train)
        branch3x3_b = BasicConv2d(out_channels=384,
                                  kernel_size=(3, 1),
                                  padding=((1, 1), (0, 0)),
                                  params_dict=get(self.params_dict, 'branch3x3_2b'),
                                  dtype=self.dtype)(branch3x3, train)
        branch3x3 = jnp.concatenate((branch3x3_a, branch3x3_b), axis=-1)

        branch3x3dbl = BasicConv2d(out_channels=448,
                                   kernel_size=(1, 1),
                                   params_dict=get(self.params_dict, 'branch3x3dbl_1'),
                                   dtype=self.dtype)(x, train)
        branch3x3dbl = BasicConv2d(out_channels=384,
                                   kernel_size=(3, 3),
                                   padding=((1, 1), (1, 1)),
                                   params_dict=get(self.params_dict, 'branch3x3dbl_2'),
                                   dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl_a = BasicConv2d(out_channels=384,
                                     kernel_size=(1, 3),
                                     padding=((0, 0), (1, 1)),
                                     params_dict=get(self.params_dict, 'branch3x3dbl_3a'),
                                     dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl_b = BasicConv2d(out_channels=384,
                                     kernel_size=(3, 1),
                                     padding=((1, 1), (0, 0)),
                                     params_dict=get(self.params_dict, 'branch3x3dbl_3b'),
                                     dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl = jnp.concatenate((branch3x3dbl_a, branch3x3dbl_b), axis=-1)

        branch_pool = self.pooling(x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        branch_pool = BasicConv2d(out_channels=192,
                                  kernel_size=(1, 1),
                                  params_dict=get(self.params_dict, 'branch_pool'),
                                  dtype=self.dtype)(branch_pool, train)
        
        output = jnp.concatenate((branch1x1, branch3x3, branch3x3dbl, branch_pool), axis=-1)
        return output


class InceptionAux(nn.Module):
    num_classes: int
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        x = avg_pool(x, window_shape=(5, 5), strides=(3, 3))
        x = BasicConv2d(out_channels=128,
                        kernel_size=(1, 1),
                        params_dict=get(self.params_dict, 'conv0'),
                        dtype=self.dtype)(x, train)
        x = BasicConv2d(out_channels=768,
                        kernel_size=(5, 5),
                        params_dict=get(self.params_dict, 'conv1'),
                        dtype=self.dtype)(x, train)
        x = jnp.mean(x, axis=(1, 2))
        x = jnp.reshape(x, newshape=(x.shape[0], -1))
        x = Dense(features=self.num_classes,
                  params_dict=get(self.params_dict, 'fc'),
                  dtype=self.dtype)(x)
        return x
    
def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(nn.Module):
    """BatchNorm Module.
    Taken from: https://github.com/google/flax/blob/master/flax/linen/normalization.py
    Attributes:
        use_running_average: if True, the statistics stored in batch_stats
                             will be used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
               When the next layer is linear (also e.g. nn.relu), this can be disabled
               since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
               devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
                       representing subsets of devices to reduce over (default: None). For
                       example, `[[0, 1], [2, 3]]` would independently batch-normalize over
                       the examples on the first two and last two devices. See `jax.lax.psum`
                       for more details.
    """
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    mean_init: Callable[[Shape], Array] = lambda s: jnp.zeros(s, jnp.float32)
    var_init: Callable[[Shape], Array] = lambda s: jnp.ones(s, jnp.float32)
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Normalizes the input using batch statistics.
        
        NOTE:
        During initialization (when parameters are mutable) the running average
        of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don't need to match that of the actual input
        distribution and the reduction axis (set with `axis_name`) does not have
        to exist.
        Args:
            x: the input to be normalized.
            use_running_average: if true, the statistics stored in batch_stats
                                 will be used instead of computing the batch statistics on the input.
        Returns:
            Normalized inputs (the same shape as inputs).
        """
        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average)
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # see NOTE above on initialization behavior
        initializing = self.is_mutable_collection('params')

        ra_mean = self.variable('batch_stats', 'mean',
                                self.mean_init,
                                reduced_feature_shape)
        ra_var = self.variable('batch_stats', 'var',
                               self.var_init,
                               reduced_feature_shape)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(
                        concatenated_mean,
                        axis_name=self.axis_name,
                        axis_index_groups=self.axis_index_groups), 2)
            var = mean2 - lax.square(mean)

            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param('scale',
                               self.scale_init,
                               reduced_feature_shape).reshape(feature_shape)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param('bias',
                              self.bias_init,
                              reduced_feature_shape).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
    """
    Taken from: https://github.com/google/flax/blob/main/flax/linen/pooling.py

    Helper function to define pooling functions.
    Pooling functions are implemented using the ReduceWindow XLA op.
    NOTE: Be aware that pooling is not generally differentiable.
    That means providing a reduce_fn that is differentiable does not imply
    that pool is differentiable.
    Args:
      inputs: input data with dimensions (batch, window dims..., features).
      init: the initial value for the reduction
      reduce_fn: a reduce function of the form `(T, T) -> T`.
      window_shape: a shape tuple defining the window to reduce over.
      strides: a sequence of `n` integers, representing the inter-window
          strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
    Returns:
      The output of the reduction for each window slice.
    """
    strides = strides or (1,) * len(window_shape)
    assert len(window_shape) == len(strides), (
        f"len({window_shape}) == len({strides})")
    strides = (1,) + strides + (1,)
    dims = (1,) + window_shape + (1,)

    is_single_input = False
    if inputs.ndim == len(dims) - 1:
      # add singleton batch dimension because lax.reduce_window always
      # needs a batch dimension.
      inputs = inputs[None]
      is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    if not isinstance(padding, str):
      padding = tuple(map(tuple, padding))
      assert(len(padding) == len(window_shape)), (
        f"padding {padding} must specify pads for same number of dims as "
        f"window_shape {window_shape}")
      assert(all([len(x) == 2 for x in padding])), (
        f"each entry in padding {padding} must be length 2")
      padding = ((0,0),) + padding + ((0,0),)
    y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
      y = jnp.squeeze(y, axis=0)
    return y


def avg_pool(inputs, window_shape, strides=None, padding='VALID'):
    """
    Pools the input by taking the average over a window.

    In comparison to flax.linen.avg_pool, this pooling operation does not
    consider the padded zero's for the average computation.

    Args:
      inputs: input data with dimensions (batch, window dims..., features).
      window_shape: a shape tuple defining the window to reduce over.
      strides: a sequence of `n` integers, representing the inter-window
          strides (default: `(1, ..., 1)`).
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension (default: `'VALID'`).
    Returns:
      The average for each window slice.
    """
    assert inputs.ndim == 4
    assert len(window_shape) == 2

    y = pool(inputs, 0., jax.lax.add, window_shape, strides, padding)
    ones = jnp.ones(shape=(1, inputs.shape[1], inputs.shape[2], 1)).astype(inputs.dtype)
    counts = jax.lax.conv_general_dilated(ones,
                                          jnp.expand_dims(jnp.ones(window_shape).astype(inputs.dtype), axis=(-2, -1)),
                                          window_strides=(1, 1),
                                          padding=((1, 1), (1, 1)),
                                          dimension_numbers=nn.linear._conv_dimension_numbers(ones.shape),
                                          feature_group_count=1)
    y = y / counts 
    return y

if __name__ == '__main__':
    # Example usage
    import numpy as np

    #samples = np.load('cfg_2_UMD_B_4_fid.npy', allow_pickle=True).item()
    with gfile.GFile("gs://tpu_bucket_central_2/big_vision/final_release/UMD_B_4/sample_cfg_2fid_samples/samples_123863", 'rb') as f:
        samples = pickle.load(f)
        print(samples.shape)
    
    ref_arr_path = "/home/philippehansen_utexas_edu/64x64_fid_stats_validation_i1k.npy"
    fid_fn = create_fid_score_fn(1000, ref_arr_path)
    fid, inception_score = fid_fn(samples)
    print(f'FID: {fid}, Inception Score: {inception_score}')

    '''
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import numpy as np
    from jax.experimental import mesh_utils
    import big_vision.sharding as bv_sharding
    val_ds = tfds.load('imagenet2012', split='validation', as_supervised=True)

    raw_images = []
    raw_labels = []
    for image, label in tqdm(val_ds):  # Adjust the number taken based on memory capacity
        image = tf.image.resize(image, (64, 64))
        #image = tf.image.resize(image, (256, 256))
        raw_images.append(image)
        raw_labels.append(label)
    
    raw_images = np.array(raw_images).astype(np.uint8)
    rng = jax.random.PRNGKey(0)
    model = InceptionV3(pretrained=True)
    rng, rng_init = jax.random.split(rng)
    params_shape = jax.eval_shape(model.init, rng_init, jnp.ones((1, 299, 299, 3)))
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = jax.sharding.Mesh(devices, ("data",))
    repl_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    params_sharding = bv_sharding.infer_sharding(
        params_shape, mesh, axis_name="data",
        strategy="replicated",
        extra_strategy_args={},
        )
    params = jax.jit(model.init, out_shardings=params_sharding)(rng_init, jnp.ones((1, 299, 299, 3)))
    apply_fn = functools.partial(model.apply, train=False)

    @jax.jit
    def inception_forward(arr):
        x = arr / 255.0
        x = 2 * x - 1
        pred = apply_fn(params, jax.lax.stop_gradient(x))
        return pred
    
    mu, sigma, inception_score = compute_statistics(raw_images, inception_forward)
    print(f'Inception Score: {inception_score}')
    np.save('64x64_fid_stats_validation_i1k.npy', {'mu': mu, 'sigma': sigma, 'inception_score': inception_score})
    '''