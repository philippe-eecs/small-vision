import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
import big_vision.sharding as bv_sharding
from diffusers.models import FlaxAutoencoderKL
import einops as eo
import functools
from tqdm import trange
import multiprocessing.pool
import importlib
import tensorflow as tf

latent_dim = 32 * 32 * 4

def parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([latent_dim], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    latents = example['image']
    latents = tf.reshape(latents, [32, 32, 4])  # Reshape to (32, 32, 4)
    label = example['label']
    return {'image': latents, 'label': label}

# Function to load and concatenate latents from multiple TFRecord files
def load_latents_from_tfrecords(file_pattern, batch_size):
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
    dataset = dataset.cache()
    dataset = dataset.repeat(None)
    dataset = dataset.shuffle(250000) # Shuffle buffer size of 250,000
    dataset = dataset.map(parse_example, num_parallel_calls=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(64)
    return dataset

def serialize_tensor(tensor, label):
    feature = {
        'image': tf.train.Feature(float_list=tf.train.FloatList(value=tensor.flatten())),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def load_vae():
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="vae", 
        revision="flax",
        dtype=jnp.float32,
    )
    # monkey-patch encode to use channels-last (it returns a FlaxDiagonalGaussianDistribution object, which is already
    # channels-last)
    vae.encode = lambda self, sample, *args, **kwargs: FlaxAutoencoderKL.encode(
        self, eo.rearrange(sample, "b h w c -> b c h w"), *args, **kwargs
    ).latent_dist

    # monkey-patch decode to use channels-last (it already accepts channels-last input)
    vae.decode = lambda self, latents, *args, **kwargs: eo.rearrange(
        FlaxAutoencoderKL.decode(self, latents, *args, **kwargs).sample,
        "b c h w -> b h w c",
    )

    # HuggingFace places vae_params committed onto the CPU -_-
    # this one took me awhile to figure out...
    vae_params = jax.device_get(vae_params)

    def vae_encode(vae_params, key, sample, scale=False):
        # handle the case where `sample` is multiple images stacked
        batch_size = sample.shape[0]
        sample = eo.rearrange(sample, "n h w (x c) -> (n x) h w c", c=3)
        latents = vae.apply({"params": vae_params}, sample, method=vae.encode).sample(
            key
        )
        latents = eo.rearrange(latents, "(n x) h w c -> n h w (x c)", n=batch_size)
        latents = jax.lax.cond(
            scale, lambda: latents * vae.config.scaling_factor, lambda: latents
        )
        return latents

    def vae_decode(vae_params, latents, scale=True):
        # handle the case where `latents` is multiple images stacked
        batch_size = latents.shape[0]
        latents = eo.rearrange(
            latents, "n h w (x c) -> (n x) h w c", c=vae.config.latent_channels
        )
        latents = jax.lax.cond(
            scale, lambda: latents / vae.config.scaling_factor, lambda: latents
        )
        sample = vae.apply({"params": vae_params}, latents, method=vae.decode)
        sample = eo.rearrange(sample, "(n x) h w c -> n h w (x c)", n=batch_size)
        return sample

    return vae_params, vae_encode, vae_decode

if __name__ == "__main__":
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import numpy as np
    from jax.experimental import mesh_utils
    import big_vision.input_pipeline as input_pipeline
    import big_vision.sharding as bv_sharding
    import ml_collections as mlc

    
    #
    num_epochs_encode = 4 # Save 4 views of the latents for each example

    devices = mesh_utils.create_device_mesh((jax.device_count(),))

    config = mlc.ConfigDict()

    config.input = dict()
    config.input.data = dict(
        name='imagenet2012',
        split='train[:99%]',
    )
    config.input.batch_size = 1024
    config.input.cache_raw = True  # Needs up to 120GB of RAM!
    config.input.shuffle_buffer_size = 250_000

    pp_common = (
        '|value_range(-1, 1)'
        '|keep("image", "label")'
    )
    config.input.pp = f'decode_jpeg_and_inception_crop(size={256}, area_min={80}, antialias=True)|flip_lr' + pp_common.format(lbl='label')
    #pp_eval = f'decode|resize_small({256}, antialias=True)|central_crop({256})' + pp_common

    config.input.prefetch = 8
    pool = multiprocessing.pool.ThreadPool()
    for m in config.get("pp_modules", ["ops_general", "ops_image"]):
        importlib.import_module(f"big_vision.pp.{m}")

    train_ds, ntrain_img = input_pipeline.training(config.input)
    train_iter = input_pipeline.start_global(train_ds, devices, 4)

    vae_params, vae_encode, vae_decode = load_vae()
    def dummy(params):
      return params
    vae_params_shape = jax.eval_shape(dummy, vae_params)
    mesh = jax.sharding.Mesh(devices, ("data",))
    repl_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    vae_params_sharding = bv_sharding.infer_sharding(
      vae_params_shape, mesh, axis_name="data",
      strategy="replicated",
      extra_strategy_args={})
    
    train_state_sharding = {"vae_params": vae_params_sharding, "rng": repl_sharding}
    train_state = {"vae_params": vae_params, "rng": jax.random.PRNGKey(0)}

    encoded_latents = []
    total_steps = int(ntrain_img // config.input.batch_size * num_epochs_encode)

    @functools.partial(
      jax.jit,
      donate_argnums=(0,),
      out_shardings=(train_state_sharding, repl_sharding))
    def update_fn(train_state, batch):
        """Update step."""
        images = batch["image"]
        rng = train_state["rng"]

        rng, vae_rng = jax.random.split(rng)
        latents = vae_encode(train_state["vae_params"], vae_rng, images, scale=True)
        new_train_state = {"vae_params": train_state["vae_params"], "rng": rng}
        
        return new_train_state, latents
    
    base_filename = "gs://tpu_bucket_central_2/imagenet_vae_preprocessed/stability_vae_train_i1k_latents_32x32x4_4_views"
    chunk_idx = 0
    writer = tf.io.TFRecordWriter(f"{base_filename}/{chunk_idx}.tfrecord")
    samples_per_chunk = 10
    print("Writing to", base_filename)
    print("Total steps:", total_steps)
    for step, batch in zip(trange(0 + 1, total_steps + 1), train_iter):
        train_state, latents = update_fn(train_state, batch)
        latents = jax.device_get(latents)
        labels = jax.device_get(batch['label'])

        for latent, label in zip(latents, labels):
            serialized_latent = serialize_tensor(latent, label)
            writer.write(serialized_latent)
        
        if step % samples_per_chunk == 0:
            print(f"Chunk {chunk_idx} written")
            chunk_idx += 1
            writer.close()
            writer = tf.io.TFRecordWriter(f"{base_filename}/{chunk_idx}.tfrecord")
    
    writer.close()
    pool.close()
    pool.join()
    
