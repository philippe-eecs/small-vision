# pylint: disable=consider-using-from-import
# pylint: disable=logging-fstring-interpolation

import jax
import functools
import importlib
import multiprocessing.pool
import os
import big_vision.optax as bv_optax
from absl import app
from absl import logging
import big_vision.evaluators.common as eval_common
import big_vision.input_pipeline as input_pipeline
import big_vision.sharding as bv_sharding
import big_vision.utils as u
import flax
import flax.linen as nn
from jax.experimental import mesh_utils
from jax.experimental.array_serialization import serialization as array_serial
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import wandb
from tensorflow.io import gfile
from tqdm import trange
import copy
from big_vision.gaussian_diffusion import create_gaussian_diffusion, q_sample, snr

from functools import partial

class LinearCLS(nn.Module):
  num_classes: int = 1000

  @nn.compact
  def __call__(self, rep, train=True):
    norm = partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        use_scale=False,
        use_bias=False,
    )
    x = norm(name="bn")(rep)
    logits = nn.Dense(self.num_classes)(x)
    return logits


def main(argv, flags):
  del argv

  tf.config.set_visible_devices([], "GPU")
  devices = mesh_utils.create_device_mesh((jax.device_count(),))
  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir
  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  save_ckpt_path = None
  if workdir:  # Always create if requested, even if we may not write into it.
    gfile.makedirs(workdir)
    save_ckpt_path = os.path.join(workdir, "checkpoint.bv")

  pool = multiprocessing.pool.ThreadPool()
  for m in config.get("pp_modules", ["ops_general", "ops_image"]):
    importlib.import_module(f"big_vision.pp.{m}")

  fillin = lambda s: s
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  if jax.process_index() == 0:
    wandb.init(project=config.get("wandb_project", None),
               id=config.get("wandb_id", None),
               entity=config.get("entity", None),
               mode=config.get("wandb_mode", "online"), 
               config=config, dir=workdir)
    wandb.config.update(config)

  write_note("Initializing train dataset...")
  batch_size = config.input.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must "
                     f"be divisible by device number ({jax.device_count()})")
  info("Global batch size %d on %d hosts results in %d local batch size. With "
       "%d dev per host (%d dev total), that's a %d per-device batch size.",
       batch_size, jax.process_count(), batch_size // jax.process_count(),
       jax.local_device_count(), jax.device_count(),
       batch_size // jax.device_count())

  train_ds, ntrain_img = input_pipeline.training(config.input)

  total_steps = u.steps("total", config, ntrain_img, batch_size)
  def get_steps(name, default=ValueError, cfg=config):
    return u.steps(name, cfg, ntrain_img, batch_size, total_steps, default)

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  # Start input pipeline as early as possible.
  n_prefetch = config.get("prefetch_to_device", 1)
  if config.get("latent_diffusion", False):
    from big_vision.vae_utils import load_vae, load_latents_from_tfrecords
    vae_params, vae_encode, vae_decode = load_vae()
    def dummy(params):
      return params
    vae_params_shape = jax.eval_shape(dummy, vae_params)
  train_iter = input_pipeline.start_global(train_ds, devices, n_prefetch)

  write_note("Creating model...")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model = model_mod.Model(**config.get("model", {}))

  def init(rng):
    bs = batch_size // jax.device_count()
    no_image = jnp.zeros((bs,) + config.diffusion_space, jnp.float32)
    ts = jnp.zeros((bs, 1), jnp.int32)
    rngs = {"params": rng, "dropout": rng, "mae_noise": rng, "cfg": rng}
    params = flax.core.unfreeze(model.init(rngs, no_image, t=ts, train=True, mask=True))["params"]
    return params
  
  #rng = jax.random.PRNGKey(u.put_cpu(config.get("seed", 0)))
  rng = jax.random.PRNGKey(config.get("seed", 0))
  write_note("Inferring parameter shapes...")
  rng, rng_init = jax.random.split(rng)
  params_shape = jax.eval_shape(init, rng_init)

  if jax.process_index() == 0:
    num_params = sum(np.prod(p.shape) for p in jax.tree_leaves(params_shape))
    print(f"Total number of parameters: {num_params}")

  write_note("Creating device mesh...")
  mesh = jax.sharding.Mesh(devices, ("data",))
  repl_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  write_note("Inferring shardings...")
  params_sharding = bv_sharding.infer_sharding(
      params_shape, mesh, axis_name="data",
      strategy=config.get("param_sharding", "replicated"),
      extra_strategy_args=config.get("param_sharding_args", {}))

  write_note("Transferring train_state to devices...")
  # RNG is always replicated
  rng_init = u.reshard(rng_init, repl_sharding)

  # Parameters and the optimizer are now global (distributed) jax arrays.
  
  params = jax.jit(init, out_shardings=params_sharding)(rng_init)

  rng, rng_loop = jax.random.split(rng, 2)
  rng_loop = u.reshard(rng_loop, repl_sharding)

  # At this point we have everything we need to form a train state. It contains
  # all the parameters that are passed and updated by the main training step.
  train_state_sharding = {
      "params": params_sharding, "rng": repl_sharding}
  train_state = {
    "params": params, "rng": rng_loop}
  del params, rng_loop # Delete to avoid memory leak or accidental reuse.

  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(f"{save_ckpt_path}-LAST"):
    resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
    resume_ckpt_path = fillin(config.resume)

  ckpt_mngr = None
  if save_ckpt_path or resume_ckpt_path:
    ckpt_mngr = array_serial.GlobalAsyncCheckpointManager()

  if resume_ckpt_path:
    write_note(f"Resuming training from checkpoint {resume_ckpt_path}...")
    shardings = {
        'params': train_state_sharding['params'],
    }
    loaded = u.load_checkpoint_ts(
        resume_ckpt_path, tree=shardings, shardings=shardings)
    
    #train_state = {key: loaded[key] for key in ["params", "rng"]}
    model_params = loaded["params"]

    del loaded, train_state # We only need the model params
  elif config.get("model_init"):
    write_note(f"Initialize model from {config.model_init}...")
    train_state["params"] = model_mod.load(
        train_state["params"], config.model_init, config.get("model"),
        **config.get("model_load", {}))

    train_state["params"] = u.reshard(
        train_state["params"], train_state_sharding["params"])

  linear_model = LinearCLS(num_classes=config.num_classes)
  lr = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                            peak_value=0.1 * (config.input.batch_size / 256),
                                            warmup_steps=int(0.05 * config.total_epochs) * ntrain_img // batch_size,
                                            decay_steps=total_steps)
  
  tx = optax.lars(
                learning_rate=lr,
                weight_decay=config.wd,
                momentum=0.9,
            )
  
  def init(rng):
    bs = batch_size // jax.device_count()
    rep = jnp.zeros((bs, config.width), jnp.float32)
    rngs = {"params": rng}
    variables = flax.core.unfreeze(linear_model.init(rngs, rep, train=True))
    params, batch_stats = variables["params"], variables["batch_stats"]
    return params, batch_stats
  
  rng, rng_init = jax.random.split(rng)
  rng_init = u.reshard(rng_init, repl_sharding)
  rng, rng_loop = jax.random.split(rng, 2)
  rng_loop = u.reshard(rng_loop, repl_sharding)
  params_shape, batch_stats_shape = jax.eval_shape(init, rng_init)
  opt_shape = jax.eval_shape(tx.init, params_shape)
  write_note("Inferring shardings...")
  params_sharding = bv_sharding.infer_sharding(
      params_shape, mesh, axis_name="data",
      strategy=config.get("param_sharding", "replicated"),
      extra_strategy_args=config.get("param_sharding_args", {}))
  opt_sharding = bv_sharding.infer_sharding(
      opt_shape, mesh, axis_name="data",
      strategy=config.get("optim_sharding", "replicated"),
      extra_strategy_args=config.get("optim_sharding_args", {}))
  
  batch_stats_sharding = bv_sharding.infer_sharding(
        batch_stats_shape, mesh, axis_name="data",
        strategy="replicated",
        extra_strategy_args={})
  
  params, batch_stats = jax.jit(init, out_shardings=(params_sharding, batch_stats_sharding))(rng_init)
  opt = jax.jit(tx.init, out_shardings=opt_sharding)(params)

  train_state_sharding = {"params": params_sharding, "batch_stats": batch_stats_sharding, 
                          "model_params": train_state_sharding["params"], 
                          "opt": opt_sharding, "rng": repl_sharding}
  train_state = {
    "params": params, "model_params": model_params, "opt": opt, "rng": rng_loop, "batch_stats": batch_stats}

  del params, opt, rng_loop, batch_stats, rng # Delete to avoid memory leak or accidental reuse.

  if config.get("latent_diffusion", False):
    vae_params_sharding = bv_sharding.infer_sharding(
      vae_params_shape, mesh, axis_name="data",
      strategy=config.get("vae_param_sharding", "replicated"),
      extra_strategy_args=config.get("vae_param_sharding_args", {}))
    
    train_state_sharding = {**train_state_sharding, "vae_params": vae_params_sharding}
    train_state = {**train_state, "vae_params": vae_params}

  @functools.partial(
      jax.jit,
      donate_argnums=(0,),
      out_shardings=(train_state_sharding, repl_sharding))
  def update_fn(train_state, batch):
    """Update step."""

    images = batch["image"]
    B = images.shape[0]
    rng = train_state["rng"]
    if config.get("latent_diffusion", False) and not config.get("use_preprocessed_latents", False):
      rng, vae_rng = jax.random.split(rng)
      images = vae_encode(train_state["vae_params"], vae_rng, images, scale=True)

    rng, noise_rng = jax.random.split(rng, 2)

    if config.use_noised_pred:
      batched_t = jnp.ones((B, 1), jnp.int32) * 50
      noise = jax.random.normal(noise_rng, images.shape)
      images = q_sample(gd=train_state['gd'], 
                    x_start=images, 
                    t=batched_t, 
                    noise=noise)
    else:
      batched_t = jnp.zeros((B, 1), jnp.int32)

    batch_stats = train_state["batch_stats"]
    _, out = model.apply(
          {"params": train_state['model_params']}, 
          images, 
          t=batched_t,
          train=False)
    
    rep = jax.lax.stop_gradient(out['pre_logits'])
    
    def loss_fn(params):
      logits, new_state = linear_model.apply({"params": params, "batch_stats": batch_stats}, 
                                  rep,
                                  mutable=["batch_stats"], 
                                  train=True)
      loss = optax.softmax_cross_entropy(logits, batch["labels"]).mean()
      acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(batch["labels"], axis=-1))
      return loss, {"batch_stats": new_state["batch_stats"], "training_accuracy": acc}

    params, opt = train_state["params"], train_state["opt"]
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)
    measurements = {"training_loss": loss, "training_accuracy": aux["training_accuracy"]}

    new_train_state = {"params": params, "model_params": train_state['model_params'], 
            "opt": opt, "rng": rng, "batch_stats": aux['batch_stats']}

    if config.get("latent_diffusion", False):
      new_train_state = {**new_train_state, "vae_params": train_state["vae_params"]}
    return new_train_state, measurements
  
  def eval_logits_fn(train_state, batch):
    B = batch["image"].shape[0]
    images = batch["image"]
    rng = train_state["rng"]
    if config.get("latent_diffusion", False):
      _, vae_rng = jax.random.split(rng)
      images = vae_encode(train_state["vae_params"], vae_rng, images, scale=True)
    
    rng, noise_rng = jax.random.split(rng, 2)

    if config.use_noised_pred:
      batched_t = jnp.ones((B, 1), jnp.int32) * 50
      noise = jax.random.normal(noise_rng, images.shape)
      images = q_sample(gd=train_state['gd'], 
                    x_start=images, 
                    t=batched_t, 
                    noise=noise)
    else:
      batched_t = jnp.zeros((B, 1), jnp.int32)

    _, out = model.apply({"params": train_state["model_params"]},
                        images, 
                        t=batched_t,
                        train=False)
    logits = linear_model.apply({"params": train_state["params"], 
                                      "batch_stats": train_state['batch_stats']}, 
                                      out['pre_logits'], 
                                      train=False, 
                                      mutable=False)
    return logits, out
  
  eval_fns = {
      "predict": eval_logits_fn,
  }

  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config, eval_fns,
        lambda s: write_note(f"Init evaluator: {s}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
        devices,
    )
  
  first_step_device = bv_optax.get_count(train_state["opt"], jittable=True)
  first_step = int(jax.device_get(first_step_device))

  if first_step in (total_steps, 0):
    for (name, evaluator, _, prefix) in evaluators():
      if config.evals[name].get("skip_first") and first_step != total_steps:
        continue
      with mesh, nn.logical_axis_rules([("act_batch", "data")]):
        eval_results = evaluator.run(train_state)
        for key, value in eval_results:
          value = u.gather_metrics(value)
          if jax.process_index() != 0:
            continue
          if key == "batch" or "wandbimage" in key: #Logs each of the batches to get an idea of what the data looks like
            grid = u.make_grid(value, num_samples=config.num_samples, unnormalize=True)
            img = wandb.Image(grid)
            wandb.log({f"{prefix}{key}": img}, step=0)
            del grid, img
          else:
            wandb.log({f"{prefix}{key}": value}, step=0)

  prof = None  # Keeps track of start/stop of profiler state.
  if config.get("profile_flops", False):
    example_batch = next(iter(train_iter))
    compiled_train_step = update_fn.lower(train_state, example_batch).compile()
    flops = compiled_train_step.cost_analysis()[0]['flops']
    gflops = flops / 1e9
    write_note(f"Estimated GFLOPs per step: {gflops}")
    wandb.log({"gflops": gflops}, step=0)

  write_note("Starting training loop, compiling the first step...")
  for step, batch in zip(trange(first_step + 1, total_steps + 1), train_iter):

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      with mesh, nn.logical_axis_rules([("act_batch", "data")]):
        train_state, measurements = update_fn(train_state, batch)

    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, get_steps("log_training"))

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0) and jax.process_index() == 0):
      measurements = jax.device_get(measurements)
      wandb.log(measurements, step=step)

    # Checkpoint saving
    keep_ckpt_steps = get_steps("keep_ckpt", None) or total_steps
    if save_ckpt_path and config.get("save_ckpt", True) and (
        (keep := u.itstime(step, keep_ckpt_steps, total_steps, first=False))
        or u.itstime(step, get_steps("ckpt", None), total_steps, first=True)
    ):

      ckpt = {**train_state}
      u.save_checkpoint_ts(ckpt_mngr, ckpt, save_ckpt_path, step, keep)

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=False, last=True):
        with mesh, nn.logical_axis_rules([("act_batch", "data")]):
          eval_results = evaluator.run(train_state)
          for key, value in eval_results:
            value = u.gather_metrics(value)
            if jax.process_index() != 0:
              continue
            if "wandbimage" in key:
              grid = u.make_grid(value, num_samples=config.num_samples, unnormalize=True)
              img = wandb.Image(grid)
              wandb.log({f"{prefix}{key}": img}, step=step)
              del grid, img
            elif key == "batch":
              continue
            else:
              wandb.log({f"{prefix}{key}": value}, step=step)

  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)

  pool.close()
  pool.join()

  if ckpt_mngr:
    ckpt_mngr.wait_until_finished()

  # Make sure all hosts stay up until the end of main.
  u.sync()
  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)

if __name__ == "__main__":
  app.run(main)