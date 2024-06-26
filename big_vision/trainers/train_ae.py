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
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serial
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from tensorflow.io import gfile
from tqdm import trange
import copy
import pickle
from big_vision.evaluators.fid import create_fid_score_fn
from big_vision.gaussian_diffusion import create_gaussian_diffusion, q_sample, _predict_eps_from_xstart, ddim_sample_loop, _predict_xstart_from_eps

def main(argv, flags):
  del argv

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

  xid, wid = -1, -1
  fillin = lambda s: s
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)

  u.chrono.inform(measure=mw.measure, write_note=write_note)

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
  
  u.chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size)

  # Start input pipeline as early as possible.
  n_prefetch = config.get("prefetch_to_device", 1)
  if config.get("latent_diffusion", False):
    from big_vision.vae_utils import load_vae, load_latents_from_tfrecords
    vae_params, vae_encode, vae_decode = load_vae()
    def dummy(params):
      return params
    vae_params_shape = jax.eval_shape(dummy, vae_params)

    if config.get('preprocessed_latents_pattern', False):
      del train_ds
      train_ds = load_latents_from_tfrecords(config.preprocessed_latents_pattern, config.input.batch_size)
  
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
  
  rng = jax.random.PRNGKey(config.get("seed", 0))
  write_note("Inferring parameter shapes...")
  rng, rng_init = jax.random.split(rng)
  params_shape = jax.eval_shape(init, rng_init)

  write_note("Inferring optimizer state shapes...")
  if 'adafactor' in config.optax_name:
    tx, sched_fns = bv_optax.make(config, params_shape, sched_kw=dict(
        total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))
    opt_shape = jax.eval_shape(tx.init, params_shape)
  else:
    def get_weight_decay_mask(params):
      flattened_params = flax.traverse_util.flatten_dict(
          flax.core.frozen_dict.unfreeze(params)
      )
      def decay(key):
          return all([k not in model.no_decay_list for k in key])

      return flax.traverse_util.unflatten_dict(
          {key: decay(key) for key in flattened_params.keys()}
      )
    lr = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                            peak_value=config.peak_lr * batch_size / 256, #Scale by batch size
                                            warmup_steps=config.warmup_epochs * ntrain_img // batch_size,
                                            decay_steps=total_steps)
    betas = config.get("betas", (0.9, 0.95))
    optimizer = optax.adamw(
        learning_rate = lr, 
        weight_decay = config.wd,
        mask=get_weight_decay_mask,
        b1=betas[0],
        b2=betas[1],
        mu_dtype=config.get("mu_dtype", 'bfloat16'))

    tx = optax.chain(
        optax.clip_by_global_norm(config.get("clip_norm", 1.0)), 
        optimizer
    )
    opt_shape = jax.eval_shape(tx.init, params_shape)

  if jax.process_index() == 0:
    num_params = sum(np.prod(p.shape) for p in jax.tree_leaves(params_shape))
    mw.measure("num_params", num_params)

  write_note("Creating device mesh...")
  mesh = jax.sharding.Mesh(devices, ("data",))
  repl_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  write_note("Inferring shardings...")
  params_sharding = bv_sharding.infer_sharding(
      params_shape, mesh, axis_name="data",
      strategy=config.get("param_sharding", "replicated"),
      extra_strategy_args=config.get("param_sharding_args", {}))
  opt_sharding = bv_sharding.infer_sharding(
      opt_shape, mesh, axis_name="data",
      strategy=config.get("optim_sharding", "replicated"),
      extra_strategy_args=config.get("optim_sharding_args", {}))

  write_note("Transferring train_state to devices...")
  # RNG is always replicated
  rng_init = u.reshard(rng_init, repl_sharding)

  # Parameters and the optimizer are now global (distributed) jax arrays.
  params = jax.jit(init, out_shardings=params_sharding)(rng_init)
  opt = jax.jit(tx.init, out_shardings=opt_sharding)(params)

  rng, rng_loop = jax.random.split(rng, 2)
  rng_loop = u.reshard(rng_loop, repl_sharding)

  gd = create_gaussian_diffusion(beta_type=config.diff_schedule.beta_schedule,
                                 training_steps=config.diff_schedule.timesteps)
  gd_loop = u.reshard(gd, repl_sharding)
  del gd, rng

  # At this point we have everything we need to form a train state. It contains
  # all the parameters that are passed and updated by the main training step.
  
  train_state_sharding = {
      "params": params_sharding, "opt": opt_sharding, "rng": repl_sharding}
  train_state = {
    "params": params, "opt": opt, "rng": rng_loop}
  del params, rng_loop, opt # Delete to avoid memory leak or accidental reuse.

  if config.get("ema_decay", None):
    train_state_sharding = {
      **train_state_sharding,
      "ema_params": params_sharding
    }

  if config.get("finetune", False) and gfile.exists(f"{workdir}/checkpoint_finetune.bv-LAST"):
    save_ckpt_path = os.path.join(workdir, "checkpoint_finetune.bv")

  write_note(f"Checkpoint Location...{save_ckpt_path}")
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
        **train_state_sharding,
    }
    if config.get("finetune", False) and not gfile.exists(f"{workdir}/checkpoint_finetune.bv-LAST"):
      write_note("Finetuning model with no prior training...")
      assert config.get("num_classes", None) is not None, "Need to provide class labels for fine-tuning"
      init_label_emb, init_label_trunk = copy.deepcopy(train_state["params"]["label_emb"]), copy.deepcopy(train_state["params"]["label_trunk"])
      del shardings["opt"], shardings["params"]["label_emb"], shardings["params"]["label_trunk"], shardings["ema_params"]
    elif config.get("force_eval", False):
      del shardings["opt"]
      
    jax.tree_map(lambda x: x.delete(), train_state)
    del train_state 

    loaded = u.load_checkpoint_ts(
        resume_ckpt_path, tree=shardings, shardings=shardings)

    write_note("Checkpoint loaded successfully.")
    write_note(f"Loaded Keys: {list(shardings.keys())}")
    train_state = {key: loaded[key] for key in shardings.keys()}
    del loaded

    if config.get("finetune", False) and not gfile.exists(f"{workdir}/checkpoint_finetune.bv-LAST"):
      write_note("Finetuning model with no prior training...")
      train_state['params']['label_emb'], train_state['params']['label_trunk'] = copy.deepcopy(init_label_emb), copy.deepcopy(init_label_trunk)
      del init_label_emb, init_label_trunk
      opt = jax.jit(tx.init, out_shardings=opt_sharding)(train_state["params"]) #need to reinitialize optimizer
      train_state = {
          **train_state,
          "opt": opt
      }
      #Need to reinitialize sharding after deleting
      params_sharding = bv_sharding.infer_sharding(
      params_shape, mesh, axis_name="data",
      strategy=config.get("param_sharding", "replicated"),
      extra_strategy_args=config.get("param_sharding_args", {}))
      train_state_sharding = {
      "params": params_sharding, "opt": opt_sharding, "rng": repl_sharding}
      if config.get("ema_decay", None):
        train_state_sharding = {
          **train_state_sharding,
          "ema_params": params_sharding
        }
  
  train_state_sharding = {**train_state_sharding, "gd": repl_sharding}
  train_state = {**train_state, "gd": gd_loop}
  del gd_loop

  if config.get("latent_diffusion", False):
    vae_params_sharding = bv_sharding.infer_sharding(
      vae_params_shape, mesh, axis_name="data",
      strategy=config.get("vae_param_sharding", "replicated"),
      extra_strategy_args=config.get("vae_param_sharding_args", {}))
    
    train_state_sharding = {**train_state_sharding, "vae_params": vae_params_sharding}
    train_state = {**train_state, "vae_params": vae_params}
  
  with jax.spmd_mode('allow_all'):
    if config.get("ema_decay", None) and "ema_params" not in train_state:
      write_note("Creating EMA params...")
      train_state = {
          **train_state,
          "ema_params": copy.deepcopy(train_state["params"])
      }

  if config.get("finetune", False):
    save_ckpt_path = os.path.join(workdir, "checkpoint_finetune.bv")
    
  @functools.partial(
      jax.jit,
      donate_argnums=(0,),
      out_shardings=(train_state_sharding, repl_sharding))
  def update_fn(train_state, batch):
    """Update step."""
    images = batch["image"]
    rng = train_state["rng"]
    gd = train_state["gd"]
    B = images.shape[0]

    if config.get("latent_diffusion", False) and not config.get("use_preprocessed_latents", False):
      rng, vae_rng = jax.random.split(rng)
      images = vae_encode(train_state["vae_params"], vae_rng, images, scale=True)

    rng, rng_model, t_rng, noise_rng, mae_noise_rng, cfg_rng = jax.random.split(rng, 6)
    rng, rng_model_noise, mae_noise_rng_noise, cfg_rng_noise = jax.random.split(rng, 4)
    n_no_noise = int(B * config.no_noise_prob)
    n_noise = B - n_no_noise

    x_0_noise = images[:n_noise]
    x_0_no_noise = images[n_noise:]

    if config.get("use_labels", False):
      labels = batch["label"]
      labels_t = labels[:n_noise]
    else:
      labels_t = None

    batched_t = jax.random.randint(t_rng, shape=(n_noise, 1), dtype=jnp.int32, minval=0, maxval=len(train_state['gd']["betas"]))
    noise = jax.random.normal(noise_rng, x_0_noise.shape)
    x_t_noise = q_sample(gd=train_state['gd'], 
                   x_start=x_0_noise, 
                   t=batched_t, 
                   noise=noise)
    
    def loss_fn(params):
      if n_no_noise > 0:
        pred, out = model.apply(
          {"params": params}, x_0_no_noise,
          t=jnp.zeros((n_no_noise, 1), dtype=jnp.int32),
          train=True, mask=config.mask_ratio_no_noise,
          rngs={"dropout": rng_model,
                "cfg": cfg_rng,
                "mae_noise": mae_noise_rng})
        
        pred_x0 = pred[..., :config.diffusion_space[-1]]
        x0_se = (pred_x0 - x_0_no_noise)**2
        mae_loss = jnp.mean(x0_se * out['mask']) / jnp.mean(out['mask'])
      else:
        mae_loss = 0.0
      if n_noise > 0:
        pred, out = model.apply(
            {"params": params}, x_t_noise, 
            t=batched_t + 1, 
            y=labels_t,
            train=True, mask=config.mask_ratio,
            rngs={"dropout": rng_model_noise,
                  "cfg": cfg_rng_noise,
                  "mae_noise": mae_noise_rng_noise})
        
        pred_x0 = pred[..., :config.diffusion_space[-1]]
        pred_eps = pred[..., config.diffusion_space[-1]:]
        x0_se = (pred_x0 - x_0_noise)**2
        eps_se = (pred_eps - noise)**2
        if out['mask'] is not None:
          eps_loss = jnp.mean(eps_se * (1 - out['mask'])) / jnp.mean(1 - out['mask'])
          x0_loss = jnp.mean(x0_se * out['mask']) / jnp.mean(out['mask'])
          dit_loss = (eps_loss + x0_loss) / 2
        else:
          dit_loss = (jnp.mean(eps_se) + jnp.mean(x0_se)) / 2
      else:
        dit_loss = 0.0
      loss = dit_loss * (1 - n_no_noise / B) + mae_loss * (n_no_noise / B)
      return loss

    params, opt = train_state["params"], train_state["opt"]
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)
    measurements = {"training_loss": loss}
    ps = jax.tree_util.tree_leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.sum(p * p) for p in ps]))
    us = jax.tree_util.tree_leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.sum(u * u) for u in us]))

    if "ema_params" in train_state:
      ema_params = optax.incremental_update(params, train_state["ema_params"], config.ema_decay)
      new_train_state = {"params": params, "ema_params": ema_params, "opt": opt, "rng": rng, "gd": gd}
    else:
      new_train_state = {"params": params, "opt": opt, "rng": rng, "gd": gd}

    if config.get("latent_diffusion", False):
      new_train_state = {**new_train_state, "vae_params": train_state["vae_params"]}
    
    return new_train_state, measurements
  
  def predict_fn(train_state, batch):
      images = batch["image"]
      rng = train_state["rng"]
      if config.get("latent_diffusion", False):
        _, vae_rng = jax.random.split(rng)
        images = vae_encode(train_state["vae_params"], vae_rng, images, scale=True)
      _, out = model.apply({"params": train_state["params"]}, 
                                images, 
                                t=jnp.zeros((images.shape[0], 1), dtype=jnp.int32))
      return None, out
  
  def create_noised_pred_fn(t):
    def predict_fn(train_state, batch):
      images = batch["image"]
      rng = train_state["rng"]
      if config.get("latent_diffusion", False):
        rng, vae_rng = jax.random.split(rng)
        images = vae_encode(train_state["vae_params"], vae_rng, images, scale=True)
      B = images.shape[0]
      _, noise_rng = jax.random.split(rng)
      batched_t = jnp.ones((B, 1), dtype=jnp.int32) * t
      noise = jax.random.normal(noise_rng, images.shape)
      x_t = q_sample(gd=train_state["gd"], 
                    x_start=images, 
                    t=batched_t, 
                    noise=noise)
      _, out = model.apply({"params": train_state["params"]}, 
                                x_t, 
                                t=batched_t + 1)
      return None, out
    return predict_fn

  def eval_patch_fn(train_state, batch):
    B = batch["image"].shape[0]
    images = batch["image"]
    rng = train_state["rng"]
    if config.get("latent_diffusion", False):
      rng, vae_rng = jax.random.split(rng)
      images = vae_encode(train_state["vae_params"], vae_rng, images, scale=True)
    _, mae_noise_rng = jax.random.split(rng, 2)
    pred, out = model.apply({"params": train_state["params"]}, 
                                      images, 
                                      t=jnp.zeros((B, 1), dtype=jnp.int32), 
                                      mask=config.mask_ratio_no_noise, 
                                      rngs={"mae_noise": mae_noise_rng})
    
    pred_x0 = pred[..., :config.diffusion_space[-1]]
    if config.get("latent_diffusion", False):
      pred_x0 = vae_decode(train_state["vae_params"], pred_x0, scale=True)
      mask = jax.image.resize(out['mask'], shape=(B, config.size, config.size, 1), method='nearest')
    else:
      mask = out['mask']
    return pred_x0, mask
  
  def eval_loss_fn(train_state, batch):
    images = batch["image"]
    rng = train_state["rng"]
    if config.get("latent_diffusion", False):
      rng, vae_rng = jax.random.split(rng)
      images = vae_encode(train_state["vae_params"], vae_rng, images, scale=True)
    B = images.shape[0]
    if config.get("use_labels", False):
      labels = batch["label"]
    else:
      labels = None
    t_rng, noise_rng = jax.random.split(rng, 2)
    gd = train_state["gd"]
    batched_t = jax.random.randint(t_rng, shape=(B, 1), dtype = jnp.int32, minval=0, maxval=len(gd["betas"]))
    noise = jax.random.normal(noise_rng, images.shape)
    x_t = q_sample(gd=gd, 
                   x_start=images, 
                   t=batched_t, 
                   noise=noise)
    pred, _ = model.apply({"params": train_state["params"]}, 
                            x_t,
                            y=labels, 
                            t=batched_t + 1)
    
    pred_eps = pred[..., config.diffusion_space[-1]:]
    pred_x0 = pred[..., :config.diffusion_space[-1]]
    loss = (jnp.mean((pred_eps - noise)**2) + jnp.mean((pred_x0 - images)**2)) / 2
    pred_x0_eps = _predict_xstart_from_eps(gd, x_t, batched_t, pred_eps)
    if config.get("latent_diffusion", False):
      pred_x0 = vae_decode(train_state["vae_params"], pred_x0, scale=True)
      pred_x0_eps = vae_decode(train_state["vae_params"], pred_x0_eps, scale=True)
      x_t = vae_decode(train_state["vae_params"], x_t, scale=True)
    return loss, x_t, pred_x0, pred_x0_eps
  
  def create_apply_fn(train_state, eps_pred=True):
    def apply_fn(*, x_t, t, rng, y=None, cfg_scale=None):
        pred, _ = model.apply({"params": train_state["ema_params"]}, 
                                                  x_t, 
                                                  t=t + 1, 
                                                  y=y, 
                                                  cfg_scale=cfg_scale)
        if eps_pred:
          return pred[..., config.diffusion_space[-1]:]
        else:
          return _predict_eps_from_xstart(train_state['gd'], x_t, t, pred[..., :config.diffusion_space[-1]])
    return apply_fn

  def create_sample_fn(num_classes=None, manual_ys=None, cfg_scale=None, unnormalize=True, eps_pred=True):
    def eval_sample_fn(train_state, rng):
      return_dict = {}
      rng, key = jax.random.split(rng)
      fid_samples = []
      num_samples = config.num_samples_per_call
      if num_classes is not None and manual_ys is None:
        assert num_samples >= num_classes
        ys = jnp.arange(num_classes)
        if num_samples > num_classes:
          ys = jnp.concatenate([ys, jax.random.randint(key, shape=(num_samples - num_classes, ), dtype=jnp.int32, minval=0, maxval=num_classes)])
      elif manual_ys is not None:
        ys = jnp.array(manual_ys)
      else:
        ys = None
      input_shape = jnp.zeros((num_samples, config.diffusion_space[0], config.diffusion_space[1], config.diffusion_space[2]), dtype=jnp.float32)
      dic, _ = ddim_sample_loop(train_state["gd"],
                          create_apply_fn(train_state, eps_pred=eps_pred),
                          rng, 
                          input_shape,
                          ys=ys,
                          sampling_steps=config.diff_schedule.sampling_timesteps,
                          clip_denoised=config.diff_schedule.clip_denoised,
                          eta=config.diff_schedule.eta,
                          cfg_scale=cfg_scale)
      
      rng, key = jax.random.split(dic['rng'])
      select_indices = jax.random.randint(key, shape=(config.num_samples, ), dtype=jnp.int32, minval=0, maxval=num_samples)
      rng, key = jax.random.split(rng)      
      fid_samples = dic["sample"]

      if config.get("latent_diffusion", False):
        fid_samples = vae_decode(train_state["vae_params"], fid_samples, scale=True)

      if unnormalize:
        fid_samples = jnp.clip(fid_samples, -1, 1)
        fid_samples = fid_samples * 0.5 + 0.5 #brings -1, 1 to 0, 1
        fid_samples = jnp.clip(fid_samples * 255, 0, 255).astype(jnp.uint8)

      return_dict = {"fid_samples": fid_samples,
                      "wandbimage_examples": fid_samples[select_indices],
                      "ys": ys,
                      "rng": rng}
      return return_dict
    return eval_sample_fn

  eval_fns = {
      "predict": predict_fn,
      "noised_predict": create_noised_pred_fn(50), #Optimal noising level for representation
      'patch': eval_patch_fn,
      'loss': eval_loss_fn,
      'uncond_eps': create_sample_fn(),
      'cond_eps': create_sample_fn(num_classes=config.num_classes),
      'cfg_eps_1.0': create_sample_fn(cfg_scale=1.0, num_classes=config.num_classes),
      'cfg_eps_1.5': create_sample_fn(cfg_scale=1.5, num_classes=config.num_classes),
      'cfg_eps_2.0': create_sample_fn(cfg_scale=2.0, num_classes=config.num_classes),
      'cfg_eps_4.0': create_sample_fn(cfg_scale=4.0, num_classes=config.num_classes),
      'cfg_x0_2.0': create_sample_fn(cfg_scale=2.0, num_classes=config.num_classes, eps_pred=False),
      'cfg_x0_4.0': create_sample_fn(cfg_scale=4.0, num_classes=config.num_classes, eps_pred=False),
  }

  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config, eval_fns,
        lambda s: write_note(f"Init evaluator: {s}…\n{u.chrono.note}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
        devices,
    )
  
  if config.get("force_eval", False):
    first_step_device=1e8
  else:
    first_step_device = bv_optax.get_count(train_state["opt"], jittable=True)
  first_step = int(jax.device_get(first_step_device))
  u.chrono.inform(first_step=first_step)
  current_epochs = first_step * batch_size / ntrain_img
  if jax.process_index() == 0:
    wandb.log({"epochs": current_epochs}, step=first_step)

  if first_step in (total_steps, 0) or config.get("force_eval", False):
    write_note("Running initial or final evals...")
    mw.step_start(first_step)
    for (name, evaluator, _, prefix) in evaluators():
      if config.evals[name].get("skip_first") and first_step != total_steps:
        continue
      write_note(f"{name} evaluation...\n{u.chrono.note}")
      with u.chrono.log_timing(f"z/secs/eval/{name}"):
        with mesh, nn.logical_axis_rules([("act_batch", "data")]):
          eval_results = evaluator.run(train_state)
          for key, value in eval_results:
            if "fid_samples" not in key:
              value = u.gather_metrics(value)
            if "wandbimage" in key: #Logs each of the batches to get an idea of what the data looks like
              if jax.process_index() != 0:
                continue
              grid = u.make_grid(value, num_samples=config.num_samples)
              img = wandb.Image(grid)
              wandb.log({f"{prefix}{key}": img}, step=first_step)
              del grid, img
            elif 'batch' in key:
              if jax.process_index() != 0:
                continue
              grid = u.make_grid(value, num_samples=config.num_samples)
              img = wandb.Image(grid)
              wandb.log({f"{prefix}{key}": img}, step=first_step)
              del grid, img
            elif "fid_samples" in key:
              write_note(f"Calculating FID and Inception Score for {key}... with shape{value['samples'].shape}")
              fid_fn = create_fid_score_fn(1024, config.inception_reference_path)
              fid_score, inception_score = fid_fn(value['samples'])
              del fid_fn
              mw.measure(f"{prefix}{key}_fid_score", fid_score)
              mw.measure(f"{prefix}{key}_inception_score", inception_score)
              if jax.process_index() != 0:
                continue
              samples_bytes = pickle.dumps(value['samples'], protocol=5)
              samples_labels = pickle.dumps(value['ys'], protocol=5)
              with gfile.GFile(f'{workdir}/{name}{key}/samples_{first_step}', 'wb') as f:
                f.write(samples_bytes)
              with gfile.GFile(f'{workdir}/{name}{key}/labels_{first_step}', 'wb') as f:
                f.write(samples_labels)
              wandb.log({f"{prefix}{key}_fid_score": fid_score}, step=first_step)
              wandb.log({f"{prefix}{key}_inception_score": inception_score}, step=first_step)
            else:
              mw.measure(f"{prefix}{key}", value)
              if jax.process_index() != 0:
                continue
              wandb.log({f"{prefix}{key}": value}, step=first_step)
              

  prof = None  # Keeps track of start/stop of profiler state.

  if config.get("profile_flops", False):
    example_batch = next(iter(train_iter))
    compiled_train_step = update_fn.lower(train_state, example_batch).compile()
    flops = compiled_train_step.cost_analysis()[0]['flops']
    gflops = flops / 1e9
    write_note(f"Estimated GFLOPs per step: {gflops}")
    if jax.process_index() == 0:
      wandb.log({"gflops": gflops}, step=0)
  
  #Collect example training samples

  write_note("Starting training loop, compiling the first step...")
  for step, batch in zip(trange(first_step + 1, total_steps + 1), train_iter):
    mw.step_start(step)
    current_epochs = step * batch_size / ntrain_img

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      with mesh, nn.logical_axis_rules([("act_batch", "data")]):
        train_state, measurements = update_fn(train_state, batch)

    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, get_steps("log_training"))

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0) and jax.process_index() == 0):
      measurements['epochs'] = current_epochs
      measurements = jax.device_get(measurements)
      wandb.log(measurements, step=step)
      for name, value in measurements.items():
        mw.measure(name, value)
      u.chrono.tick(step)
      if not np.isfinite(measurements["training_loss"]):
        raise RuntimeError(f"The loss became nan or inf somewhere within steps "
                           f"[{step - get_steps('log_training')}, {step}]")

    # Checkpoint saving
    keep_ckpt_steps = get_steps("keep_ckpt", None) or total_steps
    if save_ckpt_path and config.get("save_ckpt", True) and (
        (keep := u.itstime(step, keep_ckpt_steps, total_steps, first=False))
        or u.itstime(step, get_steps("ckpt", None), total_steps, first=True)
    ):
      u.chrono.pause(wait_for=train_state)
      ckpt = {**train_state}
      with jax.transfer_guard("allow"):
        chrono_ckpt = multihost_utils.broadcast_one_to_all(u.chrono.save())
      chrono_shardings = jax.tree_map(lambda _: repl_sharding, chrono_ckpt)
      ckpt = ckpt | {"chrono": u.reshard(chrono_ckpt, chrono_shardings)}

      u.save_checkpoint_ts(ckpt_mngr, ckpt, save_ckpt_path, step, keep)
      u.chrono.resume()

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=False, last=True):
        u.chrono.pause(wait_for=train_state)
        u.chrono.tick(step)  # Record things like epoch number, core hours etc.
        write_note(f"{name} evaluation...\n{u.chrono.note}")
        with u.chrono.log_timing(f"z/secs/eval/{name}"):
          with mesh, nn.logical_axis_rules([("act_batch", "data")]):
            eval_results = evaluator.run(train_state)
            for key, value in eval_results:
              if "fid_samples" not in key:
                value = u.gather_metrics(value)
              if "wandbimage" in key:
                if jax.process_index() != 0:
                  continue
                grid = u.make_grid(value, num_samples=config.num_samples)
                img = wandb.Image(grid)
                wandb.log({f"{prefix}{key}": img}, step=step)
                del grid, img
              elif key == "batch":
                continue
              elif "fid_samples" in key:
                write_note(f"Calculating FID and Inception Score for {key}... with shape{value['samples'].shape}")
                fid_fn = create_fid_score_fn(1024, config.inception_reference_path)
                fid_score, inception_score = fid_fn(value['samples'])
                del fid_fn
                mw.measure(f"{prefix}{key}_fid_score", fid_score)
                mw.measure(f"{prefix}{key}_inception_score", inception_score)
                if jax.process_index() != 0:
                  continue
                samples_bytes = pickle.dumps(value['samples'], protocol=5)
                samples_labels = pickle.dumps(value['ys'], protocol=5)
                with gfile.GFile(f'{workdir}/{name}{key}/samples_{step}', 'wb') as f:
                  f.write(samples_bytes)
                with gfile.GFile(f'{workdir}/{name}{key}/labels_{step}', 'wb') as f:
                  f.write(samples_labels)
                wandb.log({f"{prefix}{key}_fid_score": fid_score}, step=step)
                wandb.log({f"{prefix}{key}_inception_score": inception_score}, step=step)
              else:
                mw.measure(f"{prefix}{key}", jax.device_get(value))
                if jax.process_index() != 0:
                  continue
                wandb.log({f"{prefix}{key}": value}, step=step)
        u.chrono.resume()
                
    mw.step_end()

  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)
  
  write_note(f"Done!\n{u.chrono.note}")

  pool.close()
  pool.join()
  mw.close()

  if ckpt_mngr:
    ckpt_mngr.wait_until_finished()

  # Make sure all hosts stay up until the end of main.
  u.sync()
  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)

if __name__ == "__main__":
  app.run(main)