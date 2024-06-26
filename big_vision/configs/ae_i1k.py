# pylint: disable=line-too-long
import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc

def get_config(arg=None):
  """Config for training."""
  arg = bvcc.parse_arg(arg, variant='B/4', scan=True, fsdp=False, batch_size=1024, use_labels=False,
                       mask_ratio=0.375, no_noise_prob=0.5, mask_ratio_no_noise=0.75, finetune=False,
                       lr=15e-5, wd=5e-2, beta2=0.95, size=64, adaln=True, epochs=800, area_min=80,
                       use_preprocessed_latents=False, latent_diffusion=False, wandb_mode='online', save_ckpt=True)
  config = mlc.ConfigDict()
  config.finetune = arg.finetune #Saves checkpoint named checkpoint_finetune.bv, primary purpose is finetune supervised model with labels
  config.size = arg.size
  config.latent_diffusion = arg.latent_diffusion
  if arg.latent_diffusion:
    assert arg.size == 256, "Latent Diffusion only supports 256x256 images"
    config.diffusion_space = (32, 32, 4)
    if arg.use_preprocessed_latents:
      config.use_preprocessed_latents = True
      config.preprocessed_latents_pattern = 'gs://tpu_bucket_central_2/imagenet_vae_preprocessed/stability_vae_train_i1k_latents_32x32x4_4_views/*.tfrecord'
  else:
    config.diffusion_space = (config.size, config.size, 3)
  config.resize = int(config.size * (256/246)) #Keep same ratio
  config.seed = 0
  config.wandb = True
  config.wandb_id = None #Should give random name
  config.wandb_project = "AE"
  config.entity = "diffusion_pi"
  config.total_epochs = arg.epochs
  if arg.use_labels:
    config.num_classes = 1000
    config.ema_decay = 0.0001 * (arg.batch_size / 256)
  else:
    config.num_classes = None
  config.num_samples = 36
  config.no_noise_prob = arg.no_noise_prob
  config.mask_ratio = arg.mask_ratio
  config.mask_ratio_no_noise = arg.mask_ratio_no_noise
  config.batch_size = arg.batch_size
  config.wandb_mode = arg.wandb_mode
  config.use_labels = arg.use_labels
  config.diff_schedule = dict()
  config.diff_schedule.eta = 1.0
  if arg.latent_diffusion:
    config.diff_schedule.beta_schedule = 'linear'
    config.diff_schedule.clip_denoised = False
  else:
    config.diff_schedule.beta_schedule = 'cosine'
    config.diff_schedule.clip_denoised = True
  config.diff_schedule.timesteps = 1000
  config.diff_schedule.sampling_timesteps = 125 #About same FID/IS as 250
  config.num_samples_per_call = 1024
  config.patch_size = int(arg.variant.split('/')[1])
  config.input = dict()
  config.input.data = dict(
      name='imagenet2012',
      split='train[:99%]',
  )
  config.input.batch_size = arg.batch_size
  config.input.cache_raw = True  # Needs up to 120GB of RAM!
  config.input.shuffle_buffer_size = 250_000

  pp_common = (
      '|value_range(-1, 1)'
      '|keep("image", "label")'
  )
  config.input.pp = f'decode_jpeg_and_inception_crop(size={config.size}, area_min={arg.area_min})|flip_lr' + pp_common.format(lbl='label')
  pp_eval = f'decode|resize_small({config.size})|central_crop({config.size})' + pp_common

  config.input.prefetch = 16
  config.prefetch_to_device = 8

  config.log_training_steps = 100
  config.ckpt_steps = 5000

  config.save_ckpt = arg.save_ckpt

  # Model section
  config.model_name = 'ae' #generic vit based auto-encoder
  config.model = dict(
      num_classes=config.num_classes,
      variant=arg.variant,
      scan=arg.scan,
      adaln=arg.adaln,
      channels=config.diffusion_space[-1],
      img_size=config.diffusion_space[0],
      remat_policy='nothing_saveable',
  )

  config.optax_name = 'adamw' #TODO: Test adafactor implementation in big vision
  config.clip_norm = 1.0
  config.warmup_epochs = int(0.05 * arg.epochs)
  config.peak_lr = arg.lr
  config.wd = arg.wd
  config.betas = (0.9, arg.beta2)

  def get_eval(split, dataset="imagenet2012"):
    return dict(
        type='diffusion_loss',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        log_steps=25000,
        pred='loss',
        cache_final=True,
    )
  
  def get_mae_eval(split, dataset="imagenet2012"):
    return dict(
        type='mae_reconstruction',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        log_steps=25000,
        pred='patch',
        cache_final=True,
    )

  def get_sample_eval(pred='samples'):
    return dict(
        type='diffusion_sampling',
        pred=pred,
        total_samples=10000,
        log_steps=25000,
    )
  
  config.evals = {}
  if config.no_noise_prob < 1.0:
    config.evals.val = get_eval('validation')

  if config.mask_ratio > 0.0 or config.no_noise_prob > 0.0:
    config.evals.mae_val = get_mae_eval('validation')

  if config.no_noise_prob > 0.0: #Very good proxy for the full LP task
    pred = 'predict'
  else:
    pred = 'noised_predict'
  config.evals.fewshot = get_fewshot_lsr(runlocal=False,
                                          target_resolution=config.size,
                                          resize_resolution=config.resize,
                                          datasets={'imagenet': ('imagenet2012', 'imagenet2012', 'train[:100000]', 'validation')},
                                          pred=pred)
  config.evals.fewshot.log_steps = 10000 
  
  if arg.fsdp: #VERY slow on PCI GPUs, not worth it just set scan=True for gradient checkpointing
    config.param_sharding = 'fully_sharded'
    config.optim_sharding = 'fully_sharded'
    config.model.scan = True

  if config.no_noise_prob < 1.0 and arg.use_labels:
    config.evals.sample_cond = get_sample_eval(pred='cond_eps')
    config.evals.sample_cfg_1_5 = get_sample_eval(pred='cfg_eps_2.0')
    config.evals.sample_cfg_4 = get_sample_eval(pred='cfg_eps_4.0')
    config.inception_reference_path = f'gs://tpu_bucket_central_2/fid_stats/{arg.size}x{arg.size}_fid_stats_validation_i1k.npy'
  return config