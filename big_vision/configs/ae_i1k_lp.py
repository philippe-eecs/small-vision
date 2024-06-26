# pylint: disable=line-too-long
import big_vision.configs.common as bvcc
import ml_collections as mlc

def get_config(arg=None):
  """Config for training."""
  arg = bvcc.parse_arg(arg, variant='L/2', scan=True, fsdp=False, batch_size=4096,
                       size=256, adaln=True, epochs=100, area_min=80, width=1024, wd=5e-5,
                       use_noised_pred=False, latent_diffusion=True, wandb_mode='online', save_ckpt=False)
  config = mlc.ConfigDict()
  config.size = arg.size
  config.use_noised_pred = arg.use_noised_pred
  config.latent_diffusion = arg.latent_diffusion
  if arg.latent_diffusion:
    assert arg.size == 256, "Latent Diffusion only supports 256x256 images"
    config.diffusion_space = (32, 32, 4)
    config.use_preprocessed_latents = False
  else:
    config.diffusion_space = (config.size, config.size, 3)
  config.resize = int(config.size * (256/246)) #Keep same ratio
  config.seed = 0
  config.wandb = True
  config.wandb_id = None #Should give random name
  config.wandb_project = "Linear_AE"
  config.entity = "diffusion_pi"
  config.total_epochs = arg.epochs
  config.width = arg.width
  config.num_classes = 1000
  config.num_samples = 36
  config.batch_size = arg.batch_size
  config.wandb_mode = arg.wandb_mode
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
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )
  config.input.pp = f'decode_jpeg_and_inception_crop(size={config.size}, area_min={arg.area_min}, antialias=True)|flip_lr' + pp_common.format(lbl='label')
  pp_eval = f'decode|resize_small({config.size}, antialias=True)|central_crop({config.size})' + pp_common

  config.input.prefetch = 16
  config.prefetch_to_device = 8

  config.log_training_steps = 100
  config.ckpt_steps = 5000

  config.save_ckpt = arg.save_ckpt

  config.wd = arg.wd

  # Model section
  config.model_name = 'ae' #generic vit based auto-encoder
  config.model = dict(
      num_classes=None, #Should be self-supervised
      variant=arg.variant,
      scan=arg.scan,
      adaln=arg.adaln,
      channels=config.diffusion_space[-1],
      img_size=config.diffusion_space[0],
      remat_policy='nothing_saveable',
  )
  
  if arg.fsdp: #VERY slow on PCI GPUs, not worth it just set scan=True for gradient checkpointing
    config.param_sharding = 'fully_sharded'
    config.optim_sharding = 'fully_sharded'
    config.model.scan = True

  def get_eval(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        log_steps=100,  # Very fast O(seconds) so it's fine to run it often.
        cache_final=True,
    )
  config.evals = {}
  config.evals.train = get_eval('train[:2%]')
  config.evals.minival = get_eval('train[99%:]')
  config.evals.val = get_eval('validation')
  return config