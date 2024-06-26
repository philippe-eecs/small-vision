# pylint: disable=line-too-long
import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc

def get_config(arg=None):
  """Config for training."""
  arg = bvcc.parse_arg(arg, variant='L/2', scan=True, batch_size=1024,
                       size=256, adaln=True, finetune=False, sample=True, probe=False,
                       noised_pred=False, sampling_timesteps=250,
                       latent_diffusion=True, wandb_mode='online', runlocal=False)
  config = mlc.ConfigDict()
  config.size = 64
  config.finetune = arg.finetune
  config.latent_diffusion = arg.latent_diffusion
  if arg.latent_diffusion:
    assert arg.size == 256, "Latent Diffusion only supports 256x256 images"
    config.diffusion_space = (32, 32, 4)
  else:
    config.diffusion_space = (config.size, config.size, 3)
  config.force_eval = True
  config.resize = int(config.size * (256/246)) #Keep same ratio
  config.seed = 0
  config.wandb = True
  config.wandb_id = None #Should give random name
  config.wandb_project = "Eval_AE"
  config.entity = "diffusion_pi"
  config.total_epochs = 0 #No Training just eval
  if arg.sample:
    config.num_classes = 1000
    config.ema_decay = 0.00025
  else:
    config.num_classes = None
  config.num_samples = 36
  config.num_samples_per_call = 1024
  config.batch_size = arg.batch_size
  config.wandb_mode = arg.wandb_mode

  #TODO: Depending on the model, this might need to be changed
  config.diff_schedule = dict()
  config.diff_schedule.eta = 1.0
  if arg.latent_diffusion:
    config.diff_schedule.beta_schedule = 'linear'
    config.diff_schedule.clip_denoised = False
  else:
    config.diff_schedule.beta_schedule = 'cosine'
    config.diff_schedule.clip_denoised = True
  config.diff_schedule.timesteps = 1000
  config.diff_schedule.sampling_timesteps = arg.sampling_timesteps
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
  config.input.pp = f'decode_jpeg_and_inception_crop(size={config.size}, area_min=80)|flip_lr' + pp_common.format(lbl='label')
  config.input.prefetch = 8
  config.prefetch_to_device = 4

  config.log_training_steps = 10
  config.ckpt_steps = 1000

  config.save_ckpt = False

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

  config.optax_name = 'adamw'
  config.clip_norm = 1.0
  config.warmup_epochs = 0
  config.peak_lr = 0.001
  config.wd = 0.0
  config.gradient_accum = 1
  config.betas = (0.9, 0.999)

  def get_sample_eval(pred='sample_uncond'):
    return dict(
        type='diffusion_sampling',
        pred=pred,
        log_steps=25000,
        total_samples=50000,
    )
  
  config.evals = {}
  if arg.probe:
    if arg.noised_pred:
      pred = 'noised_predict'
    else:
      pred = 'predict'
    config.evals.fewshot = get_fewshot_lsr(runlocal=False,
                                           target_resolution=config.size,
                                           resize_resolution=config.resize,
                                           datasets={'imagenet': ('imagenet2012', 'imagenet2012', 'train[:100000]', 'validation'),
                                                     'cifar100': ('cifar100', 'cifar100', 'train', 'test'),
                                                     'stl10': ('stl10', 'stl10', 'train', 'test'),
                                            'dtd': ('dtd', 'dtd', 'train', 'test'),
                                            'pets': ('oxford_iiit_pet', 'oxford_iiit_pet', 'train', 'test'),
                                            'flowers': ('oxford_flowers102', 'oxford_flowers102', 'train', 'test'),
                                            'food': ('food101', 'food101', 'train', 'validation'),
                                            'stanford_dogs': ('stanford_dogs', 'stanford_dogs', 'train', 'test'),
                                            'imagenet_v2': ('imagenet2012', 'imagenet_v2', 'train[:100000]', 'test'),
                                            'uc_merced': ('uc_merced', 'uc_merced', 'train[:1000]', 'train[1000:]')},
                                            pred=pred,
                                           shots=(10, 100))
    config.evals.fewshot.log_steps = 10000 

  if arg.sample:
    #config.evals.cond_eps = get_sample_eval(pred='cond_eps')
    config.evals.cfg_eps_1_5 = get_sample_eval(pred='cfg_eps_1.5')
    #config.evals.cfg_eps_2 = get_sample_eval(pred='cfg_eps_2.0')
    #config.evals.cfg_eps_4 = get_sample_eval(pred='cfg_eps_4.0')
    config.inception_reference_path = f'gs://tpu_bucket_central_2/fid_stats/{arg.size}x{arg.size}_fid_stats_validation_i1k.npy'
  return config