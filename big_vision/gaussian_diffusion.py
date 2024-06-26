import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import functools

def mean_flat(arr):
    return arr.mean(axis=list(range(1, len(arr.shape))))

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)

    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        # This will raise an error during the JIT compilation if this branch is taken
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def create_gaussian_diffusion(beta_type='cosine', training_steps=1000):
    betas = get_beta_schedule(beta_type, num_diffusion_timesteps=training_steps)
    betas = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
    posterior_variance = betas * (1.0 - alphas_cumprod) / (1.0 - alphas_cumprod[-1])
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    ) if len(posterior_variance) > 1 else np.array([])
    posterior_mean_coef1 = (
        betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    return dict(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        alphas_cumprod_next=alphas_cumprod_next,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2)

def snr(*, gd, t):
  return (_extract_into_tensor(gd["sqrt_alphas_cumprod"], t, t.shape)**2 /
          _extract_into_tensor(gd["sqrt_one_minus_alphas_cumprod"], t, t.shape)**2)

def q_mean_variance(gd, x_start, t):
    """
    Get the distribution q(x_t | x_0).
    :param x_start: the [N x C x ...] tensor of noiseless inputs.
    :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    :return: A tuple (mean, variance, log_variance), all of x_start's shape.
    """
    mean = _extract_into_tensor(gd["sqrt_alphas_cumprod"], t, x_start.shape) * x_start
    variance = _extract_into_tensor(1.0 - gd["alphas_cumprod"], t, x_start.shape)
    log_variance = _extract_into_tensor(gd["log_one_minus_alphas_cumprod"], t, x_start.shape)
    return mean, variance, log_variance
    
def q_sample(*, gd, x_start, t, noise):
    """
    Diffuse the data for a given number of diffusion steps.
    In other words, sample from q(x_t | x_0).
    :param x_start: the initial data batch.
    :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    :param noise: if specified, the split-out normal noise.
    :return: A noisy version of x_start.
    """

    return (
        _extract_into_tensor(gd["sqrt_alphas_cumprod"], t, x_start.shape) * x_start
        + _extract_into_tensor(gd["sqrt_one_minus_alphas_cumprod"], t, x_start.shape) * noise
    )

def q_posterior_mean_variance(gd, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(gd["posterior_mean_coef1"], t, x_t.shape) * x_start
            + _extract_into_tensor(gd["posterior_mean_coef2"], t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(gd["posterior_variance"], t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            gd["posterior_log_variance_clipped"], t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

def _predict_xstart_from_eps(gd, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(gd["sqrt_recip_alphas_cumprod"], t, x_t.shape) * x_t
            - _extract_into_tensor(gd["sqrt_recipm1_alphas_cumprod"], t, x_t.shape) * eps
        )

def _predict_eps_from_xstart(gd, x_t, t, pred_xstart):
    return (
        _extract_into_tensor(gd["sqrt_recip_alphas_cumprod"], t, x_t.shape) * x_t - pred_xstart
    ) / _extract_into_tensor(gd["sqrt_recipm1_alphas_cumprod"], t, x_t.shape)

def p_mean_variance(gd, p_apply, x, t, rng, clip_denoised=False, denoised_fn=None, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    
    B, C = x.shape[:2]
    assert t.shape == (B, 1)
    model_output = p_apply(x_t=x, 
                           t=t, 
                           rng=rng,
                           **model_kwargs)
    if isinstance(model_output, tuple):
        model_output, extra = model_output
    else:
        extra = None

    def process_xstart(x):
        if denoised_fn is not None:
            x = denoised_fn(x)
        if clip_denoised:
            return x.clip(-1, 1)
        return x

    pred_xstart = process_xstart(
        _predict_xstart_from_eps(gd, x_t=x, t=t, eps=model_output)
    )
    model_mean, _, _ = q_posterior_mean_variance(gd, x_start=pred_xstart, x_t=x, t=t)

    return {
        "mean": model_mean,
        "pred_xstart": pred_xstart,
        "extra": extra,
    }

def ddim_sample(
    gd,
    p_apply,
    x,
    t,
    t_next,
    rng,
    clip_denoised=False,
    denoised_fn=None,
    model_kwargs=None,
    eta=1.0, #
):
    out = p_mean_variance(
        gd,
        p_apply,
        x,
        t,
        rng,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs,
    )

    eps = _predict_eps_from_xstart(gd, x, t, out["pred_xstart"])
    
    alpha_bar = _extract_into_tensor(gd["alphas_cumprod"], t, x.shape)
    if t_next is not None:
        alpha_bar_prev = _extract_into_tensor(gd["alphas_cumprod"], t_next, x.shape)
    else:
        alpha_bar_prev = _extract_into_tensor(gd["alphas_cumprod_prev"], t, x.shape)

    sigma = (
        eta
        * jnp.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        * jnp.sqrt(1 - alpha_bar / alpha_bar_prev)
    )
    # Equation 12.
    rng, noise_rng = jax.random.split(rng, 2)
    noise = jax.random.normal(noise_rng, x.shape)
    mean_pred = (
        out["pred_xstart"] * jnp.sqrt(alpha_bar_prev)
        + jnp.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    )
    t = t[:, : , None, None]
    sample = mean_pred + (t > 0) * sigma * noise
    return {"sample": sample, "pred_xstart": out["pred_xstart"], "rng": rng}

def ddim_sample_loop(
    gd,
    apply_fn,
    rng,
    shape,
    ys=None,
    clip_denoised=False,
    sampling_steps=250,
    denoised_fn=None,
    cfg_scale=None,
    eta=1.0,
):
    batch_size = shape.shape[0]
    if ys is not None:
        assert ys.shape[0] == batch_size, "ys must have the same batch size as shape"

    model_kwargs = dict(
        y=ys,
        cfg_scale=cfg_scale,
    )

    shape = shape.shape

    rng, noise_rng = jax.random.split(rng, 2)
    img = jax.random.normal(noise_rng, shape)
    
    reference_timesteps = jnp.arange(len(gd["betas"]) - 1, 0, step=-len(gd["betas"])//sampling_steps, dtype=jnp.int32)
    reference_timesteps = jnp.append(reference_timesteps, 0)

    def ddim_sample_step(carry, t):
        rng, img = carry

        t_curr = jnp.ones((img.shape[0], 1), dtype=jnp.int32) * reference_timesteps[t]
        t_next = jnp.ones((img.shape[0], 1), dtype=jnp.int32) * reference_timesteps[t + 1]

        out = ddim_sample(
            gd,
            apply_fn,
            img,
            t_curr,
            t_next,
            rng,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            eta=eta,
        )
        return (out["rng"], out["sample"]), None
    
    (rng, sample), _ = jax.lax.scan(ddim_sample_step, (rng, img), jnp.arange(0, sampling_steps))

    final_out = ddim_sample(
                gd,
                apply_fn,
                sample,
                jnp.zeros((sample.shape[0], 1), dtype=jnp.int32),
                None,
                rng,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
    
    return_dict = {
        "sample": final_out['pred_xstart'],
        "rng": final_out['rng'],
        "y": ys,
    }

    return return_dict, rng

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    selection = arr[timesteps]
    selection = selection.reshape(-1, *([1] * (len(broadcast_shape) - 1)))
    return selection