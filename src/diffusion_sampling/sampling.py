"""A library baseline and independently implemented deterministic integrators.

Custom algorithms do not call Diffusers scheduler.step. The scheduler supplies
only the checkpoint's training noise coefficients. All training queries are
integer timesteps; no hidden interpolation or extra network evaluation is used.
"""

from contextlib import contextmanager

import torch

from .parameterization import prediction_to_x0, ve_epsilon_to_x0, ve_to_vp, vp_to_ve
from .schedules import make_plan


@contextmanager
def count_model_calls(model):
    counter = {"calls": 0}

    def increment(_module, _args):
        counter["calls"] += 1

    handle = model.register_forward_pre_hook(increment)
    try:
        yield counter
    finally:
        handle.remove()


@torch.inference_mode()
def sample_ddim(model, scheduler, initial_noise, num_steps, trace=None):
    scheduler.set_timesteps(num_steps, device=initial_noise.device)
    sample = initial_noise.clone() * scheduler.init_noise_sigma
    with count_model_calls(model) as counter:
        for index, timestep in enumerate(scheduler.timesteps):
            model_input = scheduler.scale_model_input(sample, timestep)
            epsilon = model(model_input, timestep).sample
            sample = scheduler.step(epsilon, timestep, sample, eta=0.0, return_dict=False)[0]
            if trace is not None:
                trace(index, int(timestep), sample)
    if counter["calls"] != num_steps:
        raise RuntimeError(f"Unexpected DDIM NFE: {counter['calls']} != {num_steps}")
    return sample, counter["calls"]


def ddim_step(sample, epsilon, alpha_t, beta_t, alpha_next, beta_next,
              clip_sample=False, recompute_epsilon=False):
    denoised = prediction_to_x0(sample, epsilon, alpha_t, beta_t)
    if clip_sample:
        denoised = denoised.clamp(-1, 1)
        if recompute_epsilon:
            epsilon = (sample - alpha_t * denoised) / beta_t
    return alpha_next * denoised + beta_next * epsilon


def dpmpp_2m_step(sample, denoised, alpha_t, beta_t, alpha_next, beta_next,
                  previous_denoised=None, previous_lambda=None, terminal=False):
    """Deterministic DPM-Solver++ 2M, midpoint form, in VP coordinates.

    A first-order startup supplies history. The clean terminal step has the
    analytic limit x_next=x0_hat and never evaluates log(0) or a new network call.
    """
    if terminal:
        return denoised
    lambda_t = alpha_t.log() - beta_t.log()
    lambda_next = alpha_next.log() - beta_next.log()
    h = lambda_next - lambda_t
    corrected = denoised
    if previous_denoised is not None:
        h_previous = lambda_t - previous_lambda
        corrected = denoised + 0.5 * (h / h_previous) * (denoised - previous_denoised)
    return (beta_next / beta_t) * sample - alpha_next * torch.expm1(-h) * corrected


@torch.inference_mode()
def sample_custom(model, scheduler, initial_noise, config, trace=None):
    plan = make_plan(config, scheduler)
    device, dtype = initial_noise.device, initial_noise.dtype
    if dtype != torch.float32:
        raise ValueError("Custom integrators are verified in float32 only")
    times = torch.tensor(plan.timesteps, device=device, dtype=torch.long)
    sigmas = plan.sigmas.to(device)
    alpha_bar = plan.alpha_bar.to(device)
    if config.sampler == "dpmpp_2m":
        # Match the coefficients reconstructed from VE sigma in the library
        # reference. Mathematically these equal sqrt(alpha_bar) and sqrt(1-a_bar).
        alphas = torch.rsqrt(1 + sigmas.square())
        betas = sigmas * alphas
    else:
        alphas, betas = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
    sample = initial_noise.clone()
    if config.sampler == "heun":
        sample = vp_to_ve(sample, sigmas[0])
    previous_denoised = previous_lambda = None

    def denoise_ve(value, sigma, timestep):
        epsilon = model(ve_to_vp(value, sigma), timestep).sample
        denoised = ve_epsilon_to_x0(value, epsilon, sigma)
        return denoised.clamp(-1, 1) if config.clip_sample else denoised

    with count_model_calls(model) as counter:
        for index, timestep in enumerate(times):
            terminal = index == len(plan.timesteps) - 1
            if config.sampler == "ddim":
                epsilon = model(sample, timestep).sample
                sample = ddim_step(
                    sample, epsilon, alphas[index], betas[index], alphas[index + 1], betas[index + 1],
                    clip_sample=config.clip_sample, recompute_epsilon=config.schedule != "native",
                )
            elif config.sampler == "dpmpp_2m":
                epsilon = model(sample, timestep).sample
                denoised = prediction_to_x0(sample, epsilon, alphas[index], betas[index])
                if config.clip_sample:
                    denoised = denoised.clamp(-1, 1)
                sample = dpmpp_2m_step(
                    sample, denoised, alphas[index], betas[index], alphas[index + 1], betas[index + 1],
                    previous_denoised=previous_denoised, previous_lambda=previous_lambda, terminal=terminal,
                )
                previous_denoised = denoised
                previous_lambda = alphas[index].log() - betas[index].log()
            elif config.sampler == "heun":
                sigma, sigma_next = sigmas[index], sigmas[index + 1]
                denoised = denoise_ve(sample, sigma, timestep)
                derivative = (sample - denoised) / sigma
                delta = sigma_next - sigma
                trial = sample + delta * derivative
                if terminal:
                    # Analytic Euler endpoint is denoised; avoid cancellation in
                    # sample - sigma*((sample-denoised)/sigma).
                    sample = denoised
                else:
                    denoised_next = denoise_ve(trial, sigma_next, times[index + 1])
                    derivative_next = (trial - denoised_next) / sigma_next
                    sample = sample + 0.5 * delta * (derivative + derivative_next)
            else:
                raise ValueError(f"Unknown custom sampler: {config.sampler}")
            if trace is not None:
                traced = ve_to_vp(sample, sigmas[index + 1]) if config.sampler == "heun" else sample
                trace(index, plan.timesteps[index], traced)
    if counter["calls"] != plan.nfe:
        raise RuntimeError(f"Unexpected NFE: {counter['calls']} != {plan.nfe}")
    return sample, counter["calls"]


def sample_configured(model, scheduler, initial_noise, config, trace=None):
    if config.sampler == "diffusers_ddim":
        return sample_ddim(model, scheduler, initial_noise, config.num_steps, trace=trace)
    return sample_custom(model, scheduler, initial_noise, config, trace=trace)
