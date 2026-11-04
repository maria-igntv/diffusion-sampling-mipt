"""Library references used only for correctness checks, not custom integration.

Every reference receives the explicit query grid and the SAME VP initial state.
This avoids confusing solver error with differences in library default priors.
"""

from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler
import torch

from .parameterization import ve_to_vp, vp_to_ve
from .sampling import count_model_calls, sample_ddim
from .schedules import make_plan


@torch.inference_mode()
def sample_reference(model, base, initial_noise, config, trace=None):
    device = initial_noise.device
    plan = make_plan(config, base)
    if config.sampler == "ddim":
        if config.schedule != "native":
            raise ValueError("DDIM library trajectory verification uses native leading spacing")
        scheduler = DDIMScheduler.from_config(base.config, clip_sample=config.clip_sample)
        return sample_ddim(model, scheduler, initial_noise, config.num_steps, trace=trace)
    if config.sampler == "dpmpp_2m":
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=len(base.betas), trained_betas=base.betas.cpu().numpy(),
            algorithm_type="dpmsolver++", solver_order=2, solver_type="midpoint",
            lower_order_final=False, final_sigmas_type="zero",
            prediction_type="sample" if config.clip_sample else "epsilon",
        )
        scheduler.set_timesteps(timesteps=list(plan.timesteps), device=device)
        sample = initial_noise.clone()
        with count_model_calls(model) as counter:
            for index, timestep in enumerate(scheduler.timesteps):
                prediction = model(sample, timestep).sample
                if config.clip_sample:
                    sigma = scheduler.sigmas[index].to(device)
                    alpha = torch.rsqrt(1 + sigma.square())
                    beta = sigma * alpha
                    prediction = ((sample - beta * prediction) / alpha).clamp(-1, 1)
                sample = scheduler.step(prediction, timestep, sample, return_dict=False)[0]
                if trace is not None:
                    trace(index, plan.timesteps[index], sample)
        return sample, counter["calls"]
    if config.sampler == "heun":
        scheduler = HeunDiscreteScheduler(
            num_train_timesteps=len(base.betas), trained_betas=base.betas.cpu().numpy(),
            prediction_type="epsilon", clip_sample=config.clip_sample, timestep_spacing="leading",
        )
        scheduler.set_timesteps(timesteps=list(plan.timesteps), device=device)
        # Heun.set_timesteps moves sigmas back to CPU. In the tested PyTorch/MPS
        # runtime, mixing these CPU scalar tensors with MPS tensors can produce
        # incorrect corrector derivatives with clipping. Keep all coefficients
        # on the selected device; the library step formulas remain unchanged.
        scheduler.sigmas = scheduler.sigmas.to(device)
        sample = vp_to_ve(initial_noise.clone(), plan.sigmas[0].to(device))
        interval = 0
        with count_model_calls(model) as counter:
            for index, timestep in enumerate(scheduler.timesteps):
                epsilon = model(scheduler.scale_model_input(sample, timestep), timestep).sample
                sample = scheduler.step(epsilon, timestep, sample, return_dict=False)[0]
                if index % 2 == 1 or index == len(scheduler.timesteps) - 1:
                    if trace is not None:
                        trace(interval, plan.timesteps[interval], ve_to_vp(sample, plan.sigmas[interval + 1].to(device)))
                    interval += 1
        return sample, counter["calls"]
    raise ValueError(f"No library verification adapter for {config.sampler}")
