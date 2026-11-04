"""The baseline delegates DDIM mathematics to the pinned Diffusers version."""

from contextlib import contextmanager

import torch


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
def sample_ddim(model, scheduler, initial_noise, num_steps):
    scheduler.set_timesteps(num_steps, device=initial_noise.device)
    sample = initial_noise.clone() * scheduler.init_noise_sigma
    with count_model_calls(model) as counter:
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(sample, timestep)
            epsilon = model(model_input, timestep).sample
            sample = scheduler.step(epsilon, timestep, sample, eta=0.0, return_dict=False)[0]
    if counter["calls"] != num_steps:
        raise RuntimeError(f"Unexpected DDIM NFE: {counter['calls']} != {num_steps}")
    return sample, counter["calls"]
