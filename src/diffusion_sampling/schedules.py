"""Discrete training-compatible schedules with explicit nodes and NFE.

Standard common grids start at training timestep T-1, finish at timestep 0,
then append a clean endpoint (alpha=1, sigma=0) that is NEVER sent to the model.
One CPU training table supplies coefficients for all algorithms.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch


CUSTOM_SAMPLERS = {"ddim", "dpmpp_2m", "heun"}


def expected_nfe(sampler, num_steps):
    return 2 * num_steps - 1 if sampler == "heun" else num_steps


def points_for_budget(sampler, budget):
    if type(budget) is not int or budget <= 0:
        raise ValueError("NFE budget must be a positive integer")
    if sampler == "heun":
        if budget % 2 != 1:
            raise ValueError("Heun with one terminal Euler step requires an odd NFE budget")
        return (budget + 1) // 2
    if sampler not in CUSTOM_SAMPLERS:
        raise ValueError(f"Unknown custom sampler: {sampler}")
    return budget


@dataclass(frozen=True)
class SamplingPlan:
    sampler: str
    schedule: str
    timesteps: tuple
    alpha_bar: torch.Tensor
    sigmas: torch.Tensor
    clip_sample: bool
    adjusted_points: int = 0

    @property
    def nfe(self):
        return expected_nfe(self.sampler, len(self.timesteps))

    def to_dict(self):
        return {
            "sampler": self.sampler, "schedule": self.schedule,
            "timesteps": list(self.timesteps), "sigmas_with_clean_endpoint": self.sigmas.tolist(),
            "num_inference_points": len(self.timesteps), "expected_nfe": self.nfe,
            "clip_sample": self.clip_sample, "quantization_adjusted_points": self.adjusted_points,
            "denoiser_policy": ("native-ddim-original-epsilon" if self.schedule == "native" else "clipped-x0-consistent-ode") if self.clip_sample else "unclipped",
            "initialization": "x_vp=z; y_ve=sqrt(1+sigma_start**2)*z; same saved z for all methods",
            "terminal_endpoint": "sigma=0, alpha=1; no model call",
            "ddim_clipping": "original epsilon (Diffusers compatibility)" if self.schedule == "native" else "epsilon re-derived after x0 clipping (common denoiser)",
        }


def validate_timesteps(timesteps, count, training_steps):
    if len(timesteps) != count or any(type(t) is not int for t in timesteps):
        raise ValueError("Custom timesteps must contain exactly num_steps integers")
    if any(t < 0 or t >= training_steps for t in timesteps):
        raise ValueError("Timesteps lie outside the checkpoint training range")
    if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
        raise ValueError("Timesteps must be strictly decreasing, with no duplicates")
    return tuple(timesteps)


def _quantize_unique(target_sigmas, training_sigmas):
    # Nearest point in log-sigma, with order constraints leaving enough distinct
    # indices for all remaining points. Record how many collision fixes were made.
    log_training = np.log(training_sigmas)
    targets = np.log(target_sigmas)
    right = np.searchsorted(log_training, targets).clip(1, len(log_training) - 1)
    left = right - 1
    raw = np.where(np.abs(targets - log_training[left]) <= np.abs(targets - log_training[right]), left, right)
    chosen = []
    upper = len(training_sigmas) - 1
    for index, candidate in enumerate(raw):
        lower = len(raw) - index - 1
        value = min(max(int(candidate), lower), upper)
        chosen.append(value)
        upper = value - 1
    return chosen, sum(int(a) != b for a, b in zip(raw, chosen))


def make_plan(config, scheduler):
    alpha_bar_table = scheduler.alphas_cumprod.detach().cpu().float()
    total = len(alpha_bar_table)
    count = config.num_steps
    if count > total:
        raise ValueError("More inference points than training timesteps")
    if not torch.isfinite(alpha_bar_table).all() or not torch.all((alpha_bar_table > 0) & (alpha_bar_table < 1)):
        raise ValueError("Expected finite VP training alpha_bar strictly between zero and one")
    if not torch.all(alpha_bar_table[:-1] > alpha_bar_table[1:]):
        raise ValueError("Training alpha_bar must be strictly decreasing")
    training_sigmas = torch.sqrt((1 - alpha_bar_table) / alpha_bar_table)
    adjusted = 0
    if config.schedule == "native":
        if config.sampler == "diffusers_ddim":
            scheduler.set_timesteps(count)
            nodes = tuple(int(t) for t in scheduler.timesteps.tolist())
        else:
            if config.sampler != "ddim" or config.timestep_spacing != "leading":
                raise ValueError("Custom native DDIM compatibility supports leading spacing only")
            nodes = tuple(reversed([index * (total // count) for index in range(count)]))
    elif config.schedule == "custom":
        nodes = validate_timesteps(config.timesteps, count, total)
    elif count == 1:
        nodes = (total - 1,)
    elif config.schedule == "uniform_t":
        nodes = tuple(np.linspace(total - 1, 0, count).round().astype(int).tolist())
    else:
        sigma_min, sigma_max = float(training_sigmas[0]), float(training_sigmas[-1])
        ramp = np.linspace(0, 1, count)
        if config.schedule == "log_sigma":
            target = np.exp(math.log(sigma_max) + ramp * (math.log(sigma_min) - math.log(sigma_max)))
        elif config.schedule == "karras":
            target = (sigma_max ** (1 / config.rho) + ramp * (sigma_min ** (1 / config.rho) - sigma_max ** (1 / config.rho))) ** config.rho
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")
        nodes, adjusted = _quantize_unique(target, training_sigmas.numpy())
        nodes = tuple(nodes)
    nodes = validate_timesteps(list(nodes), count, total)
    selected = alpha_bar_table[list(nodes)]
    alphas = torch.cat([selected, torch.ones(1)])
    sigmas = torch.cat([training_sigmas[list(nodes)], torch.zeros(1)])
    return SamplingPlan(config.sampler, config.schedule, nodes, alphas, sigmas, config.clip_sample, adjusted)
