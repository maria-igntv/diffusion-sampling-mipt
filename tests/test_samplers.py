"""Independent math checks and step-by-step comparisons with Diffusers 0.35.2."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler
import numpy as np
import torch

from diffusion_sampling.config import RunConfig
from diffusion_sampling.parameterization import prediction_to_epsilon, prediction_to_x0, ve_to_vp, vp_to_ve
from diffusion_sampling.references import sample_reference
from diffusion_sampling.sampling import sample_configured, sample_ddim
from diffusion_sampling.schedules import make_plan, points_for_budget


class SmoothDenoiser(torch.nn.Module):
    def forward(self, sample, timestep):
        time = torch.as_tensor(timestep, dtype=sample.dtype, device=sample.device) / 1000
        return SimpleNamespace(sample=0.13 * sample.tanh() + 0.02 * time.square())


class OracleDenoiser(torch.nn.Module):
    """Known clean image makes VP/VE scaling errors observable, not just self-consistent."""
    def __init__(self, alpha_bar, clean):
        super().__init__()
        self.register_buffer("alpha_bar", alpha_bar)
        self.clean = clean

    def forward(self, sample, timestep):
        a = self.alpha_bar[timestep].sqrt()
        b = (1 - self.alpha_bar[timestep]).sqrt()
        return SimpleNamespace(sample=(sample - a * self.clean) / b)


def reference_dpm(model, base, noise, nodes, solver_order=2, clip_sample=False):
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=len(base.betas), trained_betas=base.betas.numpy(),
        algorithm_type="dpmsolver++", solver_order=solver_order, solver_type="midpoint",
        lower_order_final=False, final_sigmas_type="zero", prediction_type="sample" if clip_sample else "epsilon",
    )
    scheduler.set_timesteps(timesteps=list(nodes))
    sample = noise.clone()
    trajectory = []
    with torch.inference_mode():
        for index, time in enumerate(scheduler.timesteps):
            epsilon = model(sample, time).sample
            if clip_sample:
                sigma = scheduler.sigmas[index]
                alpha = torch.rsqrt(1 + sigma.square())
                epsilon = ((sample - sigma * alpha * epsilon) / alpha).clamp(-1, 1)
            sample = scheduler.step(epsilon, time, sample, return_dict=False)[0]
            trajectory.append(sample.clone())
    return sample, trajectory


def reference_heun(model, base, noise, plan):
    scheduler = HeunDiscreteScheduler(
        num_train_timesteps=len(base.betas), trained_betas=base.betas.numpy(),
        prediction_type="epsilon", clip_sample=plan.clip_sample, timestep_spacing="leading",
    )
    scheduler.set_timesteps(timesteps=list(plan.timesteps))
    # Force the SAME VP initial state, rather than compare differing priors.
    sample = vp_to_ve(noise.clone(), scheduler.sigmas[0])
    trajectory = []
    with torch.inference_mode():
        for index, time in enumerate(scheduler.timesteps):
            epsilon = model(scheduler.scale_model_input(sample, time), time).sample
            sample = scheduler.step(epsilon, time, sample, return_dict=False)[0]
            if index % 2 == 1 or index == len(scheduler.timesteps) - 1:
                trajectory.append(ve_to_vp(sample, plan.sigmas[len(trajectory) + 1]).clone())
    return sample, trajectory


class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.base = DDIMScheduler()
        self.noise = torch.randn((2, 3, 8, 8), generator=torch.Generator().manual_seed(23))
        self.model = SmoothDenoiser().eval()

    def config(self, **updates):
        return replace(RunConfig(sampler="ddim", schedule="uniform_t", num_steps=7,
                                 clip_sample=False, purpose="pilot", device="cpu"), **updates)

    def test_prediction_parameterizations_recover_known_clean_image_and_noise(self):
        clean = self.noise / 3
        epsilon = self.noise.flip(-1)
        alpha, beta = torch.tensor(0.6), torch.tensor(0.8)
        sample = alpha * clean + beta * epsilon
        velocity = alpha * epsilon - beta * clean
        for kind, prediction in [("epsilon", epsilon), ("sample", clean), ("v_prediction", velocity)]:
            with self.subTest(kind=kind):
                torch.testing.assert_close(prediction_to_x0(sample, prediction, alpha, beta, kind), clean)
                torch.testing.assert_close(prediction_to_epsilon(sample, prediction, alpha, beta, kind), epsilon)
        sigma = beta / alpha
        ve = vp_to_ve(sample, sigma)
        torch.testing.assert_close(ve, clean + sigma * epsilon)
        torch.testing.assert_close(ve_to_vp(ve, sigma), sample)

    def test_standard_schedules_are_strict_and_share_endpoints(self):
        for schedule in ["uniform_t", "log_sigma", "karras"]:
            for count in [1, 2, 19, 39, 79, 1000]:
                with self.subTest(schedule=schedule, count=count):
                    plan = make_plan(self.config(schedule=schedule, num_steps=count), self.base)
                    self.assertEqual(len(set(plan.timesteps)), count)
                    self.assertEqual(plan.timesteps[0], 999)
                    if count > 1:
                        self.assertEqual(plan.timesteps[-1], 0)
                    self.assertTrue(torch.all(plan.sigmas[:-1] > plan.sigmas[1:]))
                    self.assertEqual(float(plan.sigmas[-1]), 0)

    def test_invalid_custom_schedule_and_even_heun_budget_are_rejected(self):
        for nodes in [[999, 10, 10], [0, 10, 999], [1000, 10, 0], [999, 0.5, 0]]:
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                self.config(schedule="custom", timesteps=nodes, num_steps=3)
        with self.assertRaisesRegex(ValueError, "odd NFE"):
            points_for_budget("heun", 20)

    def test_native_custom_ddim_matches_every_library_step_with_and_without_clipping(self):
        for count in [1, 7, 20, 100]:
            for clipping in [False, True]:
                with self.subTest(count=count, clipping=clipping):
                    config = self.config(schedule="native", num_steps=count, clip_sample=clipping)
                    custom_trace, reference_trace = [], []
                    actual, nfe = sample_configured(self.model, self.base, self.noise, config,
                                                    trace=lambda i, t, x: custom_trace.append(x.clone()))
                    reference, _ = sample_ddim(self.model, DDIMScheduler(clip_sample=clipping), self.noise, count,
                                               trace=lambda i, t, x: reference_trace.append(x.clone()))
                    self.assertEqual(nfe, count)
                    for left, right in zip(custom_trace, reference_trace):
                        torch.testing.assert_close(left, right, rtol=3e-6, atol=3e-6)
                    torch.testing.assert_close(actual, reference, rtol=3e-6, atol=3e-6)

    def test_dpmpp_2m_matches_library_on_nonuniform_grid_step_by_step(self):
        for schedule in ["uniform_t", "log_sigma", "karras", "custom"]:
            with self.subTest(schedule=schedule):
                config = self.config(sampler="dpmpp_2m", schedule=schedule,
                                     timesteps=[999, 900, 730, 300, 80, 5, 0] if schedule == "custom" else None)
                plan = make_plan(config, self.base)
                actual_trace = []
                actual, nfe = sample_configured(self.model, self.base, self.noise, config,
                                                trace=lambda i, t, x: actual_trace.append(x.clone()))
                reference, reference_trace = reference_dpm(self.model, self.base, self.noise, plan.timesteps)
                for left, right in zip(actual_trace, reference_trace):
                    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-4)
                torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-4)
                self.assertEqual(nfe, config.num_steps)
                first_order, _ = reference_dpm(self.model, self.base, self.noise, plan.timesteps, solver_order=1)
                self.assertGreater(float((actual - first_order).abs().max()), 1e-3)

    def test_heun_matches_library_with_same_initial_state_and_clipping(self):
        for count in [1, 2, 10]:
            for clipping in [False, True]:
                with self.subTest(count=count, clipping=clipping):
                    config = self.config(sampler="heun", schedule="karras", num_steps=count, clip_sample=clipping)
                    plan = make_plan(config, self.base)
                    actual_trace = []
                    actual, nfe = sample_configured(self.model, self.base, self.noise, config,
                                                    trace=lambda i, t, x: actual_trace.append(x.clone()))
                    reference, reference_trace = reference_heun(self.model, self.base, self.noise, plan)
                    self.assertEqual(len(actual_trace), len(reference_trace))
                    for left, right in zip(actual_trace, reference_trace):
                        torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-4)
                    torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-4)
                    self.assertEqual(nfe, 2 * count - 1)
                    self.assertTrue(torch.isfinite(actual).all())

    def test_common_ddim_matches_library_first_order_on_arbitrary_grid(self):
        # DDIM eta=0 and first-order DPM-Solver++ solve the same update.
        # This checks nonuniform transitions without DDIMScheduler's fixed
        # previous-timestep convention or our own step implementation.
        for schedule in ["uniform_t", "log_sigma", "karras"]:
            for clipping in [False, True]:
                with self.subTest(schedule=schedule, clipping=clipping):
                    config = self.config(schedule=schedule, num_steps=19, clip_sample=clipping)
                    plan = make_plan(config, self.base)
                    actual_steps = []
                    actual, _ = sample_configured(self.model, self.base, self.noise, config,
                                                  trace=lambda i, t, x: actual_steps.append(x.clone()))
                    expected, expected_steps = reference_dpm(self.model, self.base, self.noise,
                                                             plan.timesteps, solver_order=1, clip_sample=clipping)
                    for left, right in zip(actual_steps, expected_steps):
                        torch.testing.assert_close(left, right, atol=3e-4, rtol=3e-5)
                    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-5)

    def test_known_denoiser_recovers_the_clean_image_for_all_methods(self):
        oracle = OracleDenoiser(self.base.alphas_cumprod, clean=0.25)
        for sampler in ["ddim", "dpmpp_2m", "heun"]:
            for count in [1, 10]:
                with self.subTest(sampler=sampler, count=count):
                    actual, _ = sample_configured(oracle, self.base, self.noise,
                                                  self.config(sampler=sampler, num_steps=count))
                    # At sigma~157, recovering x0 in a single call subtracts
                    # large float32 values. Bound the conditioning error by the
                    # input scale instead of assuming float64-level accuracy.
                    scale = max(1.0, float(make_plan(self.config(sampler=sampler, num_steps=count), self.base).sigmas[0]))
                    tolerance = 2 * torch.finfo(torch.float32).eps * scale * float(self.noise.abs().max()) if count == 1 else 1e-5
                    torch.testing.assert_close(actual, torch.full_like(actual, 0.25), rtol=0, atol=tolerance)

    def test_equal_nfe_budgets_have_different_point_counts_but_same_range(self):
        for budget in [19, 39, 79]:
            plans = [make_plan(self.config(sampler=s, num_steps=points_for_budget(s, budget)), self.base)
                     for s in ["ddim", "dpmpp_2m", "heun"]]
            self.assertEqual({plan.nfe for plan in plans}, {budget})
            self.assertEqual({plan.timesteps[0] for plan in plans}, {999})
            self.assertEqual({plan.timesteps[-1] for plan in plans}, {0})

    def test_reference_adapters_support_clipped_dpm_and_heun_without_extra_model_calls(self):
        for sampler in ["dpmpp_2m", "heun"]:
            with self.subTest(sampler=sampler):
                config = self.config(sampler=sampler, schedule="log_sigma", num_steps=10, clip_sample=True)
                actual, nfe = sample_configured(self.model, self.base, self.noise, config)
                reference, reference_nfe = sample_reference(self.model, self.base, self.noise, config)
                self.assertEqual(nfe, reference_nfe)
                torch.testing.assert_close(actual, reference, atol=3e-4, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
