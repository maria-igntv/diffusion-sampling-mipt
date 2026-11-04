"""Check all custom integrators against Diffusers on the actual frozen checkpoint.

This is a numerical correctness check, not a quality benchmark or a timing study.
The short trajectories use two fixed initial noises and compare every full step.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from diffusion_sampling.config import RunConfig
from diffusion_sampling.model import load_model
from diffusion_sampling.noise import make_noise
from diffusion_sampling.references import sample_reference
from diffusion_sampling.runtime import configure_runtime, environment, select_device, synchronize
from diffusion_sampling.sampling import sample_configured
from diffusion_sampling.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/verification/sampler_parity.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configure_runtime(4)
    device = select_device(args.device)
    base_config = RunConfig(num_samples=2, batch_size=2, purpose="pilot", device=args.device)
    model, base, contract = load_model(base_config, root, device, args.local_files_only)
    noise = make_noise(2, 10000).to(device)
    cases = [
        replace(base_config, sampler="ddim", schedule="native", num_steps=20),
        replace(base_config, sampler="dpmpp_2m", schedule="karras", num_steps=19, clip_sample=False),
        replace(base_config, sampler="heun", schedule="karras", num_steps=10, clip_sample=False),
        replace(base_config, sampler="dpmpp_2m", schedule="log_sigma", num_steps=19, clip_sample=True),
        replace(base_config, sampler="heun", schedule="log_sigma", num_steps=10, clip_sample=True),
    ]
    rows = []
    for config in cases:
        actual_steps, reference_steps = [], []
        actual, nfe = sample_configured(model, base, noise, config,
                                        trace=lambda i, t, x: actual_steps.append(x.detach().clone()))
        reference, reference_nfe = sample_reference(model, base, noise, config,
                                                    trace=lambda i, t, x: reference_steps.append(x.detach().clone()))
        synchronize(device)
        if nfe != reference_nfe or nfe != config.expected_nfe or len(actual_steps) != len(reference_steps):
            raise AssertionError("Reference and custom solver use different NFE/step counts")
        maxima = []
        for index, (left, right) in enumerate(zip(actual_steps, reference_steps)):
            # Float32 roundoff is amplified by sigma_max~157 and recurrent model
            # evaluation. Keep a small absolute + relative bound and save actual errors.
            torch.testing.assert_close(left, right, atol=5e-4, rtol=5e-4,
                                       msg=f"{config.sampler} differs at completed step {index}")
            maxima.append(float((left - right).abs().max()))
        difference = (actual - reference).float()
        row = {"sampler": config.sampler, "schedule": config.schedule,
               "clip_sample": config.clip_sample, "nfe": nfe,
               "max_step_abs_error": max(maxima), "max_final_abs_error": float(difference.abs().max()),
               "final_mse": float(difference.square().mean()), "passed": True}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"kind": "PRETRAINED_SAMPLER_PARITY_CHECK", "not_a_quality_benchmark": True,
              "model_revision": contract["resolved_revision"], "weights_sha256": contract["weights_sha256"],
              "num_images": 2, "environment": environment(device), "rows": rows, "passed": True}
    write_json(root / args.output, report)
    print(root / args.output)


if __name__ == "__main__":
    main()
