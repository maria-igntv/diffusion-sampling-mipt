"""Integration check with real Inception weights; this is NOT a CIFAR-10 FID.

Compare a generated pilot directory with itself. This exercises Inception,
the actual FID implementation and reading its cached NumPy statistics.
"""

import argparse
from pathlib import Path

import torch
from torch_fidelity import calculate_metrics

from diffusion_sampling.evaluation import fid_cache_compatibility
from diffusion_sampling.runtime import configure_runtime, environment
from diffusion_sampling.storage import file_sha256, object_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="results/pilot_ddim_20/images")
    args = parser.parse_args()
    configure_runtime(4)
    root = Path(__file__).resolve().parents[1]
    torch.hub.set_dir(str(root / ".cache" / "torch" / "hub"))
    images = Path(args.images).resolve()
    paths = sorted(images.glob("*.png"))
    if len(paths) < 2:
        raise ValueError("Generate the pilot first")
    identity = object_sha256({"images": [(p.name, file_sha256(p)) for p in paths],
                              "environment": environment(torch.device("cpu"))})
    with fid_cache_compatibility():
        metrics = calculate_metrics(
            input1=str(images), input2=str(images),
            input1_cache_name=identity, input2_cache_name=identity,
            cache_root=str(root / ".cache" / "fid-self-check"), cache=True,
            fid=True, isc=False, kid=False, ppl=False, cuda=False, batch_size=8,
            save_cpu_ram=True, samples_find_ext="png", samples_find_deep=False,
            samples_shuffle=False, verbose=True,
        )
    value = float(metrics["frechet_inception_distance"])
    if abs(value) > 0.01:
        raise AssertionError(f"Self-distance should be numerically near zero, got {value}")
    report = {"kind": "FID_IMPLEMENTATION_SELF_CHECK", "not_a_cifar10_baseline": True,
              "images": len(paths), "self_distance": value, "passed": True,
              "environment": environment(torch.device("cpu"))}
    write_json(root / "results" / "verification" / "fid_self_check.json", report)
    print(report)


if __name__ == "__main__":
    main()
