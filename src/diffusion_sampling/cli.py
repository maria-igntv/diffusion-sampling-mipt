"""Command line used by both local runs and the Colab notebook."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import os

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="CIFAR-10 diffusion sampling project")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Inspect Python, packages and GPU availability")

    noise = subparsers.add_parser("prepare-noise", help="Save disjoint fixed noise banks")
    noise.add_argument("--output", default="data/noise")
    noise.add_argument("--fid-count", type=int, default=5000)

    data = subparsers.add_parser("prepare-data", help="Download and verify CIFAR-10 train")
    data.add_argument("--output", default="data/cifar10")

    run = subparsers.add_parser("run", help="Generate or resume the DDIM baseline")
    run.add_argument("--config", required=True)
    run.add_argument("--root", default=".", help="Base directory for paths inside config")
    run.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--max-batches", type=int, help="Stop cleanly after this many new batches")

    fid = subparsers.add_parser("evaluate", help="Evaluate a completed run against CIFAR-10 train")
    fid.add_argument("--run", required=True)
    fid.add_argument("--data-root", default="data/cifar10")
    fid.add_argument("--cache-root", default=".cache/fidelity")
    fid.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    fid.add_argument("--batch-size", type=int, default=64)
    fid.add_argument("--download", action="store_true", help="Allow downloading CIFAR-10 if absent")
    args = parser.parse_args(argv)

    # Keep downloads within the project, including torch-fidelity's Inception weights.
    cache = Path(args.root if args.command == "run" else ".").resolve() / ".cache"
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface-home"))

    if args.command == "doctor":
        import torch
        from .runtime import environment, select_device
        result = environment(select_device("auto"))
        result.update(cuda_available=torch.cuda.is_available(), mps_available=torch.backends.mps.is_available())
        if torch.cuda.is_available():
            result["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    elif args.command == "prepare-noise":
        from .noise import prepare_splits
        result = prepare_splits(args.output, args.fid_count)
    elif args.command == "prepare-data":
        from .evaluation import prepare_cifar10
        result = prepare_cifar10(args.output)
    elif args.command == "run":
        from .experiment import run_experiment
        config = load_config(args.config)
        if args.device:
            config = replace(config, device=args.device)
        result = run_experiment(config, root=args.root, local_files_only=args.local_files_only, max_batches=args.max_batches)
    else:
        from .evaluation import evaluate_fid
        result = evaluate_fid(args.run, args.data_root, args.cache_root, args.device, args.batch_size, args.download)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
