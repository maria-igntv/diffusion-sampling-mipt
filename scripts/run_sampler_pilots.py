"""Run the three 19-NFE pilots and optionally 500/1000-NFE numerical references."""

import argparse
from dataclasses import replace
from pathlib import Path

from diffusion_sampling.config import load_config
from diffusion_sampling.experiment import run_experiment
from diffusion_sampling.study import compare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--references", choices=["none", "500", "1000", "both"], default="none")
    parser.add_argument("--output-dir", default="results/sampler_pilots", help="Use a new directory after changing source versions")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs = []
    for name in ["pilot_ddim_19.json", "pilot_dpmpp_19.json", "pilot_heun_19.json"]:
        config = load_config(root / "configs" / name)
        config = replace(config, device=args.device, output_dir=str(Path(args.output_dir) / Path(config.output_dir).name))
        run_experiment(config, root, local_files_only=args.local_files_only)
        runs.append(root / config.output_dir)
    output = root / args.output_dir
    compare_runs(runs, output / "comparison.json")
    reference_runs = {}
    counts = {"none": [], "500": [500], "1000": [1000], "both": [500, 1000]}[args.references]
    for count in counts:
        config = load_config(root / "configs" / f"reference_ddim_{count}_debug.json")
        config = replace(config, device=args.device, output_dir=str(Path(args.output_dir) / Path(config.output_dir).name))
        run_experiment(config, root, local_files_only=args.local_files_only)
        reference_runs[count] = root / config.output_dir
    if reference_runs:
        compare_runs(runs, output / "comparison_with_reference.json", reference=reference_runs[max(counts)])
    if args.references == "both":
        compare_runs([reference_runs[500]], output / "reference_convergence.json", reference=reference_runs[1000])
    print(output)


if __name__ == "__main__":
    main()
