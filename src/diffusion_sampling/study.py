"""Matched-NFE configurations and comparisons of committed generation results."""

import csv
from dataclasses import replace
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw
import torch

from .config import RunConfig
from .experiment import load_shards
from .images import to_uint8
from .schedules import CUSTOM_SAMPLERS, points_for_budget
from .storage import read_json, write_json


def create_sweep(output, name="sampler_study", budgets=(19, 39, 79),
                 schedules=("uniform_t", "log_sigma", "karras"),
                 samplers=("ddim", "dpmpp_2m", "heun"), num_samples=16,
                 batch_size=8, purpose="pilot", device="auto", noise_path=None, noise_seed=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Study name must contain letters, numbers, underscores or hyphens")
    if not budgets or not schedules or not samplers:
        raise ValueError("Select at least one budget, schedule and sampler")
    if any(len(set(values)) != len(values) for values in (budgets, schedules, samplers)):
        raise ValueError("Sweep axes must not contain duplicates")
    if any(s not in CUSTOM_SAMPLERS for s in samplers):
        raise ValueError("Sweep supports the three custom samplers")
    if any(s not in {"uniform_t", "log_sigma", "karras"} for s in schedules):
        raise ValueError("Sweep uses only standard common-grid schedules")
    if noise_path is None:
        noise_path = "data/noise/fid.pt" if purpose == "baseline" else "data/noise/debug.pt"
    if noise_seed is None:
        noise_seed = 50000 if purpose == "baseline" else 10000
    base = RunConfig(num_samples=num_samples, batch_size=batch_size, purpose=purpose,
                     device=device, noise_path=noise_path, noise_seed=noise_seed, clip_sample=False)
    prepared = []
    output = Path(output)
    for sampler in samplers:
        for schedule in schedules:
            for budget in budgets:
                points = points_for_budget(sampler, budget)
                label = f"{sampler}_{schedule}_nfe{budget}"
                config = replace(base, sampler=sampler, schedule=schedule, num_steps=points,
                                 output_dir=f"results/{name}/{label}")
                path = output / f"{label}.json"
                if path.exists() and read_json(path) != config.to_dict():
                    raise ValueError(f"Existing config differs; choose another sweep directory: {path}")
                prepared.append((path, config))
    for path, config in prepared:
        write_json(path, config.to_dict())
    manifest = {
        "schema_version": 1, "kind": "matched_nfe_sweep", "name": name,
        "samplers": list(samplers), "schedules": list(schedules), "budgets": list(budgets),
        "num_samples": num_samples, "clip_sample": False,
        "configs": [path.name for path, _ in prepared],
        "note": "num_steps counts query nodes; Heun NFE=2*num_steps-1. All standard grids start at t=999 and end at t=0 then sigma=0.",
    }
    write_json(output / "sweep.json", manifest)
    return manifest


def _load_run(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    summary = read_json(directory / "summary.json")
    shards = load_shards(directory, manifest)
    count = sum(s["end"] - s["start"] for s in shards)
    if count != manifest["config"]["num_samples"] or not summary["generation_complete"]:
        raise ValueError(f"Incomplete generation: {directory}")
    if summary["experiment_id"] != manifest["experiment_id"]:
        raise ValueError("Summary belongs to a different experiment")
    if summary["generated_images"] != count:
        raise ValueError("Summary image count differs from committed tensors")
    elapsed = sum(shard["seconds"] for shard in shards)
    actual_nfe = sum(shard["nfe"] * (shard["end"] - shard["start"]) for shard in shards) / count
    if not math.isclose(summary["generation_seconds"], elapsed, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Summary timing differs from committed batches")
    if summary["nfe_per_image"] != actual_nfe or not math.isclose(summary["seconds_per_image"], elapsed / count, rel_tol=1e-12):
        raise ValueError("Summary NFE or throughput differs from committed batches")
    return manifest, summary, torch.cat([shard["samples"] for shard in shards])


def _comparison_contract(manifest):
    model = manifest["model"]
    plan = model.get("sampling_plan")
    if plan is None:
        raise ValueError("Comparison requires explicit sampling plans; regenerate with the current version")
    return {
        "model_id": model["model_id"], "revision": model["resolved_revision"],
        "weights_sha256": model["weights_sha256"], "prediction_type": model["prediction_type"],
        "noise_sha256": manifest["noise_sha256"], "num_samples": manifest["config"]["num_samples"],
        "clip_sample": manifest["config"]["clip_sample"], "dtype": manifest["config"]["dtype"],
        "denoiser_policy": plan["denoiser_policy"],
        "start_timestep": plan["timesteps"][0], "last_query_timestep": plan["timesteps"][-1],
        "terminal_sigma": plan["sigmas_with_clean_endpoint"][-1],
        "initialization": plan["initialization"],
    }


def compare_runs(runs, output, reference=None):
    if not runs:
        raise ValueError("Select at least one run")
    output = Path(output)
    rows, previews = [], []
    common = timing_environment = batch_size = fid_protocol = None
    reference_samples = reference_contract = None
    if reference is not None:
        ref_manifest, _, reference_samples = _load_run(reference)
        reference_contract = _comparison_contract(ref_manifest)
    for directory in runs:
        manifest, summary, samples = _load_run(directory)
        contract = _comparison_contract(manifest)
        if common is None:
            common = contract
            timing_environment = manifest["environment"]
            batch_size = manifest["config"]["batch_size"]
        if contract != common:
            raise ValueError("Runs differ in weights, noise, precision, clipping or trajectory endpoints")
        if manifest["environment"] != timing_environment or manifest["config"]["batch_size"] != batch_size:
            raise ValueError("Timing comparison requires the same environment and batch size")
        row = {
            "run": str(directory), "sampler": manifest["config"]["sampler"],
            "schedule": manifest["config"].get("schedule", "native"),
            "num_images": len(samples), "nfe": summary["nfe_per_image"],
            "generation_seconds": summary["generation_seconds"],
            "seconds_per_image": summary["seconds_per_image"], "fid": summary.get("fid"),
            "mse_to_reference": None, "mse_standard_error": None,
        }
        if row["fid"] is not None:
            fid = read_json(Path(directory) / "fid.json")
            if fid["experiment_id"] != manifest["experiment_id"] or fid["fid"] != row["fid"]:
                raise ValueError("FID report does not match the run")
            protocol = {key: fid[key] for key in ["implementation", "version", "feature_extractor", "feature_dimension", "preprocessing", "generated_count", "environment"]}
            protocol["reference"] = {key: fid["reference"][key] for key in ["dataset", "split", "count"]}
            if fid_protocol is not None and protocol != fid_protocol:
                raise ValueError("FID protocols differ; do not rank these scores together")
            fid_protocol = protocol
        if reference_samples is not None:
            if reference_contract != common:
                raise ValueError("Reference differs in weights, noise, precision, clipping or endpoints")
            per_image = (samples - reference_samples).square().flatten(1).mean(1).double()
            row["mse_to_reference"] = float(per_image.mean())
            row["mse_standard_error"] = float(per_image.std() / math.sqrt(len(per_image))) if len(per_image) > 1 else None
        rows.append(row)
        previews.append(to_uint8(samples[:8]))
    report = {
        "schema_version": 1, "kind": "matched_conditions_comparison", "contract": common,
        "environment": timing_environment, "batch_size": batch_size,
        "fid_protocol": fid_protocol,
        "reference": str(reference) if reference is not None else None,
        "budgets": sorted({row["nfe"] for row in rows}), "rows": rows,
        "interpretation": "Compare methods within each NFE group. MSE measures fidelity to a numerical reference, not distributional quality. Missing FID means quality is not yet evaluated.",
    }
    write_json(output, report)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    width = 220 + max(len(p) for p in previews) * 96
    montage = Image.new("RGB", (width, len(rows) * 104), "white")
    draw = ImageDraw.Draw(montage)
    for index, (row, images) in enumerate(zip(rows, previews)):
        draw.text((8, index * 104 + 20), f"{row['sampler']} / {row['schedule']}\nNFE={row['nfe']:g}", fill="black")
        for column, array in enumerate(images):
            image = Image.fromarray(array).resize((96, 96), Image.Resampling.NEAREST)
            montage.paste(image, (220 + column * 96, index * 104))
    montage.save(output.with_suffix(".png"))
    return report
