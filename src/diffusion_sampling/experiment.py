"""Measured sampling, crash-safe batches and strict resume validation."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import tempfile
import time

import torch

from .images import export_images
from .noise import ensure_noise
from .runtime import configure_runtime, environment, select_device, synchronize
from .sampling import sample_configured
from .schedules import expected_nfe
from .storage import object_sha256, read_json, save_tensor_file, tensor_sha256, write_json


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    # Google Drive is a FUSE mount: put the advisory lock on the local filesystem.
    # This guards processes in one runtime, not independent Colab machines.
    locks = Path(tempfile.gettempdir()) / "mipt-diffusion-sampling-locks"
    locks.mkdir(exist_ok=True)
    lock_path = locks / (object_sha256(str(output.resolve())) + ".lock")
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is already writing this run") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_shards(output, manifest):
    config = manifest["config"]
    shards = []
    next_start = 0
    for path in sorted((Path(output) / "batches").glob("*.pt")):
        shard = torch.load(path, weights_only=True, map_location="cpu")
        expected_end = min(next_start + config["batch_size"], config["num_samples"])
        if next_start >= config["num_samples"]:
            raise ValueError("More saved batches than expected")
        if path.name != f"{next_start:08d}.pt" or shard["start"] != next_start or shard["end"] != expected_end:
            raise ValueError(f"Noncontiguous or incorrectly sized batch: {path}")
        if shard["experiment_id"] != manifest["experiment_id"]:
            raise ValueError(f"Batch belongs to another experiment: {path}")
        samples = shard["samples"]
        if samples.shape != (expected_end - next_start, 3, 32, 32) or samples.dtype != torch.float32:
            raise ValueError(f"Invalid samples shape/dtype: {path}")
        if not torch.isfinite(samples).all() or tensor_sha256(samples) != shard["samples_sha256"]:
            raise ValueError(f"Samples checksum mismatch or nonfinite values: {path}")
        if shard["nfe"] != expected_nfe(config["sampler"], config["num_steps"]) or not math.isfinite(shard["seconds"]) or shard["seconds"] <= 0:
            raise ValueError(f"Invalid measured metrics: {path}")
        shards.append(shard)
        next_start = expected_end
    return shards


def summarize(output, manifest, shards):
    generated = sum(shard["end"] - shard["start"] for shard in shards)
    seconds = sum(shard["seconds"] for shard in shards)
    sample_evaluations = sum(shard["nfe"] * (shard["end"] - shard["start"]) for shard in shards)
    summary = {
        "experiment_id": manifest["experiment_id"], "purpose": manifest["config"]["purpose"],
        "sampler": manifest["config"]["sampler"], "schedule": manifest["config"].get("schedule", "native"),
        "num_inference_points": manifest["config"]["num_steps"],
        "expected_nfe": expected_nfe(manifest["config"]["sampler"], manifest["config"]["num_steps"]),
        "generation_complete": generated == manifest["config"]["num_samples"],
        "generated_images": generated, "requested_images": manifest["config"]["num_samples"],
        "generation_seconds": seconds,
        "seconds_per_image": seconds / generated if generated else None,
        "images_per_second": generated / seconds if seconds else None,
        "nfe_per_image": sample_evaluations / generated if generated else None,
        "batched_model_forward_calls": sum(shard["nfe"] for shard in shards),
        "sample_equivalent_model_evaluations": sample_evaluations,
        "peak_gpu_memory_allocated_bytes": max((shard.get("peak_gpu_memory_allocated_bytes", 0) for shard in shards), default=0) if manifest["environment"]["device"] == "cuda" else None,
        "fid": None, "baseline_metrics_complete": False, "visual_review_required": True,
        "timing_scope": "Denoising including scheduler setup/steps; excludes model load, warmup, input device transfer, CPU output transfer, file IO and FID.",
    }
    metrics_path = Path(output) / "fid.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        if metrics["experiment_id"] == manifest["experiment_id"] and summary["generation_complete"]:
            summary["fid"] = metrics["fid"]
            summary["baseline_metrics_complete"] = metrics["eligible_for_baseline_metrics"]
    write_json(Path(output) / "summary.json", summary)
    return summary


def run_experiment(config, root=".", local_files_only=False, max_batches=None, model_loader=None):
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    root = Path(root).resolve()
    output = root / config.output_dir
    configure_runtime(config.cpu_threads)
    device = select_device(config.device)
    dtype = getattr(torch, config.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("Use float32 for CPU generation")
    runtime = environment(device)

    with run_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if previous["config"] != config.to_dict():
                raise ValueError("Config changed: select a new output_dir rather than mixing experiments")
            if previous["environment"] != runtime:
                raise ValueError("Runtime/device changed: select a new output_dir for comparable timings")
        elif any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError("Nonempty output_dir without a manifest; refusing to overwrite")

        bank = ensure_noise(root / config.noise_path, config.num_samples, config.noise_seed)
        noise = bank["noise"][:config.num_samples]
        noise_hash = tensor_sha256(noise)
        model = scheduler = None
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["noise_sha256"] != noise_hash:
                raise ValueError("Initial noise changed; refusing to mix samples")
        else:
            if model_loader is None:
                from .model import load_model
                model_loader = load_model
            model, scheduler, contract = model_loader(config, root, device, local_files_only)
            identity = {"config": config.to_dict(), "environment": runtime, "noise_sha256": noise_hash, "model": contract}
            manifest = {"schema_version": 1, **identity, "experiment_id": object_sha256(identity),
                        "created_at_utc": datetime.now(timezone.utc).isoformat()}
            write_json(manifest_path, manifest)
            write_json(output / "model_contract.json", contract)
            if "sampling_plan" in contract:
                write_json(output / "sampling_plan.json", contract["sampling_plan"])

        shards = load_shards(output, manifest)
        start = shards[-1]["end"] if shards else 0
        summarize(output, manifest, shards)
        if start < config.num_samples:
            if model is None:
                if model_loader is None:
                    from .model import load_model
                    model_loader = load_model
                model, scheduler, contract = model_loader(config, root, device, local_files_only)
                if contract != manifest["model"]:
                    raise ValueError("Loaded checkpoint/scheduler differs from the saved run")
            warmup = noise[start:min(start + config.batch_size, config.num_samples)].to(device=device, dtype=dtype)
            for _ in range(config.warmup_batches):
                sample_configured(model, scheduler, warmup, config)
            synchronize(device)
            del warmup

            for batch_index, offset in enumerate(range(start, config.num_samples, config.batch_size)):
                if max_batches is not None and batch_index >= max_batches:
                    break
                end = min(offset + config.batch_size, config.num_samples)
                initial = noise[offset:end].to(device=device, dtype=dtype)
                synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                began = time.perf_counter()
                samples, nfe = sample_configured(model, scheduler, initial, config)
                synchronize(device)
                seconds = time.perf_counter() - began
                peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                samples = samples.detach().cpu().float()
                if not torch.isfinite(samples).all():
                    raise RuntimeError("Nonfinite samples; batch has not been committed")
                shard = {
                    "experiment_id": manifest["experiment_id"], "start": offset, "end": end,
                    "seconds": seconds, "nfe": nfe, "samples": samples,
                    "samples_sha256": tensor_sha256(samples), "peak_gpu_memory_allocated_bytes": peak,
                }
                save_tensor_file(output / "batches" / f"{offset:08d}.pt", shard)
                shards.append(shard)
                summarize(output, manifest, shards)
                print(f"Generated {end}/{config.num_samples}; {seconds:.3f}s/batch; NFE={nfe}", flush=True)
                del initial

        # PNGs are derived from committed tensors, so interruption during export is recoverable.
        export_images(shards, output)
        return summarize(output, manifest, shards)
