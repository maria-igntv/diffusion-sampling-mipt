"""FID against all 50,000 CIFAR-10 train images with an explicit protocol."""

from importlib.metadata import version
from contextlib import contextmanager
import math
from pathlib import Path
import time
import warnings

import torch

from .experiment import load_shards, run_lock, summarize
from .images import export_images
from .runtime import configure_runtime, environment, select_device
from .storage import file_sha256, object_sha256, read_json, write_json


@contextmanager
def fid_cache_compatibility():
    """Allow only NumPy array/dtype objects in torch-fidelity's own stats cache.

    torch-fidelity 0.3 saves NumPy mean/covariance arrays using torch.save.
    PyTorch >=2.6 loads in weights-only mode by default. Keep that mode and
    narrowly allow these array types instead of disabling safe loading globally.
    """
    import numpy as np
    core = np._core if hasattr(np, "_core") else np.core
    allowed = [core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype("float32")), type(np.dtype("float64"))]
    with torch.serialization.safe_globals(allowed):
        yield


def prepare_cifar10(root, download=True):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=str(root), train=True, download=download)
    return {"dataset": "CIFAR-10", "split": "train", "count": len(dataset), "root": str(root)}


def evaluate_fid(run_dir, data_root, cache_root, device="auto", batch_size=64, download=False):
    if batch_size <= 0:
        raise ValueError("FID batch size must be positive")
    output = Path(run_dir)
    if not (output / "manifest.json").is_file():
        raise ValueError("No generation manifest found")
    with run_lock(output):
        manifest = read_json(output / "manifest.json")
        configure_runtime(manifest["config"]["cpu_threads"])
        shards = load_shards(output, manifest)
        count = sum(shard["end"] - shard["start"] for shard in shards)
        if count != manifest["config"]["num_samples"]:
            raise ValueError("Finish generation before evaluating FID")
        if count < 2:
            raise ValueError("FID requires at least two samples")
        if count < 2048:
            warnings.warn("Small-sample pilot FID is unstable; do not use it as the final baseline", stacklevel=2)
        export_images(shards, output)
        paths = sorted((output / "images").glob("*.png"))
        expected = [f"{index:08d}.png" for index in range(count)]
        if [path.name for path in paths] != expected:
            raise ValueError("Unexpected files in images/; refusing a contaminated FID evaluation")
        reference = prepare_cifar10(data_root, download=download)
        image_hash = object_sha256([(path.name, file_sha256(path)) for path in paths])
        selected_device = select_device(device)
        # torch-fidelity 0.3 supports CUDA or CPU, not MPS.
        use_cuda = selected_device.type == "cuda"
        metric_environment = environment(torch.device("cuda" if use_cuda else "cpu"))
        from torch_fidelity import calculate_metrics
        began = time.perf_counter()
        with fid_cache_compatibility():
            metrics = calculate_metrics(
                input1=str(output / "images"), input2="cifar10-train", fid=True,
                isc=False, kid=False, ppl=False, cuda=use_cuda, batch_size=batch_size,
                datasets_root=str(data_root), datasets_download=False,
                cache_root=str(Path(cache_root) / object_sha256(metric_environment)), cache=True,
                input1_cache_name=f"generated-{image_hash}",
                feature_extractor="inception-v3-compat", feature_layer_fid="2048",
                samples_find_deep=False, samples_find_ext="png", samples_shuffle=False, verbose=True,
            )
        value = float(metrics["frechet_inception_distance"])
        if not math.isfinite(value):
            raise RuntimeError("FID returned a nonfinite value")
        report = {
            "experiment_id": manifest["experiment_id"], "fid": value,
            "generated_count": count, "reference": reference,
            "implementation": "torch-fidelity", "version": version("torch-fidelity"),
            "feature_extractor": "inception-v3-compat", "feature_dimension": 2048,
            "preprocessing": "native 32x32 RGB PNG uint8; resizing/normalization inside torch-fidelity",
            "generated_images_sha256": image_hash,
            "device": "cuda" if use_cuda else "cpu", "batch_size": batch_size,
            "environment": metric_environment,
            "evaluation_seconds": time.perf_counter() - began,
            "small_sample_pilot": count < 2048,
            "eligible_for_baseline_metrics": manifest["config"]["purpose"] == "baseline" and count >= 5000,
            "visual_review_required": True,
        }
        write_json(output / "fid.json", report)
        summarize(output, manifest, shards)
        return report
