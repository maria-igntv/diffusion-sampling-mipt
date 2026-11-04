"""Device selection and the environment recorded with every experiment."""

from importlib.metadata import PackageNotFoundError, version
import platform
from pathlib import Path
import sys

import torch


def select_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU runtime in Colab")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this Python environment")
    return torch.device(requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def environment(device):
    from .storage import file_sha256, object_sha256
    packages = {}
    for package in ("torch", "torchvision", "diffusers", "huggingface-hub", "torch-fidelity", "numpy", "Pillow", "scipy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "cuda_runtime": torch.version.cuda, "packages": packages,
        "cpu_threads": torch.get_num_threads(),
        "source_sha256": object_sha256([
            (path.name, file_sha256(path))
            for path in sorted(Path(__file__).parent.glob("*.py"))
        ]),
    }


def configure_runtime(cpu_threads):
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
