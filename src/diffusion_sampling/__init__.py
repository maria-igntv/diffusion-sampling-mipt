"""Experiments with a frozen CIFAR-10 DDPM and interchangeable samplers."""

import os

# Must be set before the first CUDA operation when deterministic algorithms are used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
