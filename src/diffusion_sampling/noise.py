"""CPU-generated per-image noise, independent of generation batch size."""

from pathlib import Path

import torch

from .storage import save_tensor_file, tensor_sha256, write_json


SPLITS = {
    "debug": (16, 10000),
    "calibration": (64, 20000),
    "validation": (64, 30000),
    "evaluation_mse": (1024, 40000),
    "fid": (5000, 50000),
}


def make_noise(count, seed, shape=(3, 32, 32)):
    if count <= 0 or seed < 0 or seed + count >= 2**63:
        raise ValueError("Invalid noise count or seed range")
    # A generator per image makes the prefix independent of count and batching.
    return torch.stack([
        torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed + i))
        for i in range(count)
    ])


def load_noise(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    noise = payload["noise"]
    if payload.get("schema_version") != 1:
        raise ValueError("Unknown noise bank format")
    if noise.dtype != torch.float32 or tuple(noise.shape[1:]) != (3, 32, 32) or noise.ndim != 4:
        raise ValueError("Noise bank must contain N x 3 x 32 x 32 float32 tensors")
    if len(noise) != payload["count"] or not torch.isfinite(noise).all():
        raise ValueError("Invalid noise bank contents")
    if tensor_sha256(noise) != payload["sha256"]:
        raise ValueError("Noise bank checksum mismatch")
    return payload


def ensure_noise(path, count, seed):
    path = Path(path)
    if path.exists():
        payload = load_noise(path)
        if payload["seed"] != seed or len(payload["noise"]) < count:
            raise ValueError(f"{path}: existing noise bank has a different seed or insufficient samples")
        return payload
    noise = make_noise(count, seed)
    payload = {
        "schema_version": 1, "seed": seed, "count": count,
        "shape": list(noise.shape), "sha256": tensor_sha256(noise), "noise": noise,
    }
    save_tensor_file(path, payload)
    write_json(path.with_suffix(".json"), {key: value for key, value in payload.items() if key != "noise"})
    return payload


def prepare_splits(output, fid_count=5000):
    definitions = dict(SPLITS)
    definitions["fid"] = (fid_count, SPLITS["fid"][1])
    intervals = sorted((seed, seed + count, name) for name, (count, seed) in definitions.items())
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(f"Seed ranges overlap: {left[2]}, {right[2]}")
    result = {}
    for name, (count, seed) in definitions.items():
        payload = ensure_noise(Path(output) / f"{name}.pt", count, seed)
        result[name] = {key: value for key, value in payload.items() if key != "noise"}
    write_json(Path(output) / "splits.json", result)
    return result
