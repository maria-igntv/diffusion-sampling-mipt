"""Portable result archives, with incremental batches and a persistent index.

This module creates files only; the Colab notebook requests browser downloads.
An archive being created/requested does NOT prove that a browser saved it.
No model weights, CIFAR-10 dataset, credentials or library caches are included.
"""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image
import torch

from .experiment import load_shards
from .images import to_uint8
from .noise import load_noise
from .storage import file_sha256, object_sha256, read_json, tensor_sha256, write_json


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _tensor_bytes(value):
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _preview(shards):
    images = []
    for shard in shards:
        for array in to_uint8(shard["samples"][:max(0, 64 - len(images))]):
            images.append(Image.fromarray(array))
        if len(images) >= 64:
            break
    if not images:
        return None
    columns = min(8, len(images))
    grid = Image.new("RGB", (columns * 32, math.ceil(len(images) / columns) * 32))
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * 32, (index // columns) * 32))
    return grid.resize((grid.width * 4, grid.height * 4), Image.Resampling.NEAREST)


def source_bundle_fingerprint(path):
    """Use the same package-file hash convention as runtime.environment."""
    prefix = "diffusion-sampling/src/diffusion_sampling/"
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist()
                 if name.startswith(prefix) and name.endswith(".py") and "/" not in name[len(prefix):]]
        if not names:
            raise ValueError("Source bundle does not contain the experiment package")
        return object_sha256([(Path(name).name, hashlib.sha256(archive.read(name)).hexdigest())
                              for name in sorted(names)])


class _Archive:
    def __init__(self, path):
        self.path = path
        self.temporary = path.with_suffix(".zip.tmp")
        self.files = {}

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.zip = ZipFile(self.temporary, "w", compression=ZIP_DEFLATED, compresslevel=3)
        return self

    def add(self, name, data):
        self.zip.writestr(name, data)
        self.files[name] = hashlib.sha256(data).hexdigest()

    def __exit__(self, error_type, error, traceback):
        self.zip.close()
        if error_type is None:
            self.temporary.replace(self.path)
        else:
            self.temporary.unlink(missing_ok=True)


def export_noise_banks(root, output, noise_paths, source_bundle=None):
    """Back up the EXACT saved initial noises before any long GPU generation."""
    root, output = Path(root).resolve(), Path(output)
    banks = []
    for relative in noise_paths:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Noise bank paths must be relative to the project")
        bank = load_noise(root / path)
        banks.append((path, bank))
    metadata = {"kind": "fixed_initial_noise_banks", "banks": [],
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    with _Archive(output) as archive:
        for relative, bank in banks:
            archive.add(relative.as_posix(), (root / relative).read_bytes())
            details = {key: value for key, value in bank.items() if key != "noise"}
            archive.add(relative.with_suffix(".json").as_posix(), _json_bytes(details))
            metadata["banks"].append({"path": relative.as_posix(), **details})
        if source_bundle is not None:
            archive.add("source.zip", Path(source_bundle).read_bytes())
        metadata["files_sha256"] = dict(archive.files)
        archive.add("export_manifest.json", _json_bytes(metadata))
    return output


class CheckpointExporter:
    def __init__(self, root, export_dir, source_bundle=None):
        self.root = Path(root).resolve()
        self.export_dir = Path(export_dir).resolve()
        self.source_bundle = Path(source_bundle).resolve() if source_bundle is not None else None
        self.source_fingerprint = source_bundle_fingerprint(self.source_bundle) if self.source_bundle is not None else None
        self._noise_cache = {}

    def _noise(self, manifest):
        key = manifest["noise_sha256"]
        if key not in self._noise_cache:
            config = manifest["config"]
            bank = load_noise(self.root / config["noise_path"])
            noise = bank["noise"][:config["num_samples"]]
            if bank["seed"] != config["noise_seed"] or len(noise) != config["num_samples"] or tensor_sha256(noise) != key:
                raise ValueError("Saved initial noise does not match the generation manifest")
            self._noise_cache[key] = {
                "schema_version": 1, "seed": config["noise_seed"], "count": len(noise),
                "shape": list(noise.shape), "sha256": key, "noise": noise,
            }
        return self._noise_cache[key]

    def _snapshot(self, run_dir, manifest, shards, summary):
        run_dir = Path(run_dir)
        if manifest is None:
            manifest = read_json(run_dir / "manifest.json")
        if shards is None:
            shards = load_shards(run_dir, manifest)
        if summary is None:
            summary = read_json(run_dir / "summary.json")
        generated = sum(shard["end"] - shard["start"] for shard in shards)
        if summary["experiment_id"] != manifest["experiment_id"] or generated != summary["generated_images"]:
            raise ValueError("Snapshot summary differs from committed batches")
        if self.source_fingerprint is not None and self.source_fingerprint != manifest["environment"]["source_sha256"]:
            raise ValueError("Source archive does not match the code that generated this run")
        return manifest, tuple(shards), summary, generated

    def _index(self, manifest):
        path = self.export_dir / f"{manifest['experiment_id'][:16]}_index.json"
        if path.exists():
            index = read_json(path)
            if index["experiment_id"] != manifest["experiment_id"]:
                raise ValueError("Export index belongs to another run")
        else:
            index = {"schema_version": 1, "experiment_id": manifest["experiment_id"], "delta_end": 0, "archives": []}
        return path, index

    def delta(self, run_dir, manifest=None, shards=None, summary=None):
        return self._export(run_dir, "delta", "progress", manifest, shards, summary)

    def full(self, run_dir, stage="final"):
        return self._export(run_dir, "full", stage, None, None, None)

    def _export(self, run_dir, kind, stage, manifest, shards, summary):
        run_dir = Path(run_dir)
        manifest, shards, summary, end = self._snapshot(run_dir, manifest, shards, summary)
        index_path, index = self._index(manifest)
        start = index["delta_end"] if kind == "delta" else 0
        if start > end:
            raise ValueError("Export index is ahead of the committed generation")
        if start == end:
            return None
        selected = [shard for shard in shards if shard["start"] >= start]
        if not selected or selected[0]["start"] != start:
            raise ValueError("An incremental archive must start on a committed batch boundary")
        # Confirm the exact noise bank even for a delta. The initial-noise ZIP
        # is downloaded separately; full archives include this bank themselves.
        noise = self._noise(manifest)
        safe_stage = "".join(c for c in stage if c.isalnum() or c in "_-") or "snapshot"
        name = f"{run_dir.name}_{kind}_{safe_stage}_{start:06d}-{end:06d}_{uuid4().hex[:8]}.zip"
        output = self.export_dir / name
        metadata = {
            "schema_version": 1, "kind": kind, "stage": safe_stage,
            "experiment_id": manifest["experiment_id"], "start_image": start, "end_image_exclusive": end,
            "images_in_archive": end - start, "generation_complete": summary["generation_complete"],
            "baseline_metrics_complete": summary["baseline_metrics_complete"],
            "noise_sha256": manifest["noise_sha256"], "includes_initial_noise": kind == "full",
            "requires_initial_noise_backup": kind == "delta", "includes_source": self.source_bundle is not None,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "download_status": "archive_created; browser_save_not_confirmed",
        }
        with _Archive(output) as archive:
            archive.add("run/manifest.json", _json_bytes(manifest))
            archive.add("run/config.json", _json_bytes(manifest["config"]))
            archive.add("run/model_contract.json", _json_bytes(manifest["model"]))
            archive.add("run/summary.json", _json_bytes(summary))
            if "sampling_plan" in manifest["model"]:
                archive.add("run/sampling_plan.json", _json_bytes(manifest["model"]["sampling_plan"]))
            for filename in ("fid.json", "visual_review.md", "generation.log"):
                path = run_dir / filename
                if path.is_file():
                    archive.add(f"run/{filename}", path.read_bytes())
            for path in sorted((run_dir / "logs").glob("*.log")):
                archive.add(f"run/logs/{path.name}", path.read_bytes())
            for shard in selected:
                filename = f"{shard['start']:08d}.pt"
                archive.add(f"run/batches/{filename}", (run_dir / "batches" / filename).read_bytes())
                for offset, array in enumerate(to_uint8(shard["samples"])):
                    archive.add(f"run/images/{shard['start'] + offset:08d}.png", _png_bytes(Image.fromarray(array)))
            preview = _preview(shards)
            if preview is not None:
                archive.add("run/preview.png", _png_bytes(preview))
            if kind == "full":
                archive.add("initial_noise.pt", _tensor_bytes(noise))
            if self.source_bundle is not None:
                archive.add("source.zip", self.source_bundle.read_bytes())
            metadata["files_sha256"] = dict(archive.files)
            archive.add("export_manifest.json", _json_bytes(metadata))
        record = {key: metadata[key] for key in ["kind", "stage", "start_image", "end_image_exclusive", "images_in_archive"]}
        record.update(filename=output.name, bytes=output.stat().st_size, sha256=file_sha256(output),
                      download_status="archive_created; browser_save_not_confirmed")
        index["archives"].append(record)
        if kind == "delta":
            index["delta_end"] = end
        write_json(index_path, index)
        return output
