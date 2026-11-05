"""Restore a full experiment export using only Python's standard library."""

from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile


def _relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def _destination(root, name):
    path = root.joinpath(*_relative_path(name).parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves the project: {name}")
    return path


def _source_fingerprint(source):
    prefix = "diffusion-sampling/src/diffusion_sampling/"
    records = []
    for name in sorted(source.namelist()):
        relative = name.removeprefix(prefix)
        if name.startswith(prefix) and relative.endswith(".py") and "/" not in relative:
            records.append((relative, hashlib.sha256(source.read(name)).hexdigest()))
    if not records:
        raise ValueError("Source ZIP has no experiment package")
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def restore_full_archive(archive_path, project, restore_source=False, noise_equal=None):
    """Verify a full ZIP and restore its run, exact noise and optional source.

    Existing runs must be identical to the uploaded snapshot. Conflicting files
    cause an error before restoration. Run directories are published atomically.
    """
    project = Path(project).resolve()
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate entry names")
        for name in names:
            _relative_path(name)
        metadata = json.loads(archive.read("export_manifest.json"))
        hashes = metadata["files_sha256"]
        if set(names) != set(hashes) | {"export_manifest.json"}:
            raise ValueError("ZIP entries differ from the export manifest")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}")
        if metadata["kind"] != "full" or metadata["start_image"] != 0:
            raise ValueError("Upload a full archive starting at image zero")
        manifest = json.loads(archive.read("run/manifest.json"))
        summary = json.loads(archive.read("run/summary.json"))
        experiment_id = manifest["experiment_id"]
        count = summary["generated_images"]
        if metadata["experiment_id"] != experiment_id or summary["experiment_id"] != experiment_id:
            raise ValueError("Experiment identifiers differ")
        if metadata["end_image_exclusive"] != count or metadata["images_in_archive"] != count:
            raise ValueError("Snapshot image counts differ")
        expected_images = {f"run/images/{index:08d}.png" for index in range(count)}
        if {name for name in names if name.startswith("run/images/")} != expected_images:
            raise ValueError("Snapshot PNG sequence has gaps or extra entries")
        config = manifest["config"]
        expected_batches = {
            f"run/batches/{index:08d}.pt"
            for index in range(0, count, config["batch_size"])
        }
        if {name for name in names if name.startswith("run/batches/")} != expected_batches:
            raise ValueError("Snapshot batch sequence has gaps or extra entries")
        run_relative = _relative_path(config["output_dir"])
        noise_relative = _relative_path(config["noise_path"])
        if run_relative.parts[0] != "results" or noise_relative.parts[0] != "data":
            raise ValueError("Expected results/ and data/ paths in the saved config")
        run_dir = _destination(project, run_relative.as_posix())
        noise_path = _destination(project, noise_relative.as_posix())
        noise_bytes = archive.read("initial_noise.pt")
        if noise_path.exists() and noise_path.read_bytes() != noise_bytes:
            if noise_equal is None or not noise_equal(noise_path, BytesIO(noise_bytes)):
                raise ValueError("An existing noise bank differs from the uploaded bank")
        source_bytes = archive.read("source.zip")
        source_files = {}
        with ZipFile(BytesIO(source_bytes)) as source:
            if len(source.namelist()) != len(set(source.namelist())):
                raise ValueError("Source ZIP contains duplicate entry names")
            if _source_fingerprint(source) != manifest["environment"]["source_sha256"]:
                raise ValueError("Source ZIP differs from the generation source hash")
            for entry in source.infolist():
                relative = _relative_path(entry.filename)
                if relative.parts[0] != "diffusion-sampling" or len(relative.parts) < 2:
                    raise ValueError("Unexpected source ZIP root")
                if not entry.is_dir():
                    source_files[PurePosixPath(*relative.parts[1:]).as_posix()] = source.read(entry)
        if restore_source:
            for name, data in source_files.items():
                path = _destination(project, name)
                if path.exists() and path.read_bytes() != data:
                    raise ValueError(f"Existing source differs: {name}")
        else:
            package = project / "src/diffusion_sampling"
            records = [(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
                       for path in sorted(package.glob("*.py"))]
            fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            if fingerprint != manifest["environment"]["source_sha256"]:
                raise ValueError("Restore the matching project source before this run")
        run_files = {name.removeprefix("run/"): name for name in names if name.startswith("run/")}
        if run_dir.exists():
            existing = {path.relative_to(run_dir).as_posix()
                        for path in run_dir.rglob("*") if path.is_file()}
            if existing != set(run_files):
                raise ValueError("Existing run differs from the uploaded snapshot")
            for relative, name in run_files.items():
                if _destination(run_dir, relative).read_bytes() != archive.read(name):
                    raise ValueError(f"Existing run differs: {relative}")
        project.mkdir(parents=True, exist_ok=True)
        if restore_source:
            for name, data in source_files.items():
                path = _destination(project, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        if not noise_path.exists():
            with tempfile.NamedTemporaryFile(dir=noise_path.parent, delete=False) as stream:
                temporary_noise = Path(stream.name)
                stream.write(noise_bytes)
            temporary_noise.replace(noise_path)
        if not run_dir.exists():
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=run_dir.parent))
            try:
                for relative, name in run_files.items():
                    path = _destination(staging, relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(name))
                staging.replace(run_dir)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
    return {"run_dir": run_dir, "noise_path": noise_path, "manifest": manifest,
            "summary": summary, "source_bytes": source_bytes}
