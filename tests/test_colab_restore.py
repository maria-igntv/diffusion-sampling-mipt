"""Exercise restoration, integrity checks and conflicts with small ZIP fixtures."""

from io import BytesIO
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_restore", ROOT / "scripts/colab_restore.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


class ColabRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = (self.root / "project").resolve()
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("diffusion-sampling/src/diffusion_sampling/__init__.py", "# fixture\n")
            archive.writestr("diffusion-sampling/pyproject.toml", "# fixture\n")
        self.source = buffer.getvalue()
        with ZipFile(BytesIO(self.source)) as archive:
            fingerprint = restore._source_fingerprint(archive)
        self.manifest = {
            "experiment_id": "fixture-only",
            "config": {"output_dir": "results/run", "noise_path": "data/noise/fid.pt",
                       "batch_size": 2},
            "environment": {"source_sha256": fingerprint},
        }
        self.summary = {"experiment_id": "fixture-only", "generated_images": 3}

    def tearDown(self):
        self.temporary.cleanup()

    def make_archive(self, extra=None, kind="full", corrupt=False):
        files = {
            "run/manifest.json": json.dumps(self.manifest).encode(),
            "run/summary.json": json.dumps(self.summary).encode(),
            "run/batches/00000000.pt": b"batch zero fixture",
            "run/batches/00000002.pt": b"batch two fixture",
            "initial_noise.pt": b"noise fixture",
            "source.zip": self.source,
        }
        for index in range(3):
            files[f"run/images/{index:08d}.png"] = f"image fixture {index}".encode()
        if extra:
            files.update(extra)
        metadata = {
            "kind": kind, "start_image": 0, "end_image_exclusive": 3,
            "images_in_archive": 3, "experiment_id": "fixture-only",
            "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }
        if corrupt:
            files["run/images/00000000.png"] = b"damaged fixture"
        path = self.root / "full.zip"
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
            archive.writestr("export_manifest.json", json.dumps(metadata))
        return path

    def test_restores_source_run_and_exact_noise_and_is_idempotent(self):
        path = self.make_archive()
        result = restore.restore_full_archive(path, self.project, restore_source=True)
        self.assertEqual(result["run_dir"], self.project / "results/run")
        self.assertEqual(result["noise_path"].read_bytes(), b"noise fixture")
        self.assertEqual((self.project / "src/diffusion_sampling/__init__.py").read_text(), "# fixture\n")
        restore.restore_full_archive(path, self.project, restore_source=True)
        self.assertEqual(len(list(result["run_dir"].rglob("*.png"))), 3)

    def test_corruption_is_rejected_before_writing_anything(self):
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            restore.restore_full_archive(self.make_archive(corrupt=True), self.project, restore_source=True)
        self.assertFalse(self.project.exists())

    def test_traversal_and_delta_archives_are_rejected(self):
        for extra, kind in [({"../outside.txt": b"unsafe"}, "full"), (None, "delta")]:
            with self.subTest(extra=extra, kind=kind), self.assertRaises(ValueError):
                restore.restore_full_archive(self.make_archive(extra=extra, kind=kind),
                                             self.project, restore_source=True)
            self.assertFalse(self.project.exists())

    def test_conflicting_run_is_preserved(self):
        path = self.make_archive()
        restore.restore_full_archive(path, self.project, restore_source=True)
        target = self.project / "results/run/images/00000000.png"
        target.write_bytes(b"existing modified file")
        with self.assertRaisesRegex(ValueError, "Existing run differs"):
            restore.restore_full_archive(path, self.project, restore_source=True)
        self.assertEqual(target.read_bytes(), b"existing modified file")

    def test_wrong_source_hash_is_rejected(self):
        self.manifest["environment"]["source_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Source ZIP differs"):
            restore.restore_full_archive(self.make_archive(), self.project, restore_source=True)
        self.assertFalse(self.project.exists())

    def test_config_cannot_escape_results_directory(self):
        self.manifest["config"]["output_dir"] = "results/../../outside"
        with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
            restore.restore_full_archive(self.make_archive(), self.project, restore_source=True)
        self.assertFalse(self.project.exists())

    def test_partial_snapshot_and_restore_without_reextracting_source(self):
        path = self.make_archive()
        restore.restore_full_archive(path, self.project, restore_source=True)
        self.manifest["config"]["output_dir"] = "results/another_method"
        result = restore.restore_full_archive(self.make_archive(), self.project)
        self.assertTrue((result["run_dir"] / "summary.json").exists())

    def test_logically_equal_noise_can_have_different_serialization(self):
        path = self.make_archive()
        result = restore.restore_full_archive(path, self.project, restore_source=True)
        result["noise_path"].write_bytes(b"same tensor, different container fixture")
        observed = []

        def equal(left, right):
            observed.append((left.read_bytes(), right.read()))
            return True

        restore.restore_full_archive(path, self.project, noise_equal=equal)
        self.assertEqual(len(observed), 1)
        self.assertEqual(result["noise_path"].read_bytes(), b"same tensor, different container fixture")


if __name__ == "__main__":
    unittest.main()
