"""Check portable incremental backups without Colab, network or model weights."""

from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from diffusers import DDIMScheduler
import torch

from diffusion_sampling.config import RunConfig
from diffusion_sampling.experiment import run_experiment
from diffusion_sampling.exports import CheckpointExporter, export_noise_banks
from diffusion_sampling.noise import ensure_noise
from diffusion_sampling.storage import read_json, tensor_sha256


class ZeroDenoiser(torch.nn.Module):
    def forward(self, sample, timestep):
        return SimpleNamespace(sample=torch.zeros_like(sample))


def export_test_loader(config, root, device, local_files_only):
    return ZeroDenoiser().eval(), DDIMScheduler(), {"architecture": "TEST_DOUBLE_ONLY"}


class ExportTests(unittest.TestCase):
    def config(self):
        return RunConfig(num_samples=9, num_steps=3, batch_size=2, noise_seed=123,
                         noise_path="data/noise.pt", output_dir="results/run", purpose="pilot",
                         device="cpu", cpu_threads=2, warmup_batches=0)

    def inspect_archive(self, path):
        with ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            metadata = json.loads(archive.read("export_manifest.json"))
            for name, expected in metadata["files_sha256"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
            return metadata, archive.namelist()

    def test_incremental_archives_have_no_gaps_duplicates_or_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exporter = CheckpointExporter(root, root / "results/exports")
            created = []

            def checkpoint(run, manifest, shards, summary):
                archive = exporter.delta(run, manifest, shards, summary)
                if archive is not None:
                    created.append(archive)

            result = run_experiment(self.config(), root, model_loader=export_test_loader,
                                    on_checkpoint=checkpoint, checkpoint_every_batches=2)
            self.assertEqual(result["nfe_per_image"], 3)
            self.assertEqual(len(created), 4)  # batches 1,2,4 and final partial batch 5
            ids = []
            for path in created:
                metadata, names = self.inspect_archive(path)
                self.assertEqual(metadata["kind"], "delta")
                self.assertFalse(metadata["includes_initial_noise"])
                ids.extend(int(Path(n).stem) for n in names if n.startswith("run/images/"))
                self.assertFalse(any(".cache" in n or ".venv" in n for n in names))
            self.assertEqual(ids, list(range(9)))
            uninterrupted = replace(self.config(), output_dir="results/no_backups")
            run_experiment(uninterrupted, root, model_loader=export_test_loader)
            for batch in (root / "results/run/batches").glob("*.pt"):
                original = torch.load(batch, weights_only=True)["samples"]
                plain = torch.load(root / "results/no_backups/batches" / batch.name, weights_only=True)["samples"]
                self.assertTrue(torch.equal(original, plain))

    def test_resume_catches_up_after_backup_failure_without_recomputing_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config()
            with self.assertRaisesRegex(RuntimeError, "backup failed"):
                run_experiment(config, root, model_loader=export_test_loader,
                               on_checkpoint=lambda *args: (_ for _ in ()).throw(RuntimeError("backup failed")))
            committed = root / "results/run/batches/00000000.pt"
            before = committed.read_bytes()
            exporter = CheckpointExporter(root, root / "results/exports")
            created = []

            def callback(run, manifest, shards, summary):
                result = exporter.delta(run, manifest, shards, summary)
                if result:
                    created.append(result)

            run_experiment(config, root, model_loader=export_test_loader, on_checkpoint=callback,
                           checkpoint_every_batches=2)
            self.assertEqual(committed.read_bytes(), before)
            ids = []
            for path in created:
                _, names = self.inspect_archive(path)
                ids.extend(int(Path(n).stem) for n in names if n.startswith("run/images/"))
            self.assertEqual(ids, list(range(9)))
            # Restarting the exporter reads the persistent index and does not
            # request another delta for images already archived.
            restored = CheckpointExporter(root, root / "results/exports")
            self.assertIsNone(restored.delta(root / "results/run"))

    def test_full_archive_contains_exact_noise_raw_samples_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_experiment(self.config(), root, model_loader=export_test_loader)
            exporter = CheckpointExporter(root, root / "results/exports")
            path = exporter.full(root / "results/run", stage="final")
            metadata, names = self.inspect_archive(path)
            self.assertEqual(metadata["images_in_archive"], 9)
            self.assertTrue(metadata["generation_complete"])
            self.assertTrue(metadata["includes_initial_noise"])
            with ZipFile(path) as archive:
                bank = torch.load(BytesIO(archive.read("initial_noise.pt")), weights_only=True)
                manifest = json.loads(archive.read("run/manifest.json"))
                self.assertEqual(tensor_sha256(bank["noise"]), manifest["noise_sha256"])
                self.assertIn("run/config.json", names)
                self.assertEqual(len([name for name in names if name.startswith("run/batches/")]), 5)
                self.assertFalse(metadata["baseline_metrics_complete"])
                self.assertEqual(metadata["download_status"], "archive_created; browser_save_not_confirmed")

    def test_noise_archive_preserves_the_original_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = ensure_noise(root / "data/noise.pt", 9, 123)
            path = export_noise_banks(root, root / "noises.zip", ["data/noise.pt"])
            metadata, _ = self.inspect_archive(path)
            self.assertEqual(metadata["banks"][0]["sha256"], bank["sha256"])
            with ZipFile(path) as archive:
                self.assertEqual(archive.read("data/noise.pt"), (root / "data/noise.pt").read_bytes())

    def test_source_mismatch_is_rejected_before_an_archive_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_experiment(self.config(), root, model_loader=export_test_loader)
            source = root / "source.zip"
            with ZipFile(source, "w") as archive:
                archive.writestr("diffusion-sampling/src/diffusion_sampling/__init__.py", "# wrong source\n")
            exporter = CheckpointExporter(root, root / "results/exports", source_bundle=source)
            with self.assertRaisesRegex(ValueError, "Source archive does not match"):
                exporter.full(root / "results/run")
            self.assertFalse((root / "results/exports").exists())


if __name__ == "__main__":
    unittest.main()
