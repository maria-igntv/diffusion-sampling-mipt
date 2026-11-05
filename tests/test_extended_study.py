"""Check the Colab workflow with CPU test doubles and temporary artifacts."""

from functools import partial
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from diffusers import DDIMScheduler
import torch

from diffusion_sampling.experiment import load_shards, run_experiment, summarize
from diffusion_sampling.schedules import make_plan
from diffusion_sampling.storage import write_json


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("extended_study", ROOT / "scripts/extended_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class ZeroDenoiser(torch.nn.Module):
    def forward(self, sample, timestep):
        return SimpleNamespace(sample=torch.zeros_like(sample))


def fixture_loader(config, root, device, local_files_only):
    scheduler = DDIMScheduler()
    return ZeroDenoiser().eval(), scheduler, {
        "model_id": config.model_id, "resolved_revision": "TEST_ONLY",
        "weights_sha256": "TEST_ONLY", "prediction_type": "epsilon",
        "sampling_plan": make_plan(config, scheduler).to_dict(),
    }


def fixture_evaluator(directory, data_root, cache_root, device, download):
    manifest, _, _ = study._load_run(directory)
    config = manifest["config"]
    report = {
        "experiment_id": manifest["experiment_id"],
        "fid": {"uniform_t": 3.0, "log_sigma": 2.0, "karras": 1.0}[config["schedule"]],
        "eligible_for_baseline_metrics": True, "implementation": "TEST_FIXTURE",
        "version": "TEST_FIXTURE", "feature_extractor": "TEST_FIXTURE",
        "feature_dimension": 2, "preprocessing": "TEST_FIXTURE",
        "generated_count": config["num_samples"], "environment": manifest["environment"],
        "reference": {"dataset": "TEST_FIXTURE", "split": "TEST_FIXTURE", "count": 5},
    }
    write_json(Path(directory) / "fid.json", report)
    shards = load_shards(directory, manifest)
    summarize(Path(directory), manifest, shards)
    return report


class ExtendedStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        self.source = self.project / "source.zip"
        with ZipFile(self.source, "w") as archive:
            for path in sorted((ROOT / "src/diffusion_sampling").glob("*.py")):
                archive.write(path, f"diffusion-sampling/src/diffusion_sampling/{path.name}")
        self.options = dict(device="cpu", budgets=(3, 5, 7), selection_budget=5,
                            fid_count=5, mse_count=3, confirm_count=4, debug_count=2,
                            reference_budgets=(5, 7), batch_size=2)
        self.plan = study.prepare_plan(self.project, "fixture", **self.options)
        self.runner = partial(run_experiment, model_loader=fixture_loader)

    def tearDown(self):
        self.temporary.cleanup()

    def advance(self, phase):
        return study.run_next_block(self.project, self.plan, phase, self.source,
                                    max_batches=2, runner=self.runner, evaluator=fixture_evaluator)

    def finish(self, phase):
        for _ in range(120):
            if self.advance(phase) is None:
                return
        self.fail("Fixture phase did not terminate")

    def test_design_has_matched_nfe_and_can_be_reloaded(self):
        self.assertEqual(len(self.plan["phases"]["grid"]), 27)
        for task in self.plan["phases"]["grid"]:
            config = study.RunConfig(**task["config"])
            self.assertIn(config.expected_nfe, (3, 5, 7))
            self.assertFalse(config.clip_sample)
        again = study.prepare_plan(self.project, "fixture", **self.options)
        self.assertEqual(again, self.plan)
        changed = {**self.options, "mse_count": 4}
        with self.assertRaisesRegex(ValueError, "Design or environment changed"):
            study.prepare_plan(self.project, "fixture", **changed)

    def test_reference_phase_exports_snapshots_and_reports_mse(self):
        self.finish("convergence")
        report = study.write_phase_report(self.project, self.plan, "convergence")
        self.assertEqual(len(report["rows"]), 3)
        self.assertTrue(all(row["mse_to_reference"] is not None for row in report["rows"]))
        self.assertIsNone(self.advance("convergence"))
        support = study.support_backup(self.project, self.plan, self.source)
        with tempfile.TemporaryDirectory() as restored_root:
            restored = study.restore_support(support, Path(restored_root), self.plan["environment"])
            self.assertEqual(restored, self.plan)
            self.assertTrue((Path(restored_root) / self.plan["banks"]["debug"]["path"]).is_file())

    def test_partial_generation_is_resumed_and_selection_is_fixed_before_confirmation(self):
        checkpoint = self.advance("grid")
        self.assertTrue(checkpoint.is_file())
        rows = study.phase_progress(self.project, self.plan, "grid")
        self.assertEqual(rows[0]["images"], 4)
        self.assertFalse(rows[0]["done"])
        with self.assertRaisesRegex(ValueError, "Finish all tasks"):
            study.lock_selection(self.project, self.plan)
        self.finish("grid")
        self.plan = study.lock_selection(self.project, self.plan)
        self.assertEqual(len(self.plan["selection"]), 6)
        self.assertEqual(len(self.plan["phases"]["independent"]), 12)
        self.assertEqual(len(self.plan["phases"]["mse1024"]), 7)
        self.assertIs(study.lock_selection(self.project, self.plan), self.plan)
        support = study.support_backup(self.project, self.plan, self.source)
        with tempfile.TemporaryDirectory() as restored_root:
            restored = study.restore_support(support, Path(restored_root), self.plan["environment"])
            self.assertEqual(restored["selection"], self.plan["selection"])
        self.finish("independent")
        report = study.write_phase_report(self.project, self.plan, "independent")
        self.assertEqual(len(report["aggregate"]), 6)
        self.assertTrue(all(row["banks"] == 2 for row in report["aggregate"]))
        self.finish("mse1024")
        report = study.write_phase_report(self.project, self.plan, "mse1024")
        self.assertEqual(len(report["rows"]), 6)

    def test_mse_grid_produces_per_image_errors_and_plots(self):
        self.finish("mse64")
        report = study.write_phase_report(self.project, self.plan, "mse64")
        self.assertEqual(len(report["rows"]), 27)
        directory = self.project / "results/fixture/reports"
        self.assertTrue((directory / "mse64_curves.png").is_file())
        self.assertEqual(len((directory / "mse64_per_image.csv").read_text().splitlines()), 1 + 27 * 3)

    def test_support_rejects_changed_runtime(self):
        support = study.support_backup(self.project, self.plan, self.source)
        with self.assertRaisesRegex(ValueError, "Support environment differs"):
            study.restore_support(support, self.project, {"different": "runtime"})


if __name__ == "__main__":
    unittest.main()
