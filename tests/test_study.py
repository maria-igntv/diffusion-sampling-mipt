"""Integration checks of matched budgets, saved trajectories and comparison guards."""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from diffusers import DDIMScheduler
import torch

from diffusion_sampling.config import RunConfig, load_config
from diffusion_sampling.experiment import run_experiment
from diffusion_sampling.schedules import make_plan, points_for_budget
from diffusion_sampling.storage import read_json, write_json
from diffusion_sampling.study import compare_runs, create_sweep


class ZeroDenoiser(torch.nn.Module):
    def forward(self, sample, timestep):
        return SimpleNamespace(sample=torch.zeros_like(sample))


def study_test_loader(config, root, device, local_files_only):
    scheduler = DDIMScheduler(clip_sample=config.clip_sample)
    contract = {
        "model_id": "TEST_DOUBLE_ONLY", "resolved_revision": "test", "weights_sha256": "test",
        "prediction_type": "epsilon", "sampling_plan": make_plan(config, scheduler).to_dict(),
    }
    return ZeroDenoiser().eval(), scheduler, contract


class StudyTests(unittest.TestCase):
    def config(self, **updates):
        return replace(RunConfig(
            sampler="ddim", schedule="uniform_t", num_steps=3, num_samples=5, batch_size=2,
            noise_seed=123, noise_path="noise.pt", output_dir="ddim", purpose="pilot",
            device="cpu", cpu_threads=2, warmup_batches=0, clip_sample=False,
        ), **updates)

    def test_sweep_encodes_equal_actual_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = create_sweep(Path(tmp) / "configs", budgets=[19, 39], schedules=["uniform_t"], num_samples=5)
            self.assertEqual(len(report["configs"]), 6)
            for filename in report["configs"]:
                config = load_config(Path(tmp) / "configs" / filename)
                expected = int(filename.rsplit("nfe", 1)[1].split(".")[0])
                self.assertEqual(config.expected_nfe, expected)
                self.assertFalse(config.clip_sample)
                if config.sampler == "heun":
                    self.assertEqual(config.num_steps, (expected + 1) // 2)

    def test_invalid_sweep_does_not_leave_partially_written_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "configs"
            with self.assertRaises(ValueError):
                create_sweep(output, budgets=[19, 20])
            self.assertFalse(output.exists())

    def test_existing_sweep_is_not_overwritten_with_other_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_sweep(tmp, budgets=[19], schedules=["uniform_t"], samplers=["ddim"])
            path = Path(tmp) / "ddim_uniform_t_nfe19.json"
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Existing config differs"):
                create_sweep(tmp, budgets=[19], schedules=["uniform_t"], samplers=["ddim"], num_samples=8)
            self.assertEqual(path.read_bytes(), before)

    def test_heun_shards_resume_and_compare_at_actual_nfe(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for sampler in ["ddim", "dpmpp_2m", "heun"]:
                config = self.config(sampler=sampler, output_dir=sampler, num_steps=points_for_budget(sampler, 3))
                partial = run_experiment(config, tmp, max_batches=1, model_loader=study_test_loader)
                self.assertEqual(partial["nfe_per_image"], 3)
                summary = run_experiment(config, tmp, model_loader=study_test_loader)
                self.assertTrue(summary["generation_complete"])
                paths.append(Path(tmp) / sampler)
            reference = self.config(num_steps=20, output_dir="reference")
            run_experiment(reference, tmp, model_loader=study_test_loader)
            output = Path(tmp) / "comparison.json"
            report = compare_runs(paths, output, reference=Path(tmp) / "reference")
            self.assertEqual(report["budgets"], [3])
            self.assertEqual(len(report["rows"]), 3)
            self.assertTrue(output.with_suffix(".csv").is_file())
            self.assertTrue(output.with_suffix(".png").is_file())
            self.assertTrue(all(row["fid"] is None for row in report["rows"]))
            self.assertTrue(all(row["mse_to_reference"] is not None for row in report["rows"]))

    def test_mismatched_clipping_and_endpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_experiment(self.config(), tmp, model_loader=study_test_loader)
            run_experiment(self.config(clip_sample=True, output_dir="clipped"), tmp, model_loader=study_test_loader)
            run_experiment(self.config(schedule="custom", timesteps=[990, 250, 0], output_dir="other_start"), tmp, model_loader=study_test_loader)
            for incompatible in ["clipped", "other_start"]:
                with self.subTest(run=incompatible), self.assertRaisesRegex(ValueError, "Runs differ"):
                    compare_runs([Path(tmp) / "ddim", Path(tmp) / incompatible], Path(tmp) / "comparison.json")

    def test_tampered_summary_cannot_change_the_reported_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_experiment(self.config(), tmp, model_loader=study_test_loader)
            path = Path(tmp) / "ddim/summary.json"
            summary = read_json(path)
            summary["nfe_per_image"] = 100
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, "Summary NFE"):
                compare_runs([path.parent], Path(tmp) / "comparison.json")


if __name__ == "__main__":
    unittest.main()
