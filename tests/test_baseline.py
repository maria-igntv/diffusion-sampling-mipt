"""Offline checks for scientific comparability and interruption recovery.

The tiny/random and zero-output models below are test doubles, not baseline results.
No pretrained checkpoint or dataset is downloaded by this suite.
"""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel

from diffusion_sampling.config import RunConfig, load_config
from diffusion_sampling.evaluation import evaluate_fid, fid_cache_compatibility
from diffusion_sampling.experiment import run_experiment
from diffusion_sampling.images import to_uint8
from diffusion_sampling.noise import ensure_noise, make_noise, prepare_splits
from diffusion_sampling.sampling import count_model_calls, sample_ddim
from diffusion_sampling.storage import save_tensor_file


class ZeroDenoiser(torch.nn.Module):
    def forward(self, sample, timestep):
        return SimpleNamespace(sample=torch.zeros_like(sample))


def test_loader(config, root, device, local_files_only):
    return ZeroDenoiser().eval(), DDIMScheduler(clip_sample=config.clip_sample), {"architecture": "TEST_DOUBLE_ONLY"}


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **updates):
        return replace(RunConfig(
            num_samples=5, num_steps=3, batch_size=2, noise_seed=123,
            noise_path="noise.pt", output_dir="run", purpose="pilot", device="cpu", cpu_threads=2,
        ), **updates)

    def test_noise_prefix_is_independent_of_requested_count(self):
        self.assertTrue(torch.equal(make_noise(2, 10), make_noise(5, 10)[:2]))
        self.assertFalse(torch.equal(make_noise(2, 10), make_noise(2, 20)))

    def test_saved_noise_cannot_be_silently_reseeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "noise.pt"
            ensure_noise(path, 3, 10)
            with self.assertRaisesRegex(ValueError, "different seed"):
                ensure_noise(path, 3, 11)

    def test_noise_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "noise.pt"
            bank = ensure_noise(path, 3, 10)
            bank["noise"][0, 0, 0, 0] += 1
            save_tensor_file(path, bank)
            with self.assertRaisesRegex(ValueError, "checksum"):
                ensure_noise(path, 3, 10)

    def test_splits_have_disjoint_seed_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = prepare_splits(tmp, fid_count=8)
            used = set()
            for split in result.values():
                seeds = set(range(split["seed"], split["seed"] + split["count"]))
                self.assertFalse(used & seeds)
                used |= seeds

    def test_unknown_config_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"num_step": 20}))
            with self.assertRaisesRegex(ValueError, "Unknown config"):
                load_config(path)

    def test_ddim_matches_official_pipeline(self):
        torch.manual_seed(8)
        unet = UNet2DModel(
            sample_size=32, in_channels=3, out_channels=3, layers_per_block=1,
            block_out_channels=(8,), down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",), norm_num_groups=4,
        ).eval()
        scheduler = DDIMScheduler()
        raw, nfe = sample_ddim(unet, scheduler, make_noise(1, 123), 4)
        pipe = DDIMPipeline(unet=unet, scheduler=DDIMScheduler())
        pipe.set_progress_bar_config(disable=True)
        expected = pipe(generator=torch.Generator().manual_seed(123), batch_size=1,
                        num_inference_steps=4, eta=0.0, output_type="np").images
        actual = (raw / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(nfe, 4)

    def test_hook_is_removed_after_model_failure(self):
        model = ZeroDenoiser()
        with self.assertRaises(RuntimeError):
            with count_model_calls(model):
                raise RuntimeError("interrupted")
        self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_uint8_conversion_clips_and_preserves_rgb(self):
        samples = torch.tensor([-2., 0., 2.]).reshape(1, 3, 1, 1)
        self.assertEqual(to_uint8(samples).tolist(), [[[[0, 128, 255]]]])

    def test_resume_matches_uninterrupted_generation_and_repairs_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config()
            partial = run_experiment(config, tmp, max_batches=1, model_loader=test_loader)
            self.assertEqual(partial["generated_images"], 2)
            self.assertFalse(partial["generation_complete"])
            (Path(tmp) / "run/images/00000000.png").unlink()
            # An uncommitted temp file must never be treated as a completed batch.
            (Path(tmp) / "run/batches/00000002.pt.tmp").write_bytes(b"interrupted")
            resumed = run_experiment(config, tmp, model_loader=test_loader)
            self.assertTrue(resumed["generation_complete"])
            self.assertEqual(resumed["nfe_per_image"], 3)
            self.assertEqual(resumed["batched_model_forward_calls"], 9)
            self.assertEqual(resumed["sample_equivalent_model_evaluations"], 15)
            self.assertTrue((Path(tmp) / "run/images/00000000.png").is_file())
            fresh_config = replace(config, output_dir="fresh")
            run_experiment(fresh_config, tmp, model_loader=test_loader)
            for filename in ["00000000.pt", "00000002.pt", "00000004.pt"]:
                left = torch.load(Path(tmp) / "run/batches" / filename, weights_only=True)
                right = torch.load(Path(tmp) / "fresh/batches" / filename, weights_only=True)
                self.assertTrue(torch.equal(left["samples"], right["samples"]))
            # Already finished runs must not load the model again.
            run_experiment(config, tmp, model_loader=lambda *args: self.fail("Unnecessary model load"))

    def test_changed_config_cannot_mix_with_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config()
            run_experiment(config, tmp, max_batches=1, model_loader=test_loader)
            with self.assertRaisesRegex(ValueError, "Config changed"):
                run_experiment(replace(config, num_steps=4), tmp, model_loader=test_loader)

    def test_corrupt_saved_batch_is_not_silently_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config()
            run_experiment(config, tmp, max_batches=1, model_loader=test_loader)
            path = Path(tmp) / "run/batches/00000000.pt"
            shard = torch.load(path, weights_only=True)
            shard["samples"][0, 0, 0, 0] += 0.25
            save_tensor_file(path, shard)
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_experiment(config, tmp, model_loader=test_loader)

    def test_fid_rejects_incomplete_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_experiment(self.config(), tmp, max_batches=1, model_loader=test_loader)
            with self.assertRaisesRegex(ValueError, "Finish generation"):
                evaluate_fid(Path(tmp) / "run", Path(tmp) / "data", Path(tmp) / "cache", device="cpu")

    def test_fid_protocol_and_small_sample_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_experiment(self.config(), tmp, model_loader=test_loader)
            with patch("diffusion_sampling.evaluation.prepare_cifar10", return_value={"split": "train", "count": 50000}), \
                 patch("torch_fidelity.calculate_metrics", return_value={"frechet_inception_distance": 99.0}) as metric:
                with self.assertWarnsRegex(UserWarning, "Small-sample"):
                    report = evaluate_fid(Path(tmp) / "run", Path(tmp) / "data", Path(tmp) / "cache", device="cpu")
            self.assertEqual(metric.call_args.kwargs["input2"], "cifar10-train")
            self.assertFalse(metric.call_args.kwargs["datasets_download"])
            self.assertTrue(report["small_sample_pilot"])
            self.assertFalse(report["eligible_for_baseline_metrics"])
            self.assertTrue(report["visual_review_required"])

    def test_upstream_numpy_stats_cache_can_be_reloaded_on_modern_pytorch(self):
        from torch_fidelity.utils import cache_lookup_one_recompute_on_miss
        with tempfile.TemporaryDirectory() as tmp:
            expected = {"mu": np.zeros(3, dtype=np.float32), "sigma": np.eye(3, dtype=np.float64)}
            with fid_cache_compatibility():
                cache_lookup_one_recompute_on_miss("stats", lambda: expected, cache_root=tmp, cache=True, verbose=False)
                actual = cache_lookup_one_recompute_on_miss(
                    "stats", lambda: self.fail("Expected a cache hit"), cache_root=tmp, cache=True, verbose=False,
                )
            np.testing.assert_array_equal(actual["mu"], expected["mu"])
            np.testing.assert_array_equal(actual["sigma"], expected["sigma"])


if __name__ == "__main__":
    unittest.main()
