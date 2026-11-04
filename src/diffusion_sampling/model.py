"""Load the frozen denoiser and its training coefficients, plus an explicit sampling plan."""

from pathlib import Path
import json

from diffusers import DDIMScheduler, UNet2DModel
from huggingface_hub import snapshot_download
import torch

from .storage import file_sha256
from .schedules import make_plan


def load_model(config, root, device, local_files_only=False):
    snapshot = Path(snapshot_download(
        repo_id=config.model_id, revision=config.revision,
        cache_dir=str(Path(root) / ".cache" / "huggingface"),
        allow_patterns=["config.json", "scheduler_config.json", "diffusion_pytorch_model.safetensors"],
        local_files_only=local_files_only,
    ))
    dtype = getattr(torch, config.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("Use float32 for the CPU baseline")
    model = UNet2DModel.from_pretrained(str(snapshot), use_safetensors=True, torch_dtype=dtype)
    model.eval().requires_grad_(False).to(device)
    scheduler = DDIMScheduler.from_pretrained(
        str(snapshot), prediction_type="epsilon", clip_sample=config.clip_sample,
        timestep_spacing=config.timestep_spacing, set_alpha_to_one=True,
        thresholding=False, rescale_betas_zero_snr=False,
    )
    if model.config.sample_size != 32 or model.config.in_channels != 3 or model.config.out_channels != 3:
        raise ValueError("Unexpected checkpoint input/output contract")
    contract = {
        "model_id": config.model_id, "requested_revision": config.revision,
        "resolved_revision": snapshot.name, "architecture": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "prediction_type": "epsilon", "unconditional": True,
        "image_shape": [3, 32, 32], "clean_image_range": [-1, 1],
        "noisy_input_range": "unbounded", "epsilon_prediction_range": "unbounded",
        "training_timesteps": scheduler.config.num_train_timesteps,
        "scheduler": json.loads(scheduler.to_json_string()),
        "weights_sha256": file_sha256(snapshot / "diffusion_pytorch_model.safetensors"),
        "sampling_plan": make_plan(config, scheduler).to_dict(),
    }
    return model, scheduler, contract
