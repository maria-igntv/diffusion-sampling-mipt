"""Small, strict JSON configs: no silently ignored experiment parameters."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import math

from .schedules import CUSTOM_SAMPLERS, expected_nfe


@dataclass(frozen=True)
class RunConfig:
    model_id: str = "google/ddpm-cifar10-32"
    revision: str = "267b167"
    sampler: str = "diffusers_ddim"
    num_steps: int = 100
    num_samples: int = 5000
    batch_size: int = 64
    noise_seed: int = 50000
    noise_path: str = "data/noise/fid.pt"
    output_dir: str = "results/ddim_100"
    purpose: str = "baseline"
    device: str = "auto"
    dtype: str = "float32"
    timestep_spacing: str = "leading"
    clip_sample: bool = True
    warmup_batches: int = 1
    cpu_threads: int = 4
    schedule: str = "native"
    timesteps: list[int] | None = None
    rho: float = 7.0

    def __post_init__(self):
        for name in ("num_steps", "num_samples", "batch_size", "cpu_threads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("noise_seed", "warmup_batches"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.noise_seed + self.num_samples >= 2**63:
            raise ValueError("Noise seeds must stay below 2**63")
        if self.num_steps > 1000:
            raise ValueError("This checkpoint has only 1000 training timesteps")
        if self.model_id != "google/ddpm-cifar10-32":
            raise ValueError("This baseline currently supports google/ddpm-cifar10-32 only")
        if not self.revision or self.revision == "main":
            raise ValueError("Pin an immutable checkpoint revision, not main")
        if self.sampler not in {"diffusers_ddim", *CUSTOM_SAMPLERS}:
            raise ValueError("sampler must be diffusers_ddim, ddim, dpmpp_2m or heun")
        if self.schedule not in {"native", "uniform_t", "log_sigma", "karras", "custom"}:
            raise ValueError("Unknown noise schedule")
        if self.sampler == "diffusers_ddim" and self.schedule != "native":
            raise ValueError("The original Diffusers baseline only supports its native schedule")
        if self.sampler == "diffusers_ddim" and self.timestep_spacing != "leading":
            raise ValueError("The verified native DDIM reference uses leading spacing")
        if self.sampler in {"dpmpp_2m", "heun"} and self.schedule == "native":
            raise ValueError("Select a common schedule: uniform_t, log_sigma, karras or custom")
        if self.sampler == "ddim" and self.schedule == "native" and self.timestep_spacing != "leading":
            raise ValueError("Native custom DDIM supports leading spacing only; use a common schedule otherwise")
        if self.sampler in CUSTOM_SAMPLERS and self.dtype != "float32":
            raise ValueError("Custom samplers currently require float32 for verified numerical behavior")
        if type(self.rho) not in (int, float) or not math.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be finite and positive")
        if self.schedule == "custom":
            from .schedules import validate_timesteps
            if not isinstance(self.timesteps, list):
                raise ValueError("Custom schedule requires an explicit timesteps list")
            validate_timesteps(self.timesteps, self.num_steps, 1000)
        elif self.timesteps is not None:
            raise ValueError("timesteps may only be supplied for schedule=custom")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")
        if self.dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be float32 or float16")
        if self.timestep_spacing not in {"leading", "trailing", "linspace"}:
            raise ValueError("Unknown DDIM timestep spacing")
        if type(self.clip_sample) is not bool:
            raise ValueError("clip_sample must be a boolean")
        if self.purpose not in {"pilot", "baseline"}:
            raise ValueError("purpose must be pilot or baseline")
        for name in ("noise_path", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty path string")

    def to_dict(self):
        return asdict(self)

    @property
    def expected_nfe(self):
        return expected_nfe(self.sampler, self.num_steps)


def load_config(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Config must be a JSON object")
    unknown = set(values) - {field.name for field in fields(RunConfig)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return RunConfig(**values)
