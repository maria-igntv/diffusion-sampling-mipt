"""Colab orchestration for references, matched budgets and independent noise banks.

This script is embedded in notebook 04. The archived experiment package supplies
the numerical methods, generation, FID, comparison and portable run exports.
"""

import csv
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from diffusion_sampling.config import RunConfig
from diffusion_sampling.evaluation import evaluate_fid
from diffusion_sampling.experiment import run_experiment
from diffusion_sampling.exports import CheckpointExporter
from diffusion_sampling.noise import ensure_noise, load_noise
from diffusion_sampling.runtime import configure_runtime, environment
from diffusion_sampling.schedules import points_for_budget
from diffusion_sampling.storage import file_sha256, read_json, write_json
from diffusion_sampling.study import _load_run, compare_runs


METHODS = ("ddim", "dpmpp_2m", "heun")
SCHEDULES = ("uniform_t", "log_sigma", "karras")


def task_label(method, schedule, budget):
    return f"{method}_{schedule}_nfe{budget}"


def _task(plan, phase, label, method, schedule, budget, bank_name, count, fid=False):
    bank = plan["banks"][bank_name]
    config = RunConfig(
        sampler=method, schedule=schedule, num_steps=points_for_budget(method, budget),
        num_samples=count, batch_size=plan["batch_size"], device=plan["device"],
        clip_sample=False, noise_path=bank["path"], noise_seed=bank["seed"],
        purpose="baseline" if fid else "pilot",
        output_dir=f"results/{plan['name']}/{phase}/{label}",
    )
    return {"label": label, "config": config.to_dict(), "fid_required": fid}


def prepare_plan(project, name, archived_runs=(), device="cuda", budgets=(19, 39, 79),
                 selection_budget=39, fid_count=5000, mse_count=64, confirm_count=1024,
                 debug_count=16, reference_budgets=(500, 1000), batch_size=64):
    """Create a fixed design, or load the same design after session recovery."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Use letters, numbers, underscores or hyphens for the study name")
    if selection_budget not in budgets:
        raise ValueError("The selection budget must be included in the grid")
    for budget in budgets:
        points_for_budget("heun", budget)
    project = Path(project)
    configure_runtime(4)
    current = environment(torch.device(device))
    design = {"budgets": list(budgets), "selection_budget": selection_budget,
              "fid_count": fid_count, "mse_count": mse_count, "confirm_count": confirm_count,
              "debug_count": debug_count, "reference_budgets": list(reference_budgets),
              "batch_size": batch_size, "device": device}
    path = project / f"results/{name}/extended_plan.json"
    if path.exists():
        plan = read_json(path)
        if plan["design"] != design or plan["environment"] != current:
            raise ValueError("Design or environment changed. Choose a new STUDY_NAME.")
        return plan
    prefix = f"data/noise/{name}"
    plan = {"schema_version": 1, "name": name, "design": design, "environment": current,
            "device": device, "batch_size": batch_size, "selection": [],
            "banks": {
                "debug": {"path": f"{prefix}/debug.pt", "seed": 10000, "count": debug_count},
                "mse64": {"path": f"{prefix}/mse64.pt", "seed": 30000, "count": mse_count},
                "mse1024": {"path": f"{prefix}/mse1024.pt", "seed": 40000, "count": confirm_count},
                "fid": {"path": "data/noise/fid.pt", "seed": 50000, "count": fid_count},
                "fid60000": {"path": f"{prefix}/fid60000.pt", "seed": 60000, "count": fid_count},
                "fid70000": {"path": f"{prefix}/fid70000.pt", "seed": 70000, "count": fid_count},
            }, "phases": {"convergence": [], "mse64": [], "grid": [],
                          "mse1024": [], "independent": []}}
    intervals = sorted((bank["seed"], bank["seed"] + bank["count"]) for bank in plan["banks"].values())
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("Noise-bank seed ranges overlap")
    for bank in plan["banks"].values():
        payload = ensure_noise(project / bank["path"], bank["count"], bank["seed"])
        if payload["count"] != bank["count"]:
            raise ValueError("This study requires the exact saved bank size")
        bank["sha256"] = payload["sha256"]
    for budget in reference_budgets:
        plan["phases"]["convergence"].append(_task(
            plan, "convergence", f"reference_{budget}", "ddim", "uniform_t",
            budget, "debug", debug_count))
    for method in METHODS:
        plan["phases"]["convergence"].append(_task(
            plan, "convergence", task_label(method, "uniform_t", selection_budget),
            method, "uniform_t", selection_budget, "debug", debug_count))
    plan["phases"]["mse64"].append(_task(
        plan, "mse64", "reference", "ddim", "uniform_t", reference_budgets[-1], "mse64", mse_count))
    old_runs = {}
    for directory in archived_runs:
        manifest, summary, _ = _load_run(directory)
        config = RunConfig(**manifest["config"])
        if (manifest["environment"] == current and not config.clip_sample
                and config.schedule == "uniform_t" and config.num_samples == fid_count
                and config.batch_size == batch_size and config.noise_seed == 50000
                and manifest["noise_sha256"] == plan["banks"]["fid"]["sha256"]
                and summary["baseline_metrics_complete"] and config.expected_nfe in budgets):
            old_runs[task_label(config.sampler, config.schedule, config.expected_nfe)] = config.to_dict()
    for method in METHODS:
        for schedule in SCHEDULES:
            for budget in budgets:
                label = task_label(method, schedule, budget)
                plan["phases"]["mse64"].append(_task(
                    plan, "mse64", label, method, schedule, budget, "mse64", mse_count))
                task = _task(plan, "grid", label, method, schedule, budget, "fid", fid_count, fid=True)
                if label in old_runs:
                    task["config"] = old_runs[label]
                    task["reused"] = True
                plan["phases"]["grid"].append(task)
    write_json(path, plan)
    return plan


def _summary(project, plan, task):
    directory = Path(project) / task["config"]["output_dir"]
    if not (directory / "manifest.json").exists():
        return None
    manifest = read_json(directory / "manifest.json")
    if manifest["config"] != task["config"] or manifest["environment"] != plan["environment"]:
        raise ValueError("Saved config or runtime differs. Restore its original environment or use a new study name.")
    return read_json(directory / "summary.json")


def phase_progress(project, plan, phase):
    rows = []
    for task in plan["phases"][phase]:
        summary = _summary(project, plan, task)
        done = bool(summary and summary["generation_complete"]
                    and (not task["fid_required"] or summary["baseline_metrics_complete"]))
        rows.append({"label": task["label"], "done": done,
                     "images": summary["generated_images"] if summary else 0,
                     "requested": task["config"]["num_samples"]})
    return rows


def run_next_block(project, plan, phase, source_bundle, max_batches=16,
                   runner=run_experiment, evaluator=evaluate_fid):
    """Advance one task by bounded batches and publish a full portable snapshot."""
    project = Path(project)
    configure_runtime(4)
    if environment(torch.device(plan["device"])) != plan["environment"]:
        raise ValueError("Runtime changed. Choose a new study name for new measurements.")
    tasks = plan["phases"][phase]
    if not tasks:
        raise ValueError("Run the grid and lock selection before this phase")
    for task, progress in zip(tasks, phase_progress(project, plan, phase)):
        if progress["done"]:
            continue
        config = RunConfig(**task["config"])
        bank = next(bank for bank in plan["banks"].values() if bank["path"] == config.noise_path)
        if load_noise(project / config.noise_path)["sha256"] != bank["sha256"]:
            raise ValueError("Noise bank changed. Restore the exact bank from the support ZIP.")
        print("Task", phase, task["label"], flush=True)
        block_batches = min(max_batches, 1) if config.expected_nfe >= 500 else max_batches
        result = runner(config, root=project, max_batches=block_batches)
        directory = project / config.output_dir
        if result["generation_complete"] and task["fid_required"] and not result["baseline_metrics_complete"]:
            gc.collect()
            if plan["device"] == "cuda":
                torch.cuda.empty_cache()
            evaluator(directory, project / "data/cifar10", project / ".cache/fidelity",
                      device=plan["device"], download=True)
        logs = directory / "logs"
        logs.mkdir(exist_ok=True)
        write_json(logs / "extended_plan.log", plan)
        exporter = CheckpointExporter(project, project / f"results/exports/{plan['name']}",
                                      source_bundle=source_bundle)
        result = read_json(directory / "summary.json")
        stage = "final_fid" if result["baseline_metrics_complete"] else "progress"
        output = exporter.full(directory, stage=f"{phase}_{stage}")
        print("Images", result["generated_images"], "/", config.num_samples, flush=True)
        print("Full checkpoint", output, flush=True)
        return output
    print("Phase complete", phase)
    return None


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mse_details(project, tasks, reference, output):
    _, _, reference_samples = _load_run(reference)
    rows = []
    for task in tasks:
        directory = Path(project) / task["config"]["output_dir"]
        _, _, samples = _load_run(directory)
        errors = (samples - reference_samples).square().flatten(1).mean(1).double()
        rows.extend({"label": task["label"], "image_index": index, "mse": float(error)}
                    for index, error in enumerate(errors))
    _write_csv(output, rows)


def _plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["rows"]
    has_fid = all(row["fid"] is not None for row in rows)
    metric = "fid" if has_fid else "mse_to_reference"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        for schedule in SCHEDULES:
            selected = sorted((row for row in rows if row["sampler"] == method
                               and row["schedule"] == schedule), key=lambda row: row["nfe"])
            if not selected:
                continue
            label = f"{method} / {schedule}"
            axes[0].plot([row["nfe"] for row in selected], [row[metric] for row in selected],
                         marker="o", label=label)
            axes[1].plot([row["seconds_per_image"] for row in selected],
                         [row[metric] for row in selected], marker="o", label=label)
    axes[0].set_xlabel("NFE per image")
    axes[1].set_xlabel("Denoising seconds per image")
    for axis in axes:
        axis.set_ylabel("FID-5k / real-50k" if has_fid else "MSE to DDIM reference")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_phase_report(project, plan, phase):
    project = Path(project)
    progress = phase_progress(project, plan, phase)
    if not progress or not all(row["done"] for row in progress):
        raise ValueError("Finish all tasks in this phase before generating its report")
    directory = project / f"results/{plan['name']}/reports"
    directory.mkdir(parents=True, exist_ok=True)
    tasks = plan["phases"][phase]
    runs = [project / task["config"]["output_dir"] for task in tasks]
    if phase == "convergence":
        low, high = runs[:2]
        compare_runs([low], directory / "reference_convergence.json", reference=high)
        report = compare_runs(runs[2:], directory / "convergence.json", reference=high)
        compare_runs(runs[2:], directory / "convergence_to_500.json", reference=low)
        _mse_details(project, tasks[2:], high, directory / "convergence_per_image.csv")
    elif phase in ("mse64", "mse1024"):
        report = compare_runs(runs[1:], directory / f"{phase}.json", reference=runs[0])
        _mse_details(project, tasks[1:], runs[0], directory / f"{phase}_per_image.csv")
        _plots(report, directory / f"{phase}_curves.png")
    elif phase == "grid":
        report = compare_runs(runs, directory / "grid.json")
        _plots(report, directory / "grid_curves.png")
    else:
        combined = []
        for seed in (60000, 70000):
            selected = [project / task["config"]["output_dir"] for task in tasks
                        if task["config"]["noise_seed"] == seed]
            result = compare_runs(selected, directory / f"independent_{seed}.json")
            combined.extend({**row, "noise_seed": seed} for row in result["rows"])
        aggregate = []
        for method, schedule, budget in sorted({(r["sampler"], r["schedule"], r["nfe"]) for r in combined}):
            values = [r["fid"] for r in combined if (r["sampler"], r["schedule"], r["nfe"]) == (method, schedule, budget)]
            aggregate.append({"sampler": method, "schedule": schedule, "nfe": budget,
                              "banks": len(values), "mean_fid": sum(values) / len(values),
                              "min_fid": min(values), "max_fid": max(values)})
        report = {"rows": combined, "aggregate": aggregate, "selection": plan["selection"]}
        write_json(directory / "independent.json", report)
        _write_csv(directory / "independent.csv", aggregate)
    return report


def lock_selection(project, plan):
    """Choose up to six fixed configs on the primary bank before confirmation."""
    if plan["selection"]:
        return plan
    report = write_phase_report(project, plan, "grid")
    budget = plan["design"]["selection_budget"]
    selected = set()
    for method in METHODS:
        rows = [row for row in report["rows"] if row["sampler"] == method and row["nfe"] == budget]
        best = min(rows, key=lambda row: (row["fid"], row["schedule"]))
        selected.add((method, "uniform_t", budget))
        selected.add((method, best["schedule"], budget))
    plan["selection"] = [task_label(*values) for values in sorted(selected)]
    plan["selection_source_sha256"] = file_sha256(Path(project) / f"results/{plan['name']}/reports/grid.json")
    count = plan["design"]["confirm_count"]
    high = plan["design"]["reference_budgets"][-1]
    plan["phases"]["mse1024"] = [_task(plan, "mse1024", "reference", "ddim", "uniform_t", high, "mse1024", count)]
    for method, schedule, budget in sorted(selected):
        label = task_label(method, schedule, budget)
        plan["phases"]["mse1024"].append(_task(plan, "mse1024", label,
                                              method, schedule, budget, "mse1024", count))
        for seed in (60000, 70000):
            plan["phases"]["independent"].append(_task(
                plan, "independent", f"{label}_seed{seed}", method, schedule,
                budget, f"fid{seed}", plan["design"]["fid_count"], fid=True))
    write_json(Path(project) / f"results/{plan['name']}/extended_plan.json", plan)
    return plan


def support_backup(project, plan, source_bundle):
    """Save the fixed design and all exact noise banks for another Colab session."""
    project = Path(project)
    directory = project / f"results/exports/{plan['name']}"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{plan['name']}_support_{uuid4().hex[:8]}.zip"
    hashes = {}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        data = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode()
        archive.writestr("extended_plan.json", data)
        hashes["extended_plan.json"] = hashlib.sha256(data).hexdigest()
        for bank in plan["banks"].values():
            payload = load_noise(project / bank["path"])
            if payload["sha256"] != bank["sha256"]:
                raise ValueError("Noise bank changed")
            data = (project / bank["path"]).read_bytes()
            archive.writestr(bank["path"], data)
            hashes[bank["path"]] = hashlib.sha256(data).hexdigest()
        data = Path(source_bundle).read_bytes()
        archive.writestr("source.zip", data)
        hashes["source.zip"] = hashlib.sha256(data).hexdigest()
        archive.writestr("support_manifest.json", json.dumps({"files_sha256": hashes}))
    return output


def restore_support(path, project, current_environment):
    project = Path(project)
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("support_manifest.json"))
        hashes = metadata["files_sha256"]
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(hashes) | {"support_manifest.json"}:
            raise ValueError("Support ZIP entries differ from its manifest")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Support checksum mismatch: {name}")
        plan = json.loads(archive.read("extended_plan.json"))
        if plan["environment"] != current_environment:
            raise ValueError("Support environment differs. Create a new study name for this runtime.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", plan["name"]):
            raise ValueError("Unsafe study name")
        bank_paths = {bank["path"] for bank in plan["banks"].values()}
        for tasks in plan["phases"].values():
            for task in tasks:
                config = RunConfig(**task["config"])
                output = project / config.output_dir
                if (not output.resolve().is_relative_to(project.resolve())
                        or not config.output_dir.startswith("results/")
                        or config.noise_path not in bank_paths):
                    raise ValueError("Unsafe task path in support plan")
        for bank in plan["banks"].values():
            destination = project / bank["path"]
            if not destination.resolve().is_relative_to(project.resolve()) or not bank["path"].startswith("data/"):
                raise ValueError("Unsafe noise bank path")
            data = archive.read(bank["path"])
            incoming = load_noise(BytesIO(data))
            if incoming["sha256"] != bank["sha256"] or incoming["count"] != bank["count"] or incoming["seed"] != bank["seed"]:
                raise ValueError("Saved noise differs from the support plan")
            if destination.exists() and destination.read_bytes() != data:
                existing = load_noise(destination)
                if existing["sha256"] != bank["sha256"]:
                    raise ValueError("An existing noise bank differs")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        write_json(project / f"results/{plan['name']}/extended_plan.json", plan)
    return plan


def report_backup(project, plan, source_bundle):
    project = Path(project)
    directory = project / f"results/exports/{plan['name']}"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{plan['name']}_reports_{uuid4().hex[:8]}.zip"
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        state = project / f"results/{plan['name']}/extended_plan.json"
        archive.write(state, "extended_plan.json")
        reports = project / f"results/{plan['name']}/reports"
        for path in sorted(reports.glob('*')):
            if path.is_file():
                archive.write(path, Path("reports") / path.name)
        archive.write(source_bundle, "source.zip")
    return output
