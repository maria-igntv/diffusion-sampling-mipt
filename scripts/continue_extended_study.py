"""Paste into the already initialized notebook 04; no uploads or setup required."""

from pathlib import Path
from time import monotonic


def continue_study(through="independent", max_batches=16, max_minutes=None):
    """Resume existing runs, complete dependencies and export reports automatically.

    A time limit is checked between blocks; a running batch/FID is not killed.
    Downloads belong in a separate idle-kernel cell, not in this loop.
    """
    global plan, latest_checkpoint, support_zip, reports_zip

    phases = ("convergence", "mse64", "grid", "mse1024", "independent")
    if through not in phases:
        raise ValueError(f"through must be one of {phases}")
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError("max_batches must be a positive integer")
    if max_minutes is not None and max_minutes <= 0:
        raise ValueError("max_minutes must be positive or None")
    if not Path(PROJECT).is_dir() or not Path(SOURCE_BUNDLE).is_file():
        raise RuntimeError("The Colab runtime no longer contains the restored project/source ZIP")

    started = monotonic()
    status = "complete"
    current_phase = None
    try:
        for phase in phases[:phases.index(through) + 1]:
            current_phase = phase
            if phase == "mse1024" and not plan.get("selection"):
                plan = lock_selection(PROJECT, plan)
                support_zip = support_backup(PROJECT, plan, SOURCE_BUNDLE)
                print("Выбор зафиксирован; support ZIP:", support_zip, flush=True)

            progress = phase_progress(PROJECT, plan, phase)
            if not progress:
                raise RuntimeError(f"Фаза {phase} не содержит задач")
            print(f"\n{phase}: готово {sum(r['done'] for r in progress)}/{len(progress)}", flush=True)

            while not all(row["done"] for row in progress):
                if max_minutes is not None and monotonic() - started >= max_minutes * 60:
                    status = "time_limit"
                    print("Лимит времени достигнут. Повторный вызов продолжит с сохранённого места.", flush=True)
                    return {"status": status, "phase": phase}

                before = [(r["label"], r["images"], r["done"]) for r in progress]
                latest_checkpoint = run_next_block(
                    PROJECT, plan, phase, SOURCE_BUNDLE, max_batches=max_batches,
                )
                progress = phase_progress(PROJECT, plan, phase)
                after = [(r["label"], r["images"], r["done"]) for r in progress]
                if before == after:
                    raise RuntimeError(f"Нет прогресса в {phase}; выполнение остановлено")
                print(f"{phase}: готово {sum(r['done'] for r in progress)}/{len(progress)}", flush=True)

            write_phase_report(PROJECT, plan, phase)
            reports_zip = report_backup(PROJECT, plan, SOURCE_BUNDLE)
            print("Фаза завершена; reports ZIP:", reports_zip, flush=True)

        print("Все фазы до", through, "включительно завершены.", flush=True)
        return {"status": status, "phase": through}
    except KeyboardInterrupt:
        print("Остановлено. Готовые батчи сохранены; повторный вызов продолжит работу.", flush=True)
        return {"status": "interrupted", "phase": current_phase}
