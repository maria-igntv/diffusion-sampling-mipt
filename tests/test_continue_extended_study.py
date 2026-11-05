"""Orchestration tests; no model, dataset, CUDA or downloads are required."""

from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/continue_extended_study.py"


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source_zip = root / "source.zip"
        source_zip.touch()
        self.rows = {name: [{"label": name, "images": 0, "done": False}]
                     for name in ("convergence", "mse64", "grid", "mse1024", "independent")}
        self.rows["convergence"][0].update(images=2, done=True)
        self.events = []
        self.ns = {"PROJECT": root, "SOURCE_BUNDLE": source_zip,
                   "plan": {"name": "fixture", "selection": []}}
        exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), self.ns)

        def progress(project, plan, phase):
            if phase in ("mse1024", "independent"):
                self.assertTrue(plan["selection"], "selection must precede confirmation")
            return [dict(row) for row in self.rows[phase]]

        def advance(project, plan, phase, source, max_batches):
            self.events.append(("advance", phase))
            row = self.rows[phase][0]
            row["images"] += 1
            row["done"] = row["images"] >= 2
            return root / f"{phase}.zip"

        def report(project, plan, phase):
            self.assertTrue(all(row["done"] for row in self.rows[phase]))
            self.events.append(("report", phase))

        def select(project, plan):
            self.assertTrue(self.rows["grid"][0]["done"])
            self.events.append(("selection", "grid"))
            return {**plan, "selection": ["fixed"]}

        self.ns.update(phase_progress=progress, run_next_block=advance,
                       write_phase_report=report, lock_selection=select,
                       support_backup=lambda *args: root / "support.zip",
                       report_backup=lambda *args: root / "reports.zip")

    def test_all_dependencies_run_and_completed_tasks_are_skipped(self):
        result = self.ns["continue_study"]()
        self.assertEqual(result, {"status": "complete", "phase": "independent"})
        self.assertNotIn(("advance", "convergence"), self.events)
        self.assertLess(self.events.index(("report", "grid")),
                        self.events.index(("selection", "grid")))
        self.assertEqual(len([e for e in self.events if e[0] == "advance"]), 8)
        self.events.clear()
        self.ns["continue_study"]()
        self.assertFalse(any(e[0] in ("advance", "selection") for e in self.events))

    def test_stop_after_mse64_does_not_start_grid(self):
        self.ns["continue_study"](through="mse64")
        self.assertFalse(any(phase == "grid" for _, phase in self.events))

    def test_no_progress_is_an_error(self):
        self.ns["run_next_block"] = lambda *args, **kwargs: None
        with self.assertRaisesRegex(RuntimeError, "Нет прогресса"):
            self.ns["continue_study"](through="mse64")

    def test_interrupt_preserves_batches_and_resume(self):
        advance = self.ns["run_next_block"]

        def interrupt(*args, **kwargs):
            advance(*args, **kwargs)
            raise KeyboardInterrupt

        self.ns["run_next_block"] = interrupt
        result = self.ns["continue_study"](through="mse64")
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(self.rows["mse64"][0]["images"], 1)
        self.ns["run_next_block"] = advance
        self.ns["continue_study"](through="mse64")
        self.assertEqual(self.rows["mse64"][0]["images"], 2)

    def test_time_limit_stops_at_block_boundary(self):
        times = iter([0, 61])
        self.ns["monotonic"] = lambda: next(times)
        result = self.ns["continue_study"](through="mse64", max_minutes=1)
        self.assertEqual(result["status"], "time_limit")
        self.assertFalse(any(e[0] == "advance" for e in self.events))


if __name__ == "__main__":
    unittest.main()
