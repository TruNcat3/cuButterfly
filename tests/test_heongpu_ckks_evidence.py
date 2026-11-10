import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/workloads" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("run_heongpu_ckks_e2e", "run_heongpu_ckks_e2e.py")
summary = load_module("summarize_heongpu_ckks_e2e", "summarize_heongpu_ckks_e2e.py")
GPU = "GPU-00000000-0000-0000-0000-000000000003"
OTHER_GPU = "GPU-00000000-0000-0000-0000-000000000004"


def probe(stdout="", returncode=0):
    return {"returncode": returncode, "stdout": stdout, "stderr": ""}


def row(uuid=GPU, pid=999):
    return f"{uuid}, {pid}, foreign-job, 1024\n"


class FakeProcess:
    pid = 123

    def __init__(self, blocking=False):
        self.returncode = None
        self.blocking = blocking
        self.stopped = threading.Event()
        self.terminate = Mock(side_effect=self._terminate)
        self.kill = Mock(side_effect=self._kill)
        self.stdout = self

    def read(self, count):
        if self.blocking and not self.stopped.wait(3):
            raise RuntimeError("monitor failed to stop its child")
        return b""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def _terminate(self):
        self.returncode = -15
        self.stopped.set()

    def _kill(self):
        self.returncode = -9
        self.stopped.set()


class EvidenceTests(unittest.TestCase):
    def test_preparation_handoff_rejects_failed_changed_and_foreign_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bench"
            binary.write_text("compiled fixture")
            library = root / "libcuntt.so"
            library.write_text("library fixture")
            path = root / "preparation.json"
            command = [str(binary), "--sizes", "32768", "--stream-mode", "nonblocking", "--graph"]
            env = {"CUDA_VISIBLE_DEVICES": GPU, "CUBUTTERFLY_RNS_CACHE": str(root / "cache"),
                   "CUBUTTERFLY_RNS_TUNE_REPETITIONS": "12"}
            meta = {"status": "completed", "returncode": 0, "measurement_kind": "preparation-diagnostic",
                    "benchmark_argv": [*command[:-1], "--prepare-only", "--prepare-backend", "both"],
                    "CUDA_VISIBLE_DEVICES": GPU, "rns_preparation": env,
                    "artifacts": {"benchmark_binary": runner.file_identity(binary),
                                  "libcuntt": runner.file_identity(library)}}
            path.write_text(json.dumps(meta))
            runner.await_preparation(path, command, env, 1)
            for changes in ({"status": "failed", "returncode": 2}, {"CUDA_VISIBLE_DEVICES": OTHER_GPU},
                            {"measurement_kind": "ordinary"}):
                path.write_text(json.dumps({**meta, **changes}))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    runner.await_preparation(path, command, env, 1)
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "sizes"):
                runner.await_preparation(path, [x if x != "32768" else "65536" for x in command], env, 1)
            binary.write_text("different executable")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                runner.await_preparation(path, command, env, 1)

    def execute(self, observations, process=None, visible=GPU, diagnostic=False, preparation=False):
        process = process or FakeProcess()
        remaining = iter(observations)
        last = observations[-1]
        calls = []

        def read_probe(command):
            calls.append(command)
            if any(option.startswith("--query-compute-apps=") for option in command):
                return next(remaining, last)
            return probe("identity")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.csv"
            with patch.object(runner, "read_only_probe", side_effect=read_probe), \
                    patch.object(runner, "file_identity", return_value={}), \
                    patch.object(runner, "GPU_MONITOR_INTERVAL", 0.001), \
                    patch.object(runner.subprocess, "Popen", return_value=process) as launch:
                result = runner.run_with_evidence(
                    ["benchmark", *(["--trace-measurement"] if diagnostic else []),
                     *(["--prepare-only"] if preparation else [])], {"CUDA_VISIBLE_DEVICES": visible,
                                    "CUBUTTERFLY_COMPILE_MODE": "research"},
                    output, Path(directory), Path(directory) / "libcuntt.so",
                    Path(directory) / "linked.a", Path(directory) / "original.a", True)
                metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            return result, metadata, launch, process, calls

    def test_clean_success(self):
        result, meta, launch, process, calls = self.execute([probe(), probe()])
        self.assertEqual(result, 0)
        self.assertEqual(meta["status"], "completed")
        self.assertTrue(meta["performance_eligible"])
        self.assertIn("cannot detect every transient", meta["monitor_scope"]["coverage"])
        self.assertEqual(meta["monitor_scope"]["target_gpu_uuid"], GPU)
        launch.assert_called_once()
        process.terminate.assert_not_called()
        for command in calls:
            if command[0] == "nvidia-smi":
                self.assertIn(f"--id={GPU}", command)

    def test_other_gpu_and_own_pid_are_ignored(self):
        result, meta, _, process, _ = self.execute([
            probe(row(OTHER_GPU)), probe(row(OTHER_GPU) + row(pid=FakeProcess.pid))])
        self.assertEqual(result, 0)
        self.assertTrue(meta["performance_eligible"])
        process.terminate.assert_not_called()

    def test_successful_profile_is_not_performance_evidence(self):
        result, meta, _, _, _ = self.execute([probe(), probe()], diagnostic=True)
        self.assertEqual(result, 0)
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["measurement_kind"], "profiler-diagnostic")
        self.assertFalse(meta["performance_eligible"])
        with self.assertRaisesRegex(ValueError, "ineligible"):
            summary.require_performance_evidence(meta, "trace.csv")

    def test_busy_target_never_launches(self):
        result, meta, launch, _, _ = self.execute([probe(row())])
        self.assertNotEqual(result, 0)
        self.assertEqual(meta["status"], "blocked_gpu_busy")
        self.assertFalse(meta["performance_eligible"])
        launch.assert_not_called()

    def test_preparation_has_no_performance_claim(self):
        result, meta, _, _, _ = self.execute([probe(), probe()], preparation=True)
        self.assertEqual(result, 0)
        self.assertEqual(meta["measurement_kind"], "preparation-diagnostic")
        self.assertFalse(meta["performance_eligible"])
        with self.assertRaisesRegex(ValueError, "ineligible"):
            summary.require_performance_evidence(meta, "prepare.csv")

    def test_foreign_process_during_run_stops_only_child(self):
        child = FakeProcess(blocking=True)
        result, meta, launch, process, _ = self.execute([
            probe(), probe(row(pid=child.pid) + row() + row(OTHER_GPU)), probe()], child)
        self.assertNotEqual(result, 0)
        self.assertEqual(meta["status"], "invalid_gpu_contention")
        self.assertFalse(meta["performance_eligible"])
        self.assertEqual(meta["child_stop"]["pid"], child.pid)
        self.assertEqual([p["pid"] for p in meta["gpu_violation"]["foreign_processes"]], [999])
        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        launch.assert_called_once()

    def test_preflight_query_failure_never_launches(self):
        result, meta, launch, _, _ = self.execute([probe(returncode=1)])
        self.assertNotEqual(result, 0)
        self.assertEqual(meta["status"], "failed")
        self.assertFalse(meta["performance_eligible"])
        launch.assert_not_called()

    def test_monitor_query_failure_stops_child(self):
        result, meta, _, process, _ = self.execute(
            [probe(), probe(returncode=1), probe()], FakeProcess(blocking=True))
        self.assertNotEqual(result, 0)
        self.assertEqual(meta["status"], "failed")
        self.assertFalse(meta["performance_eligible"])
        process.terminate.assert_called_once()

    def test_final_query_failure_rejects_completed_child(self):
        result, meta, _, _, _ = self.execute([probe(), probe(returncode=1)])
        self.assertNotEqual(result, 0)
        self.assertEqual(meta["status"], "failed")
        self.assertFalse(meta["performance_eligible"])

    def test_explicit_single_uuid_required(self):
        for visible in ("", "0", "0,1", GPU + "," + OTHER_GPU, "GPU-1111"):
            with self.subTest(visible=visible):
                result, meta, launch, _, _ = self.execute([probe()], visible=visible)
                self.assertNotEqual(result, 0)
                self.assertEqual(meta["status"], "failed")
                self.assertFalse(meta["performance_eligible"])
                launch.assert_not_called()

    def test_stop_child_escalates_only_after_timeout(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("benchmark", 5), -9]
        action = runner.stop_own_process(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(action, {"pid": 123, "terminate_requested": True, "kill_requested": True})

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.csv"
            output.write_text("prior evidence")
            with patch.object(runner.subprocess, "Popen") as launch:
                with self.assertRaises(FileExistsError):
                    runner.run_with_evidence([], {}, output, output, output, output, output, True)
                launch.assert_not_called()
            self.assertEqual(output.read_text(), "prior evidence")

    def test_existing_selection_sidecars_are_not_overwritten(self):
        for suffix in (".selection.csv", ".samples.csv"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run.csv"
                prior = Path(str(output) + suffix)
                prior.write_text("prior measurements")
                with patch.object(runner.subprocess, "Popen") as launch:
                    with self.assertRaises(FileExistsError):
                        runner.run_with_evidence([], {}, output, output, output, output, output, True)
                    launch.assert_not_called()
                self.assertEqual(prior.read_text(), "prior measurements")


class SummaryEvidenceTests(unittest.TestCase):
    def metadata(self, before="", after=""):
        return {"status": "completed", "returncode": 0, "CUDA_VISIBLE_DEVICES": GPU,
                "compute_processes_before": probe(before), "compute_processes_after": probe(after)}

    def test_new_ineligible_record_is_rejected(self):
        meta = self.metadata()
        meta["performance_eligible"] = False
        with self.assertRaisesRegex(ValueError, "ineligible"):
            summary.require_performance_evidence(meta, "new.csv")

    def test_legacy_other_gpu_and_own_pid_pass(self):
        meta = self.metadata(row(OTHER_GPU), row(OTHER_GPU) + row(pid=123))
        meta["benchmark_pid"] = 123
        summary.require_performance_evidence(meta, "legacy.csv")

    def test_legacy_target_contention_rejected_with_pid(self):
        for boundary in ("compute_processes_before", "compute_processes_after"):
            with self.subTest(boundary=boundary):
                meta = self.metadata()
                meta[boundary] = probe(row(pid=264824))
                with self.assertRaisesRegex(ValueError, "264824"):
                    summary.require_performance_evidence(meta, "legacy.csv")

    def test_legacy_failed_or_missing_queries_rejected(self):
        for invalid in ({}, probe(returncode=1), {"returncode": 0}):
            with self.subTest(invalid=invalid):
                meta = self.metadata()
                meta["compute_processes_before"] = invalid
                with self.assertRaisesRegex(ValueError, "query is missing or failed"):
                    summary.require_performance_evidence(meta, "legacy.csv")


if __name__ == "__main__":
    unittest.main()
