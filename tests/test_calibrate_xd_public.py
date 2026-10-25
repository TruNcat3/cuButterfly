import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/calibrate_xd_public.py"
SPEC = importlib.util.spec_from_file_location("calibrate_xd_public", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def plans(source):
    return [{"direction": direction, "transform": transform, "selection_source": source,
             "mapping": {"kind": "xd", "physical": {"axis": 1}, "contract": {"batch": 8}}}
            for direction, transform in (("forward", "r2c"), ("inverse", "c2r"))]


class CalibrationTests(unittest.TestCase):
    def test_gate_requires_both_cache_hits(self):
        for index in (0, 1):
            replay = plans("user_cache")
            replay[index]["selection_source"] = "static_model"
            with self.assertRaisesRegex(ValueError, "user_cache"):
                runner.cache_gate(plans("runtime_measured"), replay)

    def test_gate_identity_and_direction_order(self):
        before, after = plans("runtime_measured"), plans("user_cache")
        self.assertTrue(runner.cache_gate(before, after[::-1])["passed"])
        after[1]["mapping"]["physical"]["axis"] = 2
        with self.assertRaisesRegex(ValueError, "mismatch"):
            runner.cache_gate(before, after)
        after = plans("user_cache")
        after[0]["mapping"]["contract"]["batch"] = 16
        with self.assertRaisesRegex(ValueError, "mismatch"):
            runner.cache_gate(before, after)

    def test_missing_physical_and_duplicate_direction_rejected(self):
        records = plans("user_cache")
        del records[0]["mapping"]["physical"]
        with self.assertRaisesRegex(ValueError, "physical"):
            runner.cache_gate(plans("runtime_measured"), records)
        with self.assertRaisesRegex(ValueError, "exactly"):
            runner.cache_gate(plans("runtime_measured"), [plans("user_cache")[0]] * 2)

    def test_exclusive_guard_ignores_own_live_context_only(self):
        with patch.object(runner.subprocess, "check_output", return_value="GPU-test, 123, python\n"):
            runner.exclusive_gpu("GPU-test", own_pid=123)
            with self.assertRaisesRegex(RuntimeError, "other compute"):
                runner.exclusive_gpu("GPU-test", own_pid=124)

    def test_library_identity_blocks_changed_library(self):
        with patch.object(runner, "sha256", return_value="same"):
            runner.require_library_identity(Path("library"), "same")
            with self.assertRaisesRegex(RuntimeError, "library changed"):
                runner.require_library_identity(Path("library"), "different")

    def test_checkpoint_and_dry_run_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            stream = StringIO()
            with redirect_stdout(stream), patch.object(runner, "exclusive_gpu", side_effect=AssertionError("GPU touched")):
                self.assertEqual(runner.main(["--library", "/not-built.so", "--output", str(output),
                                              "--gpu-uuid", "GPU-test", "--dry-run"]), 0)
            document = json.loads(stream.getvalue())
            self.assertEqual(len(document["cases"]), 3)
            self.assertIsNone(document["protocol"]["float_pass_tolerance"])
            self.assertFalse(output.exists())
            runner.checkpoint(output, document)
            self.assertEqual(json.loads(output.read_text()), document)
            with self.assertRaises(SystemExit):
                runner.main(["--library", "/not-built.so", "--output", str(output),
                             "--gpu-uuid", "GPU-test", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
