import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import fft


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_public_radix4_pair_experiment.py"
SPEC = importlib.util.spec_from_file_location("public_radix4_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PublicRadix4RunnerTests(unittest.TestCase):
    def test_legacy_matrix_unchanged(self):
        cases = runner.cases()
        self.assertEqual(len(cases), 8)
        self.assertEqual(cases[0]["name"], "rank2_b1_forward")
        self.assertEqual([(c["shape"], c["batch"]) for c in cases[::2]],
                         [([32, 32], 1), ([32, 32], 64), ([32, 32], 256), ([8, 32, 32], 2)])
        self.assertTrue(all(c["precision"] == "fp32" and c["fft_type"] == "c2c" for c in cases))

    def test_bounded_matrix_and_unique_identity(self):
        cases = runner.cases(("fp32", "fp64"), "all")
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({c["name"] for c in cases}), len(cases))
        real = runner.cases(("fp32", "fp64"), "compact-real")
        self.assertEqual(len(real), 8)
        self.assertEqual({tuple(c["shape"]) for c in real}, {(32, 32, 64)})
        self.assertEqual({c["batch"] for c in real}, {1, 4})
        self.assertEqual({c["fft_type"] for c in real}, {"r2c", "c2r"})

    def test_same_precision_and_legal_hermitian(self):
        for case in runner.cases(("fp32", "fp64"), "all"):
            with self.subTest(case=case["name"]):
                data, reference = runner.cpu_arrays(case, 42)
                real = np.dtype("float64" if case["precision"] == "fp64" else "float32")
                complex_ = np.dtype("complex128" if case["precision"] == "fp64" else "complex64")
                self.assertEqual(data.dtype, real if case["fft_type"] == "r2c" else complex_)
                self.assertEqual(reference.dtype, real if case["fft_type"] == "c2r" else complex_)
                self.assertTrue(data.flags.c_contiguous and reference.flags.c_contiguous)
                self.assertTrue(np.isfinite(data).all() and np.isfinite(reference).all())
                if case["fft_type"] != "c2c":
                    full = (case["batch"], 32, 32, 64)
                    half = (case["batch"], 32, 32, 33)
                    self.assertEqual(data.shape, full if case["fft_type"] == "r2c" else half)
                    self.assertEqual(reference.shape, half if case["fft_type"] == "r2c" else full)
                if case["fft_type"] == "c2r":
                    # Identical seed recreates the exact real source used to
                    # generate the input, including both Hermitian endpoint planes.
                    source = np.random.default_rng(42).uniform(-1, 1, reference.shape).astype(real)
                    expected = fft.rfftn(source, axes=(1, 2, 3), workers=1, norm="backward")
                    np.testing.assert_array_equal(data, expected)

    def test_provider_families_do_not_substitute(self):
        for case in runner.cases(("fp32", "fp64"), "compact-real"):
            self.assertIn("torch_cufft", runner.providers_for(case))
            self.assertIn("cufft_whole_external", runner.providers_for(case))
            self.assertEqual(runner.family_for("cufft_whole_external", case), "xd:cufft-whole-real")
            self.assertEqual(runner.family_for("radix4", case), "xd:pair-tiled-radix4")
        self.assertNotIn("cufft_whole_external", runner.providers_for(runner.cases()[-1]))

    def test_cpu_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared, boundary = runner.prepare_cpu(Path(temporary), ("fp64",), "compact-real")
            self.assertEqual(len(prepared), 4)
            self.assertEqual(boundary[1].dtype, np.complex128)
            for case, data, reference in prepared:
                with np.load(Path(temporary) / (case["name"] + ".cpu.npz")) as saved:
                    np.testing.assert_array_equal(saved["input"], data)
                    np.testing.assert_array_equal(saved["cpu_reference"], reference)

    def test_strict_cli_parameters(self):
        base = [str(SCRIPT), "--library", "unused", "--gpu", "GPU-unused", "--output", "unused"]
        for extra in [["--precisions", "fp16"], ["--precisions", "fp32", "fp32"],
                      ["--suite", "unknown"]]:
            with self.subTest(extra=extra), mock.patch("sys.argv", base + extra):
                with self.assertRaises(SystemExit) as raised:
                    runner.main()
                self.assertEqual(raised.exception.code, 2)

    def test_prepare_only_never_queries_gpu(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            argv = [str(SCRIPT), "--library", str(SCRIPT), "--gpu", "GPU-unused",
                    "--output", str(output), "--suite", "compact-real",
                    "--precisions", "fp32", "fp64", "--prepare-only"]
            with mock.patch("sys.argv", argv), mock.patch.object(runner, "gpu_state") as gpu, \
                    mock.patch.object(runner, "run_gpu") as execute:
                self.assertEqual(runner.main(), 0)
                gpu.assert_not_called()
                execute.assert_not_called()
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["status"], "cpu_prepared")
            self.assertEqual(len(report["prepared_cases"]), 8)
            self.assertTrue(all(len(c["cpu_artifact_sha256"]) == 64 for c in report["prepared_cases"]))

    def test_errors_are_report_only(self):
        metrics, absolute = runner.errors(np.array([100.0], dtype=np.float32),
                                          np.array([1.0], dtype=np.float32))
        self.assertEqual(metrics["nonfinite"], 0)
        self.assertEqual(metrics["absolute_error_quantiles"]["max"], 99.0)
        np.testing.assert_array_equal(absolute, [99.0])


if __name__ == "__main__":
    unittest.main()
