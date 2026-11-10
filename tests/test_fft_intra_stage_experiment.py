import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_fft_intra_stage_experiment.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("fft_intra_stage_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
UUID = "GPU-00000000-0000-0000-0000-000000000007"


class IntraStageTests(unittest.TestCase):
    def test_runner_only_revision_reuses_artifacts_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module = directory / "module.so"
            module.write_bytes(b"CPU fixture")
            old = directory / "old"
            new = directory / "new"
            args = ["--library", str(SCRIPT), "--gpu", UUID, "--prepare-only"]
            with mock.patch.object(runner, "cpu_arrays", return_value=(np.ones(4), np.ones(4))), \
                    mock.patch.object(runner, "compile_mapping", return_value=module):
                self.assertEqual(runner.main(args + ["--output", str(old)]), 0)
            original = json.loads((old / "summary.json").read_text())
            original["identity"]["sources"]["scripts/run_fft_intra_stage_experiment.py"] = "old-runner-hash"
            original["identity_sha256"] = hashlib.sha256(json.dumps(original["identity"], sort_keys=True).encode()).hexdigest()
            runner.save(old / "summary.json", original)
            before = (old / "summary.json").read_bytes()
            with mock.patch.object(runner, "cpu_arrays") as cpu, \
                    mock.patch.object(runner, "compile_mapping") as compiler, \
                    mock.patch.object(runner, "gpu_state") as gpu:
                self.assertEqual(runner.main(args + ["--output", str(new), "--reuse-preparation-from", str(old)]), 0)
                cpu.assert_not_called()
                compiler.assert_not_called()
                gpu.assert_not_called()
            self.assertEqual(before, (old / "summary.json").read_bytes())
            reused = json.loads((new / "summary.json").read_text())
            self.assertEqual(reused["preparation_reused_from"]["identity"], original["identity"])
            self.assertEqual(reused["prepared_cases"][0]["artifact"], original["prepared_cases"][0]["artifact"])
            original["cases"] = [dict(name="measured-prior")]
            runner.save(old / "summary.json", original)
            with self.assertRaisesRegex(RuntimeError, "no prior measurements"):
                runner.main(args + ["--output", str(directory / "not-created"), "--reuse-preparation-from", str(old)])
            self.assertFalse((directory / "not-created").exists())

    def test_prefetch_shared_is_bounded_by_local_group_not_workload(self):
        case = runner.cases()[-1]
        mapping = runner.mappings_for(case)["factor_t4_p2"]
        limit = 166912
        rejection = runner.resource_rejection(case, mapping, limit)
        self.assertEqual(rejection["status"], "infeasible_resource")
        self.assertEqual(rejection["resource_requirement"]["max_dynamic_shared_lower_bound_bytes"], 256 * 1024)
        self.assertNotIn("mapping", rejection)
        self.assertEqual(rejection["trial_us"], [])
        single = dict(mapping, prefetch_depth=1)
        self.assertEqual(runner.resource_lower_bounds(case, single)["max_dynamic_shared_lower_bound_bytes"], 192 * 1024)
        self.assertIsNone(runner.resource_rejection(case, single, limit))
        smaller = dict(mapping, factor_columns=1)
        bound = runner.resource_lower_bounds(case, smaller)
        self.assertEqual(bound["max_dynamic_shared_lower_bound_bytes"], 48 * 1024)
        self.assertEqual(bound, runner.resource_lower_bounds(dict(case, batch=65536, logN=24), smaller))
        providers = {name: dict(mapping=point) for name, point in runner.mappings_for(case).items()}
        providers["factor_t4_p2"] = rejection
        runner.check_factor_differences(providers)

    def test_exact_bounded_matrix_and_factor_controls(self):
        cases = runner.cases()
        self.assertEqual(len(cases), 4)
        self.assertEqual({(case["logN"], case["batch"], case["precision"]) for case in cases},
                         {(n, 16, p) for n in (16, 20) for p in ("fp32", "fp64")})
        self.assertTrue(all(not case["inverse"] and not case["normalize_inverse"] for case in cases))
        self.assertTrue(all(case["compute_family"] == "fft-h" and
                            case["storage_precision"] == case["compute_precision"] == case["precision"]
                            for case in cases))
        self.assertEqual(sum(len(runner.mappings_for(case)) for case in cases), 24)
        for case in cases:
            mappings = runner.mappings_for(case)
            self.assertEqual(tuple(mappings), runner.PROVIDERS)
            self.assertEqual(mappings["register_radix4"], runner.register_mapping(case, "radix4"))
            for provider, (tiles, depth) in zip(runner.PROVIDERS[1:-1], runner.FACTOR_POINTS):
                point = mappings[provider]
                self.assertEqual(point["factor_partition"], [case["logN"] - 10, 10])
                self.assertEqual((point["factor_columns"], point["factor_ept"]),
                                 (4, 8 if case["logN"] == 16 else 16))
                self.assertEqual([(1 << size) // point["factor_ept"] * point["factor_columns"]
                                  for size in point["factor_partition"]],
                                 [32, 512] if case["logN"] == 16 else [256, 256])
                self.assertEqual((point["data_tiles_per_cta"], point["prefetch_depth"]), (tiles, depth))
                self.assertEqual((point["stage_overlap"], point["factor_overlap"], point["factor_slices"]),
                                 (False, False, 1))
                request = runner.compile_request(dict(operator="fft", precision=case["precision"],
                    logN=case["logN"], mapping_json=point), "research")
                self.assertEqual(request["factor_partition"], point["factor_partition"])

    def test_same_precision_reference_and_report_only_errors(self):
        for precision in ("fp32", "fp64"):
            case = dict(runner.cases()[0], precision=precision, shape=[32], batch=2)
            data, reference = runner.cpu_arrays(case, 17)
            self.assertEqual(data.dtype, np.complex64 if precision == "fp32" else np.complex128)
            self.assertEqual(data.dtype, reference.dtype)
            metrics, _ = runner.errors(reference * 100, reference)
            self.assertEqual(metrics["nonfinite"], 0)
            self.assertGreater(metrics["relative_l2"], 90)

    def test_actual_factor_summary_counts_physical_groups(self):
        mapping = runner.mappings_for(runner.cases()[1])["factor_t4_p2"]
        mapping["execution_group_mappings"] = [dict(core="cufftdx-block", threads=256, ept=16)] * 2
        summary = runner.design_summary(mapping)
        self.assertEqual(summary["stage_partition"], [20])
        self.assertEqual(summary["factor_partition"], [10, 10])
        self.assertEqual(summary["physical_stage_count"], 2)
        self.assertEqual([stage["columns"] for stage in summary["stages"]], [4, 4])
        self.assertEqual([stage["threads_per_cta"] for stage in summary["stages"]], [256, 256])
        self.assertEqual(summary["parallelism"], "unknown")
        register = runner.design_summary(dict(runner.mappings_for(runner.cases()[0])["register_radix4"],
                                              data_tiles_per_cta=1, prefetch_depth=0))
        self.assertEqual([stage["columns"] for stage in register["stages"]], [4, 4])
        self.assertEqual([stage["tilegroups_per_cta"] for stage in register["stages"]], [1, 1])
        self.assertEqual([stage["prefetch_depth"] for stage in register["stages"]], [0, 0])
        self.assertEqual(runner.design_summary(dict(backend="cufft"))["stages"], "unknown")

    def test_provider_error_artifact_is_written_once_and_complete(self):
        arrays = {name: np.arange(8) for name in
                  ("actual", "absolute_error", "replay_output", "replay_absolute_error")}
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(np, "savez_compressed", wraps=np.savez_compressed) as compress:
            artifact = runner.save_provider_errors(Path(temporary), "case", "factor_t4_p1", arrays)
            compress.assert_called_once()
            self.assertEqual(Path(artifact["path"]).name, "case.factor_t4_p1.errors.npz")
            self.assertEqual(artifact["sha256"], runner.digest(artifact["path"]))
            with np.load(artifact["path"]) as saved:
                self.assertEqual(set(saved.files), set(arrays))
                for name, expected in arrays.items():
                    np.testing.assert_array_equal(saved[name], expected)

    def test_mapping_difference_guard(self):
        mappings = runner.mappings_for(runner.cases()[0])
        providers = {name: dict(mapping=point) for name, point in mappings.items()}
        runner.check_factor_differences(providers)
        providers["factor_t4_p1"]["mapping"]["factor_overlap"] = True
        with self.assertRaisesRegex(AssertionError, "differ beyond"):
            runner.check_factor_differences(providers)
        with self.assertRaises(AssertionError):
            runner.validate_mapping("cufft", dict(backend="factor-streamed"), "cufft")

    def test_prepared_module_lookup_is_exact_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module = directory / "module.so"
            module.write_bytes(b"prepared test module")
            point = dict(backend="factor-streamed", factor_partition=[6, 10], factor_ept=16)
            report = dict(identity=dict(sm=80), modules=[dict(request=point, module=str(module),
                                                             sha256=runner.digest(module))])
            summary = directory / "summary.json"
            runner.save(summary, report)
            argv = ["compile_module.py", "--mapping", json.dumps(point), "--sm", "80"]
            for key in ("root", "mathdx", "nvcc", "cache", "timeout"):
                argv += ["--" + key, "unused"]
            with mock.patch.dict(os.environ, CUBUTTERFLY_INTRA_STAGE_PREPARED=str(summary)), \
                    mock.patch.object(runner, "compile_mapping") as compiler:
                self.assertEqual(runner.module_lookup(argv), 0)
                # -S removes site packages: the proxy path needs only stdlib.
                result = subprocess.run([sys.executable, "-S", str(SCRIPT), *argv],
                                        capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout.strip(), str(module))
                self.assertTrue(os.access(SCRIPT, os.X_OK))
                changed = argv.copy()
                changed[2] = json.dumps(dict(point, factor_ept=8))
                with self.assertRaisesRegex(RuntimeError, "unprepared"):
                    runner.module_lookup(changed)
                module.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    runner.module_lookup(argv)
                compiler.assert_not_called()

    def test_prepare_and_resume_do_not_repeat_cpu_or_jit_or_query_gpu(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=False):
            directory = Path(temporary)
            module = directory / "module.so"
            module.write_bytes(b"CPU test only")
            output = directory / "prepared"
            argv = ["--library", str(SCRIPT), "--gpu", UUID, "--output", str(output), "--prepare-only"]
            def tiny_arrays(case, seed):
                dtype = np.complex64 if case["precision"] == "fp32" else np.complex128
                return np.ones((2, 8), dtype=dtype), np.ones((2, 8), dtype=dtype)
            with mock.patch.object(runner, "cpu_arrays", side_effect=tiny_arrays) as cpu, \
                    mock.patch.object(runner, "compile_mapping", return_value=module) as compiler, \
                    mock.patch.object(runner, "gpu_state") as gpu, \
                    mock.patch.object(runner, "run_gpu") as execute:
                self.assertEqual(runner.main(argv), 0)
                self.assertEqual(cpu.call_count, 4)
                self.assertEqual(compiler.call_count, 20)
                cpu.reset_mock()
                compiler.reset_mock()
                self.assertEqual(runner.main(argv + ["--resume"]), 0)
                cpu.assert_not_called()
                compiler.assert_not_called()
                gpu.assert_not_called()
                execute.assert_not_called()
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["status"], "cpu_and_jit_prepared")
            self.assertEqual(report["compute_family"], "fft-h")
            self.assertEqual(report["storage_precisions"], ["fp32", "fp64"])
            self.assertEqual(report["compute_precisions"], ["fp32", "fp64"])
            self.assertEqual(len(report["identity_sha256"]), 64)
            with mock.patch.object(runner, "cpu_arrays") as cpu, \
                    mock.patch.object(runner, "compile_mapping") as compiler, \
                    mock.patch.object(runner, "gpu_state", return_value=dict(uuid=UUID)), \
                    mock.patch.object(runner, "run_gpu") as execute:
                self.assertEqual(runner.main([arg for arg in argv if arg != "--prepare-only"] + ["--resume"]), 0)
                execute.assert_called_once()
                cpu.assert_not_called()
                compiler.assert_not_called()
            before = (output / "summary.json").read_bytes()
            with mock.patch.object(runner, "cpu_arrays") as cpu, \
                    mock.patch.object(runner, "compile_mapping") as compiler:
                self.assertEqual(runner.main(argv + ["--resume", "--chain", "9"]), 1)
                cpu.assert_not_called()
                compiler.assert_not_called()
            self.assertEqual((output / "summary.json").read_bytes(), before)
            Path(report["prepared_cases"][0]["artifact"]).write_bytes(b"corrupt")
            self.assertEqual(runner.main(argv + ["--resume"]), 1)
            self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_protocol_defaults_and_strict_cli(self):
        base = ["--library", str(SCRIPT), "--gpu", UUID, "--output", "unused"]
        args = runner.make_parser().parse_args(base)
        self.assertEqual((args.chain, args.repeats, args.trials, args.warmup), (8, 20, 5, 5))
        for extra in (["--gpu", "2"], ["--chain", "0"], ["--warmup", "-1"], ["--logns", "16"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as caught:
                runner.main(base + extra)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
