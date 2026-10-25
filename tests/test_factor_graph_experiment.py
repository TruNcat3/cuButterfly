"""CPU-only contracts for fixed-core Graph preparation and pairing."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_factor_graph_experiment as runner

CONFIG = ROOT / "configs/experiments/factor_graph_core_organization_20261025.json"


def suite():
    value = json.loads(CONFIG.read_text())
    runner.expand_cases(value)
    return value


class FactorGraphTests(unittest.TestCase):
    def test_bounded_matched_core_matrix(self):
        value = suite()
        self.assertEqual((value["logN"], value["batch"], value["precision"]), (16, 16, "fp64"))
        self.assertEqual(len(value["cases"]), 9)
        self.assertIn("matched organization control", value["purpose"])
        report = runner.check_control_groups(value)
        self.assertEqual({row["fft_core"] for row in report}, {"register-tile", "cufftdx-block"})
        for group in value["control_groups"]:
            cases = {case["id"]: case["mapping"] for case in value["cases"]}
            mappings = [cases[key] for key in group["cases"]]
            self.assertEqual([(m["data_tiles_per_cta"], m["prefetch_depth"]) for m in mappings],
                             [(1, 0), (4, 0), (4, 1), (4, 2)])
            for mapping in mappings:
                self.assertEqual(mapping["factor_partition"], [8, 8])
                self.assertEqual((mapping["factor_ept"], mapping["factor_columns"]), (16, 4))
                self.assertEqual((mapping["factor_overlap"], mapping["factor_slices"]), (False, 1))

    def test_public_projection_preserves_core_slices_and_io(self):
        value = suite()
        mapping = dict(value["cases"][0]["mapping"], factor_slices=2,
                       factor_overlap=True, factor_io_policies=["static-unrolled", "dynamic"])
        request = runner.module_request(value, mapping)
        self.assertEqual(request["fft_core"], "register-tile")
        self.assertEqual(request["factor_slices"], 2)
        self.assertEqual(request["factor_io_policies"], ["static-unrolled", "dynamic"])
        # Overlap schedules slices at runtime; it is deliberately not a module axis.
        self.assertEqual(request, runner.module_request(value, dict(mapping, factor_overlap=False)))
        cufftdx = runner.module_request(value, dict(mapping, fft_core="cufftdx-block"))
        self.assertNotIn("fft_core", cufftdx)
        self.assertNotEqual(request, cufftdx)

    def test_guard_rejects_core_shape_and_other_mapping_changes(self):
        for field, replacement in (("fft_core", "cufftdx-block"), ("factor_partition", [7, 9]),
                                   ("factor_ept", 8), ("factor_columns", 2),
                                   ("factor_slices", 2), ("factor_overlap", True),
                                   ("factor_io_policies", ["static-unrolled", "dynamic"])):
            value = suite()
            value["cases"][1]["mapping"][field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "beyond allowed"):
                runner.check_control_groups(value)
        value = suite()
        value["control_groups"][0]["allowed_differences"].append("fft_core")
        with self.assertRaisesRegex(ValueError, "invalid"):
            runner.check_control_groups(value)

    def test_observed_guard_and_legacy_config(self):
        value = suite()
        rows = copy.deepcopy(value["cases"])
        self.assertEqual(runner.check_control_groups(value), runner.check_control_groups(value, rows))
        rows[1]["mapping"]["fft_core"] = "cufftdx-block"
        with self.assertRaisesRegex(ValueError, "differs from request"):
            runner.check_control_groups(value, rows)
        rows = copy.deepcopy(value["cases"])
        for row in rows[:4]:
            row["mapping"]["fft_core"] = "cufftdx-block"
        with self.assertRaisesRegex(ValueError, "differs from request"):
            runner.check_control_groups(value, rows)
        legacy = json.loads((ROOT / "configs/experiments/factor_graph_20261025.json").read_text())
        runner.expand_cases(legacy)
        self.assertEqual(runner.check_control_groups(legacy), [])

    def test_module_lookup_only_returns_exact_prepared_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module = directory / "prepared.so"
            module.write_bytes(b"CPU fixture")
            value = suite()
            request = runner.module_request(value, value["cases"][0]["mapping"])
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps(dict(complete=True, sm=80, modules=[dict(
                point=request, module=str(module), sha256=runner.sha(module))])))
            def lookup(point, sm=80):
                return runner.module_lookup(["compile_module.py", "--mapping", json.dumps(point),
                    "--sm", str(sm), "--root", str(ROOT), "--mathdx", "unused", "--nvcc", "unused",
                    "--cache", str(directory), "--timeout", "300"])
            with mock.patch.dict(os.environ, CUBUTTERFLY_FACTOR_GRAPH_PREPARED=str(manifest)), \
                    mock.patch.object(runner, "compile_mapping") as compiler:
                runtime_request = dict(request)
                runtime_request.pop("factor_io_policies")
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(lookup(runtime_request), 0)
                self.assertEqual(output.getvalue().strip(), str(module))
                with self.assertRaisesRegex(RuntimeError, "unprepared module"):
                    lookup(dict(request, data_tiles_per_cta=4))
                with self.assertRaisesRegex(RuntimeError, "architecture"):
                    lookup(request, sm=90)
                module.write_bytes(b"modified")
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    lookup(request)
                compiler.assert_not_called()

    def test_measurement_uses_lookup_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.json"
            path.write_text("{}")
            manifest = dict(hardware_model=str(path), hardware_sha256=runner.sha(path), cache=temporary)
            env = runner.measurement_environment(manifest, path)
            self.assertEqual(env["CUBUTTERFLY_JIT_PYTHON"], str(Path(runner.__file__).resolve()))
            self.assertEqual(env["CUBUTTERFLY_FACTOR_GRAPH_PREPARED"], str(path))
            self.assertTrue(os.access(env["CUBUTTERFLY_JIT_PYTHON"], os.X_OK))

    def test_prepare_uses_public_requests_and_cpu_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / "bench"
            binary.write_bytes(b"CPU fixture")
            (directory / "libcuntt.so").write_bytes(b"library fixture")
            module = directory / "module.so"
            module.write_bytes(b"module fixture")
            hardware = directory / "hardware.json"
            hardware.write_text(json.dumps(dict(hardware=dict(compute_capability="8.0",
                shared_bytes_per_block=166912, max_threads_per_block=1024))))
            prepared = directory / "prepared.json"
            def reference(command, **kwargs):
                self.assertEqual(command[1], "--prepare-reference")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                Path(json.loads(prepared.read_text())["reference_file"]).write_bytes(b"reference fixture")
            argv = [str(runner.__file__), "prepare", "--config", str(CONFIG), "--prepared",
                    str(prepared), "--binary", str(binary), "--hardware-model", str(hardware),
                    "--cache", str(directory / "cache")]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runner, "compile_mapping", return_value=module) as compiler, \
                    mock.patch.object(runner.subprocess, "run", side_effect=reference) as process, \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(compiler.call_count, 8)
            process.assert_called_once()
            requests = [call.args[0] for call in compiler.call_args_list]
            self.assertEqual(sum(request.get("fft_core") == "register-tile" for request in requests), 4)
            manifest = json.loads(prepared.with_suffix(".manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(len(manifest["control_groups"]), 2)
            self.assertEqual(len(manifest["modules"]), 9)


if __name__ == "__main__":
    unittest.main()
