"""CPU mocks exercise persistent-cache wiring without starting CUDA."""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_public_frontend import Dll, cb

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CacheTests(unittest.TestCase):
    def test_quarot_public_final_rounding_uses_descriptor_scale(self):
        from unittest.mock import Mock
        q = load("cache_quarot_final", ROOT / "experiments/workloads/e13/quarot_adapter.py")
        own = q.QuaRotFWHTAdapter.__new__(q.QuaRotFWHTAdapter)
        own.compute_contract, own._scales, own.reuse_buffers = "public_fp32_final", {}, False
        values = q.torch.ones((2, 32), dtype=q.torch.float16)
        result = q.torch.full_like(values, 3.0)
        plan = SimpleNamespace(execute=Mock(return_value=result))
        own._plan_for = Mock(return_value=plan)
        actual = own._transform_vectors(values, 32, 128)
        self.assertEqual(actual.dtype, q.torch.float16)
        self.assertTrue(q.torch.equal(actual, result))
        args = own._plan_for.call_args.args
        self.assertEqual(args[0].dtype, q.torch.float16)
        self.assertEqual(args[1:3], (32, 2))
        self.assertEqual(args[3], float(1.0 / q.torch.tensor(128).sqrt()))

    def test_public_plan_passes_cache_before_plan_and_cleans_up_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = str(Path(directory) / "gpu.cache")
            dll = Dll()
            with patch.object(ctypes, "CDLL", return_value=dll):
                library = cb.Library("mock.so")
            with cb.Plan(library, cb.Transform(), cache_path=path) as plan:
                self.assertEqual(plan.cache_path, path)
            names = [name for name, _ in dll.calls]
            self.assertLess(names.index("SetCachePath"), names.index("CreatePlan"))
            self.assertEqual(next(args[1] for name, args in dll.calls if name == "SetCachePath"), path.encode())
            dll.fail = "SetCachePath"
            with self.assertRaises(cb.Error):
                cb.Plan(library, cb.Transform(), cache_path=path)
            self.assertEqual(dll.calls[-1][0], "Destroy")

    def test_adapters_resolve_environment_and_reject_bad_paths(self):
        import tempfile
        modules = [load("cache_quarot", ROOT / "experiments/workloads/e13/quarot_adapter.py"),
                   load("cache_real_fft", ROOT / "results/e13_assets/cubutterfly_real_fft.py")]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "gpu.cache")
            with patch.dict(os.environ, {"CUBUTTERFLY_PLAN_CACHE": path}):
                for resolver in [cb._plan_cache_path] + [m._cache_path for m in modules]:
                    self.assertEqual(resolver(), path)
                    self.assertEqual(resolver(str(Path(directory) / "explicit.cache")), str(Path(directory) / "explicit.cache"))
                    with self.assertRaises(ValueError):
                        resolver(directory)
                    with self.assertRaises(ValueError):
                        resolver(str(Path(directory) / "missing/file"))
                    with self.assertRaises(ValueError):
                        resolver("")
            with patch.dict(os.environ, {"CUBUTTERFLY_REGISTRY": str(Path(directory) / "missing.json")}):
                with self.assertRaises(ValueError):
                    modules[0]._registry_identity()

    def test_adapter_native_calls_and_complete_metadata(self):
        import tempfile
        q = load("cache_quarot_calls", ROOT / "experiments/workloads/e13/quarot_adapter.py")
        r = load("cache_real_calls", ROOT / "results/e13_assets/cubutterfly_real_fft.py")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            cache = str(Path(directory) / "gpu.cache")
            dll = Dll()
            library = Path(directory) / "mock.so"
            library.touch()
            workspace = SimpleNamespace(numel=lambda: 64, data_ptr=lambda: 123)
            with patch.object(ctypes, "CDLL", return_value=dll), patch.object(q.torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=7)), patch.object(q.torch, "empty", return_value=workspace):
                own = q.QuaRotFWHTAdapter.__new__(q.QuaRotFWHTAdapter)
                own.api, own.cache_path = q._Api(library), cache
                values = SimpleNamespace(dtype=q.torch.float32, device=q.torch.device("cuda:0"))
                plan = own._make_plan(values, 16, 3, 7)
                self.assertEqual(plan.metadata["descriptor"], {"from": "C library"})
                self.assertEqual(plan.metadata["mapping"], {"from": "C library"})
                self.assertEqual(plan.metadata["cache"]["path"], cache)
                self.assertEqual(plan.metadata["selection_source"], "measured_table")
                plan.close()
                real = r.RealFFTAdapter.__new__(r.RealFFTAdapter)
                real.api, real.cache_path = r._Api(library), cache
                real.shape, real.batch, real.device = (8, 8), 1, values.device
                real.native_real, real._volume = True, 64
                real._forward = real._make_plan(False)
                real._inverse = real._make_plan(True)
                metadata = real.plan_metadata()
                self.assertEqual([x["transform"] for x in metadata], ["r2c", "c2r"])
                self.assertTrue(all(x["descriptor"] and x["mapping"] and x["cache"]["path"] == cache for x in metadata))
                real.close()
            setters = [args for name, args in dll.calls if name == "SetCachePath"]
            self.assertEqual(len(setters), 3)
            self.assertTrue(all(args[1] == cache.encode() for args in setters))

    def test_adapter_cache_setter_failure_destroys_context(self):
        def check(status, operation):
            if status:
                raise RuntimeError(operation)

        q = load("cache_quarot_failure", ROOT / "experiments/workloads/e13/quarot_adapter.py")
        r = load("cache_real_failure", ROOT / "results/e13_assets/cubutterfly_real_fft.py")
        for module, adapter_type, invoke in (
            (q, q.QuaRotFWHTAdapter, lambda a: a._make_plan(None, 16, 1, 7)),
            (r, r.RealFFTAdapter, lambda a: a._make_plan(False)),
        ):
            dll = Dll()
            dll.fail = "SetCachePath"
            adapter = adapter_type.__new__(adapter_type)
            adapter.api = SimpleNamespace(lib=dll, check=check)
            adapter.cache_path = "/tmp/gpu.cache"
            with self.assertRaises(RuntimeError):
                invoke(adapter)
            self.assertEqual(dll.calls[-1][0], "Destroy")


if __name__ == "__main__":
    unittest.main()
