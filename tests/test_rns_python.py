"""CPU-only contract tests for the optional RNS ctypes frontend."""

import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cubutterfly import rns


CONFIG = {
    "schema_version": 1, "log_n": 12, "inverse": False,
    "ring": "negacyclic", "input_order": "natural", "output_order": "bit_reversed",
    "domains": [{"modulus": 1152921504606830593, "root": 123}],
    "polynomials": [{"domain": 0, "input_offset": 0, "output_offset": 0}],
}


class Function:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *arguments):
        return self.implementation(*arguments)


class FakeAbi:
    def __init__(self):
        self.calls = []
        self.request = None
        self.failure = None
        for name in ("Create", "Prepare", "Execute", "InfoJson", "Destroy", "LastError"):
            setattr(self, "cubutterflyRns" + name, Function(getattr(self, name)))

    def Create(self, payload, result):
        self.request = json.loads(payload)
        ct.cast(result, ct.POINTER(ct.c_void_p))[0] = 42
        return 0

    def Prepare(self, *arguments):
        self.calls.append(("Prepare", arguments))
        return 2 if self.failure else 0

    def Execute(self, *arguments):
        self.calls.append(("Execute", arguments))
        return 0

    def InfoJson(self, handle, buffer, size):
        payload = json.dumps({"request": self.request, "cache_hit": True}).encode() + b"\0"
        ct.cast(size, ct.POINTER(ct.c_size_t))[0] = len(payload)
        if buffer is not None:
            ct.memmove(buffer, payload, len(payload))
        return 0

    def Destroy(self, handle):
        self.calls.append(("Destroy", handle.value))
        return 0

    def LastError(self):
        return self.failure or b"fake error"


class RnsPythonTests(unittest.TestCase):
    def setUp(self):
        self.abi = FakeAbi()
        with patch.object(rns.ct, "CDLL", return_value=self.abi):
            self.library = rns.Library("fake.so")

    def test_import_loads_no_cuda_or_torch(self):
        script = """
import ctypes, sys
def forbidden(*args, **kwargs):
    raise AssertionError('import tried to load shared library')
ctypes.CDLL = forbidden
import cubutterfly.rns
assert 'torch' not in sys.modules
assert not any(name.startswith(('cuda', 'cupy')) for name in sys.modules)
"""
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "python"))
        subprocess.run([sys.executable, "-c", script], env=environment, check=True)

    def test_json_roundtrip_and_lifecycle(self):
        with rns.Plan(self.library, json.dumps(CONFIG)) as plan:
            self.assertEqual(plan.info()["request"], CONFIG)
            with self.assertRaises(rns.Error):
                plan.execute(4096, 8192)
            self.assertIs(plan.prepare(cache_directory=Path("cache"),
                                       policy=rns.CachePolicy.REQUIRE, repetitions=7, stream=24), plan)
            arguments = self.abi.calls[-1][1]
            self.assertEqual(arguments[1:5], (0, b"cache", 1, 7))
            self.assertEqual(arguments[5].value, 24)
            plan.execute(ct.c_void_p(4096), 8192, stream=32)
            self.assertEqual(self.abi.calls[-1][0], "Execute")
            with self.assertRaises(rns.Error):
                plan.prepare()
        plan.close()
        self.assertEqual(sum(call[0] == "Destroy" for call in self.abi.calls), 1)
        for operation in (lambda: plan.execute(4096, 8192), plan.info, plan.__enter__, plan.prepare):
            with self.assertRaises(rns.Error):
                operation()

    def test_pointer_validation(self):
        with rns.Plan(self.library, CONFIG) as plan:
            plan.prepare()
            for value in (None, True, 1.5, object()):
                with self.subTest(value=value), self.assertRaises(TypeError):
                    plan.execute(value, 8192)
            for value in (0, -8, 4097, 1 << (ct.sizeof(ct.c_void_p) * 8)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    plan.execute(value, 8192)
            with self.assertRaises(ValueError):
                plan.execute(4096, 4096)
            with self.assertRaises(TypeError):
                plan.execute(4096, 8192, stream=True)
        with rns.Plan(self.library, CONFIG) as plan:
            plan.prepare(placement=rns.Placement.IN_PLACE)
            plan.execute(4096, 4096)
            with self.assertRaises(ValueError):
                plan.execute(4096, 8192)

    def test_cache_errors_and_prepare_validation(self):
        with rns.Plan(self.library, CONFIG) as plan:
            for options in ({"policy": True}, {"policy": 3}, {"placement": 2},
                            {"repetitions": 0}, {"repetitions": 1 << 32},
                            {"cache_directory": "bad\0path"}):
                with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                    plan.prepare(**options)
            self.abi.failure = b"required RNS cache entry missing"
            with self.assertRaisesRegex(rns.Error, "required RNS cache entry missing"):
                plan.prepare(policy=rns.CachePolicy.REQUIRE)
            with self.assertRaises(rns.Error):
                plan.execute(4096, 8192)
            self.abi.failure = None
            plan.prepare(policy=rns.CachePolicy.REFRESH)
            self.assertEqual(self.abi.calls[-1][1][3], 2)

    def test_config_must_be_object(self):
        for value in ([], "[]", "null", 123):
            with self.subTest(value=value), self.assertRaises(TypeError):
                rns.Plan(self.library, value)

    def test_callback_restoration(self):
        class Application:
            def ntt(self, value):
                return value + 1
        application = Application()
        for fail in (False, True):
            try:
                with rns.replace_ntt_callback(application, "ntt", lambda value: value * 3):
                    self.assertEqual(application.ntt(5), 15)
                    if fail:
                        raise RuntimeError("application failure")
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual(application.ntt(5), 6)
            self.assertNotIn("ntt", vars(application))
        original = lambda value: value - 1
        application.ntt = original
        with rns.replace_ntt_callback(application, "ntt", lambda value: value):
            pass
        self.assertIs(application.ntt, original)
        with self.assertRaises(TypeError):
            with rns.replace_ntt_callback(application, "ntt", 3):
                pass


if __name__ == "__main__":
    unittest.main()
