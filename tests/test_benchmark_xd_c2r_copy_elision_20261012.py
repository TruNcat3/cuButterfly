import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "paper/tools/benchmark_xd_c2r_copy_elision_20261012.py"
spec = importlib.util.spec_from_file_location("c2r_copy_runner", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class C2RCopyRunnerTests(unittest.TestCase):
    def test_prepare_and_cpu_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = module.prepare(Path(directory))
            result = module.cpu_check(manifest)
            self.assertEqual(result["status"], "cpu-contract-pass")
            self.assertEqual(result["cases"], 8)
            data = json.loads(manifest.read_text())
            self.assertTrue(all(row["mapping"]["contract"]["fft_type"] == 2
                                for row in data["cases"]))

    def test_existing_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = module.prepare(path)
            before = manifest.read_bytes()
            self.assertEqual(module.prepare(path), manifest)
            self.assertEqual(manifest.read_bytes(), before)

    def test_input_mutation_contract_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = module.prepare(Path(directory))
            data = json.loads(manifest.read_text())
            self.assertTrue(data["correctness_contract"]["preserve_input"])
            self.assertIsNone(data["correctness_contract"]["atol"])
            self.assertIsNone(data["correctness_contract"]["rtol"])


if __name__ == "__main__":
    unittest.main()
