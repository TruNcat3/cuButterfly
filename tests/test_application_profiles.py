#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_application_profiles", ROOT / "scripts/generate_application_profiles.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ApplicationProfileGeneratorTest(unittest.TestCase):
    def test_v100_profiles_generate_exact_shape_batch_and_modulus_metadata(self):
        source = ROOT / "config/v100_application_profiles.json"
        document = json.loads(source.read_text())
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["target"]["multiprocessors"], 80)
        self.assertTrue(any(profile["operator"] == "fft" and profile["logN"] == 20
                            for profile in document["profiles"]))
        ntt32 = next(profile for profile in document["profiles"]
                     if profile["operator"] == "ntt" and profile["storage"] == "uint32")
        self.assertEqual(ntt32["modulus"], 1073692673)
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "profiles.hpp"
            MODULE.generate(source, output)
            text = output.read_text()
            self.assertIn("kApplicationProfileMultiprocessors = 80", text)
            self.assertIn("1073692673ULL", text)
            self.assertIn("cuButterfly-cuFFTDx-online", text)


if __name__ == "__main__":
    unittest.main()
