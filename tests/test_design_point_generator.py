#!/usr/bin/env python3
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_design_points import load_points, write_structured


class DesignPointGeneratorTest(unittest.TestCase):
    def test_structured_register_family_is_selected_and_emitted(self):
        points, _ = load_points(
            ROOT / "config" / "v100_design_points.json",
            {"structured-2x2"}, {"warp-register"}, {3, 12, 15})
        selected = points[("structured-2x2", "warp-register", "fp32")]
        self.assertEqual(selected, {3, 12, 15})
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "generated_register_structured.cu"
            write_structured(path, selected)
            source = path.read_text()
            self.assertIn("launch_register_structured<12>", source)
            self.assertIn("generated_register_structured_available", source)
            self.assertIn("bool broadcast", source)

    def test_manifest_declares_fp32_log3_through_log15(self):
        document = json.loads((ROOT / "config" / "v100_design_points.json").read_text())
        point = next(entry for entry in document["design_points"]
                     if entry["operator"] == "structured-2x2")
        self.assertEqual(point["precision"], "fp32")
        self.assertEqual(point["log_n"], list(range(3, 16)))


if __name__ == "__main__":
    unittest.main()
