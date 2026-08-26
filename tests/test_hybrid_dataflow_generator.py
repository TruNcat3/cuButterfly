#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_hybrid_dataflow", ROOT / "scripts" / "generate_hybrid_dataflow.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class HybridDataflowGeneratorTest(unittest.TestCase):
    def test_v100_manifest(self):
        points = GENERATOR.load_points(ROOT / "config" / "v100_hybrid_dataflow.json")
        defaults = GENERATOR.load_defaults(ROOT / "config" / "v100_hybrid_dataflow.json", points)
        self.assertEqual(223, len(points))
        self.assertEqual({(5, 5), (6, 2), (6, 4), (6, 6), (7, 7),
                          (8, 2), (8, 4), (8, 8)},
                         {(point[1], point[2]) for point in points})
        self.assertTrue(any(point[7] == "ping-pong" for point in points))
        self.assertTrue(any(point[8] == "atomic" for point in points))
        self.assertEqual({1, 2, 3, 4, 5, 6, 7, 8}, {point[9] for point in points})
        self.assertEqual({1, 3}, {point[10] for point in points})
        self.assertEqual({1, 2}, {point[11] for point in points})
        self.assertEqual(5, len(defaults))

        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "generated.hpp"
            GENERATOR.write_header(header, points, defaults)
            text = header.read_text()
            self.assertIn("CUNTT_FOR_EACH_HYBRID_DATAFLOW_KERNEL", text)
            self.assertEqual(7, text.count("X(8, 8,"))
            self.assertIn("std::array<GeneratedHybridDataflowPoint, 223>", text)
            self.assertIn("std::array<GeneratedHybridDataflowDefault, 5>", text)
            self.assertIn("ComputeUnit::Radix4", text)

    def test_rejects_illegal_word_data_pair(self):
        document = json.loads((ROOT / "config" / "v100_hybrid_dataflow.json").read_text())
        document["primary"]["data_space_by_word"]["64"] = [32]
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "invalid.json"
            spec.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "data-space/word-width mismatch"):
                GENERATOR.load_points(spec)


if __name__ == "__main__":
    unittest.main()
