#!/usr/bin/env python3
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("generate_runtime_selector", ROOT / "scripts/generate_runtime_selector.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeSelectorGeneratorTest(unittest.TestCase):
    def test_archived_summary_generates_every_runtime_table(self):
        tables = MODULE.load_tables(ROOT / "results/v100_scaling_full_summary.csv")
        self.assertEqual(set(tables), set(MODULE.TABLES))
        self.assertTrue(all(len(anchors) >= 2 for anchors in tables.values()))
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "generated.hpp"
            MODULE.generate_header(output, tables)
            text = output.read_text()
            self.assertIn("kFwht15WarpAnchors", text)
            self.assertIn("kNtt16Radix4Anchors", text)
            self.assertIn("kFft20Anchors", text)


if __name__ == "__main__":
    unittest.main()
