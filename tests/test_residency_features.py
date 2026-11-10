#!/usr/bin/env python3
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from residency_features import derive_residency_features


class ResidencyFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = json.loads((ROOT / "configs/hardware/v100_sxm2_16gb.json").read_text())

    def test_structured_log15_crosses_single_cta_boundary(self):
        log14 = derive_residency_features(self.hardware, 256, 128, 32768, 256, 64)
        log15 = derive_residency_features(self.hardware, 256, 254, 32768, 128, 128, log14)
        self.assertEqual(log14["resident_ctas_per_sm"], 2)
        self.assertEqual(log15["resident_ctas_per_sm"], 1)
        self.assertEqual(log15["allocated_registers_per_cta"], 65536)
        self.assertEqual(log15["occupancy_upper_bound"], 0.125)
        self.assertEqual(log15["crosses_single_cta_boundary"], 1)
        self.assertEqual(log15["resource_cliff"], 1)

    def test_cta_count_drop_is_not_an_occupancy_cliff(self):
        log8 = derive_residency_features(self.hardware, 32, 35, 1024, 16384, 8)
        log10 = derive_residency_features(self.hardware, 128, 32, 4096, 4096, 32, log8)
        self.assertLess(log10["resident_ctas_per_sm"], log8["resident_ctas_per_sm"])
        self.assertGreater(log10["occupancy_upper_bound"], log8["occupancy_upper_bound"])
        self.assertEqual(log10["resource_cliff"], 0)

    def test_illegal_register_point_is_prunable(self):
        point = derive_residency_features(self.hardware, 256, 300, 32768, 128, 128)
        self.assertEqual(point["hardware_feasible"], 0)
        self.assertEqual(point["resident_ctas_per_sm"], 0)
        self.assertEqual(point["limiting_resource"], "illegal")


if __name__ == "__main__":
    unittest.main()
