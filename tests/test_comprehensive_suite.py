#!/usr/bin/env python3
import copy
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_comprehensive_suite import (build_command, expected_semantics, load_manifest, ntt_prime,
                                     parse_csv_record, selected_cases)
from summarize_comprehensive_suite import summarize


class ComprehensiveSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "config" / "v100_comprehensive_suite.json"
        cls.document = load_manifest(cls.path)

    def test_manifest_has_quick_and_full_coverage(self):
        quick = selected_cases(self.document, "quick", None)
        full = selected_cases(self.document, "full", None)
        self.assertGreaterEqual(len(quick), 20)
        self.assertGreater(len(full), len(quick))
        self.assertEqual({case["operator"] for case in quick}, {"fft", "ntt", "fwht", "xor-zeta"})

    def test_each_group_has_at_most_one_declared_reference(self):
        groups = {}
        for case in self.document["cases"]:
            groups.setdefault(case["group"], []).append(case)
        for group, cases in groups.items():
            with self.subTest(group=group):
                self.assertLessEqual(sum(bool(case.get("reference")) for case in cases), 1)
                semantics = {(case["operator"], case["precision"], case.get("direction", "forward"), case["logN"])
                             for case in cases}
                self.assertEqual(len(semantics), 1)

    def test_invalid_duplicate_case_is_rejected(self):
        invalid = copy.deepcopy(self.document)
        invalid["cases"].append(copy.deepcopy(invalid["cases"][0]))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_manifest(path)

    def test_mixed_semantic_group_is_rejected(self):
        invalid = copy.deepcopy(self.document)
        case = next(case for case in invalid["cases"] if case["id"] == "fft8_cub")
        placement_index = case["args"].index("--placement") + 1
        case["args"][placement_index] = "out-of-place"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "mixes semantic contracts"):
                load_manifest(path)

    def test_structured_matrix_is_part_of_semantic_contract(self):
        invalid = copy.deepcopy(self.document)
        cases = [case for case in invalid["cases"] if case["group"] == "fwht-fp32-fwd-log8"]
        for case, matrix in zip(cases, ("1,0.25,-0.5,1", "1,0.5,-0.5,1")):
            case["operator"] = "structured-2x2"
            case["args"] += ["--stage-matrix", matrix]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "mixes semantic contracts"):
                load_manifest(path)

    def test_structured_semantics_canonicalize_repeated_matrices(self):
        case = {
            "runner": "butterfly", "operator": "structured-2x2", "precision": "fp32", "logN": 2,
            "args": ["--operator", "structured-2x2", "--stage-matrix", "1.0,0.25,-0.5,1",
                     "--stage-matrix", "2,0,0,0.5"],
        }
        semantics = expected_semantics(case)
        self.assertEqual(semantics["normalization"], "none")
        self.assertEqual(semantics["stage_matrices"], "1:0.25:-0.5:1x2:0:0:0.5")

    def test_ntt_prime_has_requested_contract(self):
        value = ntt_prime(60, 20)
        self.assertEqual(value.bit_length(), 60)
        self.assertEqual(value % (1 << 20), 1)

    def test_command_builder_applies_protocol_and_layout(self):
        case = next(case for case in self.document["cases"] if case["id"] == "fft8_stride_cub")
        command = build_command(ROOT, self.document, case, self.document["protocols"]["full"], 17, False)
        self.assertIn("--element-stride", command)
        self.assertEqual(command[command.index("--element-stride") + 1], "2")
        self.assertEqual(command[command.index("--batch") + 1], "17")
        self.assertEqual(command[command.index("--repeat") + 1], "100")

    def test_csv_parser_ignores_non_csv_output(self):
        output = "notice\ndevice,kernel_ms,correct\n\"V100\",0.125,1\n"
        self.assertEqual(parse_csv_record(output)["kernel_ms"], "0.125")

    def test_exact_case_filter_does_not_match_batch_prefix(self):
        scaling = load_manifest(ROOT / "config" / "v100_scaling_suite.json")
        cases = selected_cases(scaling, "full", ["fft8_cufft_b1"], exact=True)
        self.assertEqual([case["id"] for case in cases], ["fft8_cufft_b1"])

    def test_summary_uses_declared_reference_and_group_fastest(self):
        base = {
            "suite_tier": "quick", "suite_runner": "butterfly", "trial": "1",
            "performance_batch": "1", "preflight_correct": "1", "operator": "fft",
            "precision": "fp32", "direction": "forward", "logN": "8", "N": "256",
        }
        rows = [
            {**base, "suite_case_id": "a", "suite_group": "with-ref", "implementation": "self", "reference": "0", "kernel_ms": "0.8"},
            {**base, "suite_case_id": "b", "suite_group": "with-ref", "implementation": "ref", "reference": "1", "kernel_ms": "1.0"},
            {**base, "suite_case_id": "c", "suite_group": "no-ref", "implementation": "fast", "reference": "0", "kernel_ms": "0.5"},
            {**base, "suite_case_id": "d", "suite_group": "no-ref", "implementation": "slow", "reference": "0", "kernel_ms": "1.0"},
        ]
        summary = {row["case_id"]: row for row in summarize(rows)}
        self.assertAlmostEqual(summary["a"]["throughput_vs_basis"], 1.25)
        self.assertEqual(summary["a"]["performance_class"], "faster")
        self.assertAlmostEqual(summary["a"]["million_transforms_s"], 0.00125)
        self.assertAlmostEqual(summary["a"]["billion_points_s"], 0.00032)
        self.assertAlmostEqual(summary["d"]["throughput_vs_basis"], 0.5)
        self.assertEqual(summary["d"]["comparison_basis"], "group-fastest")
        self.assertEqual(summary["d"]["stability_class"], "stable")

    def test_summary_reports_but_does_not_center_on_one_outlier(self):
        rows = []
        for trial, kernel_ms in enumerate((1.0, 1.0, 1.01, 1.01, 1.20), 1):
            rows.append({
                "suite_case_id": "outlier", "suite_group": "outlier-group",
                "suite_tier": "full", "suite_runner": "butterfly",
                "implementation": "self", "reference": "0", "trial": str(trial),
                "performance_batch": "1", "preflight_correct": "1", "operator": "fwht",
                "precision": "fp32", "direction": "forward", "logN": "8", "N": "256",
                "kernel_ms": str(kernel_ms),
            })
        result = summarize(rows)[0]
        self.assertGreater(result["relative_range"], 0.1)
        self.assertLess(result["central_relative_range"], 0.03)
        self.assertEqual(result["stability_class"], "stable-with-outlier")


if __name__ == "__main__":
    unittest.main()
