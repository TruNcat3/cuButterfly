#!/usr/bin/env python3
import csv
import importlib.util
import io
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load("run_external_baseline_suite")
SUMMARY = load("summarize_external_baseline_suite")
THREE_WAY = load("summarize_three_way_comparison")


class ExternalBaselineSuiteTest(unittest.TestCase):
    def test_manifest_expands_matching_pairs(self):
        document = json.loads((ROOT / "config" / "v100_external_baseline_suite.json").read_text())
        cases = RUNNER.expand(document)
        self.assertEqual(len(cases), 24)
        groups = {}
        for case in cases:
            groups.setdefault(case["group"], []).append(case)
        self.assertTrue(all(len(cases_in_group) == 2 for cases_in_group in groups.values()))

    def test_parser_skips_non_csv_notices(self):
        output = "notice\nimplementation,logN,kernel_ms\nGPU-NTT,16,0.25\n"
        self.assertEqual(RUNNER.read_one_csv(output)["kernel_ms"], "0.25")

    def test_summary_uses_declared_local_reference(self):
        text = "group,implementation,reference,runner,operator,precision,direction,normalization,placement,output_order,logN,N,batch,modulus,modulus_bits,warmup,repeat,kernel_ms,correct\n"
        text += "g,local,1,ntt,ntt,uint64,forward,none,out-of-place,natural,16,65536,4,q,60,1,1,2.0,1\n"
        text += "g,external,0,gpuntt,ntt,uint64,forward,none,out-of-place,natural,16,65536,4,q,60,1,1,1.0,1\n"
        rows = list(csv.DictReader(io.StringIO(text)))
        summary = SUMMARY.summarize(rows)
        external = next(row for row in summary if row["implementation"] == "external")
        self.assertEqual(external["throughput_vs_reference"], 2.0)

    def test_three_way_manifest_has_library_base_and_searched_roles(self):
        document = json.loads((ROOT / "config" / "v100_three_way_comparison.json").read_text())
        cases = RUNNER.expand(document)
        self.assertEqual(len(cases), 27)
        fft_groups = {}
        for case in (case for case in cases if case["operator"] == "fft"):
            fft_groups.setdefault(case["group"], set()).add(THREE_WAY.role({"implementation": case["name"]}))
        self.assertTrue(all(roles == {"library", "base", "searched"} for roles in fft_groups.values()))

    def test_three_way_summary_separates_search_and_library_speedups(self):
        common = {"group": "g", "operator": "fft", "precision": "fp32", "logN": "8",
                  "N": "256", "batch": "4", "output_order": "natural",
                  "stability_class": "stable"}
        rows = [
            {**common, "implementation": "cuFFT", "median_kernel_ms": "3"},
            {**common, "implementation": "cuButterfly-base-radix2", "median_kernel_ms": "4"},
            {**common, "implementation": "cuButterfly-searched-radix4", "median_kernel_ms": "2"},
        ]
        result = THREE_WAY.build_main(rows, [])[0]
        self.assertEqual(result["searched_speedup_vs_base"], 2.0)
        self.assertEqual(result["searched_throughput_vs_library"], 1.5)
        self.assertEqual(result["searched_class_vs_library"], "faster")


if __name__ == "__main__":
    unittest.main()
