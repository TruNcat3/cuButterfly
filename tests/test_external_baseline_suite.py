#!/usr/bin/env python3
import csv
import importlib.util
import io
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load("run_external_baseline_suite")
SUMMARY = load("summarize_external_baseline_suite")


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


if __name__ == "__main__":
    unittest.main()
