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


RUNNER = load("run_fft_pipeline")
SUMMARY = load("summarize_fft_pipeline")


class FftPipelineRunnerTest(unittest.TestCase):
    def test_expansion_adds_one_cufft_per_shape(self):
        document = json.loads((ROOT / "config" / "v100_fft_pipeline_candidates.json").read_text())
        cases = RUNNER.expand(document)
        self.assertEqual(sum(case["reference"] for case in cases), 6)
        self.assertEqual(len(cases), 78)

    def test_parser_skips_notices(self):
        text = "notice\ndevice,operator,kernel_ms,correct\nV100,fft,0.2,1\n"
        self.assertEqual(RUNNER.parse_record(text)["kernel_ms"], "0.2")

    def test_summary_ranks_internal_and_compares_reference(self):
        header = "candidate_id,group,implementation,reference,kernel_ms,correct\n"
        text = header + "a,g,a,0,2.0,1\nb,g,b,0,1.0,1\nr,g,cuFFT,1,1.5,1\n"
        rows = list(csv.DictReader(io.StringIO(text)))
        summary = SUMMARY.summarize(rows)
        self.assertEqual(summary[0]["candidate_id"], "b")
        self.assertEqual(summary[0]["throughput_vs_cufft"], 1.5)


if __name__ == "__main__":
    unittest.main()
