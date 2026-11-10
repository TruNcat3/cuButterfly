#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_fft_pipeline_model", ROOT / "scripts" / "analyze_fft_pipeline_model.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FftPipelineModelTest(unittest.TestCase):
    def test_reports_recall_and_regret(self):
        rows = [
            {"group": "g", "candidate_id": "a", "static_score": "1", "median_kernel_ms": "2",
             "logN": "10", "batch": "4"},
            {"group": "g", "candidate_id": "b", "static_score": "2", "median_kernel_ms": "1",
             "logN": "10", "batch": "4"},
            {"group": "g", "candidate_id": "c", "static_score": "3", "median_kernel_ms": "3",
             "logN": "10", "batch": "4"},
        ]
        evaluations, metrics = MODULE.evaluate(rows)
        self.assertEqual(evaluations[0]["winner_static_rank"], 2)
        self.assertEqual(evaluations[0]["top1_regret"], 2.0)
        self.assertEqual(evaluations[0]["top3_regret"], 1.0)
        self.assertEqual(metrics["top1_exact_recall"], 0.0)
        self.assertEqual(metrics["top3_exact_recall"], 1.0)

    def test_rank_values_average_ties(self):
        self.assertEqual(MODULE.rank_values([2, 1, 1]), [3.0, 1.5, 1.5])

    def test_calibrated_selector_holds_out_target_batch(self):
        rows = []
        for batch, a_time, b_time in ((2, 1.0, 2.0), (4, 2.0, 3.0), (8, 4.0, 5.0)):
            for candidate, score, latency in (("a", 1, a_time), ("b", 2, b_time)):
                rows.append({
                    "group": f"g{batch}", "candidate_id": f"{candidate}{batch}",
                    "mapping_id": candidate, "static_score": str(score),
                    "median_kernel_ms": str(latency), "logN": "10", "batch": str(batch),
                })
        evaluations, metrics = MODULE.evaluate_calibrated(rows)
        self.assertEqual(len(evaluations), 3)
        self.assertTrue(all(row["top1_regret"] == 1.0 for row in evaluations))
        self.assertEqual(metrics["top1_exact_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
