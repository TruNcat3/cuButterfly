#!/usr/bin/env python3
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_numeric_followups import evaluate_followups, summary as followup_summary
from analyze_numeric_boundary_ncu import mechanisms
from analyze_numeric_coverage import analyze as analyze_coverage, summarize as coverage_summary
from analyze_numeric_regime_cliffs import aggregate_samples, detect_events
from build_numeric_piecewise_selector import (apply_followup_boundaries, build_model,
                                               evaluate_leave_one_batch_out,
                                               measured_points, select)
from evaluate_numeric_regime_selector import evaluate
from generate_numeric_boundary_ncu import generate as generate_boundary_ncu
from generate_numeric_coverage_suite import adaptive_shapes, generate as generate_coverage_suite
from generate_numeric_followup_suite import select_followups
from generate_numeric_regime_suite import expand


class NumericRegimeStudyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT / "config" / "v100_numeric_regime_space.json").read_text())
        cls.suite = expand(cls.spec)

    def test_space_covers_numeric_contracts_and_matched_batches(self):
        contracts = {(case["precision"], case["accumulation"]) for case in self.suite["cases"]}
        self.assertEqual(contracts, {
            ("fp16", "native"), ("fp16", "fp32"), ("bf16", "native"),
            ("bf16", "fp32"), ("fp32", "native"), ("fp64", "native"),
            ("uint32", "native"), ("uint64", "native"),
        })
        self.assertGreater(len([case for case in self.suite["cases"] if case["tier"] == "quick"]), 500)
        groups = {}
        for case in self.suite["cases"]:
            groups.setdefault(case["group"], set()).add(case["mapping_id"])
            for field in ("storage_bits", "compute_bits", "accumulator_bits", "grid_waves",
                          "resident_ctas_per_sm", "limiting_resource", "working_set_over_l2"):
                self.assertIn(field, case)
        self.assertTrue(all(len(mappings) >= 2 for mappings in groups.values()))

    def test_every_cell_has_explicit_baseline_status(self):
        self.assertEqual(len(self.suite["cells"]), 185)
        for cell in self.suite["cells"]:
            self.assertTrue(cell["baseline_coverage"])
            self.assertTrue(all(item.get("status") for item in cell["baseline_coverage"]))
        bf16_fft = next(cell for cell in self.suite["cells"]
                        if cell["id"] == "fft-bf16-native-log8")
        self.assertEqual(bf16_fft["baseline_coverage"][0]["status"], "unsupported-hardware")

    def test_confirmed_winner_crossover_requires_trial_separation(self):
        cases = [case for case in self.suite["cases"]
                 if case["operator"] == "fwht" and case["precision"] == "fp32"
                 and case["logN"] == 8 and case["tier"] == "quick"]
        batches = sorted({case["batch"] for case in cases})[:2]
        selected = [case for case in cases if case["batch"] in batches]
        raw = []
        for case in selected:
            first = case["mapping_id"] == "temporal-r2-t128"
            r2_wins = case["batch"] == batches[0]
            center = 1.0 if first == r2_wins else 1.2
            for trial, delta in enumerate((-0.005, 0.0, 0.005), 1):
                raw.append({"suite_case_id": case["id"], "preflight_correct": "1",
                            "trial": str(trial), "kernel_ms": str(center + delta)})
        rows = aggregate_samples(raw, self.suite)
        events, _ = detect_events(rows)
        crossovers = [event for event in events if event["event_type"] == "winner-crossover"]
        self.assertTrue(any(event["classification"] == "confirmed" for event in crossovers))

    def test_complete_regime_evaluation_can_gate_promotion(self):
        rows = []
        contracts = (("fp16", "native", 16, 16), ("fp32", "native", 32, 32))
        operators = (("fft", "floating-complex", "per-stage-twiddle"),
                     ("fwht", "floating-add", "implicit-sign"))
        for operator, pipeline, coefficient in operators:
            for precision, accumulation, storage, compute in contracts:
                for batch, waves in ((1, 0.25), (80, 1.0), (320, 4.0)):
                    for mapping, latency in (("mapping-a", 1.0), ("mapping-b", 1.2)):
                        rows.append({
                            "operator": operator, "precision": precision, "accumulation": accumulation,
                            "logN": 8, "batch": batch, "mapping_id": mapping, "median_ms": latency * batch,
                            "storage_bits": storage, "compute_bits": compute, "accumulator_bits": compute,
                            "grid_waves": waves, "working_set_over_l2": batch / 100.0,
                            "arithmetic_pipeline": pipeline, "coefficient_policy": coefficient,
                            "limiting_resource": "thread",
                        })
        _, metrics = evaluate(rows)
        self.assertEqual(metrics["runtime_selector_status"], "promotable")
        self.assertEqual(metrics["topk_recall"], 1.0)

    def test_confirmed_crossovers_generate_matched_full_followups(self):
        import csv
        with (ROOT / "results" / "v100_numeric_regime_events.csv").open(newline="") as source:
            events = list(csv.DictReader(source))
        followups = select_followups(self.suite, events)
        self.assertEqual(len(followups["source_events"]), 29)
        self.assertEqual(len(followups["cases"]), 149)
        self.assertTrue(all(case["tier"] == "full" for case in followups["cases"]))
        self.assertTrue(all(case["followup_reasons"] for case in followups["cases"]))
        mappings_by_reason = {}
        for case in followups["cases"]:
            for reason in case["followup_reasons"]:
                mappings_by_reason.setdefault(reason.split(":", 1)[0], set()).add(case["mapping_id"])
        self.assertTrue(all(len(mappings) == 2 for mappings in mappings_by_reason.values()))

    def test_full_followups_classify_quick_crossovers(self):
        manifest = json.loads((ROOT / "results" / "v100_numeric_confirmed_followup_suite.json").read_text())
        import csv
        with (ROOT / "results" / "v100_numeric_confirmed_followup_raw.csv").open(newline="") as source:
            raw = list(csv.DictReader(source))
        rows = aggregate_samples(raw, manifest)
        report = followup_summary(evaluate_followups(rows, manifest))
        self.assertEqual(report["events"], 29)
        self.assertEqual(report["status_counts"], {
            "confirmed-same-direction": 23,
            "confirmed-reversed": 3,
            "not-confirmed": 3,
        })

    def test_unstable_boundaries_generate_paired_ncu_profiles(self):
        manifest = json.loads((ROOT / "results" / "v100_numeric_confirmed_followup_suite.json").read_text())
        import csv
        with (ROOT / "results" / "v100_numeric_confirmed_followup_analysis.csv").open(newline="") as source:
            analysis = list(csv.DictReader(source))
        rendered, events, profiles = generate_boundary_ncu(manifest, analysis)
        self.assertEqual((events, profiles), (6, 12))
        self.assertEqual(sum(line.startswith("profile crossover_") for line in rendered.splitlines()), 12)
        self.assertIn('"$NTT_BIN"', rendered)
        self.assertIn('"$BUTTERFLY_BIN"', rendered)

    def test_ncu_mechanism_screen_separates_resource_and_work_effects(self):
        base = {
            "registers_per_thread": 32.0, "shared_mem_bytes": 1024.0,
            "occupancy_register_block_limit": 8.0, "occupancy_shared_block_limit": 8.0,
            "resident_cta_limit": 8.0,
            "waves_per_sm": 1.0, "barrier_stall_pct": 2.0,
            "long_scoreboard_stall_pct": 3.0, "warp_instructions": 100.0,
            "dram_total_mib": 10.0, "shared_bank_conflicts": 0.0,
        }
        changed = {**base, "registers_per_thread": 64.0, "occupancy_register_block_limit": 4.0,
                   "resident_cta_limit": 4.0,
                   "waves_per_sm": 1.5, "barrier_stall_pct": 12.0,
                   "warp_instructions": 125.0, "shared_bank_conflicts": 100.0}
        labels = mechanisms(base, changed).split("+")
        self.assertTrue({"resource-capacity", "grid-wave", "synchronization",
                         "instruction-work", "shared-conflict"}.issubset(labels))

    def test_piecewise_selector_abstains_at_mapping_boundaries(self):
        rows = []
        for batch, winner in ((1, "a"), (4, "a"), (16, "b"), (64, "b")):
            for mapping in ("a", "b"):
                latency = 1.0 if mapping == winner else 1.3
                rows.append({
                    "operator": "fft", "precision": "fp32", "accumulation": "native",
                    "logN": 8, "batch": batch, "mapping_id": mapping,
                    "median_ms": latency, "trial_min_ms": latency * 0.99,
                    "trial_max_ms": latency * 1.01,
                })
        model = build_model(measured_points(rows))
        intervals = model["contracts"][0]["intervals"]
        self.assertEqual([interval["decision"] for interval in intervals],
                         ["auto-select", "measurement-required", "auto-select"])
        self.assertEqual(select(model, "fft", "fp32", "native", 8, 2)["mapping"], "a")
        self.assertEqual(select(model, "fft", "fp32", "native", 8, 8)["decision"],
                         "measurement-required")
        self.assertEqual(select(model, "fft", "fp64", "native", 8, 2)["reason"],
                         "unseen numeric contract or length")

        boundary_points = apply_followup_boundaries(measured_points(rows), [{
            "operator": "fft", "precision": "fp32", "accumulation": "native",
            "logN": 8, "batch": 1, "status": "not-confirmed",
        }])
        boundary_model = build_model(boundary_points)
        self.assertEqual(select(boundary_model, "fft", "fp32", "native", 8, 1)["decision"],
                         "measurement-required")

    def test_piecewise_selector_reports_conditional_coverage_and_regret(self):
        rows = []
        for batch in (1, 4, 16):
            for mapping, latency in (("a", 1.0), ("b", 1.25)):
                rows.append({
                    "operator": "fwht", "precision": "fp32", "accumulation": "native",
                    "logN": 8, "batch": batch, "mapping_id": mapping,
                    "median_ms": latency, "trial_min_ms": latency * 0.99,
                    "trial_max_ms": latency * 1.01,
                })
        _, metrics = evaluate_leave_one_batch_out(measured_points(rows))
        self.assertEqual(metrics["auto_selected_shapes"], 1)
        self.assertEqual(metrics["calibrated_region_status"], "validated")
        self.assertEqual(metrics["global_runtime_status"], "measurement-required")

    def test_coverage_suite_excludes_crossovers_and_stable_anchors(self):
        cases = [case for case in self.suite["cases"]
                 if case["operator"] == "fwht" and case["precision"] == "fp32"
                 and case["logN"] == 8 and case["tier"] == "quick"]
        batches = sorted({case["batch"] for case in cases})[:4]
        rows = []
        winners = ("temporal-r2-t128", "temporal-r2-t128",
                   "temporal-r2-t128", "temporal-r4-t128")
        for batch, winner in zip(batches, winners):
            for case in (item for item in cases if item["batch"] == batch):
                latency = 1.0 if case["mapping_id"] == winner else 1.08
                rows.append({**case, "median_ms": latency, "trial_min_ms": latency * 0.99,
                             "trial_max_ms": latency * 1.01})
        points = measured_points(rows)
        selected = adaptive_shapes(points)
        self.assertEqual({point["batch"] for point in selected}, set(batches[:2]))
        document = generate_coverage_suite(self.suite, points)
        self.assertTrue(document["cases"])
        self.assertTrue(all(case["tier"] == "full" for case in document["cases"]))

    def test_coverage_analysis_separates_stable_and_near_tie_results(self):
        manifest = {"source_shapes": [{
            "operator": "fwht", "precision": "fp32", "accumulation": "native",
            "logN": 8, "batch": 1, "quick_winner": "a", "quick_margin": 0.08,
        }]}
        rows = []
        for mapping, latency in (("a", 1.0), ("b", 1.02)):
            rows.append({"operator": "fwht", "precision": "fp32", "accumulation": "native",
                         "logN": 8, "batch": 1, "mapping_id": mapping,
                         "median_ms": latency, "trial_min_ms": latency * 0.99,
                         "trial_max_ms": latency * 1.01})
        report = coverage_summary(analyze_coverage(rows, manifest))
        self.assertEqual(report["status_counts"], {"near-tie": 1})


if __name__ == "__main__":
    unittest.main()
