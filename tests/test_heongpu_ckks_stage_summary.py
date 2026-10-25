import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "paper" / "experiments"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "run_heongpu_ckks_e2e" not in sys.modules:
    load_module("run_heongpu_ckks_e2e")
summary = load_module("summarize_heongpu_ckks_e2e")


def metadata(stage=True, rotate=False, auto=False):
    argv = ["benchmark", "--sizes", "4096", "--batches", "1,8", "--graph",
            "--repetitions", "3", "--ntt-execution", "native-rns"]
    if stage:
        argv.append("--stage-batch")
    if rotate:
        argv.append("--rotate")
    if auto:
        argv.append("--auto-schedule")
    return {"status": "completed", "returncode": 0, "performance_eligible": True,
            "CUDA_VISIBLE_DEVICES": "fixture-GPU", "benchmark_argv": argv}


def rows(stage=True, rotate=False, auto=False):
    result = []
    scenes = list(summary.SCENES) if rotate else list(summary.SCENES)[:3]
    comparisons = summary.AUTO_COMPARISONS if auto else summary.STAGE_COMPARISONS if stage else (summary.LOOP,)
    for scene in scenes:
        for batch in (1, 8):
            for index, (baseline, submission) in enumerate(comparisons):
                native, treatment = ((12, 6), (12, 3), (9, 3), (12, 3.3), (9, 3.3))[index]
                result.append({
                    "scene": scene, "N": "4096", "batch": str(batch),
                    "type": "NTT", "baseline": baseline, "batch_submission": submission,
                    "mode": "cuda_graph", "ntt_execution": "native-rns",
                    "native_ms": str(native), "cubutterfly_ms": str(treatment),
                    "speedup": str(native / treatment), "exact_ciphertext": "1",
                    "mismatch_count": "0", "maxabs_native": "0.001", "maxabs_cu": "0.001",
                    "relL2_native": "0.002", "relL2_cu": "0.002", "repetitions": "3",
                    "native_kernel_nodes": "20" if index in (0, 1, 3) else "15",
                    "cubutterfly_kernel_nodes": "18" if index == 0 else "13",
                    "replacement_calls": "6", "polynomials": "12",
                    "keyswitch_method": "I", "sec_level": "none", "Q_bits": "50:40:40",
                    "P_bits": "50", "scale": "1099511627776",
                    "graph_replays_per_sample": "1",
                })
    return result


def sidecars(data, meta):
    requests = sorted({(r["scene"], r["N"], r["batch"], r["keyswitch_method"]) for r in data})
    argv = meta["benchmark_argv"]
    rounds = int(summary.option(argv, "--schedule-rounds", "7"))
    selection_replays = summary.option(argv, "--schedule-replays", "5")
    count = int(summary.option(argv, "--repetitions", "10"))
    replays = summary.option(argv, "--graph-replays", "1")
    selection, samples = [], []
    for scene, n, batch, method in requests:
        common = dict(scene=scene, N=n, batch=batch, keyswitch_method=method)
        for candidate, median in (("loop", 2), ("stage_batch", 4)):
            for r in range(rounds):
                selection.append(dict(common, candidate=candidate, round=str(r), ms=str(median),
                                      median_ms=str(median), selected=str(int(candidate == "loop")),
                                      selection_wall_ms="100", replays=selection_replays,
                                      confidence="measured-full-graph-request"))
        for variant, mean in (("native_loop", 12), ("cu_loop", 6), ("native_stage_batch", 9),
                              ("cu_stage_batch", 3), ("cu_auto", 3.3)):
            for r in range(count):
                # Selection and holdout deliberately have different timings.
                value = mean + (r - (count - 1) / 2) * 0.1
                samples.append(dict(common, variant=variant, round=str(r), ms=str(value), replays=replays))
    return selection, samples


class StageSummaryTests(unittest.TestCase):
    def parse(self, data=None, meta=None):
        return summary.parse_rows(data if data is not None else rows(),
                                  meta if meta is not None else metadata(), "fixture.csv")

    def write_sidecars(self, source, selection, samples):
        for suffix, data in ((".selection.csv", selection), (".samples.csv", samples)):
            with Path(str(source) + suffix).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)

    def validate_auto(self, selection=None, samples=None, data=None, meta=None):
        data = rows(auto=True) if data is None else data
        meta = metadata(auto=True) if meta is None else meta
        defaults = sidecars(data, meta)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.csv"
            self.write_sidecars(source, defaults[0] if selection is None else selection,
                                defaults[1] if samples is None else samples)
            return summary.validate_selection_evidence(data, meta, source)

    def run_report(self, data, meta):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.csv"
            output = Path(directory) / "report.md"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
            source.with_suffix(".metadata.json").write_text(json.dumps(meta))
            if "--auto-schedule" in meta["benchmark_argv"]:
                self.write_sidecars(source, *sidecars(data, meta))
            with patch.object(sys, "argv", ["summary", str(source), "--output", str(output)]):
                status = summary.main()
            return status, output.read_text()

    def test_three_comparisons_complete_with_rotation(self):
        parsed = self.parse(rows(rotate=True), metadata(rotate=True))
        self.assertEqual(len(parsed), 24)
        self.assertEqual(len(summary.group_rows(parsed, by_scene=False)), 6)
        self.assertEqual(len(summary.group_rows(parsed, by_scene=True)), 24)

    def test_missing_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete matrix"):
            self.parse(rows()[:-1])

    def test_duplicate_cannot_mask_missing_identity(self):
        data = rows()
        data[-1] = copy.deepcopy(data[0])
        with self.assertRaisesRegex(ValueError, "duplicate matrix identity"):
            self.parse(data)

    def test_unexpected_identity_is_rejected(self):
        for field, value in (("scene", "other"), ("N", "8192"), ("batch", "2"),
                             ("baseline", "wrong"), ("batch_submission", "wrong")):
            with self.subTest(field=field):
                data = rows()
                data[0][field] = value
                with self.assertRaisesRegex(ValueError, "unexpected matrix identity"):
                    self.parse(data)

    def test_configuration_and_counts_are_checked(self):
        for field, value in (("mode", "eager"), ("ntt_execution", "grouped"),
                             ("keyswitch_method", "II"), ("P_bits", "50:50"),
                             ("repetitions", "2"), ("repetitions", "0"),
                             ("graph_replays_per_sample", "0")):
            with self.subTest(field=field, value=value):
                data = rows()
                data[0][field] = value
                with self.assertRaises(ValueError):
                    self.parse(data)

    def test_each_comparison_speedup_is_checked(self):
        for index in range(3):
            with self.subTest(index=index):
                data = rows()
                data[index]["speedup"] = "100"
                with self.assertRaisesRegex(ValueError, "speedup disagrees"):
                    self.parse(data)

    def test_nonfinite_or_nonpositive_times_are_rejected(self):
        for field, value in (("native_ms", "nan"), ("cubutterfly_ms", "0"),
                             ("speedup", "inf")):
            with self.subTest(field=field):
                data = rows()
                data[0][field] = value
                with self.assertRaises(ValueError):
                    self.parse(data)

    def test_shared_variant_measurements_are_checked(self):
        for index, field, value in ((1, "native_ms", "24"),
                                    (2, "cubutterfly_ms", "6"),
                                    (1, "native_kernel_nodes", "21"),
                                    (2, "polynomials", "13")):
            with self.subTest(field=field):
                data = rows()
                data[index][field] = value
                data[index]["speedup"] = str(float(data[index]["native_ms"]) /
                                             float(data[index]["cubutterfly_ms"]))
                with self.assertRaisesRegex(ValueError, "shared stage measurement"):
                    self.parse(data)

    def test_groups_keep_baselines_and_batch_ranges_separate(self):
        groups = summary.group_rows(self.parse(), by_scene=False)
        for (baseline, submission, batch_range, *_), ratios in groups.items():
            index = summary.STAGE_COMPARISONS.index((baseline, submission))
            self.assertIn(batch_range, ("B=1", "B>1"))
            self.assertEqual(ratios, [(2, 4, 3)[index]] * 3)

    def test_report_names_organizations_and_replays(self):
        data, meta = rows(), metadata()
        meta["benchmark_argv"] += ["--graph-replays", "4"]
        for row in data:
            row["graph_replays_per_sample"] = "4"
        status, report = self.run_report(data, meta)
        self.assertEqual(status, 0)
        self.assertIn("HEonGPU 跨密文阶段合批", report)
        self.assertIn("cuButterfly 跨密文阶段合批", report)
        self.assertIn("| 3 / 4 |", report)
        self.assertEqual(report.count("| 全部场景 |"), 6)
        self.assertIn("全部整数密文一致：True", report)

    def test_native_batch_correctness_control_affects_report_status(self):
        data = rows()
        data[2].update(exact_ciphertext="0", mismatch_count="1")
        status, report = self.run_report(data, metadata())
        self.assertEqual(status, 1)
        self.assertIn("全部整数密文一致：False", report)

    def test_legacy_optional_columns_keep_defaults(self):
        data, meta = rows(stage=False), metadata(stage=False)
        position = meta["benchmark_argv"].index("--ntt-execution")
        del meta["benchmark_argv"][position:position + 2]
        for row in data:
            for field in ("baseline", "batch_submission", "ntt_execution", "graph_replays_per_sample"):
                del row[field]
        parsed = self.parse(data, meta)
        self.assertTrue(all(row["ntt_execution"] == "grouped" for row in parsed))
        self.assertTrue(all(row["graph_replays_per_sample"] == "1" for row in parsed))
        status, report = self.run_report(data, meta)
        self.assertEqual(status, 0)
        self.assertIn("grouped", report)
        self.assertEqual(report.count("| 全部场景 |"), 2)

    def test_main_preserves_evidence_eligibility_check(self):
        meta = metadata()
        meta["performance_eligible"] = False
        with self.assertRaisesRegex(ValueError, "ineligible"):
            self.run_report(rows(), meta)

    def test_auto_pairs_and_scene_subset(self):
        meta = metadata(auto=True)
        meta["benchmark_argv"] += ["--scenes", "keyswitch,square_rescale_keyswitch"]
        data = [row for row in rows(auto=True) if row["scene"] != "square_relinearize_rescale"]
        self.assertEqual(len(self.parse(data, meta)), 20)
        self.assertEqual(len(self.validate_auto(data=data, meta=meta)), 4)
        with self.assertRaisesRegex(ValueError, "incomplete matrix"):
            self.parse(data[:-1], meta)
        for scenes in ("keyswitch,keyswitch", "unknown", "", "rotate_rows_shift1"):
            with self.subTest(scenes=scenes):
                wrong = copy.deepcopy(meta)
                wrong["benchmark_argv"][-1] = scenes
                with self.assertRaises(ValueError):
                    self.parse(data, wrong)
        wrong = metadata(stage=False, auto=True)
        with self.assertRaisesRegex(ValueError, "requires stage-batch"):
            self.parse(rows(auto=True), wrong)

    def test_independent_holdout_regret_and_report(self):
        results = self.validate_auto()
        self.assertEqual(len(results), 6)
        for result in results:
            self.assertEqual(result["selected"], "loop")
            self.assertEqual(result["selection_wall_ms"], 100)
            self.assertAlmostEqual(result["auto_ms"], 3.3)
            self.assertAlmostEqual(result["regret"], 0.1)
        status, report = self.run_report(rows(auto=True), metadata(auto=True))
        self.assertEqual(status, 0)
        self.assertIn("独立复测", report)
        self.assertIn("选拔墙钟", report)
        self.assertIn("10.000%", report)
        self.assertEqual(report.count("| 全部场景 |"), 10)

    def test_negative_holdout_regret_is_preserved(self):
        data = rows(auto=True)
        selection, samples = sidecars(data, metadata(auto=True))
        for sample in samples:
            if sample["variant"] == "cu_auto":
                sample["ms"] = "2.5"
        for row in data:
            if row["batch_submission"] == "measured_auto_loop_or_stage_batch":
                row["cubutterfly_ms"] = "2.5"
                row["speedup"] = str(float(row["native_ms"]) / 2.5)
        for result in self.validate_auto(selection, samples, data):
            self.assertLess(result["regret"], 0)

    def test_sidecar_identity_completeness_and_duplicates(self):
        defaults = sidecars(rows(auto=True), metadata(auto=True))
        for phase in (0, 1):
            for mutation, pattern in (("missing", "incomplete"), ("duplicate", "duplicate"),
                                      ("foreign", "unexpected")):
                with self.subTest(phase=phase, mutation=mutation):
                    evidence = copy.deepcopy(defaults)
                    if mutation == "missing":
                        evidence[phase].pop()
                    elif mutation == "duplicate":
                        evidence[phase][-1] = copy.deepcopy(evidence[phase][0])
                    else:
                        evidence[phase][0]["keyswitch_method"] = "II"
                    with self.assertRaisesRegex(ValueError, pattern):
                        self.validate_auto(*evidence)

    def test_selection_statistics_flags_and_confidence(self):
        for field, value, pattern in (("ms", "nan", "positive finite"), ("median_ms", "0", "positive finite"),
                                      ("median_ms", "3", "median disagrees"), ("selected", "2", "selected flags"),
                                      ("confidence", "cached", "confidence"), ("selection_wall_ms", "inf", "positive finite"),
                                      ("selection_wall_ms", "101", "wall time differs"), ("replays", "1", "replay budget")):
            with self.subTest(field=field):
                selection, samples = sidecars(rows(auto=True), metadata(auto=True))
                selection[0][field] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    self.validate_auto(selection, samples)
        selection, samples = sidecars(rows(auto=True), metadata(auto=True))
        for sample in selection:
            sample["selected"] = str(1 - int(sample["selected"]))
        with self.assertRaisesRegex(ValueError, "selected candidate disagrees"):
            self.validate_auto(selection, samples)

    def test_selection_tie_keeps_loop(self):
        selection, samples = sidecars(rows(auto=True), metadata(auto=True))
        for sample in selection:
            sample["ms"] = sample["median_ms"] = "2"
        self.assertTrue(all(result["selected"] == "loop" for result in self.validate_auto(selection, samples)))

    def test_holdout_samples_budget_and_main_means(self):
        for field, value, pattern in (("ms", "0", "positive finite"), ("ms", "inf", "positive finite"),
                                      ("ms", "100", "holdout mean"), ("replays", "5", "replay budget"),
                                      ("round", "3", "unexpected")):
            with self.subTest(field=field):
                selection, samples = sidecars(rows(auto=True), metadata(auto=True))
                samples[0][field] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    self.validate_auto(selection, samples)
        for variant in ("native_loop", "cu_loop", "native_stage_batch", "cu_stage_batch", "cu_auto"):
            with self.subTest(variant=variant):
                selection, samples = sidecars(rows(auto=True), metadata(auto=True))
                next(sample for sample in samples if sample["variant"] == variant)["ms"] = "100"
                with self.assertRaisesRegex(ValueError, "holdout mean"):
                    self.validate_auto(selection, samples)

    def test_distinct_selection_and_holdout_budgets(self):
        meta = metadata(auto=True)
        meta["benchmark_argv"] += ["--schedule-rounds", "2", "--schedule-replays", "4", "--graph-replays", "6"]
        data = rows(auto=True)
        for row in data:
            row["graph_replays_per_sample"] = "6"
        self.assertEqual(len(self.validate_auto(data=data, meta=meta)), 6)
        selection, samples = sidecars(data, meta)
        meta["benchmark_argv"][-5] = "0"
        with self.assertRaises(ValueError):
            self.validate_auto(selection, samples, data=data, meta=meta)

    def test_auto_requires_both_separate_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.csv"
            with self.assertRaises(FileNotFoundError):
                summary.validate_selection_evidence(rows(auto=True), metadata(auto=True), source)
            selection, samples = sidecars(rows(auto=True), metadata(auto=True))
            self.write_sidecars(source, selection, samples)
            Path(str(source) + ".samples.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                summary.validate_selection_evidence(rows(auto=True), metadata(auto=True), source)
            Path(str(source) + ".samples.csv").write_text(Path(str(source) + ".selection.csv").read_text())
            with self.assertRaisesRegex(ValueError, "schema"):
                summary.validate_selection_evidence(rows(auto=True), metadata(auto=True), source)
        self.assertEqual(summary.validate_selection_evidence(rows(), metadata(), "missing.csv"), [])

    def test_metadata_hashes_bind_both_phase_files(self):
        data, meta = rows(auto=True), metadata(auto=True)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.csv"
            self.write_sidecars(source, *sidecars(data, meta))
            meta["artifacts"] = {}
            for artifact, suffix in (("selection_samples", ".selection.csv"), ("holdout_samples", ".samples.csv")):
                phase = Path(str(source) + suffix)
                meta["artifacts"][artifact] = dict(sha256=hashlib.sha256(phase.read_bytes()).hexdigest(),
                                                   size_bytes=phase.stat().st_size)
            self.assertEqual(len(summary.validate_selection_evidence(data, meta, source)), 6)
            for artifact in ("selection_samples", "holdout_samples"):
                with self.subTest(artifact=artifact):
                    wrong = copy.deepcopy(meta)
                    wrong["artifacts"][artifact]["sha256"] = "0" * 64
                    with self.assertRaisesRegex(ValueError, "differs from metadata artifact"):
                        summary.validate_selection_evidence(data, wrong, source)


if __name__ == "__main__":
    unittest.main()
