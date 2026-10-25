import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "portable_crypto_external", ROOT / "scripts/run_crypto_external.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class CryptoExternalTests(unittest.TestCase):
    def workload(self, *, name="cell", log_n=12, batch=1, word_bits=32,
                 mode="cyclic", direction="forward", moduli=(17,)):
        return {
            "id": name,
            "logN": log_n,
            "batch": batch,
            "word_bits": word_bits,
            "mode": mode,
            "direction": direction,
            "moduli": [str(modulus) for modulus in moduli],
        }

    def source_cell(self, workload, modulus, *, status="complete", correct=True,
                    cell_workload=None, mapping=None):
        mapping = mapping if mapping is not None else {
            "backend": "shared-iterative", "threads_per_block": 128,
        }
        sample = {
            "correct": correct,
            "kernel_ms": 0.0001,
            "mapping_json": json.dumps(mapping, sort_keys=True),
        }
        source_workload = cell_workload or runner.composition._cyclic_workload(workload, modulus)
        return {
            "id": runner.composition.cyclic_cell_key(workload, modulus),
            "workload": source_workload,
            "status": status,
            "selected": {"samples": [sample]},
        }

    def test_capability_screening_accepts_limits_and_rejects_outside_them(self):
        valid32 = self.workload(log_n=1, batch=1, word_bits=32,
                                moduli=(2**30 - 1,))
        valid64 = self.workload(log_n=28, batch=65535, word_bits=64,
                                moduli=(2**62 - 1,))
        self.assertIsNone(runner.unavailable_reason(valid32))
        self.assertIsNone(runner.unavailable_reason(valid64))

        rejected = (
            (dict(logN=0), "logN 1..28"),
            (dict(logN=29), "logN 1..28"),
            (dict(batch=0), "batch exceeds"),
            (dict(batch=65536), "batch exceeds"),
            (dict(word_bits=16), "single-word 32/64"),
            (dict(word_bits=32, moduli=(2**30,)), "at most 30-bit"),
            (dict(word_bits=64, moduli=(2**62,)), "at most 62-bit"),
        )
        for changes, message in rejected:
            with self.subTest(changes=changes):
                candidate = dict(valid32, **changes)
                self.assertRegex(runner.unavailable_reason(candidate), message)

    def test_pilot_is_deterministic_semantic_coverage_not_performance_selection(self):
        workloads = [
            self.workload(name="small", log_n=1, batch=1, word_bits=32,
                          direction="forward", moduli=(2**30 - 1,)),
            self.workload(name="large", log_n=28, batch=65535, word_bits=64,
                          mode="negacyclic", direction="inverse",
                          moduli=(2**62 - 1, 2**61 - 1)),
            self.workload(name="coset", log_n=16, batch=8, word_bits=32,
                          mode="coset", direction="inverse", moduli=(17, 97)),
            self.workload(name="unsupported", log_n=0, batch=1, word_bits=32,
                          moduli=(17,)),
        ]
        baseline = runner.pilot_ids(workloads)
        supported = [workload for workload in workloads
                     if runner.unavailable_reason(workload) is None]
        covered = set().union(*(runner.features(workload)
                                for workload in supported if workload["id"] in baseline))
        expected = set().union(*(runner.features(workload) for workload in supported))

        self.assertNotIn("unsupported", baseline)
        self.assertEqual(covered, expected)
        self.assertEqual(baseline, runner.pilot_ids(list(reversed(workloads))))

        changed_performance = copy.deepcopy(workloads)
        for index, workload in enumerate(changed_performance):
            workload["speedup"] = 10_000.0 if index == 0 else 0.000001
        self.assertEqual(baseline, runner.pilot_ids(changed_performance))

    def paired_samples(self):
        workload = self.workload(moduli=(17, 97))
        channels = [
            {"modulus": 17, "root": 9, "psi": 3},
            {"modulus": 97, "root": 65, "psi": 22},
        ]
        return workload, {"channels": copy.deepcopy(channels)}, {"channels": copy.deepcopy(channels)}

    def test_validate_pair_requires_exact_per_channel_roots_and_workload_order(self):
        workload, ours, baseline = self.paired_samples()
        runner.validate_pair(ours, baseline, workload)

        mutations = (
            ("root", 10, "paired channel root differs"),
            ("psi", 4, "paired channel psi differs"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(baseline)
                changed["channels"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    runner.validate_pair(ours, changed, workload)

        with self.assertRaisesRegex(ValueError, r"paired channel (?:modulus|order) differs"):
            changed = copy.deepcopy(baseline)
            changed["channels"] = list(reversed(changed["channels"]))
            runner.validate_pair(ours, changed, workload)

        with self.assertRaisesRegex(ValueError, "missing per-channel"):
            runner.validate_pair(ours, {"channels": baseline["channels"][:1]}, workload)

        with self.assertRaisesRegex(ValueError, r"paired channel (?:modulus|order) differs"):
            changed = copy.deepcopy(baseline)
            changed["channels"][1]["modulus"] = 101
            runner.validate_pair(ours, changed, workload)

    def test_mapping_proposal_requires_complete_correct_exact_semantics_and_no_old_timing(self):
        workload = self.workload(moduli=(17, 97, 193, 257))
        exact_mapping = {"backend": "shared-iterative", "threads_per_block": 256}
        wrong_semantics = runner.composition._cyclic_workload(workload, 97)
        wrong_semantics["batch"] += 1
        source_cells = {
            runner.composition.cyclic_cell_key(workload, 17): self.source_cell(
                workload, 17, mapping=exact_mapping),
            runner.composition.cyclic_cell_key(workload, 97): self.source_cell(
                workload, 97, cell_workload=wrong_semantics),
            runner.composition.cyclic_cell_key(workload, 193): self.source_cell(
                workload, 193, status="pending"),
            runner.composition.cyclic_cell_key(workload, 257): self.source_cell(
                workload, 257, correct=False),
        }
        source = {
            "path": "/old/gpu/acceptance.json",
            "sha256": "source-sha",
            "uuid": "GPU-old-proposal-is-allowed",
            "cells": source_cells,
        }

        mapping, provenance = runner.mapping_proposal(workload, [source])

        self.assertEqual(mapping, [exact_mapping, None, None, None])
        self.assertEqual(provenance[0]["source_uuid"], "GPU-old-proposal-is-allowed")
        self.assertEqual(provenance[0]["reuse"],
                         "mapping only; all target timings and correctness measured afresh")
        self.assertNotIn("kernel_ms", provenance[0])
        self.assertEqual(provenance[1], {"kind": "public-default", "cell_id":
                                         runner.composition.cyclic_cell_key(workload, 97)})

    def test_summary_uses_only_complete_pairs_for_performance_groups(self):
        workload = self.workload(mode="negacyclic", direction="inverse", word_bits=64,
                                 moduli=(97, 193))
        cells = [
            {"workload": workload, "status": "complete", "speedup": 2.0},
            {"workload": workload, "status": "failed", "speedup": 100.0,
             "error": "incorrect output"},
            {"workload": workload, "status": "unavailable", "speedup": 200.0},
            {"workload": workload, "status": "incomplete", "speedup": 300.0},
            {"workload": workload, "status": "interrupted", "speedup": 400.0},
            {"workload": workload, "status": "blocked-baseline-correctness", "speedup": 500.0},
        ]
        document = {"identity": {"device": {"uuid": "GPU-test"}}, "cells": cells}

        summary = runner.summarize(document)

        self.assertEqual(summary["counts"], {
            "complete": 1, "failed": 1, "unavailable": 1,
            "incomplete": 1, "interrupted": 1,
            "blocked-baseline-correctness": 1,
        })
        self.assertEqual(summary["groups"]["all-complete"]["count"], 1)
        self.assertEqual(summary["groups"]["all-complete"]["geometric_mean"], 2.0)
        self.assertEqual(summary["groups"]["all-complete"]["wins"], 1)
        self.assertEqual(summary["failures"], [{"id": workload["id"],
                                                "reason": "incorrect output"}])


if __name__ == "__main__":
    unittest.main()
