import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("crypto_study_test", ROOT / "scripts/run_crypto_application_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class CryptoStudyTests(unittest.TestCase):
    def workload(self):
        return dict(id="example", mode="coset", logN=8, batch=7, word_bits=32,
                    moduli=["2013265921", "2130706433"], direction="inverse", coset_generator=7)

    def sample(self):
        return dict(logN=8, batch=7, word_bits=32, mode="coset", direction="inverse",
                    moduli=[2013265921, 2130706433], coset_generator=7, gpu_uuid="GPU-test",
                    compile_mode="research", correct=True, verified_batches=7, verified_channels=2,
                    warmup=100, repeat=100, kernel_ms=.1, input_pattern="boundary", seed=20260921,
                    normalization="inverse", layout="channel-batch-coefficient", rns_stream_order="single-stream")

    def test_command_preserves_rns_and_non_power_of_two_batch(self):
        cmd = study.bench_command("bench", self.workload(), pattern="boundary", trial=1,
                                  mapping=[None, {"backend": "shared-iterative"}])
        self.assertEqual(cmd[cmd.index("--batch") + 1], "7")
        self.assertEqual(cmd[cmd.index("--moduli") + 1], "2013265921,2130706433")
        self.assertEqual(cmd[cmd.index("--direction") + 1], "inverse")
        self.assertEqual(json.loads(cmd[cmd.index("--mapping-json") + 1]),
                         [None, {"backend": "shared-iterative"}])

    def test_exact_sample(self):
        study.validate_trial(self.sample(), self.workload(), "GPU-test", "boundary", 1)

    def test_rejects_semantic_hardware_and_correctness_changes(self):
        changes = dict(batch=14, word_bits=64, mode="cyclic", direction="forward", moduli=[2013265921],
                       gpu_uuid="GPU-other", compile_mode="auto", correct=False, verified_batches=1,
                       verified_channels=1, warmup=5, repeat=10, kernel_ms=float("nan"),
                       coset_generator=3, input_pattern="random", seed=123)
        for field, value in changes.items():
            with self.subTest(field=field):
                sample = {**self.sample(), field: value}
                with self.assertRaises(ValueError):
                    study.validate_trial(sample, self.workload(), "GPU-test", "boundary", 1)

    def test_waits_for_prior_matrix_even_if_gpu_has_no_clients(self):
        with patch.object(study, "manifest", return_value={"dependencies": ["/first", "/second"]}), \
             patch.object(study.shared, "parent_resolved", return_value=False), \
             patch.object(study.controller, "query_compute_clients") as query:
            self.assertIn("waiting for", study.dependencies_ready("/study", "a100-40gb")[0])
            query.assert_not_called()

    def test_waits_for_prior_cpu_scoring_lock(self):
        with patch.object(study, "manifest", return_value={"dependencies": ["/first"]}), \
             patch.object(study.shared, "parent_resolved", return_value=True), \
             patch.object(study.controller, "acquire_lock", side_effect=study.controller.LockBusy("held")):
            self.assertIn("owns GPU scheduling", study.dependencies_ready("/study", "a100-40gb")[0])

    def test_freeze_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            study.freeze(path, {"moduli": [17, 97]})
            study.freeze(path, {"moduli": [17, 97]})
            with self.assertRaises(ValueError):
                study.freeze(path, {"moduli": [17]})

    def test_measuring_busy_gpu_never_launches_process(self):
        with patch.object(study.controller, "query_compute_clients", return_value=["123, foreign, 1024"]), \
             patch.object(study.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "not exclusive"):
                study.measure(["bench"], {}, "GPU-test", Path("unused"))
            popen.assert_not_called()

    def test_portfolio_falls_back_without_claiming_search(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, "BASIS", Path(directory) / "first"), \
             patch.object(study, "BATCH_STUDY", ROOT / "results/batch_precision_20260920"):
            # The actual batch study contains no NTT exact-composition result;
            # isolate lookup without touching any running journal.
            with patch.object(study.Path, "exists", return_value=False):
                portfolios = study.portfolios(Path(directory), "a100-40gb", self.workload())
            self.assertEqual(len(portfolios), 1)
            self.assertEqual(portfolios[0]["name"], "public-default")

    def test_composed_resume_wrong_uuid_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = "a100-40gb"
            study.freeze(base / "crypto_manifest.json", {})
            study.freeze(base / target / "composed.json", {"identity": {"gpu_uuid": "GPU-wrong"}})
            with patch.object(study, "validate_inputs", return_value={"targets": {target: "GPU-test"}}):
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    study.run_composed(base, target, {})

    def test_resume_keeps_complete_trials_and_exact_input_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            base, target = Path(directory), "a100-40gb"
            study.freeze(base / "crypto_manifest.json", {})
            workload = self.workload()
            portfolio = [dict(name="public-default", mappings=None)]
            meta = {"targets": {target: "GPU-test"}, "binary": {"path": "bench"}}

            def measured(command, env, uuid, log):
                return {**self.sample(),
                        "input_pattern": command[command.index("--input-pattern") + 1],
                        "seed": int(command[command.index("--seed") + 1])}

            with patch.object(study, "validate_inputs", return_value=meta), \
                 patch.object(study, "composed_rows", return_value=[workload]), \
                 patch.object(study, "portfolios", return_value=portfolio), \
                 patch.object(study, "measure", side_effect=measured) as measure:
                study.run_composed(base, target, {})
                self.assertEqual(measure.call_count, 6)
                study.run_composed(base, target, {})
                self.assertEqual(measure.call_count, 6)
                document = study.read(base / target / "composed.json")
                self.assertEqual(document["status"], "complete")
                self.assertEqual(len(document["cells"][0]["measurements"]), 6)
                document["cells"][0]["measurements"][0]["sample"]["seed"] += 1
                study.controller.atomic_write_json(base / target / "composed.json", document)
                with self.assertRaisesRegex(ValueError, "pattern/seed"):
                    study.run_composed(base, target, {})

    def test_mapping_request_cannot_silently_fall_back(self):
        with self.assertRaisesRegex(ValueError, "mapping request"):
            study.validate_trial(self.sample(), self.workload(), "GPU-test", "boundary", 1,
                                 [{"backend": "shared-iterative"}, None])

    def test_gpu_preflight_separates_word_width_modes_and_directions(self):
        with patch.object(study, "read", return_value={"moduli": [
                {"bits": 23, "modulus": 8380417}, {"bits": 31, "modulus": 2013265921},
                {"bits": 40, "modulus": 549755904001}, {"bits": 62, "modulus": 2305843009213693953}]}):
            rows = study.preflight_workloads("/study")
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["word_bits"] for r in rows}, {32, 64})
        self.assertEqual({r["mode"] for r in rows}, {"cyclic", "negacyclic", "coset"})
        self.assertEqual({r["direction"] for r in rows}, {"forward", "inverse"})
        self.assertTrue(all(r["batch"] == 3 and len(r["moduli"]) == 2 for r in rows))


if __name__ == "__main__":
    unittest.main()
