import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "portable_crypto_composition", ROOT / "scripts/run_crypto_composition.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeProcess:
    pid = 4321
    returncode = 0

    def __init__(self, output):
        self.output = output
        self.terminated = False

    def poll(self):
        return 0

    def communicate(self, **kwargs):
        return self.output, ""

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


class CryptoCompositionTests(unittest.TestCase):
    def workload(self):
        return {
            "id": "composition-example", "logN": 8, "batch": 3, "word_bits": 32,
            "mode": "coset", "direction": "inverse",
            "moduli": ["2013265921", "2130706433"], "coset_generator": 7,
        }

    def protocol(self):
        return {"trials": 3, "warmup": 100, "repeat": 100,
                "patterns": ["random", "boundary"], "seed": 20260920}

    def sample(self, workload=None, mapping=None, pattern="boundary", trial=1,
               gpu_uuid="GPU-test", compile_mode="research"):
        workload = workload or self.workload()
        return {
            "logN": workload["logN"], "batch": workload["batch"],
            "word_bits": workload["word_bits"], "mode": workload["mode"],
            "direction": workload["direction"], "moduli": [int(x) for x in workload["moduli"]],
            "coset_generator": workload.get("coset_generator", 7), "gpu_uuid": gpu_uuid,
            "compile_mode": compile_mode, "correct": True,
            "verified_batches": workload["batch"], "verified_channels": len(workload["moduli"]),
            "warmup": 100, "repeat": 100, "kernel_ms": 0.25,
            "input_pattern": pattern, "seed": 20260920 + trial,
            "normalization": "inverse", "layout": "channel-batch-coefficient",
            "rns_stream_order": "single-stream", "mapping_request": mapping,
        }

    def make_build(self, directory):
        build = Path(directory) / "build"
        build.mkdir()
        (build / "CMakeCache.txt").write_text("CACHE\n")
        (build / "Makefile").write_text("all:\n")
        (build / "crypto_ntt_bench").write_text("binary\n")
        (build / "cuntt_bench").write_text("cuntt\n")
        return build

    def test_preflight_covers_two_words_modes_directions_and_patterns(self):
        rows = runner.preflight_workloads([
            {"word_bits": 32, "moduli": ["17", "97"]},
            {"word_bits": 64, "moduli": ["257", "577"]},
        ])
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["word_bits"] for row in rows}, {32, 64})
        self.assertEqual({row["mode"] for row in rows}, {"cyclic", "negacyclic", "coset"})
        self.assertEqual({row["direction"] for row in rows}, {"forward", "inverse"})
        self.assertTrue(all(row["batch"] == 3 and row["logN"] == 8 for row in rows))
        single = runner.preflight_workloads([{"word_bits": 32, "moduli": ["17"]}])
        self.assertTrue(all(len(row["moduli"]) == 1 for row in single))

    def test_sample_validation_covers_semantics_and_mapping(self):
        workload = self.workload()
        mapping = [{"backend": "shared-iterative"}, None]
        runner.validate_sample(self.sample(workload, mapping=mapping), workload, "GPU-test",
                               "boundary", 1, self.protocol(), mapping)
        with self.assertRaisesRegex(runner.CompositionError, "mapping request"):
            runner.validate_sample(self.sample(workload, mapping=None), workload, "GPU-test",
                                   "boundary", 1, self.protocol(), mapping)
        invalid = self.sample(workload, mapping=mapping)
        invalid["kernel_ms"] = float("inf")
        with self.assertRaisesRegex(runner.CompositionError, "timing"):
            runner.validate_sample(invalid, workload, "GPU-test", "boundary", 1,
                                   self.protocol(), mapping)

    def test_mapping_source_requires_same_uuid_and_build_and_exact_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = self.make_build(root)
            source = root / "acceptance.json"
            workload = self.workload()
            modulus = workload["moduli"][0]
            cyclic = runner._cyclic_workload(workload, modulus)
            source.write_text(json.dumps({
                "identity": {
                    "device": {"uuid": "GPU-test"},
                    "compile_mode": "research",
                    "binaries": {"cuntt_bench": runner.sha256(build / "cuntt_bench")},
                    "protocol": {"trials": 3, "warmup": 100, "repeat": 100},
                },
                "cells": [{
                    "id": runner.cyclic_cell_key(workload, modulus), "workload": cyclic,
                    "status": "complete", "selected": {"samples": [{
                        "correct": "1", "mapping_json": '{"backend":"shared-iterative"}'
                    }]},
                }],
            }))
            portfolios = runner.mapping_portfolios(workload, "GPU-test", self.protocol(),
                                                   build, [source])
            self.assertEqual(len(portfolios), 2)
            self.assertEqual(portfolios[1]["mappings"], [{"backend": "shared-iterative"}, None])
            rejected = runner.mapping_portfolios(workload, "GPU-other", self.protocol(), build, [source])
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["name"], "public-default")

    def test_mapping_replay_accepts_real_acceptance_cell_schema_without_timing(self):
        source_path = ROOT / "results/paper_experiments_20260917/a100-40gb/acceptance/acceptance.json"
        if not source_path.exists():
            self.skipTest("repository snapshot does not contain the historical acceptance journal")
        source_document = json.loads(source_path.read_text())
        cell = next((row for row in source_document["cells"]
                     if row.get("status") == "complete"
                     and row.get("workload", {}).get("operator") == "ntt"), None)
        if cell is None:
            self.skipTest("repository snapshot has no completed NTT acceptance cell")
        selected = cell["selected"]
        selected_sample = selected.get("samples", [selected])[0]
        mapping = selected_sample.get("mapping_json")
        if mapping is None:
            self.skipTest("historical selected cell has no serialized mapping")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = self.make_build(root)
            source = root / "acceptance.json"
            identity = source_document["identity"]
            source_workload = dict(cell["workload"])
            workload = {
                "id": "real-schema-cell", "logN": source_workload["logN"],
                "batch": source_workload["batch"], "word_bits": 64,
                "mode": "cyclic", "direction": source_workload.get("direction", "forward"),
                "moduli": [str(source_workload["modulus"]), str(int(source_workload["modulus"]) + 2)],
            }
            source_payload = {
                "identity": {
                    "device": {"uuid": "GPU-test"},
                    "compile_mode": identity["compile_mode"],
                    "binaries": {"cuntt_bench": runner.sha256(build / "cuntt_bench")},
                    "protocol": {"trials": identity["protocol"]["trials"],
                                 "warmup": identity["protocol"]["warmup"],
                                 "repeat": identity["protocol"]["repeat"]},
                },
                # Deliberately retain only selected mapping/correctness.  The
                # historical timing rows never enter the portable result.
                "cells": [{"id": runner.cyclic_cell_key(workload, workload["moduli"][0]),
                            "workload": runner._cyclic_workload(workload, workload["moduli"][0]),
                            "status": "complete",
                            "selected": {"samples": [{"correct": "1", "mapping_json": mapping}]}}],
            }
            source.write_text(json.dumps(source_payload))
            portfolios = runner.mapping_portfolios(workload, "GPU-test", self.protocol(), build, [source])
            self.assertEqual(len(portfolios), 2)
            self.assertEqual(portfolios[1]["mappings"][0], json.loads(mapping))
            self.assertIsNone(portfolios[1]["mappings"][1])
            self.assertNotIn("kernel_ms", portfolios[1]["provenance"][0])

    def test_template_and_build_fingerprints_ignore_docs_results_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include").mkdir(parents=True)
            (root / "src").mkdir()
            (root / "scripts").mkdir()
            (root / "external/mathdx/nvidia/mathdx/include").mkdir(parents=True)
            for relative in ("scripts/compile_module.py", "scripts/research_compile_requests.py",
                             "scripts/resident_mapping.py"):
                (root / relative).write_text(relative)
            (root / "include/runtime.hpp").write_text("runtime")
            (root / "src/runtime.cuh").write_text("kernel")
            (root / "external/mathdx/nvidia/mathdx/include/cufftdx.hpp").write_text("mathdx")
            (root / "docs").mkdir()
            (root / "docs/paper.md").write_text("draft")
            (root / "results").mkdir()
            (root / "results/noise.json").write_text("noise")
            template_before = runner.template_fingerprint(root)
            (root / "docs/paper.md").write_text("rewritten")
            (root / "results/noise.json").write_text("new noise")
            self.assertEqual(runner.template_fingerprint(root), template_before)
            (root / "src/runtime.cuh").write_text("kernel changed")
            self.assertNotEqual(runner.template_fingerprint(root), template_before)

            build = root / "build"
            build.mkdir()
            for name in ("CMakeCache.txt", "Makefile"):
                (build / name).write_text(name)
            binary = build / "crypto_ntt_bench"
            binary.write_text("binary")
            binary.chmod(binary.stat().st_mode | 0o100)
            (build / "compiler.log").write_text("log")
            (build / "crypto_manifest.json").write_text("metadata")
            build_before = runner.build_fingerprint(build)
            (build / "compiler.log").write_text("different log")
            (build / "crypto_manifest.json").write_text("different metadata")
            self.assertEqual(runner.build_fingerprint(build), build_before)
            binary.write_text("binary changed")
            self.assertNotEqual(runner.build_fingerprint(build), build_before)

    def test_manifest_resume_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = self.make_build(root)
            binary = build / "crypto_ntt_bench"
            template = root / "templates"
            template.mkdir()
            (template / "kernel.cuh").write_text("template\n")
            workloads = root / "workloads.json"
            workloads.write_text(json.dumps({"workloads": [self.workload()]}))

            class Args:
                pass

            args = Args()
            args.binary = binary
            args.build_dir = build
            args.template_root = template
            args.workloads = workloads
            args.mapping_sources = []
            args.gpu_uuid = "GPU-test"
            args.compile_mode = "research"
            args.trials = 3
            args.warmup = 100
            args.repeat = 100
            args.patterns = ("random", "boundary")
            args.seed = 20260920
            document, rows = runner._workloads_document(workloads)
            manifest = runner._manifest(args, document, rows)
            changed = json.loads(json.dumps(manifest))
            changed["gpu_uuid"] = "GPU-other"
            with self.assertRaisesRegex(ValueError, "identity/protocol changed"):
                runner._validate_manifest(manifest, changed)

    def test_measure_rejects_foreign_client_before_launch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "foreign_clients", return_value=[{"pid": 99}]), \
                patch.object(runner.subprocess, "Popen") as popen:
            with self.assertRaises(runner.ExclusiveViolation):
                runner.measure(["bench"], {}, "GPU-test", Path(directory) / "log.json")
            popen.assert_not_called()

    def test_measure_preserves_json_output_after_immediate_child_exit(self):
        output = json.dumps({"correct": True, "kernel_ms": 0.1})
        process = FakeProcess(output)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "foreign_clients", return_value=[]), \
                patch.object(runner.subprocess, "Popen", return_value=process):
            result = runner.measure(["bench"], {}, "GPU-test", Path(directory) / "log.json")
            self.assertEqual(result["kernel_ms"], 0.1)
            self.assertEqual(json.loads((Path(directory) / "log.json").read_text())["returncode"], 0)

    def test_failure_is_recorded_as_interruption_not_missing(self):
        document = {"failures": []}
        workload = self.workload()
        portfolio = {"name": "public-default"}
        runner._record_failure(document, workload, portfolio, "random", 0,
                               runner.ExclusiveViolation("mixed"))
        self.assertEqual(document["failures"][0]["status"], "interrupted")
        self.assertEqual(document["failures"][0]["workload_id"], workload["id"])


if __name__ == "__main__":
    unittest.main()
