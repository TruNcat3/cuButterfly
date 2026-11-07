from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_crypto_composition_refresh as refresh
import run_crypto_composition as runner
from calibration_space import command_for


class CryptoRefreshTests(unittest.TestCase):
    def test_calibration_population_preserves_all_composition_channels(self):
        _, rows = runner._workloads_document(ROOT / "config/research_campaign/composed.json")
        calibration = refresh.calibration_workloads(rows, {"timing_mode": "graph", "warmup": 10,
                                                          "graph_replays": 7, "calls_per_graph": 20})
        self.assertEqual(len(calibration["workloads"]), 90)
        self.assertEqual(len(calibration["contracts"]), 90)
        self.assertEqual(len(calibration["composition_channels"]), 138)
        contracts = {item["id"]: item["semantics"] for item in calibration["contracts"]}
        for row, mapping in zip(rows, calibration["composition_channels"]):
            self.assertEqual(mapping["id"], row["id"])
            self.assertEqual(len(mapping["channels"]), len(row["moduli"]))
            for index, channel in enumerate(mapping["channels"]):
                self.assertEqual(channel["channel_index"], index)
                expected = runner._cyclic_workload(row, row["moduli"][index])
                self.assertEqual(contracts[channel["cyclic_contract_id"]], expected)
                self.assertEqual(channel["cyclic_semantics"], expected)
                argv = command_for("cuntt_bench", expected)
                self.assertEqual(argv[argv.index("--word-bits") + 1], str(row["word_bits"]))
                self.assertEqual(argv[argv.index("--batch") + 1], str(row["batch"]))
                self.assertEqual(argv[argv.index("--modulus") + 1], str(row["moduli"][index]))
                self.assertEqual("--inverse" in argv, row.get("direction", "forward") == "inverse")
        self.assertIsNone(calibration["execution"]["next_gpu_argv"])
        self.assertEqual(calibration["execution"]["status"], "blocked-interface-gap")

    def test_population_and_graph_argv(self):
        _, rows = runner._workloads_document(ROOT / "config/research_campaign/composed.json")
        self.assertEqual(len(rows), 138)
        protocol = dict(timing_mode="graph", warmup=10, repeat=20,
                        graph_replays=7, calls_per_graph=20, seed=1)
        argv = runner._command(Path("consumer"), rows[0], "random", 0, protocol, None)
        self.assertEqual(argv[argv.index("--timing-mode") + 1], "graph")
        self.assertEqual(argv[argv.index("--graph-replays") + 1], "7")
        self.assertEqual(argv[argv.index("--calls-per-graph") + 1], "20")
        self.assertEqual(argv[argv.index("--batch") + 1], str(rows[0]["batch"]))

    def test_graph_cannot_accept_event_or_missing_replays(self):
        protocol = dict(timing_mode="graph", warmup=10, repeat=20, trials=1,
                        graph_replays=7, calls_per_graph=20, seed=1, patterns=["random"])
        row = dict(logN=8, batch=1, word_bits=32, mode="cyclic", direction="forward", moduli=[17])
        sample = dict(logN=8, batch=1, word_bits=32, mode="cyclic", direction="forward", moduli=[17],
                      gpu_uuid="GPU-test", compile_mode="research", correct=True, verified_batches=1,
                      verified_channels=1, warmup=10, repeat=20, input_pattern="random", seed=1,
                      normalization="none", layout="channel-batch-coefficient", rns_stream_order="single-stream",
                      mapping_request=None, kernel_ms=1, timing_mode="graph", graph_replays=7,
                      calls_per_graph=20, graph_replay_call_ms=[1] * 7, graph_input_restore_verified=True)
        runner.validate_sample(sample, row, "GPU-test", "random", 0, protocol, None)
        runner.validate_sample({**sample, "graph_replays": 2, "graph_replay_call_ms": [1, 3], "kernel_ms": 2},
                               row, "GPU-test", "random", 0, {**protocol, "graph_replays": 2}, None)
        for field, value in (("timing_mode", "event"), ("graph_replay_call_ms", []),
                             ("graph_input_restore_verified", False)):
            invalid = {**sample, field: value}
            with self.assertRaises(runner.CompositionError):
                runner.validate_sample(invalid, row, "GPU-test", "random", 0, protocol, None)

    def test_incomplete_and_wrong_population_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            refresh.validate_results({}, {"status": "partial"}, [])
        manifest = dict(gpu_uuid="GPU-test", compile_mode="research")
        preparation = dict(manifest=manifest)
        document = dict(status="complete", failures=[], identity=runner._identity(manifest), cells=[])
        with self.assertRaisesRegex(ValueError, "population"):
            refresh.validate_results(preparation, document, [{"id": "missing"}])

    def test_prepared_run_rejects_changed_static_identity_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "binary"
            asset.write_text("current")
            runner.write_json(root / "preparation.json", {
                "manifest": {"gpu_uuid": "GPU-test"},
                "static_inputs": [{"path": str(asset), "sha256": "old"}]})
            with patch.object(refresh.subprocess, "run") as launch:
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    refresh.run_prepared(root, "GPU-test")
                launch.assert_not_called()

    def test_recovered_cyclic_report_is_explicit_and_searched_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovered = root / "recovery/results.json"
            with self.assertRaisesRegex(ValueError, "searched cohort"):
                refresh.run_prepared(root, "GPU-test", cyclic_report=recovered)
            runner.write_json(root / "preparation.json", {
                "manifest": {"gpu_uuid": "GPU-test"}, "static_inputs": [],
                "library_path": str(root / "libcuntt.so"),
                "jobs": [{"id": "composed-138-searched-graph"}]})
            import crypto_semantic_mapping_refresh as adapter
            with patch.object(adapter, "adapt", side_effect=ValueError("stop before GPU")) as adapt:
                with self.assertRaisesRegex(ValueError, "stop before GPU"):
                    refresh.run_prepared(root, "GPU-test", "searched", recovered)
                adapt.assert_called_once_with(root, recovered, root / "libcuntt.so", "GPU-test")

    def test_explicit_static_input_rebind_updates_only_changed_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "scripts/asset.py"
            asset.parent.mkdir()
            asset.write_text("current")
            runner.write_json(root / "preparation.json", {
                "manifest": {"gpu_uuid": "GPU-test"},
                "static_inputs": [{"path": str(asset), "sha256": "old"}],
                "jobs": [{"id": "frozen"}],
            })
            with patch.object(refresh, "ROOT", root):
                updated = refresh.rebind_static_inputs(root, reason="reviewed source update", source_paths=[asset])
            assert updated["jobs"] == [{"id": "frozen"}]
            assert updated["static_inputs"][0]["sha256"] == runner.sha256(asset)
            change = updated["provenance"]["static_input_rebindings"][-1]["changes"][0]
            assert change["old"] == "old"
            assert change["new"] == runner.sha256(asset)

    def test_static_input_rebind_rejects_missing_file_and_empty_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "missing"
            runner.write_json(root / "preparation.json", {
                "static_inputs": [{"path": str(asset), "sha256": "old"}]})
            with self.assertRaisesRegex(ValueError, "reason"):
                refresh.rebind_static_inputs(root, reason="", source_paths=[])
            with self.assertRaises(FileNotFoundError):
                refresh.rebind_static_inputs(root, reason="reviewed source update", source_paths=[])

    def test_static_rebind_rejects_data_and_unnamed_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scripts/helper.py"
            source.parent.mkdir()
            source.write_text("source")
            data = root / "inputs.json"
            data.write_text("data")
            runner.write_json(root / "preparation.json", {"static_inputs": [
                {"path": str(source), "sha256": "old"},
                {"path": str(data), "sha256": "old"}]})
            with patch.object(refresh, "ROOT", root):
                for allowed in ([source], [source, data], []):
                    with self.assertRaises(ValueError):
                        refresh.rebind_static_inputs(root, reason="reviewed", source_paths=allowed)

    def test_explicit_job_manifest_rebind_updates_named_inventory_and_archives_old(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workloads = root / "workloads.json"
            row = {"id": "one", "logN": 8, "batch": 1, "word_bits": 32,
                   "moduli": [17], "mode": "cyclic", "direction": "forward"}
            runner.write_json(workloads, {"workloads": [row]})
            argv = ["python", str(ROOT / "scripts/run_crypto_composition.py"),
                    "--binary", str(root / "consumer-build/crypto_ntt_bench"),
                    "--build-dir", str(root / "build"), "--workloads", str(workloads),
                    "--output-dir", str(root / "graph"), "--gpu-uuid", "GPU-test",
                    "--compile-mode", "research", "--timing-mode", "graph"]
            old = {"schema": "manifest", "gpu_uuid": "GPU-test", "compile_mode": "research",
                   "protocol": {"timing_mode": "graph"}, "workloads": {"count": 1},
                   "binary": {"name": "crypto_ntt_bench", "sha256": "same",
                              "dynamic_libraries": [{"name": "libcuntt.so", "sha256": "same"}],
                              "build": {"root": "build", "files": [{"path": "bridge.so", "sha256": "old"}]}}}
            old["binary"]["build"]["sha256"] = runner.json_sha256(old["binary"]["build"]["files"])
            current = copy.deepcopy(old)
            current["binary"]["build"]["files"][0]["sha256"] = "new"
            current["binary"]["build"]["sha256"] = runner.json_sha256(current["binary"]["build"]["files"])
            runner.write_json(root / "preparation.json", {
                "manifest": old,
                "jobs": [{"id": "composed-138-public-default-graph", "argv": argv}],
            })
            with patch.object(refresh.composition, "_manifest", return_value=current):
                updated = refresh.rebind_job_manifest(root, reason="reviewed bridge rebuild", build_files=["bridge.so"])
            assert updated["manifest"]["binary"] == current["binary"]
            assert updated["provenance"]["job_manifest_rebindings"][-1]["old_binary"] == old["binary"]
            assert (root / "recovery_manifest_rebinds/preparation.before.json").exists()
            for invalid in (dict(current, protocol={"changed": True}),
                            dict(current, binary=dict(current["binary"], sha256="changed")),
                            dict(current, binary=dict(current["binary"], dynamic_libraries=[]))):
                runner.write_json(root / "preparation.json", {"manifest": old,
                    "jobs": [{"id": "composed-138-public-default-graph", "argv": argv}]})
                with patch.object(refresh.composition, "_manifest", return_value=invalid):
                    with self.assertRaises(ValueError):
                        refresh.rebind_job_manifest(root, reason="reject", build_files=["bridge.so"])
            with patch.object(refresh.composition, "_manifest", return_value=current):
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    refresh.rebind_job_manifest(root, reason="reject", build_files=[])
                (root / "graph").mkdir()
                (root / "graph/results.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "result evidence"):
                    refresh.rebind_job_manifest(root, reason="reject", build_files=["bridge.so"])

    def test_job_manifest_rebind_rejects_non_binary_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workloads = root / "workloads.json"
            runner.write_json(workloads, {"workloads": [{"id": "one", "logN": 8, "batch": 1,
                "word_bits": 32, "moduli": [17], "mode": "cyclic", "direction": "forward"}]})
            argv = ["python", str(ROOT / "scripts/run_crypto_composition.py"), "--binary", "b",
                    "--build-dir", "b", "--workloads", str(workloads), "--output-dir", "o",
                    "--gpu-uuid", "GPU-test"]
            old = {"schema": "manifest", "gpu_uuid": "GPU-test", "compile_mode": "research",
                   "protocol": {}, "workloads": {}, "binary": {"name": "old"}}
            current = dict(old, protocol={"changed": True}, binary={"name": "new"})
            runner.write_json(root / "preparation.json", {"manifest": old,
                "jobs": [{"id": "composed-138-public-default-graph", "argv": argv}]})
            with patch.object(refresh.composition, "_manifest", return_value=current):
                with self.assertRaisesRegex(ValueError, "non-binary"):
                    refresh.rebind_job_manifest(root, reason="should reject", build_files=["bridge.so"])

    def test_zero_exit_with_missing_samples_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workloads = root / "workloads.json"
            row = {"id": "one", "logN": 8, "batch": 1, "word_bits": 32,
                   "moduli": [17], "mode": "cyclic", "direction": "forward"}
            runner.write_json(workloads, {"workloads": [row]})
            manifest = dict(gpu_uuid="GPU-test", compile_mode="research")
            result_path = root / "results.json"
            runner.write_json(result_path, dict(status="complete", failures=[], identity=runner._identity(manifest),
                                               cells=[dict(id="one", workload=row, status="complete", measurements=[])]))
            argv = ["python", "runner", "--binary", "fake", "--build-dir", "fake", "--workloads", str(workloads),
                    "--output-dir", str(root / "graph"), "--gpu-uuid", "GPU-test"]
            runner.write_json(root / "preparation.json", dict(manifest=manifest, static_inputs=[],
                protocol=dict(trials=1, patterns=["random"]),
                jobs=[dict(argv=argv, resume_argv=argv + ["--resume"], environment={}, result={"path": str(result_path)})]))
            with patch.object(runner, "_manifest", return_value=manifest), patch.object(refresh.subprocess, "run") as launch:
                launch.return_value.returncode = 0
                with self.assertRaisesRegex(ValueError, "samples missing"):
                    refresh.run_prepared(root, "GPU-test")


if __name__ == "__main__":
    unittest.main()
